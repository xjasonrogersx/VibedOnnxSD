from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPTokenizer

from .config import ConfigurationError, InferenceConfig, ModelPaths
from .scheduler import DDIMScheduler

LOGGER = logging.getLogger(__name__)
MINIMUM_MEMORY_HEADROOM_BYTES = 1_500_000_000


class OnnxStableDiffusionPipeline:
    def __init__(self, model_paths: ModelPaths) -> None:
        self.model_paths = model_paths
        LOGGER.info("Loading tokenizer from %s", model_paths.tokenizer_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(str(model_paths.tokenizer_dir))
        self.scheduler = DDIMScheduler.from_config_file(
            model_paths.scheduler_config_path
        )
        self.vae_scaling_factor = self._load_vae_scaling_factor(
            model_paths.vae_config_path
        )

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "OnnxStableDiffusionPipeline":
        return cls(ModelPaths.from_model_dir(model_dir))

    def generate(self, config: InferenceConfig) -> Path:
        config.validate()
        self._ensure_memory_budget()
        LOGGER.info("Encoding prompt text.")
        prompt_embeddings = self._encode_prompt(config.prompt, config.negative_prompt)

        LOGGER.info("Preparing scheduler and initial latents.")
        self.scheduler.set_timesteps(config.steps)
        latents = self._prepare_latents(
            height=config.height,
            width=config.width,
            seed=config.seed,
        )

        LOGGER.info("Loading UNet session.")
        unet = self._create_session(self.model_paths.unet_path)
        for index, timestep in enumerate(self.scheduler.timesteps):
            timestep_int = int(timestep)
            next_timestep = (
                int(self.scheduler.timesteps[index + 1])
                if index + 1 < len(self.scheduler.timesteps)
                else None
            )
            latent_model_input = np.concatenate([latents, latents], axis=0)
            noise_prediction = self._run_unet(
                unet,
                latent_model_input,
                timestep=timestep_int,
                encoder_hidden_states=prompt_embeddings,
            )
            noise_unconditional, noise_text = np.split(noise_prediction, 2, axis=0)
            guided_noise = noise_unconditional + config.guidance_scale * (
                noise_text - noise_unconditional
            )
            latents = self.scheduler.step(
                model_output=guided_noise,
                timestep=timestep_int,
                sample=latents,
                next_timestep=next_timestep,
            )
            if (
                index == 0
                or index == len(self.scheduler.timesteps) - 1
                or (index + 1) % max(1, config.steps // 5) == 0
            ):
                LOGGER.info(
                    "Denoising step %s/%s complete.",
                    index + 1,
                    len(self.scheduler.timesteps),
                )
        del unet
        gc.collect()

        LOGGER.info("Loading VAE decoder session.")
        LOGGER.info("Decoding image latents.")
        image = self._decode_latents(latents)
        output_path = config.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        LOGGER.info("Saved image to %s", output_path)
        return output_path

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> np.ndarray:
        tokens = self.tokenizer(
            [negative_prompt, prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )

        LOGGER.info("Loading text encoder session.")
        text_encoder = self._create_session(self.model_paths.text_encoder_path)
        try:
            ort_inputs: dict[str, np.ndarray] = {}
            for input_meta in text_encoder.get_inputs():
                if input_meta.name in tokens:
                    value = tokens[input_meta.name]
                elif "input_ids" in input_meta.name:
                    value = tokens["input_ids"]
                elif (
                    "attention_mask" in input_meta.name
                    and "attention_mask" in tokens
                ):
                    value = tokens["attention_mask"]
                else:
                    raise ConfigurationError(
                        f"Unsupported text encoder input: {input_meta.name}"
                    )
                ort_inputs[input_meta.name] = self._cast_input(value, input_meta.type)

            embeddings = text_encoder.run(None, ort_inputs)[0]
            return np.asarray(embeddings, dtype=np.float32)
        finally:
            del text_encoder
            gc.collect()

    def _run_unet(
        self,
        unet: ort.InferenceSession,
        sample: np.ndarray,
        timestep: int,
        encoder_hidden_states: np.ndarray,
    ) -> np.ndarray:
        ort_inputs: dict[str, np.ndarray] = {}
        for input_meta in unet.get_inputs():
            input_name = input_meta.name.lower()
            if "sample" in input_name:
                value = sample
            elif "timestep" in input_name:
                value = np.asarray([timestep], dtype=np.int64)
            elif "encoder_hidden_states" in input_name:
                value = encoder_hidden_states
            else:
                raise ConfigurationError(f"Unsupported UNet input: {input_meta.name}")
            ort_inputs[input_meta.name] = self._cast_input(value, input_meta.type)

        result = unet.run(None, ort_inputs)[0]
        return np.asarray(result, dtype=np.float32)

    def _decode_latents(self, latents: np.ndarray) -> Image.Image:
        scaled_latents = latents / self.vae_scaling_factor
        vae_decoder = self._create_session(self.model_paths.vae_decoder_path)
        ort_inputs: dict[str, np.ndarray] = {}
        for input_meta in vae_decoder.get_inputs():
            if "latent" not in input_meta.name.lower():
                raise ConfigurationError(
                    f"Unsupported VAE decoder input: {input_meta.name}"
                )
            ort_inputs[input_meta.name] = self._cast_input(
                scaled_latents, input_meta.type
            )

        decoded = vae_decoder.run(None, ort_inputs)[0]
        del vae_decoder
        gc.collect()
        image_tensor = np.asarray(decoded, dtype=np.float32)
        image_tensor = np.clip((image_tensor / 2.0) + 0.5, 0.0, 1.0)
        image_array = (image_tensor[0].transpose(1, 2, 0) * 255.0).round().astype(
            np.uint8
        )
        return Image.fromarray(image_array)

    @staticmethod
    def _create_session(path: Path) -> ort.InferenceSession:
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            return ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except (
            ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.InvalidGraph,
            ort.capi.onnxruntime_pybind11_state.NoSuchFile,
            ort.capi.onnxruntime_pybind11_state.RuntimeException,
        ) as error:
            raise ConfigurationError(
                f"Unable to load ONNX model at {path}: {error}"
            ) from error

    @staticmethod
    def _prepare_latents(height: int, width: int, seed: int) -> np.ndarray:
        latent_height = height // 8
        latent_width = width // 8
        generator = np.random.default_rng(seed)
        latents = generator.standard_normal(
            (1, 4, latent_height, latent_width),
            dtype=np.float32,
        )
        return latents.astype(np.float32, copy=False)

    @staticmethod
    def _cast_input(value: np.ndarray, ort_type: str) -> np.ndarray:
        if "int64" in ort_type:
            return np.asarray(value, dtype=np.int64)
        if "int32" in ort_type:
            return np.asarray(value, dtype=np.int32)
        if "float16" in ort_type:
            return np.asarray(value, dtype=np.float16)
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _load_vae_scaling_factor(config_path: Path | None) -> float:
        if config_path is None:
            return 0.18215
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return float(payload.get("scaling_factor", 0.18215))

    def _ensure_memory_budget(self) -> None:
        available_memory = self._read_available_memory_bytes()
        if available_memory is None:
            return

        estimated_required = self._estimate_required_memory_bytes()
        LOGGER.info(
            "Memory check: %.2f GiB available, %.2f GiB estimated required.",
            available_memory / (1024**3),
            estimated_required / (1024**3),
        )
        if available_memory < estimated_required:
            raise ConfigurationError(
                "Not enough memory to load the ONNX Stable Diffusion models on CPU. "
                f"Available: {available_memory / (1024**3):.2f} GiB; "
                f"estimated required: {estimated_required / (1024**3):.2f} GiB. "
                "Use a smaller ONNX model or run in an environment with more memory."
            )

    def _estimate_required_memory_bytes(self) -> int:
        model_files = (
            self.model_paths.text_encoder_path,
            self.model_paths.unet_path,
            self.model_paths.vae_decoder_path,
            self.model_paths.model_dir / "unet" / "weights.pb",
        )
        total_model_bytes = sum(
            path.stat().st_size for path in model_files if path.exists()
        )
        return total_model_bytes + MINIMUM_MEMORY_HEADROOM_BYTES

    @staticmethod
    def _read_available_memory_bytes() -> int | None:
        meminfo_path = Path("/proc/meminfo")
        if not meminfo_path.exists():
            return None

        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("MemAvailable:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                return None
            return int(parts[1]) * 1024
        return None
