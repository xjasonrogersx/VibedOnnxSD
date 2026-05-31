from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when the local model layout or CLI options are invalid."""


@dataclass(frozen=True)
class ModelPaths:
    model_dir: Path
    tokenizer_dir: Path
    text_encoder_path: Path
    unet_path: Path
    vae_decoder_path: Path
    scheduler_config_path: Path
    vae_config_path: Path | None

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "ModelPaths":
        root = Path(model_dir).expanduser().resolve()
        required_paths = {
            "model_dir": root,
            "tokenizer_dir": root / "tokenizer",
            "text_encoder_path": root / "text_encoder" / "model.onnx",
            "unet_path": root / "unet" / "model.onnx",
            "vae_decoder_path": root / "vae_decoder" / "model.onnx",
            "scheduler_config_path": root / "scheduler" / "scheduler_config.json",
        }

        missing = [
            name for name, path in required_paths.items() if not path.exists()
        ]
        if missing:
            details = ", ".join(
                f"{name}={required_paths[name]}" for name in missing
            )
            raise ConfigurationError(
                "Missing Stable Diffusion model assets: " + details
            )

        vae_config_path = root / "vae_decoder" / "config.json"
        return cls(
            model_dir=root,
            tokenizer_dir=required_paths["tokenizer_dir"],
            text_encoder_path=required_paths["text_encoder_path"],
            unet_path=required_paths["unet_path"],
            vae_decoder_path=required_paths["vae_decoder_path"],
            scheduler_config_path=required_paths["scheduler_config_path"],
            vae_config_path=vae_config_path if vae_config_path.exists() else None,
        )


@dataclass(frozen=True)
class InferenceConfig:
    prompt: str
    negative_prompt: str
    output_path: Path
    steps: int = 25
    guidance_scale: float = 7.5
    seed: int = 0
    height: int = 512
    width: int = 512

    def validate(self) -> "InferenceConfig":
        if not self.prompt.strip():
            raise ConfigurationError("Prompt must not be empty.")
        if self.steps <= 0:
            raise ConfigurationError("Inference steps must be positive.")
        if self.guidance_scale < 1:
            raise ConfigurationError("Guidance scale must be at least 1.0.")
        if self.height <= 0 or self.width <= 0:
            raise ConfigurationError("Image height and width must be positive.")
        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ConfigurationError("Image height and width must be divisible by 8.")
        if self.seed < 0:
            raise ConfigurationError("Seed must be zero or a positive integer.")
        return self
