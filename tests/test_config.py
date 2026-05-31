from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vibed_onnx_sd.config import ConfigurationError, InferenceConfig, ModelPaths
from vibed_onnx_sd.pipeline import OnnxStableDiffusionPipeline


class ModelPathsTests(unittest.TestCase):
    def test_from_model_dir_requires_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tokenizer").mkdir()
            (root / "text_encoder").mkdir()
            (root / "unet").mkdir()
            (root / "vae_decoder").mkdir()
            (root / "scheduler").mkdir()
            (root / "text_encoder" / "model.onnx").write_bytes(b"")
            (root / "unet" / "model.onnx").write_bytes(b"")
            (root / "vae_decoder" / "model.onnx").write_bytes(b"")
            (root / "scheduler" / "scheduler_config.json").write_text(
                json.dumps(
                    {
                        "num_train_timesteps": 1000,
                        "beta_start": 0.00085,
                        "beta_end": 0.012,
                        "beta_schedule": "scaled_linear",
                    }
                ),
                encoding="utf-8",
            )

            model_paths = ModelPaths.from_model_dir(root)

            self.assertEqual(model_paths.tokenizer_dir, root / "tokenizer")
            self.assertEqual(model_paths.unet_path, root / "unet" / "model.onnx")

    def test_validate_requires_multiple_of_eight_dimensions(self) -> None:
        config = InferenceConfig(
            prompt="test",
            negative_prompt="",
            output_path=Path("out.png"),
            height=510,
            width=512,
        )

        with self.assertRaises(ConfigurationError):
            config.validate()

    def test_pipeline_memory_check_raises_when_budget_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tokenizer").mkdir()
            (root / "text_encoder").mkdir()
            (root / "unet").mkdir()
            (root / "vae_decoder").mkdir()
            (root / "scheduler").mkdir()
            (root / "text_encoder" / "model.onnx").write_bytes(b"x" * 1024)
            (root / "unet" / "model.onnx").write_bytes(b"x" * 1024)
            (root / "unet" / "weights.pb").write_bytes(b"x" * 1024)
            (root / "vae_decoder" / "model.onnx").write_bytes(b"x" * 1024)
            (root / "scheduler" / "scheduler_config.json").write_text(
                json.dumps(
                    {
                        "num_train_timesteps": 1000,
                        "beta_start": 0.00085,
                        "beta_end": 0.012,
                        "beta_schedule": "scaled_linear",
                    }
                ),
                encoding="utf-8",
            )

            pipeline = object.__new__(OnnxStableDiffusionPipeline)
            pipeline.model_paths = ModelPaths.from_model_dir(root)

            original_reader = OnnxStableDiffusionPipeline._read_available_memory_bytes
            try:
                OnnxStableDiffusionPipeline._read_available_memory_bytes = staticmethod(
                    lambda: 1
                )
                with self.assertRaises(ConfigurationError):
                    pipeline._ensure_memory_budget()
            finally:
                OnnxStableDiffusionPipeline._read_available_memory_bytes = original_reader


if __name__ == "__main__":
    unittest.main()
