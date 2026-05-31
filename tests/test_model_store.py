from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vibed_onnx_sd.config import ConfigurationError
from vibed_onnx_sd.model_store import default_cache_root, default_model_dir, resolve_model_dir


def _write_model_layout(root: Path) -> None:
    (root / "tokenizer").mkdir(parents=True, exist_ok=True)
    (root / "text_encoder").mkdir(exist_ok=True)
    (root / "unet").mkdir(exist_ok=True)
    (root / "vae_decoder").mkdir(exist_ok=True)
    (root / "scheduler").mkdir(exist_ok=True)
    (root / "tokenizer" / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 77}),
        encoding="utf-8",
    )
    (root / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "merges.txt").write_text("", encoding="utf-8")
    (root / "tokenizer" / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (root / "text_encoder" / "model.onnx").write_bytes(b"")
    (root / "unet" / "model.onnx").write_bytes(b"")
    (root / "unet" / "weights.pb").write_bytes(b"")
    (root / "vae_decoder" / "model.onnx").write_bytes(b"")
    (root / "vae_decoder" / "config.json").write_text(
        json.dumps({"scaling_factor": 0.18215}),
        encoding="utf-8",
    )
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


class ModelStoreTests(unittest.TestCase):
    def test_default_cache_root_uses_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(default_cache_root(temp_dir), Path(temp_dir).resolve())

    def test_resolve_model_dir_returns_explicit_dir_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_model_dir(Path(temp_dir))
            self.assertEqual(resolved, Path(temp_dir).resolve())

    def test_resolve_model_dir_downloads_when_cache_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            target_dir = default_model_dir(
                repo_id="onnx-community/stable-diffusion-v1-5-ONNX",
                cache_dir=cache_dir,
            )

            def fake_download(**kwargs):  # type: ignore[no-untyped-def]
                self.assertEqual(Path(kwargs["local_dir"]), target_dir)
                _write_model_layout(target_dir)
                return str(target_dir)

            with patch("vibed_onnx_sd.model_store.snapshot_download", side_effect=fake_download) as mocked_download:
                resolved = resolve_model_dir(
                    None,
                    repo_id="onnx-community/stable-diffusion-v1-5-ONNX",
                    cache_dir=cache_dir,
                )

            self.assertEqual(resolved, target_dir)
            mocked_download.assert_called_once()

    def test_resolve_model_dir_redownloads_incomplete_default_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            target_dir = default_model_dir(cache_dir=cache_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "tokenizer").mkdir(exist_ok=True)
            (target_dir / "text_encoder").mkdir(exist_ok=True)
            (target_dir / "unet").mkdir(exist_ok=True)
            (target_dir / "vae_decoder").mkdir(exist_ok=True)
            (target_dir / "scheduler").mkdir(exist_ok=True)
            (target_dir / "text_encoder" / "model.onnx").write_bytes(b"")
            (target_dir / "unet" / "model.onnx").write_bytes(b"")
            (target_dir / "vae_decoder" / "model.onnx").write_bytes(b"")
            (target_dir / "scheduler" / "scheduler_config.json").write_text(
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

            def fake_download(**kwargs):  # type: ignore[no-untyped-def]
                self.assertEqual(Path(kwargs["local_dir"]), target_dir)
                _write_model_layout(target_dir)
                return str(target_dir)

            with patch("vibed_onnx_sd.model_store.snapshot_download", side_effect=fake_download) as mocked_download:
                resolved = resolve_model_dir(None, cache_dir=cache_dir)

            self.assertEqual(resolved, target_dir)
            mocked_download.assert_called_once()

    def test_resolve_model_dir_raises_on_incomplete_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"

            def fake_download(**kwargs):  # type: ignore[no-untyped-def]
                local_dir = Path(kwargs["local_dir"])
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "tokenizer").mkdir(exist_ok=True)
                return str(local_dir)

            with patch("vibed_onnx_sd.model_store.snapshot_download", side_effect=fake_download):
                with self.assertRaises(ConfigurationError):
                    resolve_model_dir(None, cache_dir=cache_dir)


if __name__ == "__main__":
    unittest.main()
