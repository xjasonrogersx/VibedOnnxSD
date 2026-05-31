from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

from .config import ConfigurationError, ModelPaths

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_REPO_ID = "onnx-community/stable-diffusion-v1-5-ONNX"
DEFAULT_MODEL_PATTERNS = (
    "tokenizer/*",
    "text_encoder/*",
    "unet/*",
    "vae_decoder/*",
    "scheduler/*",
)
DEFAULT_REQUIRED_FILES = (
    "tokenizer/tokenizer_config.json",
    "text_encoder/model.onnx",
    "unet/model.onnx",
    "unet/weights.pb",
    "vae_decoder/model.onnx",
    "scheduler/scheduler_config.json",
)


def resolve_model_dir(
    model_dir: str | Path | None,
    *,
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    cache_dir: str | Path | None = None,
) -> Path:
    if model_dir is not None:
        LOGGER.info("Using explicit model directory: %s", Path(model_dir).expanduser().resolve())
        return Path(model_dir).expanduser().resolve()

    target_dir = default_model_dir(repo_id=repo_id, cache_dir=cache_dir)
    if _has_expected_model_layout(target_dir, repo_id=repo_id):
        LOGGER.info("Found cached default model at %s", target_dir)
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading default model %s into %s", repo_id, target_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(target_dir),
            allow_patterns=list(DEFAULT_MODEL_PATTERNS),
        )
    except (
        EntryNotFoundError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OSError,
        RepositoryNotFoundError,
    ) as error:
        raise ConfigurationError(
            f"Unable to download default model assets from {repo_id}: {error}"
        ) from error

    try:
        _validate_model_dir(target_dir, repo_id=repo_id)
    except ConfigurationError as error:
        raise ConfigurationError(
            f"Downloaded assets from {repo_id} are incomplete or incompatible: {error}"
        ) from error
    LOGGER.info("Model download complete.")
    return target_dir


def default_model_dir(
    *, repo_id: str = DEFAULT_MODEL_REPO_ID, cache_dir: str | Path | None = None
) -> Path:
    cache_root = default_cache_root(cache_dir)
    return cache_root / repo_id.replace("/", "--")


def default_cache_root(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    return Path.cwd().resolve() / "models"


def _has_expected_model_layout(model_dir: Path, *, repo_id: str) -> bool:
    try:
        _validate_model_dir(model_dir, repo_id=repo_id)
    except ConfigurationError:
        return False
    return True


def _validate_model_dir(model_dir: Path, *, repo_id: str) -> None:
    ModelPaths.from_model_dir(model_dir)
    if repo_id == DEFAULT_MODEL_REPO_ID:
        missing = [
            str(model_dir / relative_path)
            for relative_path in DEFAULT_REQUIRED_FILES
            if not (model_dir / relative_path).exists()
        ]
        if missing:
            raise ConfigurationError(
                "Missing default model files: " + ", ".join(missing)
            )
