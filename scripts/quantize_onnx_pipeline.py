#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onnxruntime.quantization import QuantType, quantize_dynamic

from vibed_onnx_sd.config import ConfigurationError, ModelPaths
from vibed_onnx_sd.model_store import default_model_dir

LOGGER = logging.getLogger("quantize-onnx-pipeline")
COMPONENTS = ("text_encoder", "unet", "vae_decoder")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize a Stable Diffusion ONNX pipeline into a new model directory."
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_model_dir()),
        help="Source model directory. Defaults to the downloaded local model.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output model directory. Defaults to <input-dir>-int8.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=COMPONENTS,
        default=list(COMPONENTS),
        help="Pipeline components to quantize.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser


def quantize_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    components: list[str],
    force: bool,
) -> Path:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    LOGGER.info("Validating source model layout at %s", input_dir)
    ModelPaths.from_model_dir(input_dir)

    if output_dir.exists():
        if not force:
            raise ConfigurationError(
                f"Output directory already exists: {output_dir}. Use --force to replace it."
            )
        LOGGER.info("Removing existing output directory %s", output_dir)
        shutil.rmtree(output_dir)

    LOGGER.info("Creating quantized output directory %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_directory(input_dir / "tokenizer", output_dir / "tokenizer")
    _copy_directory(input_dir / "scheduler", output_dir / "scheduler")

    for component in COMPONENTS:
        source_dir = input_dir / component
        target_dir = output_dir / component
        target_dir.mkdir(parents=True, exist_ok=True)
        _copy_component_metadata(source_dir, target_dir)

        source_model = source_dir / "model.onnx"
        target_model = target_dir / "model.onnx"
        if component in components:
            LOGGER.info("Quantizing %s", component)
            quantize_dynamic(
                str(source_model),
                str(target_model),
                weight_type=QuantType.QInt8,
            )
        else:
            LOGGER.info("Copying %s without quantization", component)
            shutil.copy2(source_model, target_model)

    LOGGER.info("Quantized model available at %s", output_dir)
    return output_dir


def _copy_directory(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _copy_component_metadata(source_dir: Path, target_dir: Path) -> None:
    for source_path in source_dir.iterdir():
        if source_path.name == "model.onnx" or source_path.suffix == ".pb":
            continue
        target_path = target_dir / source_path.name
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[quantize] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"{input_dir.expanduser().resolve()}-int8"
    )

    try:
        quantize_pipeline(
            input_dir,
            output_dir,
            components=args.components,
            force=args.force,
        )
    except ConfigurationError as error:
        parser.exit(status=2, message=f"configuration error: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
