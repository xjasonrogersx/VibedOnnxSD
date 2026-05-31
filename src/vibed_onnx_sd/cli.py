from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from .config import ConfigurationError, InferenceConfig
from .model_store import DEFAULT_MODEL_REPO_ID, resolve_model_dir
from .pipeline import OnnxStableDiffusionPipeline

LOGGER = logging.getLogger(__name__)


class PipelineProtocol(Protocol):
    def generate(self, config: InferenceConfig) -> Path: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion 1.5-style ONNX inference on CPU."
    )
    parser.add_argument(
        "--model-dir",
        help="Directory with ONNX assets. If omitted, the default model is downloaded and cached.",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO_ID,
        help="Hugging Face model repo used when --model-dir is omitted.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Cache root for auto-downloaded model assets.",
    )
    parser.add_argument(
        "--prompt",
        help="Positive text prompt. If omitted, the CLI asks for it interactively.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt used for classifier-free guidance.",
    )
    parser.add_argument(
        "--output",
        default="output/generated.png",
        help="Image file path to write.",
    )
    parser.add_argument("--steps", type=int, default=25, help="Denoising step count.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic random seed.",
    )
    parser.add_argument("--height", type=int, default=512, help="Output image height.")
    parser.add_argument("--width", type=int, default=512, help="Output image width.")
    return parser


def resolve_prompt(
    prompt: str | None,
    *,
    stdin: object | None = None,
    prompt_input: Callable[[str], str] = input,
) -> str:
    resolved_prompt = (prompt or "").strip()
    if resolved_prompt:
        LOGGER.info("Using prompt from command line.")
        return resolved_prompt

    active_stdin = sys.stdin if stdin is None else stdin
    if not hasattr(active_stdin, "isatty") or not active_stdin.isatty():
        raise ConfigurationError(
            "Prompt was not provided and stdin is not interactive."
        )

    LOGGER.info("No prompt supplied; asking interactively.")
    resolved_prompt = prompt_input("Prompt: ").strip()
    if not resolved_prompt:
        raise ConfigurationError("Prompt must not be empty.")
    return resolved_prompt


def run(
    args: argparse.Namespace,
    *,
    stdin: object | None = None,
    prompt_input: Callable[[str], str] = input,
    model_dir_resolver: Callable[..., Path] = resolve_model_dir,
    pipeline_factory: Callable[[str | Path], PipelineProtocol] = OnnxStableDiffusionPipeline.from_model_dir,
) -> Path:
    prompt = resolve_prompt(args.prompt, stdin=stdin, prompt_input=prompt_input)
    LOGGER.info("Resolving model directory.")
    model_dir = model_dir_resolver(
        args.model_dir,
        repo_id=args.model_repo,
        cache_dir=args.cache_dir,
    )
    LOGGER.info("Using model directory: %s", model_dir)
    LOGGER.info("Loading ONNX pipeline.")
    pipeline = pipeline_factory(model_dir)
    LOGGER.info(
        "Starting image generation with %s steps, guidance %.2f, seed %s.",
        args.steps,
        args.guidance_scale,
        args.seed,
    )
    return pipeline.generate(
        InferenceConfig(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            output_path=Path(args.output),
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            height=args.height,
            width=args.width,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[vibed-onnx-sd] %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = run(args)
    except ConfigurationError as error:
        parser.exit(status=2, message=f"configuration error: {error}\n")

    print(f"saved image to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
