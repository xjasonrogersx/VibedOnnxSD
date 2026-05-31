from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from vibed_onnx_sd.cli import resolve_prompt, run
from vibed_onnx_sd.config import ConfigurationError


class _InteractiveStdin:
    def isatty(self) -> bool:
        return True


class _NonInteractiveStdin:
    def isatty(self) -> bool:
        return False


class _FakePipeline:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.generated_config = None

    def generate(self, config):  # type: ignore[no-untyped-def]
        self.generated_config = config
        return Path("/tmp/generated.png")


class CliTests(unittest.TestCase):
    def test_resolve_prompt_uses_explicit_value(self) -> None:
        resolved = resolve_prompt("  hello world  ", stdin=_NonInteractiveStdin())
        self.assertEqual(resolved, "hello world")

    def test_resolve_prompt_asks_interactively_when_missing(self) -> None:
        resolved = resolve_prompt(
            None,
            stdin=_InteractiveStdin(),
            prompt_input=lambda _: "painted castle",
        )
        self.assertEqual(resolved, "painted castle")

    def test_resolve_prompt_requires_interactive_stdin_when_missing(self) -> None:
        with self.assertRaises(ConfigurationError):
            resolve_prompt(None, stdin=_NonInteractiveStdin())

    def test_run_uses_resolved_prompt_and_model_dir(self) -> None:
        pipeline_holder: dict[str, _FakePipeline] = {}

        def pipeline_factory(model_dir: str | Path) -> _FakePipeline:
            pipeline = _FakePipeline(model_dir)
            pipeline_holder["pipeline"] = pipeline
            return pipeline

        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                prompt=None,
                negative_prompt="foggy",
                output=str(Path(temp_dir) / "result.png"),
                steps=12,
                guidance_scale=6.5,
                seed=9,
                height=512,
                width=512,
                model_dir=None,
                model_repo="example/repo",
                cache_dir=str(Path(temp_dir) / "cache"),
            )

            output_path = run(
                args,
                stdin=_InteractiveStdin(),
                prompt_input=lambda _: "robot portrait",
                model_dir_resolver=lambda *_, **__: Path(temp_dir) / "model",
                pipeline_factory=pipeline_factory,
            )

        self.assertEqual(output_path, Path("/tmp/generated.png"))
        pipeline = pipeline_holder["pipeline"]
        self.assertEqual(pipeline.model_dir, Path(temp_dir) / "model")
        self.assertIsNotNone(pipeline.generated_config)
        self.assertEqual(pipeline.generated_config.prompt, "robot portrait")
        self.assertEqual(pipeline.generated_config.negative_prompt, "foggy")
        self.assertEqual(pipeline.generated_config.steps, 12)


if __name__ == "__main__":
    unittest.main()
