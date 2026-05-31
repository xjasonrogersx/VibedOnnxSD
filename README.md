# VibedOnnxSD

CPU-only Stable Diffusion inference in Python with ONNX Runtime.

## Scope

This project targets a Stable Diffusion 1.5-style inference pipeline that runs entirely on CPU using local ONNX assets:

- CLIP tokenizer + text encoder
- UNet denoiser
- VAE decoder
- DDIM-style scheduler configuration

The current CLI can run in a simple first-run mode:

- if no prompt is supplied, it asks for one interactively
- if no model directory is supplied, it downloads a default ONNX Stable Diffusion 1.5-compatible model into the local `models/` folder

## Expected model layout

Manual model directories should use this structure:

```text
your-model/
  tokenizer/
  text_encoder/
    model.onnx
  unet/
    model.onnx
  vae_decoder/
    model.onnx
    config.json           # optional, used for scaling_factor when present
  scheduler/
    scheduler_config.json
```

The default auto-download source is `onnx-community/stable-diffusion-v1-5-ONNX` on Hugging Face. Downloaded assets are stored under `./models/` unless `--cache-dir` is provided.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

## Run

Simplest interactive run:

```bash
vibed-onnx-sd
```

The CLI will ask for a prompt and, on first run, download the default model into `./models/`.

Explicit prompt with automatic model download:

```bash
vibed-onnx-sd --prompt "portrait of a robot painted in watercolor"
```

Explicit local model directory:

```bash
vibed-onnx-sd \
  --model-dir /path/to/your-model \
  --prompt "portrait of a robot painted in watercolor" \
  --negative-prompt "blurry, distorted" \
  --steps 25 \
  --guidance-scale 7.5 \
  --seed 7 \
  --height 512 \
  --width 512 \
  --output output/robot.png
```

Override the download cache or default model repo:

```bash
vibed-onnx-sd \
  --prompt "city street at sunset" \
  --cache-dir /models/cache \
  --model-repo onnx-community/stable-diffusion-v1-5-ONNX
```

## Tests

Run the full test suite:

```bash
python3 -m unittest discover -s tests
```

Run a single test module:

```bash
python3 -m unittest tests.test_scheduler
```
