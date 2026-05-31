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

## Memory expectations

The default Stable Diffusion 1.5 ONNX model is large for CPU inference. In practice it needs roughly **6 GiB or more of available RAM** to load reliably with ONNX Runtime on CPU. If the environment is below that, the CLI now stops early with a clear configuration error instead of getting killed by the kernel during model load.

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

## Quantize a local model

You can create an int8 copy of a downloaded model with:

```bash
python3 scripts/quantize_onnx_pipeline.py --force
```

By default this reads from the local downloaded model folder and writes a sibling `-int8` directory. You can also choose specific components:

```bash
python3 scripts/quantize_onnx_pipeline.py \
  --components text_encoder vae_decoder \
  --output-dir models/onnx-community--stable-diffusion-v1-5-ONNX-int8
```

The UNet is the most memory-intensive component to quantize. In a small container it may still be killed during quantization even when the text encoder and VAE succeed.

## Tests

Run the full test suite:

```bash
python3 -m unittest discover -s tests
```

Run a single test module:

```bash
python3 -m unittest tests.test_scheduler
```
