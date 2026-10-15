---
title: Install nemotron-omni-mlx on Apple Silicon
description: How to install the nemotron-omni-mlx runtime, download the 4-bit Nemotron Omni MLX weights, fetch NVIDIA's reference config, and check the Metal memory limit on a 32 GB Mac.
---

# Install

## Requirements

- An Apple Silicon Mac. MLX only runs on Apple Silicon.
- 32 GB of memory or more. The image path peaks at about 22.1 GB. The only machine it has been
  tested on is an M5 Max with 128 GB.
- About 37 GB of free disk for the weights: the 4-bit checkpoint (about 19 GB) plus the filtered
  `text-only/` copy (about 17.8 GB). Figures are from `scripts/setup_weights.py`.
- Python with pip. TODO-VERIFY: no minimum Python version is stated in the repo; use whatever
  version the pinned `mlx` release supports.

## Check your Metal memory ceiling (32 GB Macs)

macOS caps how much memory Metal hands the GPU at roughly 2/3 of physical RAM
(`recommendedMaxWorkingSetSize`). On a 32 GB Mac that is about 21.3 GB, just under the image path's
22.1 GB peak. Text and audio may work while images fail. Check yours before downloading anything:

```swift
import Metal
let d = MTLCreateSystemDefaultDevice()!
print(Double(d.recommendedMaxWorkingSetSize) / 1_073_741_824.0)
```

Save it as `check.swift` and run `swift check.swift`. It prints the ceiling in GB. If it is under
about 23, raise it before using images:

```bash
sudo sysctl iogpu.wired_limit_mb=24576
```

This resets on reboot. If you try it on a 32 GB Mac, the open issue
[nicedreamzapp/nemotron-omni-mlx#1](https://github.com/nicedreamzapp/nemotron-omni-mlx/issues/1)
asks for exactly that report.

## 1. Clone and install

```bash
git clone https://github.com/nicedreamzapp/nemotron-omni-mlx
cd nemotron-omni-mlx
pip install -r requirements.txt
```

`requirements.txt` installs `mlx>=0.31.1`, `mlx-lm>=0.31.2`, `numpy`, `pillow`, `huggingface_hub`
and `hf_transfer`.

The processor loads the tokenizer with `transformers.AutoTokenizer`, which is not listed in
`requirements.txt`. TODO-VERIFY: it is expected to arrive as a dependency of `mlx-lm`. If you see
`No module named 'transformers'`, run `pip install transformers`.

## 2. Download the weights

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python scripts/setup_weights.py
```

The script does two things:

1. Downloads `mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit` into `model-4bit/`
   (skipped if `model-4bit/config.json` already exists). The vision and audio towers read their
   weights straight from here.
2. Builds `text-only/`: a copy holding only the `backbone.*` and `lm_head.*` tensors in one
   `model.safetensors`, with `llm_config` promoted to `config.json` and the tokenizer files copied
   over (skipped if `text-only/model.safetensors` exists).

Why the second folder exists: `mlx_lm.load()` reads every `.safetensors` file in a folder and
rejects the full checkpoint because of the extra `vision_model.*` and `sound_*` keys.

If `HF_HUB_ENABLE_HF_TRANSFER` is unset the script prints a warning and carries on. The script and
README describe the transfer backend as about 30x faster; that figure is not measured in the repo.

## 3. Fetch NVIDIA's reference config

The CLI reads the processor config, tokenizer and chat template from NVIDIA's reference repo,
which it expects in `reference/` (change it with `--ref-dir`). `setup_weights.py` does not fetch it.
The small files are enough:

```bash
huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --exclude "*.safetensors" --local-dir reference
```

## 4. Check it works without weights

```bash
python -m src.omni --prompt "Hi" --dry-run
```

This runs the processor only and prints the expanded prompt. If it works, `reference/` is in place.

## Optional extras

None of these are in `requirements.txt`.

| Need | Install | Why (from the code) |
|---|---|---|
| Video input | `pip install av` or `decord` | `_read_video_frames` tries `decord`, then PyAV |
| Audio not at 16 kHz | `pip install scipy` | resampling uses `scipy.signal.resample` |
| Non-WAV audio | `pip install librosa` or `soundfile` | without them, only PCM WAV is read (stdlib `wave`) |
| Parity tests | `torch>=2.11 transformers>=5.5 timm einops open_clip_torch pytest` | listed as a comment at the bottom of `requirements.txt` |

The parity tests also need NVIDIA's code in `reference/` and the bf16 shard 1 at
`weights-bf16/model-00001-of-00017.safetensors`. No script sets those up yet.

## What ends up on disk

All of these are in `.gitignore` and are never committed:

```
model-4bit/     4-bit checkpoint as published (towers are read from here)
text-only/      filtered LLM-only copy for mlx-lm
reference/      NVIDIA's config, tokenizer and processor files
weights-bf16/   only for the parity tests
```
