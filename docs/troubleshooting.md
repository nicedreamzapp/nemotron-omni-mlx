---
title: Troubleshooting nemotron-omni-mlx - errors, memory limits and fixes
description: Fixes for common nemotron-omni-mlx problems, including the image path failing on a 32 GB Mac, missing reference files, video decode errors, audio resampling, empty or cut-off answers, and weight mismatch errors.
---

# Troubleshooting

Each entry quotes the message the code prints or raises, so you can search this page for the text
you see.

## Image prompts fail on a 32 GB Mac, text and audio work

This is the Metal wired-memory ceiling, not a model bug. macOS lets the GPU use about 2/3 of
physical RAM, about 21.3 GB on a 32 GB Mac. The image path peaks at about 22.1 GB.

Check your ceiling with the `check.swift` snippet on the [Install](install.md#check-your-metal-memory-ceiling-32-gb-macs)
page, then raise it:

```bash
sudo sysctl iogpu.wired_limit_mb=24576
```

It resets on reboot. Nobody has reported a run on a 32 GB Mac yet. The open issue
[nicedreamzapp/nemotron-omni-mlx#1](https://github.com/nicedreamzapp/nemotron-omni-mlx/issues/1)
asks for one, with the chip, RAM, what `check.swift` printed, the pytest result, tok/s for an
image and an audio run, and whether it swapped or crashed.

TODO-VERIFY: the exact error text you see when the image path runs out of wired memory was not
captured in the repo.

## `FileNotFoundError` for `preprocessor_config.json`

`OmniProcessor.from_pretrained()` opens `<ref-dir>/preprocessor_config.json` first. The weights
script does not download it. Fetch the reference files:

```bash
huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --exclude "*.safetensors" --local-dir reference
```

Or point `--ref-dir` at wherever you put them. This also happens with `--dry-run`.

## `No module named 'transformers'`

The tokenizer is loaded with `transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True)`.
`transformers` is not listed in `requirements.txt`. Install it with `pip install transformers`.

## `could not decode '<file>': install `decord` or `av` for video support`

Raised by `_read_video_frames()` in `src/omni.py` when neither decord nor PyAV can read the file.
Install one:

```bash
pip install av
```

If PyAV is installed and you still get this, the file may have no video stream PyAV can decode.
The message ends with the underlying error. `no frames decoded` means PyAV opened the file but got
zero frames.

## Audio errors

**`No module named 'scipy'`.** Your file is not 16 kHz and neither `librosa` nor `soundfile` is
installed, so the standard-library loader needs `scipy` to resample. Run `pip install scipy`, or
convert the file to 16 kHz WAV first.

**`Unsupported wav sample width: 3`** (or another number). The standard-library fallback reads
8, 16 and 32 bit PCM only. 24-bit WAV is not supported by it. Install `soundfile` or `librosa`,
or convert to 16-bit.

**A non-WAV file fails to open (`wave.Error`).** Without `librosa` or `soundfile`, only WAV is
readable. Install one of them.

**`sequence length <t> > max_position_embeddings`.** Raised by the audio encoder when the
subsampled clip is longer than 5000 positions. At 80 ms per position that is about 400 seconds of
audio. Trim the clip. TODO-VERIFY: long-clip behaviour has not been tested in the repo.

## The answer is cut off, or only shows `<think>`

The model reasons in a `<think>` block before answering. The default `--max-tokens 512` can end
generation mid-thought. Use `--max-tokens 900` or higher.

## The answer is empty

`PLAN.md` notes that a raw prompt without the chat template returns empty output. The CLI always
applies the template, so this should only happen if you call the processor or `mlx-lm` yourself.
Use `tokenizer.apply_chat_template(..., add_generation_prompt=True)` like `build_prompt()` does.

## `mlx_lm.load()` fails on `model-4bit/`

`mlx-lm` rejects the full checkpoint because of the `vision_model.*` and `sound_*` keys. Load
`text-only/` instead, which `scripts/setup_weights.py` builds. The CLI does this by default.

## `no backbone.* tensors found — is model-4bit/ complete?`

Printed by `scripts/setup_weights.py` when none of the downloaded shards contain text backbone
weights. The download is incomplete or the folder holds something else. Delete `model-4bit/` and
run the script again. Note the script skips the download whenever `model-4bit/config.json` exists,
so a partial download with a config file will not be retried on its own.

## `! HF_HUB_ENABLE_HF_TRANSFER is unset — this will be ~30x slower.`

A warning, not an error. The download continues without the faster backend. Press Ctrl-C and
re-run with `export HF_HUB_ENABLE_HF_TRANSFER=1` if you want it.

## Weight and shard errors

**`no weights with prefix 'vision_model.' in .../model.safetensors.index.json`** (or
`'sound_encoder.'`). `--model-dir` does not point at the full 4-bit checkpoint, or its index is
missing. The towers are read from `model-4bit/`, not `text-only/`.

**`'vision_model.' spans multiple shards [...]; loader takes one`.** The loader expects every tower
key in one shard, which is true for the mlx-community 4-bit checkpoint. A different checkpoint
layout is not supported by `_shard_for_prefix()`.

**`vision weight mismatch. missing: [...] unexpected: [...]`.** Raised by `load_vision_tower()`.
The shard's vision keys do not match the module tree. Usually a different checkpoint or a
different `config.json` (for example a different register count).

**`vision tower unavailable (src/vision.py): ...`** or **`audio tower unavailable (src/audio.py): ...`.**
The tower module failed to import. The text after the colon is the real error, usually a
missing or too-old `mlx`.

## Splice errors

**`No context tokens (id=<n>) found in input_ids`** and
**`context token count (<a>) != tower embedding count (<b>) for token id <n>`.**
Raised by `splice_embeddings()` when the processor's predicted token count does not match what a
tower produced. This should not happen with the shipped processor. If it does, please open an
issue with the input's size (image dimensions, audio length or frame count) and the full output.

## `Video is not supported for batch size > 1`

An assertion in `OmniProcessor.__call__`. Only reachable from Python when you pass a list of
prompts together with a video.

## A parity test fails

Open an [issue](https://github.com/nicedreamzapp/nemotron-omni-mlx/issues/new) with your chip,
RAM and the failing test. The tests need the torch-side packages from the bottom of
`requirements.txt`, NVIDIA's code in `reference/`, and
`weights-bf16/model-00001-of-00017.safetensors`. Small differences on the GPU are expected, see
[How it works](how-it-works.md#why-gpu-results-are-not-bit-exact).
