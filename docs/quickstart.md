---
title: Quickstart - ask Nemotron Omni about an image, audio clip or video on a Mac
description: Example commands for nemotron-omni-mlx. Describe an image, transcribe a WAV file, summarize a video, or combine image and audio, all locally with MLX.
---

# Quickstart

Run these from the repo root after finishing [Install](install.md). The CLI is a Python module,
so call it as `python -m src.omni`.

## See

```bash
python -m src.omni --image photo.jpg --prompt "What's in this picture?"
```

## Hear

```bash
python -m src.omni --audio clip.wav --prompt "Transcribe this."
```

## Watch

```bash
pip install av   # video needs PyAV or decord
python -m src.omni --video clip.mp4 --prompt "What happens here?"
```

By default 8 frames are sampled evenly across the clip. Change it with `--num-frames`.

## Combine

```bash
python -m src.omni --image photo.jpg --audio clip.wav --prompt "Do these match?"
```

The CLI accepts one `--image`, one `--video` and one `--audio` per run, in any combination.

## Text only

```bash
python -m src.omni --prompt "Explain what a Conformer is in two sentences."
```

## Give it room to think

This is a reasoning model. It writes a `<think>` block before the answer. The default
`--max-tokens` is 512, which can cut it off mid-thought. The README suggests:

```bash
python -m src.omni --image photo.jpg --prompt "What's in this picture?" --max-tokens 900
```

## What the output looks like

With the default (verbose) output, the CLI prints progress lines before the answer. These come
straight from `src/omni.py`:

```text
[proc] <N> tokens (<n> image/video ctx, <n> sound ctx)
[proc] pixel_values (1, 3, <H>, <W>), num_tokens=[<n>]
[llm] loaded in <s>s
[gen] <n> tokens in <s>s = <rate> tok/s
[mem] peak <GB> GB

<think> ... </think>
...answer...
```

| Line | Meaning |
|---|---|
| `[proc]` | Prompt length after the processor expanded the image, video and audio placeholders |
| `[evs]` | Video only. How many video tokens Efficient Video Sampling kept |
| `[llm]` | Time to load the text backbone from `text-only/` |
| `[gen]` | Tokens generated and tokens per second |
| `[mem]` | Peak MLX memory (`mx.get_peak_memory()`) |

Add `-q` to print only the answer.

## Check the prompt without loading weights

```bash
python -m src.omni --image photo.jpg --prompt "Describe it." --dry-run
```

`--dry-run` runs only the processor and prints the templated prompt with the context tokens
collapsed (for example `...<image x256>...`). Nothing is loaded into the GPU. It still needs
`reference/`.

## Use it from Python

`generate()` in `src/omni.py` takes the same options as the CLI and returns the decoded text:

```python
from src.omni import generate

text = generate(
    prompt="What's in this picture?",
    image="photo.jpg",
    max_tokens=900,
    verbose=False,
)
print(text)
```

It reloads the text backbone and towers on every call. There is no long-running server in this repo.
