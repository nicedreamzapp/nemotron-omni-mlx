---
title: FAQ - running NVIDIA Nemotron Omni on a Mac with MLX
description: Answers about nemotron-omni-mlx. Which Macs it runs on, how much memory it needs, whether it works offline, video and audio support, abliterated builds, and how it relates to mlx-vlm and llama.cpp.
---

# FAQ

## Does it run fully offline?

After the weights and `reference/` files are downloaded, yes. `src/` makes no network calls.
`AutoTokenizer.from_pretrained()` is pointed at the local `reference/` folder.

## Which Macs does it run on?

Apple Silicon only, since MLX is Apple Silicon only. It has been tested on one machine, an M5 Max
MacBook Pro with 128 GB. 32 GB should work for text and audio. Images on 32 GB likely need the
wired-memory change in [Troubleshooting](troubleshooting.md#image-prompts-fail-on-a-32-gb-mac-text-and-audio-work).
Results from smaller Macs are wanted in
[nicedreamzapp/nemotron-omni-mlx#1](https://github.com/nicedreamzapp/nemotron-omni-mlx/issues/1).

## How much memory does it use?

Peak memory on the test machine, from the README: 17.9 GB text, 21.0 GB audio, 22.1 GB image.

## How fast is it?

README figures on the M5 Max: 152 tok/s text, 147 tok/s audio, 67.7 tok/s image. `PLAN.md`
records lower end-to-end CLI numbers for text (109 tok/s) and audio (110.5 tok/s). TODO-VERIFY:
re-run to settle which set is current.

## Does it support video?

Yes, with `--video`. Install `av` or `decord`. It samples 8 frames by default (`--num-frames`),
pairs them, and prunes redundant tokens with Efficient Video Sampling.

## Which audio formats work?

16 kHz PCM WAV works with only the base requirements. Other sample rates need `scipy`. Other
formats need `librosa` or `soundfile`.

## Can I send several images at once?

Not from the CLI, which takes one `--image`. `OmniProcessor` accepts a list of images if you call
it from Python, but `generate()` does not.

## Is there an OpenAI-compatible server?

No. This repo is a CLI and a `generate()` function.

## Why is the answer full of `<think>`?

It is a reasoning model. The README describes it emitting a `<think>` block before the answer.
The CLI prints the raw decoded text, including that block.

## Does it work with the abliterated (uncensored) builds?

The README says the three abliterated MLX builds published under
[divinetribe on Hugging Face](https://huggingface.co/divinetribe) (4-bit, 8-bit, bf16) run with
this same runtime or with `mlx-vlm` directly. TODO-VERIFY: `scripts/setup_weights.py` only
downloads the mlx-community 4-bit checkpoint, so using another build means downloading it yourself
and building its `text-only/` folder the same way.

## How is this different from mlx-vlm?

[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) runs and fine-tunes many vision-language models on
MLX. This repo does one model and adds its audio tower, with parity tests against NVIDIA's
reference. For other model families, start with mlx-vlm. For speech-to-text and text-to-speech,
see [mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Can I run it with llama.cpp?

Not with this repo. The README describes a GGUF conversion of the abliterated bf16 build by
[@wacomctl672](https://huggingface.co/wacomctl672), run at Q4_K_M with ggml-org's combined
audio and vision `mmproj`, and the one-line patch needed to convert it yourself.

## How do I know the port is correct?

Run `pytest tests/`. Each test runs NVIDIA's PyTorch code and the MLX code on the same input with
the same weights and compares the outputs. See [How it works](how-it-works.md#how-correctness-was-checked).

## Does the repo include the weights?

No. The runtime is MIT. The weights are NVIDIA's, under the
[NVIDIA Open Model License](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/blob/main/LICENSE),
and are downloaded from Hugging Face.
