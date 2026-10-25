---
title: Credits - the projects behind nemotron-omni-mlx
description: nemotron-omni-mlx builds on NVIDIA's Nemotron Omni model and reference code, yayr's 4-bit MLX quantization on mlx-community, Apple's MLX and mlx-lm, and Hugging Face transformers.
---

# Credits

This repo is an MLX runtime for someone else's model. The pieces it stands on:

- **[NVIDIA](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)** built
  Nemotron-3-Nano-Omni-30B-A3B and released the weights and the PyTorch reference code openly. The
  vision, audio and processor code here is a port of that reference. The vision tower is NVIDIA's
  [C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H).
- **[yayr at mlx-community](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit)**
  made the 4-bit MLX quantization this runs on.
- **[Apple's MLX team](https://github.com/ml-explore/mlx)** and the contributors of `nemotron_h` to
  **[mlx-lm](https://github.com/ml-explore/mlx-lm)**. The text backbone runs on their code
  unchanged.
- **[Hugging Face transformers](https://github.com/huggingface/transformers)**. The audio tower is a
  port of its `ParakeetEncoder` (v5.5.0), and the tokenizer is loaded through it.
- **[@wacomctl672](https://huggingface.co/wacomctl672)** converted the abliterated bf16 build to
  GGUF and documented running it on llama.cpp.
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** and **[mlx-audio](https://github.com/Blaizzy/mlx-audio)**
  by Blaizzy are the README's suggested alternatives for other model families.

The MLX runtime in `src/`, `scripts/` and `tests/` was written by
[Matt Macosko](https://github.com/nicedreamzapp) (Nice Dreamz LLC). If you want local MLX models
behind Claude Code as well, see
[claude-code-local](https://github.com/nicedreamzapp/claude-code-local).

## License

Runtime code: MIT, see [`LICENSE`](https://github.com/nicedreamzapp/nemotron-omni-mlx/blob/main/LICENSE).
Model weights are NVIDIA's and carry the
[NVIDIA Open Model License](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/blob/main/LICENSE).
This repo ships no weights.

This documentation site is built with [MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
