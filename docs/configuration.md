---
title: Configuration - every nemotron-omni-mlx CLI flag, env var and config file
description: Reference for the nemotron-omni-mlx command line (src/omni.py), the setup script's environment variable, and which values are read from NVIDIA's config.json and preprocessor_config.json.
---

# Configuration

Everything on this page is read from the source. Line references point at the files in this repo.

## CLI flags (`python -m src.omni`)

Defined in `main()` in `src/omni.py`.

| Flag | Type | Default | What the code does |
|---|---|---|---|
| `--prompt` | string | required | The user message. It is wrapped in the tokenizer's chat template with `add_generation_prompt=True`. |
| `--image` | path | none | Opened with Pillow and converted to RGB. Inserts an `<image>` placeholder before the prompt. |
| `--audio` | path | none | Loaded at 16 kHz mono. Inserts `<so_embedding>` before the prompt. The help text says "16 kHz wav", but other rates and formats work if the optional libraries are present (see below). |
| `--video` | path | none | Frames are decoded with `decord`, or PyAV as a fallback. Inserts `This is a video:` and a `<video>` placeholder. |
| `--max-tokens` | int | `512` | Maximum tokens to generate. Generation also stops at an end-of-sequence token. |
| `--temp` | float | `0.0` | Passed to `mlx_lm.sample_utils.make_sampler(temp=...)`. `0.0` means greedy decoding. |
| `--num-frames` | int | `8` | Frames sampled evenly from `--video` (`np.linspace` over the frame count). |
| `--ref-dir` | path | `<repo>/reference` | Folder holding NVIDIA's `preprocessor_config.json`, `config.json` and tokenizer. |
| `--text-only-dir` | path | `<repo>/text-only` | Folder `mlx_lm.load()` reads the text backbone from. Also where `generation_config.json` is read for end-of-sequence ids. |
| `--model-dir` | path | `<repo>/model-4bit` | Full checkpoint. The vision and audio tower weights are read from here. |
| `--dry-run` | flag | off | Runs the processor only, prints the expanded prompt, loads no weights. |
| `-q`, `--quiet` | flag | off | Hides the `[proc]`, `[evs]`, `[llm]`, `[gen]` and `[mem]` lines. |

Placeholder order in the message is fixed by `build_prompt()`: image, then video, then audio,
then your prompt, joined by newlines.

### Things the CLI does not expose

- There is no top-p, top-k, repetition penalty or seed flag. Only `temp` is passed to the sampler.
- There is no flag for tower precision. `load_vision_tower()` defaults to `dtype=mx.float32`,
  and `src/audio.py` converts every audio weight to float32 on load. TODO-VERIFY: the README's
  benchmark line says "bf16 towers", which matches how the weights are stored in the checkpoint,
  not the dtype the code computes in.
- There is no way to pass more than one image, video or audio file from the CLI, though
  `OmniProcessor` itself accepts lists.

## Python API

`src.omni.generate()` takes the same options as keyword arguments:

```python
generate(
    prompt: str,
    image: str | None = None,
    audio: str | None = None,
    video: str | None = None,
    ref_dir: str = DEFAULT_REF_DIR,
    text_only_dir: str = DEFAULT_TEXT_ONLY,
    model_dir: str = DEFAULT_MODEL_4BIT,
    max_tokens: int = 512,
    temp: float = 0.0,
    num_frames: int = 8,
    dry_run: bool = False,
    verbose: bool = True,
) -> str
```

It returns the decoded text, including any `<think>` block. With `dry_run=True` it returns `""`.

## Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `HF_HUB_ENABLE_HF_TRANSFER` | `scripts/setup_weights.py` and `huggingface_hub` | If unset, the script prints a warning before downloading. If set to `1`, `huggingface_hub` uses the `hf_transfer` backend. |

No other environment variable is read by the code in `src/` or `scripts/`. Standard Hugging Face
variables (cache location, tokens) apply to the download as they would to any `huggingface_hub`
call.

## `scripts/setup_weights.py`

The script has no flags. Its settings are constants at the top of the file:

| Constant | Value |
|---|---|
| `REPO_ID` | `mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit` |
| `MODEL_DIR` | `<repo>/model-4bit` |
| `TEXT_DIR` | `<repo>/text-only` |
| `TOKENIZER_FILES` | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json` |

If you want the weights somewhere else, edit those constants and pass matching
`--model-dir` and `--text-only-dir` to the CLI.

## Values read from NVIDIA's config files

These come from the files in `--ref-dir`. The table lists the code default used when a key is
missing. The actual values in NVIDIA's files are what run.

### `preprocessor_config.json` (image and video sizing)

Read by `OmniImageProcessor.from_pretrained()`. Unknown keys are ignored.

| Key | Code default | Used for |
|---|---|---|
| `norm_mean` | `[0.48145466, 0.4578275, 0.40821073]` | Pixel normalization. The processor applies it, not the vision tower. |
| `norm_std` | `[0.26862954, 0.26130258, 0.27577711]` | Same |
| `patch_size` | `16` | Images are resized to a multiple of this |
| `downsample_ratio` | `0.5` | Pixel-shuffle factor. Tokens per image = patches / 4 |
| `min_num_patches` | `1024` | Small images are scaled up to at least this many patches |
| `max_num_patches` | `13312` | Upper bound on patches per image |
| `max_model_len` | `16384` | Sets the overall patch budget (minus 4 reserved tokens) |
| `video_target_num_patches` | `1024` | Patches per video frame |
| `video_maintain_aspect_ratio` | `true` | Keep the frame's aspect ratio when sizing video |

TODO-VERIFY: the values actually shipped in NVIDIA's `preprocessor_config.json` were not checked
while writing this page.

### `config.json`

| Key | Code default | Read by | Used for |
|---|---|---|---|
| `video_temporal_patch_size` | `2` | processor | Frames per video token group |
| `vision_config.video_temporal_patch_size` | `2` | vision tower | Frames packed per temporal patch |
| `sound_config.sampling_rate` | `16000` | processor | Audio is loaded at this rate |
| `sound_config.subsampling_factor` | `8` | processor | Audio token count |
| `video_pruning_rate` | `0.0` if missing | `generate()` | EVS pruning rate. `PLAN.md` records `0.7` in NVIDIA's file. If 0, EVS is skipped. |
| `patch_size`, `downsample_ratio`, `ps_version`, `vit_hidden_size`, `projector_hidden_size` | `16`, `0.5`, `"v2"`, `1280`, `20480` | vision tower | Tower shape |
| `vision_config.args.cpe_max_size` | `2048` | vision tower | Position embedding grid (2048 / 16 = 128 x 128) |
| `vision_config.args.cls_token_per_teacher`, `teachers`, `cpe_num_registers`, `register_multiple` | see `VisionConfig.from_json` | vision tower | Number of class and register tokens to skip (10 for this checkpoint) |
| `llm_config.hidden_size` | `2688` | vision tower | Projector output size |
| `use_thumbnail`, `force_image_size`, `image_tag_type` | `true`, `512`, n/a | not used on the live path | Vestigial. See [How it works](how-it-works.md#traps-found-during-the-port). |

### The audio tower ignores `config.json`

`src/omni.py` calls `load_audio_tower([shard])` without a config, so the audio tower always uses
the defaults in `AudioConfig` (`src/audio.py`): 24 Conformer layers, hidden size 1024, 8 heads,
128 mel bins, 16 kHz, hop 160, `n_fft` 512, `max_position_embeddings` 5000, projection to 2688.
`AudioConfig.from_sound_config()` exists but nothing in the CLI calls it.

### `generation_config.json` (in `--text-only-dir`)

`_eos_ids()` collects `eos_token_id` (an int or a list) from this file plus the tokenizer's own
`eos_token_id`. If neither is found it falls back to token id `11`.
TODO-VERIFY: whether `11` is the right end-of-sequence id for this tokenizer was not checked.
