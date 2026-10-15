---
title: How it works - the MLX port of Nemotron Omni's vision and audio towers
description: Architecture of nemotron-omni-mlx. How the C-RADIOv4-H vision tower, Parakeet audio encoder, processor and embedding splice feed mlx-lm's nemotron_h backbone on Apple Silicon.
---

# How it works

Nemotron-3-Nano-Omni is an InternVL-style model: a language model with a vision tower and an audio
tower attached. Each tower turns pixels or a waveform into vectors the size of the language model's
hidden state (2688), and those vectors replace placeholder tokens in the prompt.

## The pipeline

This follows the docstring at the top of `src/omni.py`, which mirrors NVIDIA's
`reference/modeling.py::generate`.

```text
prompt + image/video/audio
        |
        v
1. OmniProcessor (src/processing.py)
   - applies the chat template
   - sizes the image, samples video frames, loads audio at 16 kHz
   - expands <image>, <video>, <so_embedding> into exactly as many
     context tokens as the towers will produce
   - tokenizes -> input_ids
        |
        v
2. Towers
   - vision: NemotronVisionTower (src/vision.py)   -> (N, 2688)
   - audio:  AudioTower           (src/audio.py)    -> (T', 2688)
        |
        v
3. Splice (splice_embeddings in src/omni.py)
   - embed input_ids with the LLM's own embedding table
   - scatter tower outputs into the positions of the context tokens
        |
        v
4. EVS, video only (apply_evs)
   - drop video tokens that barely change between frames
        |
        v
5. Decode
   - mlx_lm generate_step(..., input_embeddings=...) on the nemotron_h backbone
```

## The text backbone

The language model is NVIDIA's hybrid Mamba2 + mixture-of-experts + attention model. `PLAN.md`
records it as 30B total parameters, about 3B active, 128 experts with top-6 routing, hidden size
2688, 52 layers. It is not reimplemented here. `mlx-lm` already ships `nemotron_h`, and the runtime
loads it from `text-only/` with `mlx_lm.load()`.

Two small pieces make that work:

- **`text-only/`**. `mlx_lm.load()` refuses the full checkpoint because of the tower weights in
  it. `scripts/setup_weights.py` writes a copy with only `backbone.*` and `lm_head.*`.
- **`_make_omni_lm()`**. `mlx_lm`'s `generate_step` only accepts `input_embeddings` if the model's
  `__call__` has that parameter, and `nemotron_h` does not. The wrapper adds it, and during the call
  swaps `backbone.embeddings` for a stub that returns the spliced embeddings. The real embedding is
  restored in a `finally` block.

## Vision tower (`src/vision.py`)

A port of NVIDIA's C-RADIOv4-H (a ViT-H/16) plus InternVL's pixel-shuffle and the `mlp1` projector.
The module tree mirrors the checkpoint key names one to one, so weights load with a plain
`tree_unflatten` and no renaming table.

- 32 transformer blocks, width 1280, 16 heads, MLP width 5120, LayerNorm eps 1e-6.
- 10 prefix tokens (4 class tokens + 6 registers) are prepended, then dropped from the output.
- Position embeddings live on a 128 x 128 grid and are bilinearly resized to the image's patch grid.
- Pixel-shuffle with ratio 0.5 merges each 2 x 2 block of patches, so a 512 x 512 image becomes
  1024 patches and then 256 tokens.
- `mlp1` is RMSNorm, Linear (5120 to 20480), squared ReLU, Linear (20480 to 2688).
- Video packs two consecutive frames into one input by stacking them on the channel axis, which
  feeds the separate `video_embedder` weights. An odd frame count repeats the last frame.
- `load_vision_tower()` reads only `vision_model.*` and `mlp1.*` keys and raises if any are missing
  or unexpected.

Layers that exist in the reference module but are switched off by its config (final norm, feature
normalizer) are `Identity` here, and their weights are not in the checkpoint either.

## Audio tower (`src/audio.py`)

A port of the Parakeet encoder from `transformers` plus NVIDIA's `SoundProjection`, in float32.

- **Frontend**: pre-emphasis, STFT (`n_fft` 512, window 400, hop 160), mel filterbank, log, then
  per-utterance normalization. The filterbank and window are loaded from the checkpoint. This stage
  runs on MLX's CPU stream on purpose (see below).
- **Subsampling**: three stride-2 convolutions, so 8x fewer frames. Each output token covers 80 ms.
- **Encoder**: 24 Conformer blocks with Transformer-XL relative-position attention and a
  convolution module with BatchNorm in inference mode.
- **Projection**: RMSNorm, Linear (1024 to 4096), squared ReLU, Linear (4096 to 2688).

A 5 second clip at 16 kHz gives 501 mel frames and 63 audio tokens.

## Processor (`src/processing.py`)

A weight-free numpy port of NVIDIA's `NemotronH_Nano_Omni_Reasoning_V3Processor`. It needs only
the files in `reference/`.

- **Images**: one tile per image, sized by a dynamic-resolution rule on a 16 x 16 patch grid. No
  thumbnail tile and no multi-tile split.
- **Resize**: a numpy port of PyTorch's antialiased bicubic interpolation, matched to ATen's
  float32 weights and Pillow's cubic coefficient (-0.5) so pixels agree with the reference.
- **Video**: each frame is sized for about 1024 patches. Frames are grouped in pairs, each group
  gets a label like `Frame 1 and frame 2: ` followed by its tokens. `<video>` has no token of its
  own, so video positions reuse the `<image>` token id.
- **Audio**: the token count is computed exactly from the sample count
  (`1 + L // 160` mel frames, then three stride-2 stages). If it were off by one, the splice would
  raise a count mismatch.
- **Audio loading**: `librosa`, then `soundfile`, then a standard-library `wave` fallback for PCM
  WAV.

## Efficient Video Sampling (EVS)

For video, `apply_evs()` runs after the splice. It compares each spatial token with the same
position in the previous frame group and keeps the most-changed ones. The first group is always
kept. The number kept is `max(one frame's worth, T*H*W*(1 - pruning_rate))`, and the prompt is
shortened to match. EVS is skipped if the token grid is not square, which is the same assumption
NVIDIA's code makes.

## How correctness was checked

Each component has a parity test in `tests/` that runs NVIDIA's PyTorch reference and the MLX
port on the same input with the same weights (the bf16 shard 1, upcast to float32 on CPU).

| Test file | Compares | Pass bar in the test |
|---|---|---|
| `tests/test_audio_parity.py` | `transformers` ParakeetEncoder + `SoundProjection` vs `src/audio.py` | frontend max diff under 1e-3, min cosine per frame over 0.999 |
| `tests/test_vision_parity.py` | `nvidia/C-RADIOv4-H` + `mlp1` vs `src/vision.py` | cosine 0.999 (`COS_THRESHOLD`) |
| `tests/test_processing_parity.py` | NVIDIA's `AutoProcessor` vs `src/processing.py` | exact token ids, pixel diff |

### Why GPU results are not bit-exact

Metal's float32 matrix multiply is less precise than CPU float32. `PLAN.md` measures about 7.7e-4
relative error on the GPU against a float64 ground truth, versus about 1.3e-6 for both PyTorch CPU
and MLX on its CPU stream. Running the vision tower on `mx.cpu` gives cosine 1.00000000 against
the reference, and `test_cpu_stream_is_graph_exact` keeps it that way.

The audio frontend is small, so it always runs on the CPU stream to avoid that error.

RADIO also produces very large outlier activations (final features up to about 2370 against a median
of 2.5), so absolute differences look large while cosine similarity is 0.99999. The tests judge
the vision tower by cosine.

## Traps found during the port

The README lists three, documented at length in `PLAN.md`:

1. **The vision tower does not normalize its input.** NVIDIA's wrapper turns `input_conditioner`
   into an identity, so its weights in the checkpoint are unused. The processor has to apply the
   CLIP mean and std. Raw `[0, 1]` pixels give plausible but wrong output with no error.
2. **Parts of `config.json` are vestigial.** `use_thumbnail`, `force_image_size` and
   `image_tag_type` are left over from InternVL. The live image processor never reads them. Only
   dead code paths do. It is one dynamic-resolution tile, no thumbnail.
3. **`-inf` attention masking makes padded audio batches NaN.** A fully masked softmax row becomes
   NaN and spreads through later layers. The README notes NVIDIA's reference does this when two
   clips of different lengths are batched. This port masks with `-1e9` instead, which is identical
   for valid rows and finite for padded ones.

One more from `PLAN.md`: a raw prompt string, without the chat template, makes the model return
empty output. `build_prompt()` always applies the template.
