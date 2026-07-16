# Nemotron-3-Nano-Omni 30B-A3B → full omni runtime on Apple Silicon (MLX)

**Goal:** first open runtime that runs NVIDIA's tri-modal Nemotron Omni (text + vision + audio)
fully on a Mac. Weights already exist (`mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit`,
~19 GB, LLM 4-bit + towers bf16). What's missing is the vision/audio forward passes in MLX.
Ship: working demo (image caption + audio transcribe/describe on M5), HF model-card update or
companion repo, benchmark video for the channel.

## Directory layout
- `reference/` — NVIDIA's HF custom code (PyTorch, the porting spec)
- `reference-radio/` — nvidia/C-RADIOv4-H HF repo code (vision tower internals)
- `model-4bit/` — mlx-community 4-bit weights (download in progress, see download.log)
- `src/` — our MLX runtime (to build)
- `tests/` — parity tests vs PyTorch reference

## Architecture (from reference code, verified 2026-07-15)
Wrapper class `NemotronH_Nano_Omni_Reasoning_V3` (modeling.py, InternVL-style):

1. **LLM backbone** — `NemotronHForCausalLM`: hybrid Mamba2 + MoE + attention,
   30B total / ~3B active, 128-expert top-6, hidden 2688, 52 layers, 262k ctx.
   ✅ mlx_lm already has `models/nemotron_h.py` — reuse, do NOT rewrite.
   (mlx-server python: `~/.local/mlx-server/bin/python`, mlx 0.31.1, mlx_lm 0.31.2, mlx_vlm 0.4.4)
2. **Vision tower** — C-RADIOv4-H ViT-H, patch 16 (`vision_model.*` keys, bf16 in checkpoint).
   Loaded via trust_remote_code from nvidia/C-RADIOv4-H (timm VisionTransformer + CPE +
   custom `ViTPatchGenerator`, `input_conditioner`, `make_preprocessor_external()`).
   Port strategy: port the WEIGHT GRAPH, not the framework — enumerate `vision_model.*`
   keys in model.safetensors.index.json and implement exactly those modules.
   Extra: wrapper attaches a **3D patch projection for video frames** on top of the 2D
   patch generator (modeling.py ~line 101-113); InternVL pixel-shuffle (`downsample_ratio`,
   `ps_version` in config) then `mlp1` projector → LLM hidden.
3. **Audio tower** — `ParakeetEncoder` from HF transformers (Conformer: learned-filterbank
   frontend, BatchNorm conv module, Transformer-XL rel-pos attention). `sound_config` in
   config.json has all dims (num_mel_bins, subsampling_*, conv_kernel_size...).
   Reference impl = transformers' parakeet model code (upstreamed), NOT custom NVIDIA code.
   Then `SoundProjection` MLP (audio_model.py, 174 lines total — small).
4. **Processing** — processing.py (509 l): InternVL dynamic tiling for images
   (`use_thumbnail`, `force_image_size`, norm_mean/std), video via video_processing.py +
   `EfficientVideoSampling` (evs.py, `video_pruning_rate`), audio 16k mel features.
   Context tokens: `img_context_token_id`, `video_context_token_id`, `sound_context_token_id`
   — embeddings spliced at those token positions (standard InternVL splice).

## Phases
- [x] Scout: weights exist, no open runtime; mlx_lm has nemotron_h; scope confirmed
- [x] **P0 smoke:** ✅ PASSED 2026-07-16. Text-only generate works via mlx_lm.
      **152.5 tok/s, 17.9 GB peak, 1.0s load** on the M5. Reasoning model — emits `<think>`
      then `</think>` then the answer; MUST use `tok.apply_chat_template(...)` (raw string
      prompt returns EMPTY output — first gotcha).
      Recipe: `text-only/` dir holds a filtered checkpoint (only `backbone.*` + `lm_head.*`,
      17.8 GB, written by filtering all shards of `model-4bit/`) + `llm_config` promoted to
      config.json with `quantization` copied in + tokenizer files. mlx_lm's `load()` rejects
      the full checkpoint (chokes on the unexpected `vision_model.*`/`sound_*` keys), and
      symlinking shards doesn't help since it globs every .safetensors in the dir — so the
      filtered copy is required. `mlx_lm.load('text-only')` then works unmodified.
      Checkpoint key prefixes (all): backbone(726), vision_model(390), sound_encoder(710),
      mlp1(3), sound_projection(3), lm_head(3). Note: **no `language_model.*` prefix** — the
      LLM keys are `backbone.*`.
- [x] **P1 audio (easier):** port ParakeetEncoder + SoundProjection to MLX in `src/audio.py`.
      Parity: same wav → transformers ParakeetEncoder (CPU) vs MLX, cos-sim > 0.999 per frame.
      ✅ DONE 2026-07-16, see "P1 results" below. (Audio-only chat demo deferred to P3 —
      needs the P0 backbone + token splicing.)
- [x] **P2 vision:** port RADIO ViT-H forward (patch gen incl. video 3D proj, blocks, feature head)
      + pixel shuffle + mlp1. Parity: same image → PyTorch `.features` vs MLX. ✅ PASS, see "P2 results"
- [x] **P3 glue:** processor port (tiling/EVS/mel), token splicing, generation wrapper CLI.
      ✅ PASSED 2026-07-16, see "P3 results" below. Shipped as `src/omni.py`
      (`python -m src.omni --prompt ... [--image|--audio|--video] [--dry-run]`) — all three
      modalities generate end-to-end on the M5. 14/14 processor parity tests.
- [ ] **P4 ship:** README + HF upload (companion code repo or PR to mlx-vlm), record demo +
      benchmark video (tok/s, time-to-first-token, RAM), add to claude-code-local README lineup.

## P1 results (audio tower parity, 2026-07-16)
**PASS.** Full waveform→embedding path ported to pure MLX (`src/audio.py`, no torch).
Test: `~/.local/mlx-server/bin/python tests/test_audio_parity.py` (also pytest-compatible).
Reference: transformers 5.5.0 `ParakeetEncoder` + NVIDIA `SoundProjection`, CPU fp32, real
checkpoint weights from BF16 shard 1 (`weights-bf16/model-00001-of-00017.safetensors`, 3.7 GB —
ALL 713 `sound_encoder.*`/`sound_projection.*` keys live in that one shard per index.json).

- **Numbers (5 s / 16 kHz synthetic sweep fixture, `tests/fixtures/audio_5s_16k.{npy,wav}`):**
  - frontend (log-mel) max |diff|: **8.1e-6**
  - encoder cos-sim per frame: min **0.99999410**, mean 0.99999821
  - final projected cos-sim per frame: min **0.99999130**, mean 0.99999726 (target > 0.999)
  - padded batch (5 s + 3 s) vs per-item unpadded reference: min cos **0.99994** — mask paths verified
- **Shapes:** wav (1, 80000) → mel (1, 501, 128) [500 valid] → encoder (1, 63, 1024) → projected
  (1, 63, 2688). ~12.6 audio frames/sec of LLM tokens (80 ms per frame: 10 ms hop × 8x subsample).
- **Checkpoint layout:** `sound_encoder.encoder.{feature_extractor.featurizer.{fb,window},
  subsampling.{layers.{0,2,3,5,6},linear},layers.0-23.*}`, `sound_projection.{norm,linear1,linear2}`,
  all bf16. fb is (1, 128, 257) NeMo layout → transpose to (257, 128); it's just a slaney mel
  filterbank stored in bf16 (max diff vs analytic 1.2e-4), window = hann(400, periodic=False) in bf16.

**Gotchas found (P2/P3 read this):**
1. **Metal fp32 matmul/FFT are lossy** (~7e-4 relative err, fast reductions) — fine for the encoder
   (24 layers still hit 0.99999 cos) but too lossy for the log-mel frontend; the featurizer runs on
   `mx.stream(mx.cpu)` (tiny compute). P2: don't chase parity failures below ~1e-3 max-diff on GPU,
   it's the hardware, not your port.
2. **HF's -inf attention mask NaNs on padded batches**: fully-padded query rows → softmax NaN →
   poisons later layers via k/v. Even the HF torch reference produces NaN for the shorter item in a
   padded batch. MLX port masks with -1e9 instead (bit-identical for valid rows — exp underflows to
   0) and stays finite. Parity for batches must compare vs per-item unpadded torch runs.
3. **`scale_input` trap:** HF `ParakeetEncoderConfig` defaults `scale_input=True` (×√d), but NVIDIA's
   `SoundEncoder` wrapper defaults it **False** via getattr — and `sound_config` omits it. Same for
   `attention_bias`/`convolution_bias` (False → no linear/conv biases). Build configs the way the
   wrapper does, not from HF defaults.
4. **BatchNorm in the conv module** must use running stats (eval mode); implemented as an explicit
   inference-only BatchNorm (`num_batches_tracked` keys are skipped on load).
5. **Rel-pos attention details:** pos encoding is *interleaved* sin/cos (not concatenated halves),
   positions run T-1 … -(T-1); matrix_bd gets the Transformer-XL rel-shift (pad+reshape trick) and
   is scaled by d^-0.5 *separately* before being added as the attention bias.
6. **Subsampling layout:** torch flattens conv output as (B, T', C, F)→(B, T', C·F); MLX conv is
   NHWC so transpose (B,T',F,C)→(B,T',C,F) before the flatten, or the linear layer sees permuted
   features. Lengths mask is applied after every conv, but lengths only update on strided convs.
7. **Feature length formula** `(L + 2·(n_fft//2) − n_fft) // hop` yields 500 valid of 501 STFT
   frames for exact multiples — the last frame is always masked. Mel normalization is per-utterance,
   per-bin over valid frames with (n−1) variance denominator.
8. **librosa is NOT installed** in the mlx-server venv → `ParakeetFeatureExtractor` can't even
   construct. P3's processor port should use the checkpoint fb/window (already loaded in
   `src/audio.py`) instead of librosa. Verified equivalent to slaney mel to bf16 precision.

## P2 results (vision tower parity, 2026-07-16)
**PASS.** Full image/video → LLM-hidden path ported to pure MLX (`src/vision.py`, no torch).
Test: `~/.local/mlx-server/bin/python -m pytest tests/test_vision_parity.py -q -s` (7 passed, ~12 s).
Reference: real `nvidia/C-RADIOv4-H` via `AutoModel.from_config(cfg.vision_config, trust_remote_code=True)`
(the exact call modeling.py:97 makes) + the wrapper's `mlp1`, CPU fp32, real checkpoint weights from
BF16 shard 1 — **all 393 `vision_model.*`(390) + `mlp1.*`(3) keys live in that one shard**, which P1
had already downloaded, so no extra download was needed.

- **Numbers (fixtures `tests/fixtures/image_{448,512}.{npy,png}`, gradient+noise, seed 1234):**
  | case | shape | cos min | cos mean |
  |---|---|---|---|
  | image 448 `.features` | (1, 784, 1280) | **0.99999753** | 0.99999989 |
  | image 448 projected | (1, 196, 2688) | **0.99996227** | 0.99999687 |
  | image 512 projected | (1, 256, 2688) | **0.99995915** | 0.99999709 |
  | video 4 frames @448 | (2, 196, 2688) | **0.99987751** | 0.99999669 |
  | video 3 frames (padded) | (2, 196, 2688) | **0.99979961** | 0.99999509 |
  | **image 448 projected, MLX `mx.cpu` stream** | (1, 196, 2688) | **1.00000000** | 1.00000000 |
- **Shapes:** (1,3,512,512) → 1024 patches → +10 prefix tokens → ViT (1,1034,1280) → drop 10 →
  (1,1024,1280) → pixel-shuffle 0.5 → (1,256,5120) → mlp1 → **(1,256,2688)** = `num_image_token` 256/tile.
  448px → 784 patches → 196 tokens. Video: T=2 frames/temporal patch → N/T groups × 196 tokens.
- **Arch confirmed live against the torch model** (not just inferred): 32 blocks, dim 1280, 16 heads
  (scale 80^-0.5), mlp 5120, LN eps 1e-6, GELU(erf), CPE grid 128×128, num_skip=10 (4 cls + 6 registers).

**Gotchas found (P3 read this):**
1. **The vision tower does NOT normalize its own input.** The wrapper calls
   `make_preprocessor_external()`, turning `input_conditioner` into `nn.Identity` — so the
   `vision_model.radio_model.input_conditioner.norm_mean/std` keys in the checkpoint are **dead
   weight**. The processor must apply `(x/255 − norm_mean)/norm_std` using the **top-level**
   config's CLIP stats. Feed raw [0,1] pixels and you get garbage silently.
   (The MLX port keeps the conditioner loadable but off by default: `apply_conditioner=False`.)
2. **Three layers are Identity by config, and their weights are simply absent** — don't "fix" it:
   `model.norm` (args.model_norm=False → **no final LayerNorm**), `feature_normalizer`
   (feature_normalizer_config=null, despite args saying `SHIP_NORM`), `patch_normalizer`.
   Also `vitdet_window_size=null` → plain global attention; spectral reparam is pre-merged into qkv.
3. **`radio_model.summary_idxs` is in the torch module but NOT in the omni checkpoint** →
   `load_state_dict(..., strict=False)` and allow exactly that one missing key. Also: load the state
   dict **before** `make_preprocessor_external()`, else the conditioner keys become "unexpected".
4. **Metal fp32 matmul is lossy** — independently confirms P1's gotcha #1: vs a float64 ground truth,
   torch-CPU and MLX-**cpu**-stream both sit at ~1.3e-6 rel, MLX **gpu** at ~7.7e-4. That is the
   *entire* source of the GPU residual here; on `mx.stream(mx.cpu)` the port is bit-tight (cos
   1.00000000, 6.4e-6 rel). Locked in as `test_cpu_stream_is_graph_exact`. Don't chase GPU deltas.
5. **RADIO has massive outlier activations** — final features hit **|max| ≈ 2370** (median 2.5), and
   the residual stream reaches ~5570 by block 31. So `max|Δ|` looks alarming (13.1) while cosine is
   0.999999. Judge this tower by cosine/relative error, never absolute diff. Also: **bf16 max is
   ~3.4e38 so no overflow, but bf16 has ~3 decimal digits — expect real precision loss at 2370.**
   Parity here is fp32-vs-fp32; the shipped runtime runs the tower in bf16, so re-check quality there.
6. **CPE pos_embed is interpolated, not cropped**: eval path bilinearly resizes the 128×128 grid to
   `max(h,w)` (align_corners=**False**) then window-crops to (h,w). Hand-ported as
   `_interpolate_bilinear_nchw` (torch's `scale*(i+0.5)−0.5` clamped at 0); verified exact to 2e-7 on
   the real path, incl. non-square grids. Any tile size that's a multiple of 16 works (min_resolution_step).
7. **Video reuses the image ViT via a channel-stacking trick**: (N,3,H,W) → (N/T, T·3, H, W) so RADIO's
   channel-agnostic `Im2Patches` emits (·, npatch, T·3·P²) — exactly `video_embedder`'s input layout.
   Odd frame counts pad by **repeating the last frame**. Row-major reshape order matters (per-patch
   feature order is [t0,c0..c2, t1,c0..c2]); a transpose here would silently mismatch the weights.
   `video_embedder` is attached by the *wrapper*, not by C-RADIOv4-H itself.
8. **`extract_feature` accepts a list** for dynamic resolution but ends in `torch.cat(dim=0)`, so all
   entries must share a tile size (differing sizes fail on the torch side too).
9. **Deps installed into the mlx-server venv for the torch reference only:** `timm` (1.0.28),
   `einops`, `open_clip_torch` (trust_remote_code's import check hard-requires `open_clip` even
   though the adaptor catches the ImportError), `pytest`. `src/vision.py` itself imports only mlx.

## Parity harness notes
- PyTorch reference runs on CPU (bf16→fp32) — only needs tower weights, pull the specific
  shards for `vision_model.*`/`sound_encoder.*` from the BF16 repo via index.json filtering.
- Golden inputs: fixed 448px test image, 5s 16k wav, 8-frame video clip in `tests/fixtures/`.

## Gotchas / decisions
- MLX 4-bit repo README says towers are bf16 with LLM tensor layout byte-identical to
  `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` (text sibling) → mlx_lm loads LLM as-is.
- Watch memory: keep peak < 40 GB (model 20 GB + activations); Matt's rule — never crash the
  machine, check `vm.swapusage` before big loads (swap was ~17.4/18.4 GB used on 2026-07-15).
- License: nvidia-open-model-license on weights; our runtime code can be MIT, don't rehost
  NVIDIA weights beyond what mlx-community already did (Matt: never delete from HF).

---

## P3 results — processor + splice + CLI ✅ PASSED 2026-07-16

Files: `src/processing.py`, `src/omni.py`, `tests/test_processing_parity.py`,
fixtures `tests/fixtures/proc_*.png|wav|npy`. **14/14 parity tests pass** vs NVIDIA's
reference processor (`AutoProcessor.from_pretrained('reference/', trust_remote_code=True)` —
weight-free, no towers needed).

### Parity numbers (ours vs PyTorch reference)
| case | input_ids | pixel max-abs-diff |
|---|---|---|
| 448x448 image (upscaled to 512x512, 256 tok) | EXACT | **0.0** |
| 1024x768 image (native, 768 tok) | EXACT | **0.0** |
| 300x700 image (→352x800, 275 tok) | EXACT | **1.55e-06** |
| 2-image batch (mixed aspect → list of tiles) | EXACT | **< 1e-06** |
| 5 video frames 320x180 (→384x672) | EXACT | **1.19e-06** |
| 5 s / 16 kHz wav (63 sound tokens) | EXACT | 0.0 (waveform) |
| image + audio mixed | EXACT | 0.0 |
Target was < 1e-4 → **~65x margin**. `num_tokens` / `num_patches` / `imgs_sizes` all exact.
EVS retention mask is bit-identical to `reference/evs.py`.

### End-to-end (real weights, M5)
`python -m src.omni --prompt ... [--image|--audio|--video]`, all three towers spliced:
- text-only: 109 tok/s, **17.9 GB** peak
- +image: 67.7 tok/s, **22.1 GB** peak — correctly described the synthetic grid fixture
- +audio: 110.5 tok/s, **21.0 GB** peak — correctly called the 440 Hz fixture "a steady
  electronic tone, resembling a sine wave"
Token counts predicted by the processor matched both towers' real output exactly (no
shape-mismatch asserts) — P1/P2/P3 interlock confirmed.

### Tower interface contract (implemented by P1/P2, consumed by `src/omni.py`)
Towers take PREPROCESSED inputs and return embeddings **already projected to LLM hidden
(2688)** — pixel-shuffle+`mlp1` and `sound_projection` are the tower's job.
- `vision.load_vision_tower(weights_path, config_path=None, dtype=mx.float32) -> NemotronVisionTower`
  - `.extract_feature(pv)` — `pv` = mx (B,3,H,W) normalized, **or a list** of (1,3,H_i,W_i)
    for mixed aspect ratios → `(B, N_tok, 2688)`, `N_tok == num_tokens[i] == (H/16)(W/16)/4`
  - `.extract_video_feature(pv_videos)` — mx (N_frames,3,H,W), packs T=2 frames/temporal
    patch (tail padded by repeating last frame) → `(ceil(N/2), N_tok, 2688)`
- `audio.load_audio_tower(shard_paths, config=None) -> AudioTower`
  - `tower(waveform (B,L) float32 @16k, audio_lengths=None) -> (embeds (B,T',2688), mask)`
  - mel extraction lives **inside** the tower (NVIDIA passes raw `sound_clips` through);
    `T'` must equal `processor._estimate_audio_num_embeddings(L)`
- All tower weights live in **shard 1** (`model-00001-of-00017.safetensors`).
- Towers are imported lazily/defensively — `--dry-run` works with no weights at all.

### Gotchas (P3)
1. **The image path is NOT classic InternVL tiling.** `config.json` still carries
   `use_thumbnail=True` / `force_image_size=512` / `image_tag_type` from the InternVL
   lineage, but the live `NemotronH_Nano_Omni_Reasoning_V3ImageProcessor` **ignores all
   three**: ONE dynamic-resolution tile per image, **no thumbnail tile**. `use_thumbnail`/
   `force_image_size` are only read by dead code (`processing_utils.dynamic_preprocess`,
   `video_processing.py`) that `__call__` never invokes. Don't "fix" this.
2. **torch's antialiased bicubic uses Pillow's cubic a = -0.5**, NOT the a = -0.75 of plain
   `mode="bicubic"` (`HelperInterpCubic::aa_filter`). Using -0.75 costs **27/255** max-abs-diff.
3. **ATen computes the resample weights in float32** (`scalar_t`), not double. Computing them
   in float64 is *more accurate* but disagrees with torch by **5.2e-3**; matching float32
   drops it to **1.2e-4** (→ ~1.5e-6 after `/255` and norm_std). Separable order is
   **width-first, then height**.
4. **`<video>` has no real token** — it maps to id 0 (unk). The processor reuses `<image>`
   (id 18) for video positions; the model tells them apart by which `pixel_values_*` arg was
   passed. So the video splice keys on `img_context_token_id`, and `video_context_token_id`
   (131081) in config.json is **never used**.
5. **mlx_lm's nemotron_h can't take input embeddings.** `generate_step` gates on
   `does_model_support_input_embeddings` (inspects `__call__` for an `input_embeddings`
   param); nemotron_h's is `(inputs, cache)` and its backbone always does
   `self.embeddings(inputs)`. `src/omni.py::_make_omni_lm` wraps it and swaps
   `backbone.embeddings` for a constant-returning stub during the call (restored in
   `finally`; safe because MLX builds the graph eagerly and only *evaluates* lazily). The
   wrapper delegates `make_cache`/`layers` via `__getattr__`.
6. **No librosa/soundfile in the mlx-server env** → the reference's `_load_audio(path)` raises
   ImportError, so file-path audio is unusable with the reference processor. Parity tests feed
   raw waveforms; `src/processing.py` adds a stdlib `wave` fallback so `--audio x.wav` works.
7. **Audio token count is exact, not a heuristic**: `n_mel = 1 + L//160` (STFT center pad) then
   3 stride-2 conv stages. 5 s @ 16k → 501 mel frames → **63** tokens. A wrong count trips a
   shape-mismatch assert in the splice.
8. **EVS runs post-splice and shrinks the prompt** (`video_pruning_rate=0.7` → keeps
   `max(1 frame, T*H*W*0.3)` tokens), rewriting `inputs_embeds` AND `input_ids` together.
   It assumes a square token grid — same assumption NVIDIA makes.
9. `round()` in the tiling math is Python's **banker's rounding** (`round(28.5) == 28`) — the
   reference relies on it, so the port must use Python `round`, not `np.round`/`floor(x+0.5)`.
