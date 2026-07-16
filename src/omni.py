#!/usr/bin/env python
# Copyright (c) 2026. MIT-licensed runtime code.
r"""Nemotron-3-Nano-Omni on MLX — generation wrapper + CLI.

    python -m src.omni --prompt "What is in this picture?" --image cat.png
    python -m src.omni --prompt "Transcribe this." --audio clip.wav
    python -m src.omni --prompt "Describe the clip." --video clip.mp4
    python -m src.omni --prompt "Hi" --dry-run     # no weights, prints the plan

Pipeline (mirrors reference/modeling.py::generate):
  1. `OmniProcessor` builds the prompt and expands <image>/<video>/<so_embedding>
     into exactly as many context tokens as the towers will emit.
  2. Towers turn pixels/waveforms into (N, llm_hidden) embeddings.
  3. We embed input_ids with the LLM's own embedding table, then SCATTER the
     tower embeddings into the positions where `input_ids == <ctx_token_id>`.
  4. EVS (EfficientVideoSampling) prunes video tokens post-splice.
  5. `mlx_lm.generate_step(..., input_embeddings=...)` runs the decode loop.

============================================================================
TOWER INTERFACE CONTRACT  (what src/vision.py and src/audio.py must expose)
============================================================================
Both towers are pure MLX `nn.Module`s that take PREPROCESSED inputs from
`src/processing.py` and return embeddings ALREADY PROJECTED to the LLM's
hidden size (2688) — i.e. pixel-shuffle + `mlp1` for vision, and
`sound_projection` for audio, are the TOWER's job, not ours.

VISION — src/vision.py
    load_vision_tower(weights_path, config_path=None, dtype=mx.float32)
        -> NemotronVisionTower                     # reads vision_model.* + mlp1.*

    tower.extract_feature(pixel_values) -> mx.array (B, N_tok, 2688)
        pixel_values: mx.array (B, 3, H, W), already normalized with
        norm_mean/norm_std by the processor — OR a list of (1, 3, H_i, W_i)
        when dynamic resolution picks different sizes per image.
        N_tok per image == processor's `num_tokens[i]` == (H/16)*(W/16)/4.

    tower.extract_video_feature(pixel_values_videos) -> mx.array (n_groups, N_tok, 2688)
        pixel_values_videos: mx.array (N_frames, 3, H, W).
        Packs T=2 consecutive frames per temporal patch (padding the tail by
        repeating the last frame), so n_groups == ceil(N_frames / 2) — which is
        exactly the number of <img>...</img> blocks the processor emitted.

AUDIO — src/audio.py
    load_audio_tower(shard_paths, config=None) -> AudioTower
        # reads sound_encoder.* + sound_projection.*

    tower(waveform, audio_lengths=None) -> (embeds (B, T', 2688), mask (B, T'))
        waveform: mx.array (B, L) float32 @ 16 kHz — the RAW waveform. Mel
        extraction lives INSIDE the tower (it owns ParakeetFeaturizer), matching
        NVIDIA, where the processor passes raw `sound_clips` through.
        T' must equal the processor's `_estimate_audio_num_embeddings(L)`:
            n_mel = 1 + L // 160;  T' = 3x stride-2 conv subsampling of n_mel.
        For B == 1 the mask is all-True and `embeds[0]` is used as-is.

Both are imported LAZILY and defensively — this file runs (and `--dry-run`
works) even while the tower files are still in flux.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, List, Optional

import numpy as np

try:
    from .processing import OmniProcessor, compute_evs_retention_mask
except ImportError:  # run as a script rather than a module
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.processing import OmniProcessor, compute_evs_retention_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REF_DIR = os.path.join(ROOT, "reference")
DEFAULT_TEXT_ONLY = os.path.join(ROOT, "text-only")
DEFAULT_MODEL_4BIT = os.path.join(ROOT, "model-4bit")


# ==========================================================================
# lazy, defensive tower imports
# ==========================================================================
def _load_vision_tower(model_dir: str, ref_dir: str):
    """Returns a vision tower or raises RuntimeError with a readable reason."""
    try:
        from .vision import load_vision_tower  # type: ignore
    except Exception:
        try:
            from src.vision import load_vision_tower  # type: ignore
        except Exception as e:
            raise RuntimeError(f"vision tower unavailable (src/vision.py): {e}") from e
    shard = _shard_for_prefix(model_dir, "vision_model.")
    return load_vision_tower(shard, os.path.join(ref_dir, "config.json"))


def _load_audio_tower(model_dir: str, ref_dir: str):
    try:
        from .audio import load_audio_tower  # type: ignore
    except Exception:
        try:
            from src.audio import load_audio_tower  # type: ignore
        except Exception as e:
            raise RuntimeError(f"audio tower unavailable (src/audio.py): {e}") from e
    shard = _shard_for_prefix(model_dir, "sound_encoder.")
    return load_audio_tower([shard])


def _shard_for_prefix(model_dir: str, prefix: str) -> str:
    """Find the safetensors shard holding a given key prefix (all towers are in shard 1)."""
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = {v for k, v in weight_map.items() if k.startswith(prefix)}
    if not shards:
        raise RuntimeError(f"no weights with prefix {prefix!r} in {idx_path}")
    if len(shards) > 1:
        raise RuntimeError(f"{prefix!r} spans multiple shards {sorted(shards)}; loader takes one")
    return os.path.join(model_dir, shards.pop())


# ==========================================================================
# LLM wrapper: teach nemotron_h to accept input_embeddings
# ==========================================================================
def _make_omni_lm(lm):
    """Wrap an `mlx_lm` nemotron_h model so it accepts `input_embeddings=`.

    mlx_lm's `generate_step` gates on `does_model_support_input_embeddings`,
    which inspects `model.__call__` for an `input_embeddings` parameter.
    nemotron_h's `Model.__call__(inputs, cache)` has none, and its backbone
    always does `hidden_states = self.embeddings(inputs)`. So we wrap it and
    swap `backbone.embeddings` for a constant-returning stub during the call.
    The swap is restored in a `finally` — and because MLX builds the graph
    eagerly (only evaluation is lazy), the spliced embeds are already captured.
    """
    import mlx.nn as nn

    class _ConstEmbedding:
        __slots__ = ("value",)

        def __init__(self, value):
            self.value = value

        def __call__(self, _inputs):
            return self.value

    class OmniLM(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.lm = base

        def __call__(self, inputs, cache=None, input_embeddings=None):
            if input_embeddings is None:
                return self.lm(inputs, cache=cache)
            backbone = self.lm.backbone
            real = backbone.embeddings
            backbone.embeddings = _ConstEmbedding(input_embeddings)
            try:
                out = backbone(inputs, cache=cache)
            finally:
                backbone.embeddings = real
            return self.lm.lm_head(out)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "lm":
                    raise
                return getattr(self.lm, name)  # make_cache, layers, ...

    return OmniLM(lm)


def _embed_tokens(lm, input_ids):
    """Run the LLM's own embedding table over (1, N) token ids -> (1, N, C)."""
    import mlx.core as mx

    return lm.backbone.embeddings(mx.array(np.asarray(input_ids, dtype=np.int32)))


# ==========================================================================
# the splice
# ==========================================================================
def splice_embeddings(inputs_embeds, input_ids, ctx_token_id: int, tower_embeds):
    """Scatter `tower_embeds` into `inputs_embeds` where `input_ids == ctx_token_id`.

    This is the standard InternVL splice; see reference/modeling.py:496-547.

    Args:
        inputs_embeds: mx.array (1, N, C)
        input_ids:     np.ndarray (1, N)
        ctx_token_id:  the context token to replace
        tower_embeds:  mx.array (..., C) — flattened to (M, C); M must equal the
                       number of context tokens present.
    Returns: mx.array (1, N, C)
    """
    import mlx.core as mx

    c = inputs_embeds.shape[-1]
    flat = tower_embeds.reshape(-1, c)
    positions = np.where(np.asarray(input_ids).reshape(-1) == ctx_token_id)[0]
    if len(positions) == 0:
        raise ValueError(f"No context tokens (id={ctx_token_id}) found in input_ids")
    if len(positions) != flat.shape[0]:
        raise ValueError(
            f"context token count ({len(positions)}) != tower embedding count "
            f"({flat.shape[0]}) for token id {ctx_token_id}"
        )
    out = inputs_embeds.reshape(-1, c)
    out[mx.array(positions.astype(np.int32))] = flat.astype(out.dtype)
    return out.reshape(inputs_embeds.shape)


def apply_evs(inputs_embeds, input_ids, video_embeds, ctx_token_id: int, pruning_rate: float):
    """EfficientVideoSampling: drop redundant video tokens after the splice.

    Mirrors reference/modeling.py:550-566. Returns (inputs_embeds, input_ids).
    """
    import mlx.core as mx

    n_groups, n_tok, _ = video_embeds.shape
    h = w = int(round(n_tok**0.5))
    if h * w != n_tok:
        # EVS assumes a square token grid (same assumption NVIDIA makes)
        return inputs_embeds, input_ids

    evs_mask = compute_evs_retention_mask(
        np.asarray(video_embeds.astype(mx.float32)),
        (n_groups, h, w),
        spatial_merge_size=1,
        q=pruning_rate,
    )
    ids = np.asarray(input_ids).reshape(-1)
    retention = np.ones_like(ids, dtype=bool)
    retention[ids == ctx_token_id] = evs_mask
    keep = np.where(retention)[0]
    print(
        f"[evs] retained {int(evs_mask.sum())}/{evs_mask.size} video tokens "
        f"({100 * evs_mask.mean():.1f}%), prompt {len(ids)} -> {len(keep)} tokens"
    )
    idx = mx.array(keep.astype(np.int32))
    return inputs_embeds.reshape(-1, inputs_embeds.shape[-1])[idx][None], ids[keep][None]


# ==========================================================================
# prompt construction
# ==========================================================================
def build_prompt(proc, prompt: str, has_image: bool, has_audio: bool, has_video: bool) -> str:
    """Chat-template the user turn, with the modality placeholders inline.

    MANDATORY (P0 gotcha): a raw string prompt makes this reasoning model return
    EMPTY output — the chat template must be applied. We template FIRST (to a
    string), then let the processor expand the placeholders and tokenize.
    """
    parts: List[str] = []
    if has_image:
        parts.append("<image>")
    if has_video:
        # NVIDIA's processor deliberately does NOT emit this prefix — it's
        # expected to come from the client message (matches vLLM + training).
        parts.append("This is a video:\n<video>")
    if has_audio:
        parts.append("<so_embedding>")
    parts.append(prompt)
    content = "\n".join(parts)

    return proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def load_media(image: Optional[str], video: Optional[str], num_frames: int = 8):
    from PIL import Image

    img = Image.open(image).convert("RGB") if image else None
    frames = _read_video_frames(video, num_frames) if video else None
    return img, frames


def _read_video_frames(path: str, num_frames: int):
    """Uniformly sample `num_frames` PIL frames. Tries decord, then PyAV, then PIL."""
    from PIL import Image

    try:
        import decord  # type: ignore

        vr = decord.VideoReader(path)
        idx = np.linspace(0, len(vr) - 1, num_frames).round().astype(int)
        return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in idx]
    except Exception:
        pass
    try:
        import av  # type: ignore

        with av.open(path) as container:
            frames = [f.to_image().convert("RGB") for f in container.decode(video=0)]
        if not frames:
            raise RuntimeError("no frames decoded")
        idx = np.linspace(0, len(frames) - 1, min(num_frames, len(frames))).round().astype(int)
        return [frames[i] for i in idx]
    except Exception as e:
        raise RuntimeError(
            f"could not decode {path!r}: install `decord` or `av` for video support ({e})"
        ) from e


# ==========================================================================
# main
# ==========================================================================
def generate(
    prompt: str,
    image: Optional[str] = None,
    audio: Optional[str] = None,
    video: Optional[str] = None,
    ref_dir: str = DEFAULT_REF_DIR,
    text_only_dir: str = DEFAULT_TEXT_ONLY,
    model_dir: str = DEFAULT_MODEL_4BIT,
    max_tokens: int = 512,
    temp: float = 0.0,
    num_frames: int = 8,
    dry_run: bool = False,
    verbose: bool = True,
) -> str:
    proc = OmniProcessor.from_pretrained(ref_dir)
    img, frames = load_media(image, video, num_frames)

    templated = build_prompt(proc, prompt, img is not None, audio is not None, frames is not None)
    inputs = proc(
        text=templated,
        images=img,
        videos=frames,
        audio=audio,
    )
    input_ids = inputs["input_ids"]

    if verbose:
        n_img = int((input_ids == proc.image_token_id).sum())
        n_snd = int((input_ids == proc.audio_token_id).sum())
        print(
            f"[proc] {input_ids.shape[1]} tokens "
            f"({n_img} image/video ctx, {n_snd} sound ctx)"
        )
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            shp = pv.shape if isinstance(pv, np.ndarray) else [p.shape for p in pv]
            print(f"[proc] pixel_values {shp}, num_tokens={inputs['num_tokens']}")
        if "pixel_values_videos" in inputs:
            print(f"[proc] pixel_values_videos {np.asarray(inputs['pixel_values_videos']).shape}")
        if "sound_clips" in inputs:
            print(f"[proc] sound_clips {[c.shape for c in inputs['sound_clips']]}")

    if dry_run:
        print("\n--- expanded prompt (context tokens collapsed) ---")
        text = inputs["expanded_text"][0]
        for tok, name in (("<image>", "image"), ("<so_embedding>", "sound")):
            n = text.count(tok)
            if n:
                text = text.replace(tok * n, f"...<{name} x{n}>...") if tok * n in text else text
        print(text[:2000])
        return ""

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    t0 = time.time()
    lm, _tok = load(text_only_dir)
    if verbose:
        print(f"[llm] loaded in {time.time() - t0:.1f}s")

    inputs_embeds = _embed_tokens(lm, input_ids)

    # --- vision splice
    video_embeds = None
    if img is not None:
        tower = _load_vision_tower(model_dir, ref_dir)
        pv = inputs["pixel_values"]
        pv_mx = (
            [mx.array(p)[None] for p in pv] if isinstance(pv, list) else mx.array(pv)
        )
        embeds = tower.extract_feature(pv_mx)
        inputs_embeds = splice_embeddings(inputs_embeds, input_ids, proc.image_token_id, embeds)
        del tower
    if frames is not None:
        tower = _load_vision_tower(model_dir, ref_dir)
        video_embeds = tower.extract_video_feature(mx.array(inputs["pixel_values_videos"]))
        inputs_embeds = splice_embeddings(
            inputs_embeds, input_ids, proc.image_token_id, video_embeds
        )
        del tower

    # --- audio splice
    if audio is not None:
        tower = _load_audio_tower(model_dir, ref_dir)
        wav = inputs["sound_clips"][0]
        embeds, _mask = tower(mx.array(wav.astype(np.float32))[None])
        inputs_embeds = splice_embeddings(inputs_embeds, input_ids, proc.audio_token_id, embeds)
        del tower

    # --- EVS (video only, post-splice; see reference/modeling.py:550)
    pruning_rate = float(proc.config.get("video_pruning_rate", 0.0) or 0.0)
    if video_embeds is not None and pruning_rate > 0:
        inputs_embeds, input_ids = apply_evs(
            inputs_embeds, input_ids, video_embeds, proc.image_token_id, pruning_rate
        )

    mx.eval(inputs_embeds)

    # --- decode
    model = _make_omni_lm(lm)
    sampler = make_sampler(temp=temp)
    ids = mx.array(np.asarray(input_ids).reshape(-1).astype(np.int32))
    embeds_seq = inputs_embeds[0]

    eos = _eos_ids(proc.tokenizer, text_only_dir)
    out_tokens: List[int] = []
    t0 = time.time()
    for token, _lp in generate_step(
        ids,
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        input_embeddings=embeds_seq,
    ):
        t = int(token)
        if t in eos:
            break
        out_tokens.append(t)
    dt = time.time() - t0

    text = proc.tokenizer.decode(out_tokens)
    if verbose:
        print(f"[gen] {len(out_tokens)} tokens in {dt:.1f}s = {len(out_tokens) / max(dt, 1e-9):.1f} tok/s")
        print(f"[mem] peak {mx.get_peak_memory() / 1e9:.1f} GB")
    return text


def _eos_ids(tokenizer, model_dir: str) -> set:
    ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    gc = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gc):
        with open(gc) as f:
            cfg = json.load(f)
        e = cfg.get("eos_token_id")
        if isinstance(e, int):
            ids.add(e)
        elif isinstance(e, list):
            ids.update(int(x) for x in e)
    return ids or {11}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Nemotron-3-Nano-Omni on MLX")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", help="path to an image file")
    p.add_argument("--audio", help="path to a 16 kHz wav file")
    p.add_argument("--video", help="path to a video file")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--num-frames", type=int, default=8, help="frames sampled from --video")
    p.add_argument("--ref-dir", default=DEFAULT_REF_DIR)
    p.add_argument("--text-only-dir", default=DEFAULT_TEXT_ONLY)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_4BIT)
    p.add_argument("--dry-run", action="store_true", help="processor only, no weights loaded")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    text = generate(
        prompt=a.prompt,
        image=a.image,
        audio=a.audio,
        video=a.video,
        ref_dir=a.ref_dir,
        text_only_dir=a.text_only_dir,
        model_dir=a.model_dir,
        max_tokens=a.max_tokens,
        temp=a.temp,
        num_frames=a.num_frames,
        dry_run=a.dry_run,
        verbose=not a.quiet,
    )
    if text:
        print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
