"""P3 parity: our MLX/numpy processor vs NVIDIA's PyTorch reference processor.

Targets:
  * input_ids            — EXACT match
  * pixel_values         — max-abs-diff < 1e-4
  * pixel_values_videos  — max-abs-diff < 1e-4
  * num_tokens / num_patches / imgs_sizes — exact

The reference processor is instantiated straight from `reference/` (which holds
NVIDIA's HF custom code + tokenizer). It is WEIGHT-FREE — no towers needed.

Run:  ~/.local/mlx-server/bin/python -m pytest tests/test_processing_parity.py -q
      ~/.local/mlx-server/bin/python tests/test_processing_parity.py   # standalone
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.processing import OmniProcessor, resize_bicubic_antialias  # noqa: E402

REF_DIR = os.path.join(ROOT, "reference")
FIX = os.path.join(ROOT, "tests", "fixtures")
PIXEL_TOL = 1e-4


# --------------------------------------------------------------------------
# fixtures (deterministic, committed as proc_*.png / proc_*.npy / proc_*.wav)
# --------------------------------------------------------------------------
def _make_image(name: str, w: int, h: int, seed: int) -> str:
    path = os.path.join(FIX, name)
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    # smooth structure + high-frequency noise: stresses the resample kernel
    base = np.stack(
        [
            (255 * xx / max(w - 1, 1)),
            (255 * yy / max(h - 1, 1)),
            (127 + 127 * np.sin(xx / 7.0) * np.cos(yy / 11.0)),
        ],
        axis=-1,
    )
    noise = rng.integers(-40, 40, size=base.shape)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    os.makedirs(FIX, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _make_audio(name: str, seconds: float = 5.0, sr: int = 16000, seed: int = 7) -> str:
    """5 s / 16 kHz PCM_16 wav, written with stdlib `wave` (no soundfile in env)."""
    path = os.path.join(FIX, name)
    npy = os.path.join(FIX, name.replace(".wav", ".npy"))
    if os.path.exists(path) and os.path.exists(npy):
        return path
    import wave

    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    wav = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.1 * rng.standard_normal(n)
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    os.makedirs(FIX, exist_ok=True)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((wav * 32767.0).astype(np.int16).tobytes())
    np.save(npy, wav)
    return path


IMG_SQUARE = _make_image("proc_image_448.png", 448, 448, seed=1234)
IMG_WIDE = _make_image("proc_image_1024x768.png", 1024, 768, seed=99)
IMG_TALL = _make_image("proc_image_300x700.png", 300, 700, seed=5)
AUDIO_WAV = _make_audio("proc_audio_5s_16k.wav")


# --------------------------------------------------------------------------
# processors
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ref_proc():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(REF_DIR, trust_remote_code=True)


@pytest.fixture(scope="module")
def our_proc():
    return OmniProcessor.from_pretrained(REF_DIR)


def _maxdiff(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


# --------------------------------------------------------------------------
# 1. the resize kernel on its own vs torch
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "in_hw,out_hw",
    [
        ((448, 448), (512, 512)),  # upscale (antialias inert)
        ((768, 1024), (512, 688)),  # downscale (antialias active)
        ((700, 300), (688, 288)),  # anisotropic
        ((64, 64), (513, 511)),  # odd targets
    ],
)
def test_resize_matches_torch(in_hw, out_hw):
    import torch

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(3, *in_hw)).astype(np.float32)
    ours = resize_bicubic_antialias(arr, *out_hw)
    ref = (
        torch.nn.functional.interpolate(
            torch.from_numpy(arr).unsqueeze(0),
            size=out_hw,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .squeeze(0)
        .numpy()
    )
    # tolerance here is on the 0-255 scale; /255/std afterwards shrinks it ~4x
    d = _maxdiff(ours, ref)
    assert d < 1e-2, f"resize max-abs-diff {d} on 0-255 scale"


# --------------------------------------------------------------------------
# 2. image path parity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("img_path", [IMG_SQUARE, IMG_WIDE, IMG_TALL])
def test_image_parity(ref_proc, our_proc, img_path):
    img = Image.open(img_path).convert("RGB")
    text = "<image>\nDescribe this image in detail."

    ref = ref_proc(images=img, text=text, return_tensors="pt")
    ours = our_proc(text=text, images=img)

    # token ids: EXACT
    ref_ids = ref["input_ids"].numpy()
    assert ours["input_ids"].shape == ref_ids.shape
    assert np.array_equal(ours["input_ids"], ref_ids), "input_ids mismatch"

    # tiling metadata
    assert list(ours["num_tokens"]) == list(ref["num_tokens"])
    assert list(ours["num_patches"]) == list(ref["num_patches"])
    assert [tuple(s) for s in ours["imgs_sizes"]] == [tuple(s) for s in ref["imgs_sizes"]]

    # exactly one tile per image, no thumbnail
    assert ours["pixel_values"].shape[0] == 1

    d = _maxdiff(ours["pixel_values"], ref["pixel_values"].numpy())
    assert d < PIXEL_TOL, f"pixel_values max-abs-diff {d:.3e} >= {PIXEL_TOL}"

    # the number of <image> context tokens equals what the tower will emit
    n_ctx = int((ours["input_ids"] == our_proc.image_token_id).sum())
    assert n_ctx == ours["num_tokens"][0]


def test_multi_image_parity(ref_proc, our_proc):
    imgs = [Image.open(p).convert("RGB") for p in (IMG_SQUARE, IMG_TALL)]
    text = "<image>\nand<image>\nCompare them."

    ref = ref_proc(images=imgs, text=text)
    ours = our_proc(text=text, images=imgs)

    assert np.array_equal(ours["input_ids"], np.asarray(ref["input_ids"], dtype=np.int32))
    assert list(ours["num_tokens"]) == list(ref["num_tokens"])
    # different aspect ratios -> list of tiles, not a stacked tensor
    assert isinstance(ours["pixel_values"], list) and isinstance(ref["pixel_values"], list)
    for a, b in zip(ours["pixel_values"], ref["pixel_values"]):
        d = _maxdiff(a, b.numpy())
        assert d < PIXEL_TOL, f"pixel_values max-abs-diff {d:.3e}"


# --------------------------------------------------------------------------
# 3. audio path parity
# --------------------------------------------------------------------------
AUDIO_NPY = np.load(os.path.join(FIX, "proc_audio_5s_16k.npy"))


def test_audio_parity(ref_proc, our_proc):
    # NOTE: we feed the raw waveform (not the path) because the reference's
    # `_load_audio` hard-requires librosa/soundfile, neither of which is in the
    # mlx-server env — see test_audio_file_loading below for our own path.
    text = "<so_embedding>\nWhat do you hear?"

    ref = ref_proc(audio=AUDIO_NPY, text=text)
    ours = our_proc(text=text, audio=AUDIO_NPY)

    assert np.array_equal(ours["input_ids"], np.asarray(ref["input_ids"], dtype=np.int32))
    d = _maxdiff(ours["sound_clips"][0], ref["sound_clips"][0])
    assert d < 1e-6, f"waveform max-abs-diff {d:.3e}"

    # 5s @16k -> 1 + 80000//160 = 501 mel frames -> 3x stride-2 -> 63 tokens
    n_ctx = int((ours["input_ids"] == our_proc.audio_token_id).sum())
    assert n_ctx == 63, n_ctx


def test_audio_file_loading(our_proc):
    """Our stdlib-wave fallback reproduces the fixture within PCM_16 quantization."""
    ours = our_proc(text="<so_embedding>", audio=AUDIO_WAV)
    wav = ours["sound_clips"][0]
    assert wav.shape == AUDIO_NPY.shape
    # PCM_16 round-trip: quantization step 1/32767, plus the 32767-vs-32768
    # scale convention soundfile/librosa also use.
    assert _maxdiff(wav, AUDIO_NPY) < 1e-4


def test_audio_token_count_formula(our_proc):
    for n_samples in [16000, 80000, 80001, 159, 160, 161, 1_000_000]:
        assert our_proc._estimate_audio_num_embeddings(n_samples) >= 1


# --------------------------------------------------------------------------
# 4. video path parity
# --------------------------------------------------------------------------
def _frames(n=5, w=320, h=180, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        base = np.full((h, w, 3), 30 * i, dtype=np.int32)
        base += rng.integers(0, 60, size=(h, w, 3))
        out.append(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB"))
    return out


def test_video_parity(ref_proc, our_proc):
    frames = _frames()
    text = "<video>\nWhat happens?"

    ref = ref_proc(videos=frames, text=text)
    ours = our_proc(text=text, videos=frames)

    assert np.array_equal(ours["input_ids"], np.asarray(ref["input_ids"], dtype=np.int32))
    d = _maxdiff(ours["pixel_values_videos"], ref["pixel_values_videos"].numpy())
    assert d < PIXEL_TOL, f"pixel_values_videos max-abs-diff {d:.3e}"

    # one <img>..</img> block per temporal patch (tubelet), ceil(5/2) = 3
    tpt = ours["num_tokens"][0]
    n_ctx = int((ours["input_ids"] == our_proc.image_token_id).sum())
    assert n_ctx == 3 * tpt


# --------------------------------------------------------------------------
# 5. mixed modalities
# --------------------------------------------------------------------------
def test_image_plus_audio_parity(ref_proc, our_proc):
    img = Image.open(IMG_SQUARE).convert("RGB")
    text = "<image>\n<so_embedding>\nDoes the sound match the picture?"

    ref = ref_proc(images=img, audio=AUDIO_NPY, text=text)
    ours = our_proc(text=text, images=img, audio=AUDIO_NPY)

    assert np.array_equal(ours["input_ids"], np.asarray(ref["input_ids"], dtype=np.int32))
    d = _maxdiff(ours["pixel_values"], ref["pixel_values"].numpy())
    assert d < PIXEL_TOL, f"pixel_values max-abs-diff {d:.3e}"


# --------------------------------------------------------------------------
# 6. EVS
# --------------------------------------------------------------------------
def test_evs_matches_reference():
    import torch

    sys.path.insert(0, os.path.dirname(REF_DIR))
    from reference.evs import EfficientVideoSampling

    rng = np.random.default_rng(11)
    T, H, W, C = 4, 8, 8, 16
    embeds = rng.standard_normal((T, H * W, C)).astype(np.float32)

    ours = compute_evs = None
    from src.processing import compute_evs_retention_mask

    ours = compute_evs_retention_mask(embeds, (T, H, W), spatial_merge_size=1, q=0.7)
    ref = EfficientVideoSampling.compute_retention_mask(
        video_embeds=torch.from_numpy(embeds.reshape(T * H * W, C)),
        thw=(T, H, W),
        spatial_merge_size=1,
        q=0.7,
    ).numpy()
    assert np.array_equal(ours, ref), (
        f"EVS mask mismatch: {int((ours != ref).sum())} of {ours.size} differ"
    )
    assert ours[: H * W].all(), "first frame must be fully retained"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-x"]))
