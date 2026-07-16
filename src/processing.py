# Copyright (c) 2026. MIT-licensed runtime code.
#
# MLX/numpy port of NVIDIA's `NemotronH_Nano_Omni_Reasoning_V3Processor`
# (reference/processing.py + reference/image_processing.py).
#
# This module is WEIGHT-FREE: it turns (text, images, videos, audio) into
# `input_ids` + `pixel_values` + `sound_clips`, with the img/video/sound context
# tokens already expanded to the exact counts the towers will produce.
#
# Parity target (tests/test_processing_parity.py):
#   * input_ids                      — EXACT match vs the PyTorch reference
#   * pixel_values / pixel_values_videos — max-abs-diff < 1e-4
#
# --------------------------------------------------------------------------
# IMPORTANT deviations from the "classic InternVL" description
# --------------------------------------------------------------------------
# The live image path is NOT classic InternVL dynamic tiling. `config.json`
# still carries `use_thumbnail=True` / `force_image_size=512` / `image_tag_type`
# from the InternVL lineage, but the image processor registered in
# `preprocessor_config.json` (`NemotronH_Nano_Omni_Reasoning_V3ImageProcessor`)
# ignores them completely: it produces exactly ONE tile per image, sized by a
# dynamic-resolution rule on a 16x16 patch grid, with NO thumbnail tile.
# `use_thumbnail` / `force_image_size` are only read by the dead code paths
# (`processing_utils.dynamic_preprocess`, `video_processing.py`), which the
# processor's `__call__` never invokes. We follow the LIVE path.
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # PIL is only needed when images/videos are actually passed
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# ==========================================================================
# torch.nn.functional.interpolate(mode="bicubic", antialias=True) — numpy port
# ==========================================================================
# Port of ATen's `upsample_bicubic2d_aa` CPU kernel
# (aten/src/ATen/native/UpSampleKernel.cpp, `HelperInterpBase::_compute_index_ranges_weights`
# + `_separable_upsample_generic_Nd_kernel_impl`).
#
# Gotchas that MUST be preserved for bit-close parity:
#   * align_corners=False  =>  scale = in_size / out_size
#   * antialias only widens the kernel when DOWNscaling (scale >= 1.0)
#   * the C `(int)` casts truncate toward zero (not floor) — matters for the
#     `center - support + 0.5` term
#   * separable order is WIDTH first, then HEIGHT (interp_dim counts down from
#     the last dim), and the intermediate buffer is NOT clamped/rounded
#
# TWO NON-OBVIOUS GOTCHAS, each worth ~1e-2 of error if you get them wrong:
#   1. The ANTIALIAS path uses Pillow's cubic coefficient a = -0.5
#      (`HelperInterpCubic::aa_filter`), NOT the a = -0.75 that plain
#      (non-AA) `mode="bicubic"` uses. Using -0.75 costs ~27/255 max-abs-diff.
#   2. ATen computes the weights in the input's `scalar_t` (float32 here), not
#      double. Computing them in float64 is *more accurate* but disagrees with
#      torch by ~5e-3 on the 0-255 scale; matching float32 drops it to ~1e-4.
_BICUBIC_A = -0.5  # Pillow's coefficient — the AA path, see gotcha #1


def _cubic_filter(x: np.ndarray, a: float = _BICUBIC_A) -> np.ndarray:
    """Pillow/ATen `aa_filter` cubic kernel, evaluated in the input's dtype."""
    dt = x.dtype.type
    x = np.abs(x)
    out = np.zeros_like(x)
    m1 = x < 1.0
    m2 = (x >= 1.0) & (x < 2.0)
    x1 = x[m1]
    out[m1] = (dt(a + 2.0) * x1 - dt(a + 3.0)) * x1 * x1 + dt(1.0)
    x2 = x[m2]
    out[m2] = ((dt(a) * x2 - dt(5.0 * a)) * x2 + dt(8.0 * a)) * x2 - dt(4.0 * a)
    return out


def _compute_weights_aa(
    in_size: int, out_size: int, antialias: bool = True, dtype=np.float32
) -> np.ndarray:
    """Dense (out_size, in_size) resample matrix for one separable dim.

    All arithmetic is done in `dtype` (float32) to mirror ATen's `scalar_t`.
    """
    dt = np.dtype(dtype).type
    scale = dt(in_size) / dt(out_size)  # align_corners=False, no explicit scale
    interp_size = 4  # bicubic
    if antialias and scale >= 1.0:
        support = dt(interp_size * 0.5) * scale
        invscale = dt(1.0) / scale
    else:
        support = dt(interp_size * 0.5)
        invscale = dt(1.0)

    w = np.zeros((out_size, in_size), dtype=dt)
    for i in range(out_size):
        center = scale * dt(i + 0.5)
        # C-style (int) casts: truncation toward zero.
        xmin = max(int(center - support + dt(0.5)), 0)
        xmax = min(int(center + support + dt(0.5)), in_size)
        if xmax <= xmin:
            xmin = min(max(xmin, 0), in_size - 1)
            xmax = xmin + 1
        j = np.arange(xmin, xmax, dtype=dt)
        wj = _cubic_filter((j - center + dt(0.5)) * invscale)
        total = wj.sum()
        if total != 0.0:
            wj = wj / total
        w[i, xmin:xmax] = wj
    return w


def resize_bicubic_antialias(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a (C, H, W) float32 array. Mirrors torch's antialiased bicubic.

    Separable: width pass first, then height (ATen's dim order).
    """
    c, in_h, in_w = img.shape
    x = img.astype(np.float32, copy=False)
    if in_w != out_w:
        ww = _compute_weights_aa(in_w, out_w)
        x = x @ ww.T  # (C, H, out_w)
    if in_h != out_h:
        wh = _compute_weights_aa(in_h, out_h)
        # (out_h, H) x (C, H, out_w) -> (C, out_h, out_w)
        x = np.einsum("oh,chw->cow", wh, x, optimize=True)
    return np.ascontiguousarray(x, dtype=np.float32)


# ==========================================================================
# Image processor — port of NemotronH_Nano_Omni_Reasoning_V3ImageProcessor
# ==========================================================================
class OmniImageProcessor:
    """One tile per image, dynamic resolution on a `patch_size` grid.

    Port of `reference/image_processing.py` (itself a port of vLLM's
    `DynamicResolutionImageTiler`). No thumbnail, no multi-tile split.
    """

    def __init__(
        self,
        norm_mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
        norm_std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
        patch_size: int = 16,
        downsample_ratio: float = 0.5,
        min_num_patches: int = 1024,
        max_num_patches: int = 13312,
        max_model_len: int = 16384,
        video_target_num_patches: int = 1024,
        video_maintain_aspect_ratio: bool = True,
        **_ignored: Any,
    ) -> None:
        self.norm_mean = list(norm_mean)
        self.norm_std = list(norm_std)
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self._downsample_factor = int(round(1.0 / downsample_ratio))
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.max_model_len = max_model_len
        self.video_target_num_patches = video_target_num_patches
        self.video_maintain_aspect_ratio = video_maintain_aspect_ratio

    @classmethod
    def from_pretrained(cls, path: str) -> "OmniImageProcessor":
        with open(os.path.join(path, "preprocessor_config.json")) as f:
            cfg = json.load(f)
        cfg.pop("image_processor_type", None)
        cfg.pop("auto_map", None)
        return cls(**cfg)

    # -- sizing rules ------------------------------------------------------
    def _compute_target_patches(self, size_wh: Tuple[int, int], tokens_available: int):
        """Port of `DynamicResolutionImageTiler.process_media` (image path)."""
        orig_w, orig_h = size_wh
        closest_patch_h = round(orig_h / self.patch_size + 0.5)
        closest_patch_w = round(orig_w / self.patch_size + 0.5)
        patches = closest_patch_h * closest_patch_w

        factor = min(math.sqrt(tokens_available / patches), 1.0)
        target_h = math.floor(factor * closest_patch_h)
        target_w = math.floor(factor * closest_patch_w)

        if tokens_available > self.min_num_patches and target_h * target_w < self.min_num_patches:
            up = math.sqrt(self.min_num_patches / (target_h * target_w))
            target_h = math.ceil(up * target_h)
            target_w = math.ceil(up * target_w)

        divisor = self._downsample_factor
        rem_h = target_h % divisor
        if rem_h:
            inc_h = divisor - rem_h
            if (target_h + inc_h) * target_w <= tokens_available:
                target_h += inc_h
            else:
                target_h = max(divisor, target_h - rem_h)
        rem_w = target_w % divisor
        if rem_w:
            inc_w = divisor - rem_w
            if target_h * (target_w + inc_w) <= tokens_available:
                target_w += inc_w
            else:
                target_w = max(divisor, target_w - rem_w)

        return target_w, target_h

    def _compute_target_patches_video(self, size_wh: Tuple[int, int]):
        """Port of vLLM's `_compute_aspect_preserving_size` (video frames)."""
        orig_w, orig_h = size_wh
        target = self.video_target_num_patches
        divisor = self._downsample_factor
        if self.video_maintain_aspect_ratio:
            aspect_wh = orig_w / max(orig_h, 1)
            ph = max(round(math.sqrt(target / aspect_wh)), 1)
            pw = max(round(math.sqrt(target * aspect_wh)), 1)
            if divisor > 1:
                rem_h = ph % divisor
                rem_w = pw % divisor
                ph_up = ph + (divisor - rem_h if rem_h else 0)
                ph_down = ph - rem_h
                pw_up = pw + (divisor - rem_w if rem_w else 0)
                pw_down = pw - rem_w
                if ph_up * pw_up <= target:
                    ph, pw = ph_up, pw_up
                else:
                    ph = max(divisor, ph_down)
                    pw = max(divisor, pw_down)
        else:
            side = int(math.sqrt(target))
            side = max(divisor, (side // divisor) * divisor)
            ph = pw = side
        return pw, ph

    # -- main --------------------------------------------------------------
    def __call__(self, images, is_video: bool = False) -> Dict[str, Any]:
        images = _make_list_of_images(images)
        sizes = [_image_size_wh(im) for im in images]

        if is_video:
            target_sizes = [self._compute_target_patches_video(s) for s in sizes]
        else:
            num_tokens_available = self.max_model_len - 4  # match vLLM's reserve
            budget = num_tokens_available * (self._downsample_factor**2)
            budget = max(budget, self.min_num_patches * len(images))
            max_budget = (
                self.max_num_patches
                if (self.max_num_patches and self.max_num_patches > 0)
                else float("inf")
            )
            per_image_budget = [
                max(min(budget, max_budget), self.min_num_patches) for _ in images
            ]
            target_sizes = [
                self._compute_target_patches(s, b) for s, b in zip(sizes, per_image_budget)
            ]

        norm_mean = np.array(self.norm_mean, dtype=np.float32).reshape(3, 1, 1)
        norm_std = np.array(self.norm_std, dtype=np.float32).reshape(3, 1, 1)

        pixel_values_list: List[np.ndarray] = []
        num_tokens_per_image: List[int] = []
        imgs_sizes: List[Tuple[int, int]] = []
        for im, (wp, hp) in zip(images, target_sizes):
            target_w = wp * self.patch_size
            target_h = hp * self.patch_size
            arr = _image_to_uint8_hwc(im)  # (H, W, 3) uint8
            t = arr.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)
            if t.shape[-2] != target_h or t.shape[-1] != target_w:
                t = resize_bicubic_antialias(t, target_h, target_w)
            t = (t / 255.0 - norm_mean) / norm_std
            pixel_values_list.append(t.astype(np.float32))
            num_tokens_per_image.append((wp * hp) // (self._downsample_factor**2))
            imgs_sizes.append((target_h, target_w))

        all_same_shape = all(t.shape == pixel_values_list[0].shape for t in pixel_values_list)
        pixel_values = np.stack(pixel_values_list, 0) if all_same_shape else pixel_values_list

        return {
            "pixel_values": pixel_values,
            "num_patches": [1] * len(images),
            "num_tokens": num_tokens_per_image,
            "imgs_sizes": imgs_sizes,
        }


def _make_list_of_images(images) -> List[Any]:
    if Image is not None and isinstance(images, Image.Image):
        return [images]
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            return [images]
        if images.ndim == 4:
            return [images[i] for i in range(images.shape[0])]
        raise ValueError(f"Unsupported image array ndim={images.ndim}")
    if isinstance(images, (list, tuple)):
        out: List[Any] = []
        for im in images:
            out.extend(_make_list_of_images(im))
        return out
    raise ValueError(f"Unsupported image input type: {type(images)}")


def _image_size_wh(im) -> Tuple[int, int]:
    if Image is not None and isinstance(im, Image.Image):
        return im.width, im.height
    arr = np.asarray(im)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    # HWC (the reference does `np.asarray(img, dtype=np.uint8)` -> (H, W, 3)).
    return arr.shape[1], arr.shape[0]


def _image_to_uint8_hwc(im) -> np.ndarray:
    if Image is not None and isinstance(im, Image.Image):
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)
    arr = np.asarray(im)
    return arr.astype(np.uint8)


# ==========================================================================
# EfficientVideoSampling — port of reference/evs.py (numpy)
# ==========================================================================
def compute_evs_retention_mask(
    video_embeds: np.ndarray,
    thw: Tuple[int, int, int],
    spatial_merge_size: int = 1,
    q: float = 0.7,
) -> np.ndarray:
    """Port of `EfficientVideoSampling.compute_retention_mask`.

    Args:
        video_embeds: (T, H*W, C) or (T*H*W, C) float array of *vision* embeds.
        thw: (T, H, W) grid of the video features.
        spatial_merge_size: downsampling factor still to come (1 here — the
            model calls EVS on already-pixel-shuffled embeds).
        q: pruning rate (`video_pruning_rate`, 0.7 in config.json).

    Returns:
        Bool mask of shape (T*H'*W',), True = keep.
    """
    T, H, W = thw
    h, w = H // spatial_merge_size, W // spatial_merge_size
    c = video_embeds.shape[-1]
    x = np.asarray(video_embeds, dtype=np.float32).reshape(T, h, w, c)

    # cosine similarity between consecutive frames, per spatial position
    a, b = x[1:], x[:-1]
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    # torch's cosine_similarity clamps the denominator with eps=1e-8
    sim = num / np.maximum(den, 1e-8)
    dissim = 1.0 - sim

    # first frame is always fully retained (sentinel 255)
    dissim = np.concatenate([255.0 * np.ones_like(x[:1, :, :, 0]), dissim], axis=0)
    flat = dissim.reshape(-1)

    min_num_tokens = h * w  # a single frame
    evs_num_tokens = int(T * min_num_tokens * (1 - q))
    num_tokens_to_keep = max(min_num_tokens, evs_num_tokens)

    # torch.argsort(descending=True, stable=True) == stable ascending on -x
    order = np.argsort(-flat, kind="stable")
    keep = order[:num_tokens_to_keep]

    mask = np.zeros(flat.shape, dtype=bool)
    mask[keep] = True
    return mask


# ==========================================================================
# Processor — port of NemotronH_Nano_Omni_Reasoning_V3Processor
# ==========================================================================
class OmniProcessor:
    """MLX-side port of NVIDIA's omni processor.

    Usage:
        proc = OmniProcessor.from_pretrained("reference")   # or the model dir
        out = proc(text="<image>\\nDescribe.", images=pil_img)
        out["input_ids"]      -> (1, N) int32 numpy
        out["pixel_values"]   -> (1, 3, H, W) float32 numpy
    """

    def __init__(
        self,
        image_processor: OmniImageProcessor,
        tokenizer,
        config: Optional[dict] = None,
        audio_sampling_rate: int = 16000,
        audio_subsampling_factor: int = 8,
        audio_hop_length: int = 160,
        video_temporal_patch_dim: int = 2,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.config = config or {}

        self.video_temporal_patch_dim = video_temporal_patch_dim
        self.image_token = "<image>"
        self.video_token = "<video>"
        self.audio_token = "<so_embedding>"
        self.audio_start_token = "<so_start>"
        self.audio_end_token = "<so_end>"
        self.image_start_token = "<img>"
        self.image_end_token = "</img>"

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)

        self.audio_sampling_rate = audio_sampling_rate
        self.audio_subsampling_factor = audio_subsampling_factor
        self.audio_hop_length = audio_hop_length

    @classmethod
    def from_pretrained(cls, path: str, tokenizer=None) -> "OmniProcessor":
        ip = OmniImageProcessor.from_pretrained(path)
        cfg = {}
        cfg_path = os.path.join(path, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        vt = int(cfg.get("video_temporal_patch_size", 2))
        sr = int(cfg.get("sound_config", {}).get("sampling_rate", 16000))
        sub = int(cfg.get("sound_config", {}).get("subsampling_factor", 8))
        return cls(
            ip,
            tokenizer,
            config=cfg,
            audio_sampling_rate=sr,
            audio_subsampling_factor=sub,
            video_temporal_patch_dim=vt,
        )

    # -- audio -------------------------------------------------------------
    def _estimate_audio_num_embeddings(self, audio_length_samples: int) -> int:
        """Exact count of `<so_embedding>` tokens the sound encoder will emit.

        Mirrors `ParakeetFeatureExtractor` (center-padded STFT -> `1 + L // hop`
        mel frames) then `ParakeetEncoder._get_subsampling_output_length`.
        """
        n_frames = 1 + audio_length_samples // self.audio_hop_length
        kernel_size = getattr(self, "audio_subsampling_conv_kernel_size", 3)
        stride = getattr(self, "audio_subsampling_conv_stride", 2)
        num_layers = int(math.log2(self.audio_subsampling_factor))
        all_paddings = (kernel_size - 1) // 2 * 2
        add_pad = all_paddings - kernel_size
        L = n_frames
        for _ in range(num_layers):
            L = (L + add_pad) // stride + 1
        return L

    def _load_audio(self, audio_path: str, target_sr: int) -> np.ndarray:
        """Same loader-preference order as the reference (librosa, then sf).

        DEVIATION: the reference raises ImportError when neither librosa nor
        soundfile is installed (which is the case in our mlx-server env). We add
        a stdlib `wave` fallback for PCM wav so `--audio file.wav` works with no
        extra deps. Values are identical to soundfile's for PCM_16 (int16/32768).
        """
        try:
            import librosa

            waveform, _sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return waveform
        except ImportError:
            pass
        try:
            import soundfile as sf

            waveform, sr = sf.read(audio_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sr != target_sr:
                import scipy.signal

                num_samples = int(len(waveform) * target_sr / sr)
                waveform = scipy.signal.resample(waveform, num_samples)
            return waveform.astype(np.float32)
        except ImportError:
            pass
        return self._load_audio_stdlib(audio_path, target_sr)

    @staticmethod
    def _load_audio_stdlib(audio_path: str, target_sr: int) -> np.ndarray:
        import wave

        with wave.open(audio_path, "rb") as f:
            sr = f.getframerate()
            n_ch = f.getnchannels()
            width = f.getsampwidth()
            raw = f.readframes(f.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            raise ValueError(f"Unsupported wav sample width: {width}")
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if width == 1:
            data = (data - 128.0) / 128.0
        else:
            data = data / float(2 ** (8 * width - 1))
        if n_ch > 1:
            data = data.reshape(-1, n_ch).mean(axis=1)
        if sr != target_sr:
            import scipy.signal

            data = scipy.signal.resample(data, int(len(data) * target_sr / sr))
        return data.astype(np.float32)

    def _process_audio(self, audio, sampling_rate: Optional[int] = None):
        sampling_rate = sampling_rate or self.audio_sampling_rate
        if not isinstance(audio, list):
            audio = [audio]
        clips, num_tokens = [], []
        for item in audio:
            if isinstance(item, str):
                waveform = self._load_audio(item, sampling_rate)
            elif isinstance(item, np.ndarray):
                waveform = item.squeeze() if item.ndim > 1 else item
            else:  # torch tensors / mx arrays
                arr = np.asarray(item)
                waveform = arr.squeeze() if arr.ndim > 1 else arr
            clips.append(waveform)
            num_tokens.append(max(1, self._estimate_audio_num_embeddings(len(waveform))))
        return clips, num_tokens

    # -- main --------------------------------------------------------------
    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images=None,
        videos=None,
        audio=None,
        video_metadata=None,
        sampling_rate: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Dict[str, Any]:
        image_inputs: Dict[str, Any] = {}
        videos_inputs: Dict[str, Any] = {}
        audio_inputs: Dict[str, Any] = {}

        if images is not None:
            image_inputs = self.image_processor(images, is_video=False)
            image_num_tokens = image_inputs["num_tokens"]

        if videos is not None:
            videos_inputs = self.image_processor(videos, is_video=True)
            video_num_patches = [sum(videos_inputs["num_patches"])]
            videos_inputs["pixel_values_videos"] = videos_inputs.pop("pixel_values")

        audio_num_tokens: List[int] = []
        if audio is not None:
            audio_clips, audio_num_tokens = self._process_audio(audio, sampling_rate)
            audio_inputs["sound_clips"] = audio_clips

        if not isinstance(text, list):
            text = [text]
        text = list(text)

        # --- image placeholder expansion: <image> -> <img>N x <image></img>
        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    n_tokens = image_num_tokens[index]
                    text[i] = text[i].replace(
                        self.image_token,
                        self.image_start_token
                        + "<|placeholder|>" * n_tokens
                        + self.image_end_token,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # --- video placeholder expansion: one <img>...</img> per tubelet
        if videos is not None:
            assert len(text) == 1, "Video is not supported for batch size > 1"
            i = 0
            index = 0
            if self.video_token in text[i]:
                tokens_per_tubelet = videos_inputs["num_tokens"][0]
                each_group = (
                    self.image_start_token
                    + "<|placeholder|>" * tokens_per_tubelet
                    + self.image_end_token
                )
                T = self.video_temporal_patch_dim
                n_frames = video_num_patches[index]
                n_groups = (n_frames + T - 1) // T

                source_fps = getattr(video_metadata, "fps", None) if video_metadata else None
                frames_indices = (
                    getattr(video_metadata, "frames_indices", None) if video_metadata else None
                )
                if source_fps:
                    frame_duration_ms = int(1000.0 / source_fps)

                frame_labels = []
                for g in range(n_groups):
                    parts = []
                    for j in range(T):
                        fi = g * T + j
                        if fi >= n_frames:
                            break
                        prefix = "Frame" if j == 0 else "frame"
                        if source_fps and frames_indices is not None and fi < len(frames_indices):
                            ts = int(frames_indices[fi]) * frame_duration_ms / 1000.0
                            parts.append(f"{prefix} {fi+1} sampled at {ts:.2f} seconds")
                        elif source_fps:
                            ts = fi / source_fps
                            parts.append(f"{prefix} {fi+1} sampled at {ts:.2f} seconds")
                        else:
                            parts.append(f"{prefix} {fi+1}")
                    frame_labels.append(" and ".join(parts) + ": ")

                video_prompt = ""
                for g, label in enumerate(frame_labels):
                    if g > 0:
                        video_prompt += "\n"
                    video_prompt += label + each_group

                text[i] = text[i].replace(self.video_token, video_prompt, 1)
            # The tokenizer has no real `<video>` token — reuse `<image>` (id 18).
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # --- audio placeholder expansion: <so_embedding> -> <so_start>N x <so_end>
        if audio is not None:
            index = 0
            for i in range(len(text)):
                while self.audio_token in text[i]:
                    num_tokens = audio_num_tokens[index] if index < len(audio_num_tokens) else 1
                    text[i] = text[i].replace(
                        self.audio_token,
                        self.audio_start_token
                        + "<|audio_placeholder|>" * num_tokens
                        + self.audio_end_token,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)

        enc = self.tokenizer(text)
        out: Dict[str, Any] = {
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int32),
            "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int32),
            "expanded_text": text,
        }
        for k, v in image_inputs.items():
            out[k] = v
        for k, v in videos_inputs.items():
            out[k] = v
        if audio_inputs:
            out["sound_clips"] = audio_inputs["sound_clips"]
        return out

    # -- convenience -------------------------------------------------------
    def decode(self, *a, **k):
        return self.tokenizer.decode(*a, **k)

    def batch_decode(self, *a, **k):
        return self.tokenizer.batch_decode(*a, **k)


__all__ = [
    "OmniProcessor",
    "OmniImageProcessor",
    "resize_bicubic_antialias",
    "compute_evs_retention_mask",
]
