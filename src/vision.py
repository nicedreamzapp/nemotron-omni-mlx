"""C-RADIOv4-H vision tower + InternVL pixel-shuffle + mlp1 projector, in pure MLX.

Port strategy: the WEIGHT GRAPH, not the framework. The module tree here mirrors the
checkpoint key names 1:1 (`vision_model.radio_model.model.blocks.N.attn.qkv.weight`, ...)
so weights load with a plain `tree_unflatten` — no key remapping table.

Checkpoint facts (nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning, shard 1 of 17):
  vision_model.radio_model.input_conditioner.norm_mean            f32   [3,1,1]
  vision_model.radio_model.input_conditioner.norm_std             f32   [3,1,1]
  vision_model.radio_model.model.patch_generator.cls_token.token  bf16  [10, 1280]
  vision_model.radio_model.model.patch_generator.embedder.weight  bf16  [1280, 768]    # 3*16*16
  vision_model.radio_model.model.patch_generator.video_embedder.weight bf16 [1280,1536] # 2*3*16*16
  vision_model.radio_model.model.patch_generator.pos_embed        bf16  [1, 16384, 1280]  # 128x128
  vision_model.radio_model.model.blocks.{0..31}.{norm1,attn,norm2,mlp}...              # ViT-H/16
  mlp1.0.weight  bf16 [5120]          # RMSNorm(1280 * (1/0.5)^2)
  mlp1.1.weight  bf16 [20480, 5120]   # Linear, no bias
  mlp1.3.weight  bf16 [2688, 20480]   # Linear, no bias  -> LLM hidden

Notable absences (all deliberate, verified against the reference):
  * no `model.norm.*`   -> radio_model.create_model_from_args sets `model.norm = nn.Identity()`
                          because args.model_norm is False. There is NO final LayerNorm.
  * no `feature_normalizer.*` -> vision_config.feature_normalizer_config is null -> Identity,
                          despite args.feature_normalization == "SHIP_NORM".
  * no qk_norm / layer-scale / rel-pos-bias keys -> timm defaults, all off. vitdet_window_size
                          is null, so attention is plain global attention.
  * spectral reparam already merged into qkv.weight in the checkpoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

__all__ = [
    "VisionConfig",
    "NemotronVisionTower",
    "load_vision_tower",
]


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
@dataclass
class VisionConfig:
    # ViT-H/16 (timm `vit_huge_patch16_224` as redefined in reference-radio/extra_timm_models.py:
    # dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16), mlp_ratio default 4.0)
    embed_dim: int = 1280
    depth: int = 32
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 16
    layer_norm_eps: float = 1e-6  # timm VisionTransformer default norm_layer eps

    # CPE patch generator
    cpe_max_size: int = 2048  # -> pos_embed grid is 2048/16 = 128 x 128
    num_cls_tokens: int = 4  # unique teacher names (clip, siglip, dino_v2, ...)
    num_registers: int = 6  # register_multiple=10 -> 10 - (4 % 10) = 6
    # num_skip = num_cls_tokens + num_registers = 10  == cls_token.token.shape[0]

    # video
    video_temporal_patch_size: int = 2

    # InternVL head
    downsample_ratio: float = 0.5
    ps_version: str = "v2"
    vit_hidden_size: int = 1280
    projector_hidden_size: int = 20480
    llm_hidden_size: int = 2688
    rms_norm_eps: float = 1e-5

    # preprocessing (top-level config.json; NOTE: applied by the *processor*, not the tower --
    # the wrapper calls make_preprocessor_external() which swaps input_conditioner -> Identity)
    force_image_size: int = 512
    use_thumbnail: bool = True
    norm_mean: Sequence[float] = field(
        default_factory=lambda: [0.48145466, 0.4578275, 0.40821073]
    )
    norm_std: Sequence[float] = field(
        default_factory=lambda: [0.26862954, 0.26130258, 0.27577711]
    )

    @property
    def num_skip(self) -> int:
        return self.num_cls_tokens + self.num_registers

    @property
    def pos_embed_grid(self) -> int:
        return self.cpe_max_size // self.patch_size

    @property
    def num_image_token(self) -> int:
        return int(
            (self.force_image_size // self.patch_size) ** 2 * (self.downsample_ratio**2)
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "VisionConfig":
        cfg = json.loads(Path(path).read_text())
        vcfg = cfg.get("vision_config", {})
        args = vcfg.get("args", {})

        num_cls_tokens = 1
        if args.get("cls_token_per_teacher", False):
            num_cls_tokens = len({t["name"] for t in args.get("teachers", [])})
        num_registers = args.get("cpe_num_registers")
        if not num_registers:
            reg_mult = args.get("register_multiple")
            num_registers = (
                reg_mult - (num_cls_tokens % reg_mult) if reg_mult else 0
            )

        llm_hidden = cfg.get("llm_config", {}).get("hidden_size", 2688)

        return cls(
            patch_size=cfg.get("patch_size", 16),
            cpe_max_size=args.get("cpe_max_size", 2048),
            num_cls_tokens=num_cls_tokens,
            num_registers=num_registers,
            video_temporal_patch_size=vcfg.get("video_temporal_patch_size", 2),
            downsample_ratio=cfg.get("downsample_ratio", 0.5),
            ps_version=cfg.get("ps_version", "v2"),
            vit_hidden_size=cfg.get("vit_hidden_size", 1280),
            projector_hidden_size=cfg.get("projector_hidden_size", 20480),
            llm_hidden_size=llm_hidden,
            force_image_size=cfg.get("force_image_size", 512),
            use_thumbnail=cfg.get("use_thumbnail", True),
            norm_mean=cfg.get("norm_mean", [0.48145466, 0.4578275, 0.40821073]),
            norm_std=cfg.get("norm_std", [0.26862954, 0.26130258, 0.27577711]),
        )


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _interpolate_bilinear_nchw(
    x: mx.array, size: Tuple[int, int], align_corners: bool = False
) -> mx.array:
    """torch.nn.functional.interpolate(mode='bilinear', antialias=False) for NCHW.

    Matches torch's `area_pixel_compute_source_index`:
      align_corners=False -> src = scale * (i + 0.5) - 0.5, clamped at 0
      align_corners=True  -> src = (in - 1) / (out - 1) * i
    """
    b, c, h, w = x.shape
    oh, ow = size
    if (h, w) == (oh, ow):
        return x

    def _src_index(out_len: int, in_len: int) -> mx.array:
        idx = mx.arange(out_len, dtype=mx.float32)
        if align_corners:
            scale = (in_len - 1) / (out_len - 1) if out_len > 1 else 0.0
            return idx * scale
        scale = in_len / out_len
        src = scale * (idx + 0.5) - 0.5
        return mx.maximum(src, 0.0)

    sy = _src_index(oh, h)
    sx = _src_index(ow, w)

    y0 = mx.floor(sy).astype(mx.int32)
    x0 = mx.floor(sx).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)
    wy = (sy - y0.astype(mx.float32)).reshape(1, 1, oh, 1)
    wx = (sx - x0.astype(mx.float32)).reshape(1, 1, 1, ow)

    # gather rows then columns
    top = x[:, :, y0, :]  # (b, c, oh, w)
    bot = x[:, :, y1, :]
    top_l = top[:, :, :, x0]
    top_r = top[:, :, :, x1]
    bot_l = bot[:, :, :, x0]
    bot_r = bot[:, :, :, x1]

    top_i = top_l + (top_r - top_l) * wx
    bot_i = bot_l + (bot_r - bot_l) * wx
    return top_i + (bot_i - top_i) * wy


class Identity(nn.Module):
    def __call__(self, x):
        return x


class SquaredReLU(nn.Module):
    """reference/modeling.py: torch.pow(F.relu(x), 2)"""

    def __call__(self, x: mx.array) -> mx.array:
        r = nn.relu(x)
        return r * r


class RMSNorm(nn.Module):
    """reference/modeling.py RMSNorm: fp32 variance, weight-only (no bias), eps=1e-5."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dt = x.dtype
        xf = x.astype(mx.float32)
        var = mx.mean(xf * xf, axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(var + self.eps)
        return (self.weight.astype(mx.float32) * xf).astype(dt)


# --------------------------------------------------------------------------------------
# patch generator (reference-radio/vit_patch_generator.py)
# --------------------------------------------------------------------------------------
class ClsToken(nn.Module):
    """reference-radio/cls_token.py — cls tokens + registers, prepended to the patch seq."""

    def __init__(self, ndim: int, num_tokens: int, num_registers: int):
        super().__init__()
        self.token = mx.zeros((num_tokens + num_registers, ndim))

    def __call__(self, x: mx.array) -> mx.array:
        tok = mx.broadcast_to(
            self.token[None].astype(x.dtype), (x.shape[0], *self.token.shape)
        )
        return mx.concatenate([tok, x], axis=1)


class ViTPatchGenerator(nn.Module):
    """2D (image) + 3D (video) patch embedding with CPE absolute position embeddings."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        # max_input_dims == cpe_max_size, input_dims == force_image_size -> cpe_mode is True
        self.num_rows = cfg.pos_embed_grid
        self.num_cols = cfg.pos_embed_grid
        self.cpe_mode = True

        # ViTPatchLinear: Linear(3 * patch**2 -> embed_dim, bias=False)
        self.embedder = nn.Linear(3 * cfg.patch_size**2, cfg.embed_dim, bias=False)
        # attached by the omni wrapper (modeling.py ~line 108): Linear(T*3*patch**2 -> embed, bias=False)
        self.video_embedder = nn.Linear(
            cfg.video_temporal_patch_size * 3 * cfg.patch_size**2,
            cfg.embed_dim,
            bias=False,
        )
        self.pos_embed = mx.zeros((1, self.num_rows * self.num_cols, cfg.embed_dim))
        self.cls_token = ClsToken(cfg.embed_dim, cfg.num_cls_tokens, cfg.num_registers)
        # normalize_patches is False for this checkpoint (patch_embed.norm was Identity)
        self.patch_normalizer = Identity()

    # -- Im2Patches: rearrange 'b c (py yy) (px xx) -> b (py px) (c yy xx)'
    def im_to_patches(self, x: mx.array) -> mx.array:
        b, c, h, w = x.shape
        p = self.patch_size
        py, px = h // p, w // p
        x = x.reshape(b, c, py, p, px, p)
        x = x.transpose(0, 2, 4, 1, 3, 5)  # b py px c yy xx
        return x.reshape(b, py * px, c * p * p)

    def _get_pos_embeddings(self, input_dims: Tuple[int, int]) -> mx.array:
        if (self.num_rows, self.num_cols) == tuple(input_dims):
            return self.pos_embed

        pe = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).transpose(
            0, 3, 1, 2
        )
        pe = pe.astype(mx.float32)

        def window_select(pe):
            if input_dims[0] < pe.shape[-2]:
                pe = pe[..., : input_dims[0], :]
            if input_dims[1] < pe.shape[-1]:
                pe = pe[..., :, : input_dims[1]]
            return pe

        if self.cpe_mode:
            # eval branch: resize the square grid to max(input_dims), then crop.
            max_dim = max(input_dims)
            pe = _interpolate_bilinear_nchw(pe, (max_dim, max_dim), align_corners=False)
            pe = window_select(pe)
        else:
            pe = window_select(pe)

        if tuple(pe.shape[-2:]) != tuple(input_dims):
            pe = _interpolate_bilinear_nchw(pe, tuple(input_dims), align_corners=False)

        pe = pe.reshape(pe.shape[0], pe.shape[1], -1).transpose(0, 2, 1)
        return pe.astype(self.pos_embed.dtype)

    def __call__(self, x: mx.array, video: bool = False) -> mx.array:
        """x: (B, C, H, W). For video, C == T*3 and `video` selects the 3D embedder."""
        patches = self.im_to_patches(x)
        patches = (self.video_embedder if video else self.embedder)(patches)
        input_dims = (x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)
        pos = self._get_pos_embeddings(input_dims)
        patches = patches + pos.astype(patches.dtype)
        patches = self.cls_token(patches)
        return self.patch_normalizer(patches)


# --------------------------------------------------------------------------------------
# ViT blocks (timm VisionTransformer, defaults: no qk_norm, no layer-scale, GELU, LN eps 1e-6)
# --------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, c)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # timm default act_layer=nn.GELU -> exact erf GELU
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg.embed_dim, cfg.num_heads)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.mlp = Mlp(cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """timm VisionTransformer with CPE enabled (enable_cpe_support._forward_cpe).

    forward_features = patch_generator -> blocks -> norm, and `norm` is Identity here.
    """

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.patch_generator = ViTPatchGenerator(cfg)
        self.blocks = [Block(cfg) for _ in range(cfg.depth)]
        self.norm = Identity()  # create_model_from_args: model.norm = nn.Identity()

    def forward_features(self, x: mx.array, video: bool = False) -> mx.array:
        x = self.patch_generator(x, video=video)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# --------------------------------------------------------------------------------------
# RADIO wrapper (reference-radio/radio_model.py + hf_model.py)
# --------------------------------------------------------------------------------------
class InputConditioner(nn.Module):
    """(x - mean) / std. NOTE: bypassed at runtime — the omni wrapper calls
    make_preprocessor_external(), so pixel_values arrive already normalized."""

    def __init__(self):
        super().__init__()
        self.norm_mean = mx.zeros((3, 1, 1))
        self.norm_std = mx.ones((3, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.norm_mean.astype(x.dtype)) / self.norm_std.astype(x.dtype)


class RADIOModelBase(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.input_conditioner = InputConditioner()
        self.model = VisionTransformer(cfg)
        # feature_normalizer_config is null -> Identity (no weights in checkpoint)
        self.feature_normalizer = Identity()

    def __call__(
        self, x: mx.array, video: bool = False, apply_conditioner: bool = False
    ) -> mx.array:
        """Returns `.features` (NLC): the spatial tokens with all summary tokens dropped."""
        if apply_conditioner:
            x = self.input_conditioner(x)
        y = self.model.forward_features(x, video=video)
        all_feat = y[:, self.cfg.num_skip :]
        return self.feature_normalizer(all_feat)

    def summary(self, x: mx.array, video: bool = False) -> mx.array:
        """The `.summary` half of RadioOutput: the cls tokens, flattened. Unused by the omni
        wrapper (it only reads `.features`) but exposed for completeness."""
        y = self.model.forward_features(x, video=video)
        return y[:, : self.cfg.num_cls_tokens].reshape(y.shape[0], -1)


class RADIOModel(nn.Module):
    """hf_model.RADIOModel — a thin PreTrainedModel shell whose only child is `radio_model`."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.radio_model = RADIOModelBase(cfg)

    def __call__(self, x: mx.array, **kw) -> mx.array:
        return self.radio_model(x, **kw)


# --------------------------------------------------------------------------------------
# top level: tower + pixel shuffle + mlp1
# --------------------------------------------------------------------------------------
class NemotronVisionTower(nn.Module):
    """`vision_model` + `mlp1` of NemotronH_Nano_Omni_Reasoning_V3, MLX edition.

    Public API mirrors reference/modeling.py:
      extract_feature(pixel_values)        -> (B, num_image_token, llm_hidden)
      extract_video_feature(pixel_values)  -> (N/T, num_image_token, llm_hidden)
    """

    def __init__(self, cfg: Optional[VisionConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or VisionConfig()
        self.vision_model = RADIOModel(cfg)

        scale = int(1 / cfg.downsample_ratio) ** 2
        # nn.Sequential(RMSNorm, Linear, SquaredReLU, Linear) -> keys mlp1.0/.1/.3
        self.mlp1 = [
            RMSNorm(cfg.vit_hidden_size * scale, eps=cfg.rms_norm_eps),
            nn.Linear(cfg.vit_hidden_size * scale, cfg.projector_hidden_size, bias=False),
            SquaredReLU(),
            nn.Linear(cfg.projector_hidden_size, cfg.llm_hidden_size, bias=False),
        ]

    # -- InternVL pixel shuffle (reference/modeling.py:259)
    def pixel_shuffle(self, x: mx.array, scale_factor: float = 0.5) -> mx.array:
        n, w, h, c = x.shape
        x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.cfg.ps_version != "v1":
            x = x.transpose(0, 2, 1, 3)
        return x

    def _project(self, vit_embeds: mx.array, h: int, w: int) -> mx.array:
        b = vit_embeds.shape[0]
        x = vit_embeds.reshape(b, h, w, -1)
        x = self.pixel_shuffle(x, scale_factor=self.cfg.downsample_ratio)
        x = x.reshape(b, -1, x.shape[-1])
        for layer in self.mlp1:
            x = layer(x)
        return x

    def _extract_feature_single(self, pixel_values: mx.array) -> mx.array:
        vit_embeds = self.vision_model(pixel_values)
        b, _, hh, ww = pixel_values.shape
        p = self.cfg.patch_size
        return self._project(vit_embeds, hh // p, ww // p)

    def extract_feature(
        self, pixel_values: Union[mx.array, List[mx.array], Tuple[mx.array, ...]]
    ) -> mx.array:
        """pixel_values: (B, 3, H, W) already normalized with norm_mean/norm_std, or a list of
        such tensors (dynamic resolution picks different tile sizes per image)."""
        if isinstance(pixel_values, (list, tuple)):
            return mx.concatenate(
                [self._extract_feature_single(pv) for pv in pixel_values], axis=0
            )
        return self._extract_feature_single(pixel_values)

    def extract_video_feature(self, pixel_values_videos: mx.array) -> mx.array:
        """pixel_values_videos: (N_frames, 3, H, W). Packs T consecutive frames into the channel
        dim so RADIO's channel-agnostic Im2Patches yields (·, npatch, T*C*P^2) — exactly the
        `video_embedder` input layout. See reference/modeling.py:extract_video_feature."""
        cfg = self.cfg
        t = cfg.video_temporal_patch_size
        n, c, hh, ww = pixel_values_videos.shape

        if n % t != 0:
            pad_n = t - (n % t)
            pad = mx.broadcast_to(pixel_values_videos[-1:], (pad_n, c, hh, ww))
            pixel_values_videos = mx.concatenate([pixel_values_videos, pad], axis=0)
            n = pixel_values_videos.shape[0]

        # row-major reshape keeps per-patch order [t0,c0..c2, t1,c0..c2], matching the weights
        x = pixel_values_videos.reshape(n // t, t * c, hh, ww)
        vit_embeds = self.vision_model(x, video=True)
        p = cfg.patch_size
        return self._project(vit_embeds, hh // p, ww // p)

    def __call__(self, pixel_values) -> mx.array:
        return self.extract_feature(pixel_values)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_vision_tower(
    weights_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    dtype: mx.Dtype = mx.float32,
) -> NemotronVisionTower:
    """Load `vision_model.*` + `mlp1.*` from a safetensors shard into an MLX tower.

    Only the vision keys are read; the LLM tensors in the same shard are never materialized
    (mlx.core.load memory-maps and we drop the rest before eval).
    """
    cfg = VisionConfig.from_json(config_path) if config_path else VisionConfig()
    model = NemotronVisionTower(cfg)

    raw = mx.load(str(weights_path))
    weights = {
        k: v.astype(dtype)
        for k, v in raw.items()
        if k.startswith("vision_model.") or k.startswith("mlp1.")
    }
    del raw

    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing = expected - set(weights)
    unexpected = set(weights) - expected
    if missing or unexpected:
        raise ValueError(
            f"vision weight mismatch.\n  missing: {sorted(missing)[:8]}\n"
            f"  unexpected: {sorted(unexpected)[:8]}"
        )

    model.update(tree_unflatten(list(weights.items())))
    model.eval()
    mx.eval(model.parameters())
    return model
