"""Parity: PyTorch C-RADIOv4-H (+ InternVL pixel-shuffle + mlp1) vs the MLX port in src/vision.py.

Both sides load the SAME real checkpoint weights (BF16 repo shard 1, the only shard holding
`vision_model.*` / `mlp1.*`), cast to fp32, run on CPU.

Run:
    ~/.local/mlx-server/bin/python -m pytest tests/test_vision_parity.py -x -q -s
    ~/.local/mlx-server/bin/python tests/test_vision_parity.py        # standalone
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from src.vision import NemotronVisionTower, VisionConfig, load_vision_tower  # noqa: E402

WEIGHTS = ROOT / "weights-bf16" / "model-00001-of-00017.safetensors"
CONFIG = ROOT / "reference" / "config.json"
FIXTURES = ROOT / "tests" / "fixtures"

COS_THRESHOLD = 0.999


# --------------------------------------------------------------------------------------
# fixture image
# --------------------------------------------------------------------------------------
def make_test_image(size: int = 448, seed: int = 1234) -> np.ndarray:
    """Deterministic RGB gradient + noise, uint8 (H, W, 3)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(
        np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij"
    )
    img = np.stack(
        [
            xs,  # R: horizontal ramp
            ys,  # G: vertical ramp
            0.5 * (np.sin(8 * np.pi * xs) * np.cos(8 * np.pi * ys) + 1.0),  # B: texture
        ],
        axis=-1,
    )
    img = img + rng.normal(0.0, 0.05, img.shape)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def ensure_fixture(size: int = 448) -> np.ndarray:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    npy = FIXTURES / f"image_{size}.npy"
    if npy.exists():
        return np.load(npy)
    img = make_test_image(size)
    np.save(npy, img)
    try:
        from PIL import Image

        Image.fromarray(img).save(FIXTURES / f"image_{size}.png")
    except ImportError:
        pass
    return img


def preprocess(img_u8: np.ndarray, cfg: VisionConfig) -> np.ndarray:
    """uint8 HWC -> normalized (1, 3, H, W) fp32.

    Mirrors the processor: scale to [0,1] then (x - norm_mean) / norm_std with the CLIP stats
    from the TOP-LEVEL config. The tower's own input_conditioner is bypassed
    (make_preprocessor_external), so normalization must happen here.
    """
    x = img_u8.astype(np.float32) / 255.0
    x = (x - np.array(cfg.norm_mean, np.float32)) / np.array(cfg.norm_std, np.float32)
    return x.transpose(2, 0, 1)[None]


# --------------------------------------------------------------------------------------
# torch reference
# --------------------------------------------------------------------------------------
def build_torch_reference():
    """Real C-RADIOv4-H via trust_remote_code + the omni wrapper's vision head, fp32 CPU."""
    import torch
    from torch import nn
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    hf_cfg = AutoConfig.from_pretrained(str(ROOT / "reference"), trust_remote_code=True)
    vm = AutoModel.from_config(hf_cfg.vision_config, trust_remote_code=True)

    # The omni wrapper attaches a 3D patch projection for video (modeling.py ~line 108).
    pg = vm.radio_model.model.patch_generator
    t = hf_cfg.vision_config.video_temporal_patch_size
    pg.video_embedder = nn.Linear(t * 3 * pg.patch_size**2, pg.embed_dim, bias=False)

    # --- real weights
    full = load_file(str(WEIGHTS))
    vis_sd = {
        k[len("vision_model.") :]: v.float()
        for k, v in full.items()
        if k.startswith("vision_model.")
    }
    missing, unexpected = vm.load_state_dict(vis_sd, strict=False)
    # `summary_idxs` is a derived buffer, not stored in the omni checkpoint.
    assert set(missing) <= {"radio_model.summary_idxs"}, f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"

    # Must come AFTER load_state_dict, otherwise input_conditioner.* would be unexpected keys.
    vm.radio_model.make_preprocessor_external()
    vm = vm.float().eval()

    # --- mlp1 (reference/modeling.py:43-59, 126-131)
    class SquaredReLU(nn.Module):
        def forward(self, x):
            return torch.pow(torch.nn.functional.relu(x), 2)

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, h):
            dt = h.dtype
            h = h.to(torch.float32)
            var = h.pow(2).mean(-1, keepdim=True)
            h = h * torch.rsqrt(var + self.eps)
            return (self.weight.to(torch.float32) * h).to(dt)

    cfg = json.loads(CONFIG.read_text())
    ds = cfg["downsample_ratio"]
    vit_h = cfg["vit_hidden_size"] * int(1 / ds) ** 2
    mlp1 = nn.Sequential(
        RMSNorm(vit_h, eps=1e-5),
        nn.Linear(vit_h, cfg["projector_hidden_size"], bias=False),
        SquaredReLU(),
        nn.Linear(cfg["projector_hidden_size"], cfg["llm_config"]["hidden_size"], bias=False),
    )
    mlp1_sd = {k[len("mlp1.") :]: v.float() for k, v in full.items() if k.startswith("mlp1.")}
    mlp1.load_state_dict(mlp1_sd)
    mlp1 = mlp1.float().eval()

    del full
    return vm, mlp1


def torch_pixel_shuffle(x, scale_factor=0.5, ps_version="v2"):
    """reference/modeling.py:259"""
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor**2)))
    if ps_version != "v1":
        x = x.permute(0, 2, 1, 3).contiguous()
    return x


def torch_extract_feature(vm, mlp1, pixel_values, cfg: VisionConfig, video=False):
    """reference/modeling.py _extract_feature_single / extract_video_feature, fp32."""
    import torch

    with torch.no_grad():
        if video:
            t = cfg.video_temporal_patch_size
            n, c, hh, ww = pixel_values.shape
            if n % t:
                pad = pixel_values[-1:].expand(t - (n % t), -1, -1, -1)
                pixel_values = torch.cat([pixel_values, pad], 0)
                n = pixel_values.shape[0]
            x = pixel_values.reshape(n // t, t * c, hh, ww)
            pg = vm.radio_model.model.patch_generator
            orig = pg.embedder
            pg.embedder = pg.video_embedder
            try:
                feats = vm(x).features
            finally:
                pg.embedder = orig
        else:
            hh, ww = pixel_values.shape[-2:]
            feats = vm(pixel_values).features

        p = cfg.patch_size
        h, w = hh // p, ww // p
        e = feats.reshape(feats.shape[0], h, w, -1)
        e = torch_pixel_shuffle(e, cfg.downsample_ratio, cfg.ps_version)
        e = e.reshape(e.shape[0], -1, e.shape[-1])
        return feats, mlp1(e)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def cosine_per_token(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(-1, a.shape[-1]).astype(np.float64)
    b = b.reshape(-1, b.shape[-1]).astype(np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def report(name: str, ref: np.ndarray, got: np.ndarray) -> float:
    assert ref.shape == got.shape, f"{name}: shape {got.shape} != ref {ref.shape}"
    cos = cosine_per_token(ref, got)
    mad = np.abs(ref.astype(np.float64) - got.astype(np.float64)).max()
    rel = mad / (np.abs(ref).max() + 1e-12)
    print(
        f"  {name:22s} shape={tuple(ref.shape)} "
        f"cos: min={cos.min():.8f} mean={cos.mean():.8f} | max|Δ|={mad:.3e} rel={rel:.3e}"
    )
    return float(cos.min())


# --------------------------------------------------------------------------------------
# session-scoped models
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg():
    return VisionConfig.from_json(CONFIG)


@pytest.fixture(scope="module")
def mlx_model():
    assert WEIGHTS.exists(), f"missing weights: {WEIGHTS}"
    return load_vision_tower(WEIGHTS, CONFIG, dtype=mx.float32)


@pytest.fixture(scope="module")
def torch_ref():
    assert WEIGHTS.exists(), f"missing weights: {WEIGHTS}"
    return build_torch_reference()


# --------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------
def test_config_matches_checkpoint(cfg):
    """The derived config must agree with the checkpoint's actual tensor geometry."""
    assert cfg.num_skip == 10  # cls_token.token is [10, 1280]
    assert cfg.num_cls_tokens == 4 and cfg.num_registers == 6
    assert cfg.pos_embed_grid == 128  # pos_embed is [1, 16384, 1280]
    assert cfg.embed_dim == 1280 and cfg.depth == 32 and cfg.num_heads == 16
    assert cfg.num_image_token == 256  # (512/16)^2 * 0.5^2
    assert cfg.ps_version == "v2" and cfg.downsample_ratio == 0.5


@pytest.mark.parametrize("size", [448, 512])
def test_image_parity(size, cfg, mlx_model, torch_ref):
    import torch

    img = ensure_fixture(size)
    px = preprocess(img, cfg)
    vm, mlp1 = torch_ref

    ref_feats, ref_proj = torch_extract_feature(
        vm, mlp1, torch.from_numpy(px), cfg, video=False
    )
    mlx_feats = mlx_model.vision_model(mx.array(px))
    mlx_proj = mlx_model.extract_feature(mx.array(px))
    mx.eval(mlx_feats, mlx_proj)

    print(f"\n[image {size}x{size}]")
    n_patch = (size // cfg.patch_size) ** 2
    assert tuple(ref_feats.shape) == (1, n_patch, 1280)
    assert tuple(ref_proj.shape) == (1, n_patch // 4, cfg.llm_hidden_size)

    c1 = report("vit .features", ref_feats.numpy(), np.array(mlx_feats))
    c2 = report("mlp1 projected", ref_proj.numpy(), np.array(mlx_proj))
    assert c1 > COS_THRESHOLD, f"features cos {c1}"
    assert c2 > COS_THRESHOLD, f"projected cos {c2}"


def test_video_parity(cfg, mlx_model, torch_ref):
    """4 frames -> 2 temporal groups through the 3D `video_embedder`."""
    import torch

    size = 448
    frames = np.stack(
        [preprocess(make_test_image(size, seed=s), cfg)[0] for s in (11, 22, 33, 44)]
    )
    vm, mlp1 = torch_ref

    ref_feats, ref_proj = torch_extract_feature(
        vm, mlp1, torch.from_numpy(frames), cfg, video=True
    )
    mlx_proj = mlx_model.extract_video_feature(mx.array(frames))
    mx.eval(mlx_proj)

    print("\n[video 4 frames @ 448]")
    n_patch = (size // cfg.patch_size) ** 2
    assert tuple(ref_proj.shape) == (2, n_patch // 4, cfg.llm_hidden_size)
    c = report("video projected", ref_proj.numpy(), np.array(mlx_proj))
    assert c > COS_THRESHOLD, f"video projected cos {c}"


def test_video_odd_frame_padding(cfg, mlx_model, torch_ref):
    """3 frames must pad by repeating the last -> 2 groups, and match torch."""
    import torch

    size = 448
    frames = np.stack(
        [preprocess(make_test_image(size, seed=s), cfg)[0] for s in (11, 22, 33)]
    )
    vm, mlp1 = torch_ref
    _, ref_proj = torch_extract_feature(vm, mlp1, torch.from_numpy(frames), cfg, video=True)
    mlx_proj = mlx_model.extract_video_feature(mx.array(frames))
    mx.eval(mlx_proj)

    print("\n[video 3 frames @ 448 (padded to 4)]")
    assert tuple(ref_proj.shape) == (2, (size // cfg.patch_size) ** 2 // 4, cfg.llm_hidden_size)
    c = report("video projected", ref_proj.numpy(), np.array(mlx_proj))
    assert c > COS_THRESHOLD, f"video padded cos {c}"


def test_cpu_stream_is_graph_exact(cfg, torch_ref):
    """On the MLX **CPU** stream the port is numerically exact vs torch (cos == 1.0).

    This is the real proof the weight graph is right: any residual on the GPU stream is Metal's
    reduced-precision fp32 matmul (~8e-4 rel vs a float64 ground truth, where torch and MLX-CPU
    both sit at ~1e-6), NOT a porting bug. Kept as a regression guard: if the graph ever drifts,
    this test breaks long before the looser GPU cosine threshold would.
    """
    import torch

    img = ensure_fixture(448)
    px = preprocess(img, cfg)
    vm, mlp1 = torch_ref

    with mx.stream(mx.cpu):
        model = load_vision_tower(WEIGHTS, CONFIG, dtype=mx.float32)
        mlx_proj = model.extract_feature(mx.array(px))
        mx.eval(mlx_proj)

    _, ref_proj = torch_extract_feature(vm, mlp1, torch.from_numpy(px), cfg)

    print("\n[MLX cpu stream]")
    c = report("mlp1 projected", ref_proj.numpy(), np.array(mlx_proj))
    assert c > 0.9999999, f"cpu-stream cos {c} — the weight graph itself drifted"


def test_list_input_concatenates(cfg, mlx_model):
    """Dynamic-resolution path: a list of per-tile tensors concatenates along dim 0.

    NOTE: the reference does `torch.cat(outs, dim=0)`, so entries must agree on token count
    (i.e. same tile size); differing sizes would fail on the torch side too.
    """
    a = mx.array(preprocess(ensure_fixture(448), cfg))
    b = mx.array(preprocess(make_test_image(448, seed=7), cfg))
    out = mlx_model.extract_feature([a, b])
    mx.eval(out)
    assert out.shape == (2, (448 // cfg.patch_size) ** 2 // 4, cfg.llm_hidden_size)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q", "-s"]))
