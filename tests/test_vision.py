"""Unit tests for src/vision.py using a tiny random-weight ViT.

The real-weights parity check against C-RADIOv4-H lives in test_vision_parity.py. These
check config parsing, the patch/position/pixel-shuffle plumbing against independent numpy
references, the loader, and the token-count contract with the processor.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from src import vision  # noqa: E402
from src.processing import OmniImageProcessor  # noqa: E402

TINY = dict(
    embed_dim=32,
    depth=2,
    num_heads=4,
    cpe_max_size=128,  # 8x8 position grid
    vit_hidden_size=32,
    projector_hidden_size=64,
    llm_hidden_size=24,
)


def _tiny_cfg(**kw):
    return vision.VisionConfig(**{**TINY, **kw})


def _tiny_tower(seed=0, **kw):
    mx.random.seed(seed)
    tower = vision.NemotronVisionTower(_tiny_cfg(**kw))
    rng = np.random.default_rng(seed)
    pg = tower.vision_model.radio_model.model.patch_generator
    pg.pos_embed = mx.array(rng.standard_normal(pg.pos_embed.shape).astype(np.float32) * 0.1)
    pg.cls_token.token = mx.array(rng.standard_normal(pg.cls_token.token.shape).astype(np.float32))
    mx.eval(tower.parameters())
    return tower


@pytest.fixture(scope="module")
def tower():
    return _tiny_tower()


def _px(b, h, w, seed=0):
    return mx.array(np.random.default_rng(seed).standard_normal((b, 3, h, w)).astype(np.float32))


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
class TestConfig:
    def test_defaults_match_checkpoint_geometry(self):
        cfg = vision.VisionConfig()
        assert cfg.num_skip == 10  # cls_token.token is [10, 1280]
        assert cfg.pos_embed_grid == 128  # pos_embed is [1, 16384, 1280]
        assert cfg.num_image_token == 256  # (512 / 16)^2 * 0.5^2

    def _write(self, tmp_path, cfg):
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        return p

    def test_from_json_counts_unique_teachers_and_registers(self, tmp_path):
        p = self._write(
            tmp_path,
            {
                "vision_config": {
                    "args": {
                        "cls_token_per_teacher": True,
                        "teachers": [{"name": "clip"}, {"name": "siglip"}, {"name": "dino_v2"}, {"name": "clip"}, {"name": "sam"}],
                        "register_multiple": 10,
                        "cpe_max_size": 2048,
                    },
                    "video_temporal_patch_size": 2,
                },
                "llm_config": {"hidden_size": 2688},
            },
        )
        cfg = vision.VisionConfig.from_json(p)
        assert cfg.num_cls_tokens == 4
        assert cfg.num_registers == 6
        assert cfg.num_skip == 10
        assert cfg.llm_hidden_size == 2688

    def test_explicit_register_count_wins(self, tmp_path):
        p = self._write(tmp_path, {"vision_config": {"args": {"cpe_num_registers": 3, "register_multiple": 10}}})
        cfg = vision.VisionConfig.from_json(p)
        assert cfg.num_cls_tokens == 1 and cfg.num_registers == 3

    def test_no_registers(self, tmp_path):
        cfg = vision.VisionConfig.from_json(self._write(tmp_path, {}))
        assert cfg.num_cls_tokens == 1 and cfg.num_registers == 0

    def test_top_level_fields(self, tmp_path):
        p = self._write(
            tmp_path,
            {"downsample_ratio": 0.25, "ps_version": "v1", "force_image_size": 448, "llm_config": {"hidden_size": 7}},
        )
        cfg = vision.VisionConfig.from_json(p)
        assert (cfg.downsample_ratio, cfg.ps_version, cfg.force_image_size, cfg.llm_hidden_size) == (0.25, "v1", 448, 7)


# --------------------------------------------------------------------------------------
# helpers vs numpy references
# --------------------------------------------------------------------------------------
def _np_bilinear(x, oh, ow, align_corners=False):
    """Straight-loop torch bilinear (antialias=False) for NCHW."""
    _, _, h, w = x.shape

    def src(o, i, n_in, n_out):
        if align_corners:
            return o * (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
        return max((o + 0.5) * n_in / n_out - 0.5, 0.0)

    out = np.zeros(x.shape[:2] + (oh, ow), np.float64)
    for i in range(oh):
        sy = src(i, None, h, oh)
        y0 = int(np.floor(sy))
        y1 = min(y0 + 1, h - 1)
        wy = sy - y0
        for j in range(ow):
            sx = src(j, None, w, ow)
            x0 = int(np.floor(sx))
            x1 = min(x0 + 1, w - 1)
            wx = sx - x0
            top = x[:, :, y0, x0] * (1 - wx) + x[:, :, y0, x1] * wx
            bot = x[:, :, y1, x0] * (1 - wx) + x[:, :, y1, x1] * wx
            out[:, :, i, j] = top * (1 - wy) + bot * wy
    return out


class TestBilinear:
    @pytest.mark.parametrize("out_hw", [(4, 4), (13, 7), (2, 9), (1, 1)])
    @pytest.mark.parametrize("align", [False, True])
    def test_matches_reference(self, out_hw, align):
        x = np.random.default_rng(0).standard_normal((1, 2, 5, 6)).astype(np.float32)
        got = np.array(vision._interpolate_bilinear_nchw(mx.array(x), out_hw, align_corners=align))
        np.testing.assert_allclose(got, _np_bilinear(x, *out_hw, align), atol=1e-5)

    def test_known_values(self):
        x = mx.array(np.array([[[[0.0, 1.0]]]], np.float32))
        got = np.array(vision._interpolate_bilinear_nchw(x, (1, 4)))
        np.testing.assert_allclose(got[0, 0, 0], [0.0, 0.25, 0.75, 1.0])

    def test_same_size_is_passthrough(self):
        x = mx.zeros((1, 1, 3, 3))
        assert vision._interpolate_bilinear_nchw(x, (3, 3)) is x


def test_im_to_patches_layout():
    pg = vision.ViTPatchGenerator(_tiny_cfg())
    x = np.random.default_rng(0).standard_normal((2, 3, 32, 48)).astype(np.float32)
    got = np.array(pg.im_to_patches(mx.array(x)))
    # einops: 'b c (py yy) (px xx) -> b (py px) (c yy xx)'
    want = x.reshape(2, 3, 2, 16, 3, 16).transpose(0, 2, 4, 1, 3, 5).reshape(2, 6, 3 * 256)
    np.testing.assert_array_equal(got, want)


def test_pos_embed_full_grid_is_untouched():
    pg = vision.ViTPatchGenerator(_tiny_cfg())
    assert pg._get_pos_embeddings((8, 8)) is pg.pos_embed


def test_pos_embed_resize_then_crop():
    pg = vision.ViTPatchGenerator(_tiny_cfg())
    rng = np.random.default_rng(1)
    pg.pos_embed = mx.array(rng.standard_normal((1, 64, 32)).astype(np.float32))
    pe = np.array(pg._get_pos_embeddings((4, 6)))
    assert pe.shape == (1, 24, 32)
    # CPE eval path: resize the 8x8 grid to 6x6 (the max side), then keep the top 4 rows
    grid = np.array(pg.pos_embed).reshape(1, 8, 8, 32).transpose(0, 3, 1, 2)
    want = _np_bilinear(grid, 6, 6)[:, :, :4, :6].reshape(1, 32, 24).transpose(0, 2, 1)
    np.testing.assert_allclose(pe, want, atol=1e-5)


def test_cls_tokens_are_prepended():
    tok = vision.ClsToken(4, 2, 3)
    tok.token = mx.array(np.arange(20, dtype=np.float32).reshape(5, 4))
    out = np.array(tok(mx.ones((2, 7, 4))))
    assert out.shape == (2, 12, 4)
    np.testing.assert_array_equal(out[1, :5], np.arange(20).reshape(5, 4))
    np.testing.assert_array_equal(out[:, 5:], 1)


def test_rms_norm_and_squared_relu():
    norm = vision.RMSNorm(4)
    x = np.array([[1.0, -2.0, 3.0, -4.0]], np.float32)
    want = x / np.sqrt((x**2).mean() + 1e-5)
    np.testing.assert_allclose(np.array(norm(mx.array(x))), want, rtol=1e-5)
    np.testing.assert_array_equal(np.array(vision.SquaredReLU()(mx.array(x))), [[1, 0, 9, 0]])


class TestPixelShuffle:
    def _np_shuffle(self, x, s, version):
        # reference/modeling.py pixel_shuffle, transcribed to numpy
        n, w, h, c = x.shape
        x = x.reshape(n, w, int(h * s), int(c / s)).transpose(0, 2, 1, 3)
        x = x.reshape(n, int(h * s), int(w * s), int(c / (s * s)))
        if version != "v1":
            x = x.transpose(0, 2, 1, 3)
        return x

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_matches_reference(self, version):
        t = vision.NemotronVisionTower(_tiny_cfg(ps_version=version))
        x = np.arange(2 * 4 * 6 * 3, dtype=np.float32).reshape(2, 4, 6, 3)
        got = np.array(t.pixel_shuffle(mx.array(x), 0.5))
        np.testing.assert_array_equal(got, self._np_shuffle(x, 0.5, version))
        # (n, w, h, c) -> v1: (n, h/2, w/2, 4c); v2 transposes back to (n, w/2, h/2, 4c)
        assert got.shape == ((2, 2, 3, 12) if version == "v2" else (2, 3, 2, 12))

    def test_is_a_permutation(self):
        t = vision.NemotronVisionTower(_tiny_cfg())
        x = np.random.default_rng(0).standard_normal((1, 6, 6, 5)).astype(np.float32)
        got = np.array(t.pixel_shuffle(mx.array(x)))
        np.testing.assert_array_equal(np.sort(got.ravel()), np.sort(x.ravel()))


# --------------------------------------------------------------------------------------
# tower forward
# --------------------------------------------------------------------------------------
class TestTower:
    @pytest.mark.parametrize("hw", [(64, 64), (32, 96), (128, 64)])
    def test_token_count_matches_processor_contract(self, tower, hw):
        out = tower.extract_feature(_px(1, *hw))
        assert out.shape == (1, (hw[0] // 16) * (hw[1] // 16) // 4, TINY["llm_hidden_size"])

    def test_processor_output_feeds_tower(self, tower):
        proc = OmniImageProcessor(min_num_patches=16, max_num_patches=64)
        arr = np.random.default_rng(0).integers(0, 256, (100, 150, 3), dtype=np.uint8)
        pi = proc(arr)
        out = tower.extract_feature(mx.array(pi["pixel_values"]))
        assert out.shape[1] == pi["num_tokens"][0]

    def test_cls_and_registers_are_dropped(self, tower):
        feats = tower.vision_model(_px(1, 64, 64))
        assert feats.shape == (1, 16, TINY["embed_dim"])

    def test_batch_items_are_independent(self, tower):
        px = _px(2, 64, 64)
        both = np.array(tower.extract_feature(px))
        one = np.array(tower.extract_feature(px[1:2]))
        np.testing.assert_allclose(both[1], one[0], rtol=1e-4, atol=1e-4)

    def test_list_input_concatenates(self, tower):
        a, b = _px(1, 64, 64, 1), _px(1, 64, 64, 2)
        out = np.array(tower.extract_feature([a, b]))
        np.testing.assert_allclose(out[0], np.array(tower.extract_feature(a))[0], rtol=1e-4, atol=1e-4)
        assert out.shape[0] == 2
        assert np.array(tower(a)).shape == (1, 4, TINY["llm_hidden_size"])

    def test_video_groups_frames(self, tower):
        out = tower.extract_video_feature(_px(4, 64, 64))
        assert out.shape == (2, 4, TINY["llm_hidden_size"])

    def test_video_odd_frames_repeat_the_last(self, tower):
        px = np.array(_px(3, 64, 64))
        padded = np.concatenate([px, px[-1:]], 0)
        a = np.array(tower.extract_video_feature(mx.array(px)))
        b = np.array(tower.extract_video_feature(mx.array(padded)))
        assert a.shape == (2, 4, TINY["llm_hidden_size"])
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    def test_video_uses_the_3d_embedder(self, tower):
        # two identical frames through the video path != one frame through the image path
        px = _px(1, 64, 64)
        img = np.array(tower.extract_feature(px))
        vid = np.array(tower.extract_video_feature(mx.concatenate([px, px], 0)))
        assert img.shape == vid.shape
        assert not np.allclose(img, vid)

    def test_input_conditioner_is_bypassed_by_default(self, tower):
        rm = tower.vision_model.radio_model
        rm.input_conditioner.norm_mean = mx.full((3, 1, 1), 5.0)
        try:
            px = _px(1, 32, 32)
            a = np.array(rm(px))
            b = np.array(rm(px, apply_conditioner=True))
            assert not np.allclose(a, b)
            np.testing.assert_allclose(np.array(rm(px - 5.0)), b, rtol=1e-4, atol=1e-4)
        finally:
            rm.input_conditioner.norm_mean = mx.zeros((3, 1, 1))

    def test_summary_shape(self, tower):
        rm = tower.vision_model.radio_model
        assert rm.summary(_px(2, 32, 32)).shape == (2, rm.cfg.num_cls_tokens * TINY["embed_dim"])


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
@pytest.fixture
def tiny_default_config(monkeypatch):
    """load_vision_tower builds VisionConfig() when no config path is given; make it tiny."""
    real = vision.VisionConfig
    monkeypatch.setattr(vision, "VisionConfig", lambda **kw: real(**{**TINY, **kw}))


def test_load_vision_tower_round_trip(tower, tmp_path, tiny_default_config):
    weights = dict(tree_flatten(tower.parameters()))
    weights["backbone.layers.0.weight"] = mx.zeros((2, 2))  # LLM keys share the shard
    weights["sound_encoder.x"] = mx.zeros((2,))
    path = tmp_path / "shard.safetensors"
    mx.save_safetensors(str(path), {k: v.astype(mx.bfloat16) for k, v in weights.items()})

    loaded = vision.load_vision_tower(path)
    for _, p in tree_flatten(loaded.parameters()):
        assert p.dtype == mx.float32
    px = _px(1, 64, 64)
    a = np.array(tower.extract_feature(px))
    b = np.array(loaded.extract_feature(px))
    # bf16 storage round-trip
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    assert cos.min() > 0.999


def test_load_vision_tower_reports_missing_and_unexpected(tower, tmp_path, tiny_default_config):
    weights = dict(tree_flatten(tower.parameters()))
    weights.pop("mlp1.3.weight")
    weights["vision_model.radio_model.model.norm.weight"] = mx.zeros((32,))
    path = tmp_path / "shard.safetensors"
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError) as e:
        vision.load_vision_tower(path)
    assert "mlp1.3.weight" in str(e.value)
    assert "model.norm.weight" in str(e.value)


# --------------------------------------------------------------------------------------
# Metal (Apple GPU) only
# --------------------------------------------------------------------------------------
def _metal_available():
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


@pytest.mark.metal
@pytest.mark.skipif(not _metal_available(), reason="needs an Apple Silicon GPU (Metal); CI runs MLX on CPU")
def test_gpu_matches_cpu_stream(tower):
    px = _px(1, 64, 64)
    with mx.stream(mx.gpu):
        g = np.array(tower.extract_feature(px))
    with mx.stream(mx.cpu):
        c = np.array(tower.extract_feature(px))
    cos = (g * c).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(c, axis=-1))
    assert cos.min() > 0.999
