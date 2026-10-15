"""Unit tests for src/audio.py using a tiny random-weight Conformer.

The real-weights parity check against transformers' ParakeetEncoder lives in
test_audio_parity.py. These check the module wiring, the weight loader, masking, and the
contract src/omni.py relies on: the tower emits exactly as many frames as the processor
reserved <so_embedding> tokens for.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from src import audio  # noqa: E402
from src.processing import OmniImageProcessor, OmniProcessor  # noqa: E402

TINY = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    subsampling_conv_channels=8,
    num_mel_bins=16,
    projection_hidden_size=48,
    llm_hidden_size=24,
)


def _tiny_tower(seed=0):
    mx.random.seed(seed)
    cfg = audio.AudioConfig(**TINY)
    tower = audio.AudioTower(cfg)
    rng = np.random.default_rng(seed)
    feat = tower.encoder.feature_extractor.featurizer
    n_freq = cfg.n_fft // 2 + 1
    feat.fb = mx.array(rng.random((n_freq, cfg.num_mel_bins)).astype(np.float32))
    feat.window = mx.array(np.hanning(cfg.win_length).astype(np.float32))
    # non-trivial BatchNorm stats so the conv module isn't an identity-ish no-op
    for layer in tower.encoder.layers:
        layer.conv.norm.running_mean = mx.array(rng.normal(0, 0.1, cfg.hidden_size).astype(np.float32))
        layer.conv.norm.running_var = mx.array(rng.uniform(0.5, 1.5, cfg.hidden_size).astype(np.float32))
        layer.self_attn.bias_u = mx.array(rng.normal(0, 0.1, layer.self_attn.bias_u.shape).astype(np.float32))
        layer.self_attn.bias_v = mx.array(rng.normal(0, 0.1, layer.self_attn.bias_v.shape).astype(np.float32))
    mx.eval(tower.parameters())
    return tower


@pytest.fixture(scope="module")
def tower():
    return _tiny_tower()


def _wave(n, seed=0):
    t = np.arange(n) / 16000
    rng = np.random.default_rng(seed)
    return (0.4 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(n)).astype(np.float32)


def _cos(a, b):
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def test_config_from_sound_config_ignores_unknown_keys():
    cfg = audio.AudioConfig.from_sound_config(
        {"hidden_size": 512, "num_hidden_layers": 3, "model_type": "parakeet", "dropout": 0.1},
        llm_hidden_size=100,
    )
    assert cfg.hidden_size == 512 and cfg.num_hidden_layers == 3
    assert cfg.llm_hidden_size == 100
    assert cfg.num_mel_bins == 128  # default kept


def test_default_config_matches_checkpoint_geometry():
    cfg = audio.AudioConfig()
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads) == (1024, 24, 8)
    assert cfg.subsampling_factor == 8 and cfg.llm_hidden_size == 2688


# --------------------------------------------------------------------------------------
# frame-count contract with the processor
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_samples", [1600, 8000, 16000, 16160, 33333, 80000])
def test_output_length_matches_processor_estimate(tower, fake_tokenizer, n_samples):
    proc = OmniProcessor(OmniImageProcessor(), fake_tokenizer)
    embeds, mask = tower(mx.array(_wave(n_samples))[None])
    mx.eval(embeds, mask)
    assert embeds.shape == (1, proc._estimate_audio_num_embeddings(n_samples), TINY["llm_hidden_size"])
    assert mask.shape == embeds.shape[:2]
    assert not np.isnan(np.array(embeds)).any()


def test_subsampled_lengths_match_conv_output(tower):
    for n_frames in [3, 8, 64, 101, 500, 501]:
        feats = mx.zeros((1, n_frames, TINY["num_mel_bins"]))
        out = tower.encoder.subsampling(feats)
        got = int(tower.encoder._subsampled_lengths(mx.array([n_frames]))[0])
        assert got == out.shape[1], n_frames


# --------------------------------------------------------------------------------------
# frontend
# --------------------------------------------------------------------------------------
def test_featurizer_normalizes_each_mel_bin(tower):
    feats, mask = tower.featurize(mx.array(_wave(16000))[None])
    feats, mask = np.array(feats)[0], np.array(mask)[0]
    assert feats.shape == (1 + 16000 // 160, TINY["num_mel_bins"])
    valid = feats[mask]
    np.testing.assert_allclose(valid.mean(0), 0.0, atol=1e-4)
    np.testing.assert_allclose(valid.std(0, ddof=1), 1.0, atol=1e-3)
    np.testing.assert_array_equal(feats[~mask], 0.0)


def test_featurizer_ignores_padding(tower):
    w = _wave(8000)
    padded = np.concatenate([w, np.random.default_rng(9).standard_normal(4000).astype(np.float32)])
    a, ma = tower.featurize(mx.array(w)[None])
    b, mb = tower.featurize(mx.array(padded)[None], mx.array([8000]))
    a, b, ma, mb = map(np.array, (a, b, ma, mb))
    n = int(ma.sum())
    assert int(mb.sum()) == n
    # the last valid frame's window reaches into the padding, which is zeroed first in both
    np.testing.assert_allclose(a[0, :n], b[0, :n], atol=1e-4)


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------
def test_batchnorm_eval_uses_running_stats():
    bn = audio.BatchNormEval(3)
    bn.running_mean = mx.array([1.0, 2.0, 3.0])
    bn.running_var = mx.array([4.0, 1.0, 0.25])
    bn.weight = mx.array([1.0, 2.0, 3.0])
    bn.bias = mx.array([0.0, 1.0, -1.0])
    x = np.array([[3.0, 2.0, 3.5]], np.float32)
    want = (x - [1, 2, 3]) / np.sqrt(np.array([4, 1, 0.25]) + 1e-5) * [1, 2, 3] + [0, 1, -1]
    np.testing.assert_allclose(np.array(bn(mx.array(x))), want, rtol=1e-5)


def test_rel_positional_encoding_layout():
    cfg = audio.AudioConfig(**TINY)
    pe = np.array(audio.RelPositionalEncoding(cfg)(5, batch_size=2))
    assert pe.shape == (2, 9, cfg.hidden_size)
    # position runs T-1 .. -(T-1); the centre row is position 0 -> sin 0, cos 1 interleaved
    np.testing.assert_allclose(pe[0, 4, 0::2], 0.0, atol=1e-7)
    np.testing.assert_allclose(pe[0, 4, 1::2], 1.0, atol=1e-7)
    np.testing.assert_allclose(pe[0, 0, 0], np.sin(4.0), atol=1e-6)


def test_rel_positional_encoding_max_length():
    cfg = audio.AudioConfig(**{**TINY, "max_position_embeddings": 10})
    with pytest.raises(ValueError, match="max_position_embeddings"):
        audio.RelPositionalEncoding(cfg)(11, 1)


def test_rel_shift_matches_transformer_xl():
    # out[i, j] = in[i, j + T - 1 - i] for the first T columns
    b, h, t = 1, 1, 4
    scores = np.arange(t * (2 * t - 1), dtype=np.float32).reshape(b, h, t, 2 * t - 1)
    out = np.array(audio.RelPositionAttention._rel_shift(mx.array(scores)))[..., :t]
    for i in range(t):
        for j in range(t):
            assert out[0, 0, i, j] == scores[0, 0, i, j + t - 1 - i]


def test_sound_projection_relu_squared():
    cfg = audio.AudioConfig(**TINY)
    proj = audio.SoundProjectionMLX(cfg)
    x = mx.array(np.random.default_rng(0).standard_normal((1, 3, cfg.hidden_size)).astype(np.float32))
    h = proj.linear1(proj.norm(x))
    want = proj.linear2(mx.maximum(h, 0) * mx.maximum(h, 0))
    np.testing.assert_allclose(np.array(proj(x)), np.array(want), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------------------
# batching / masking
# --------------------------------------------------------------------------------------
def test_padded_batch_matches_single_runs(tower):
    long, short = _wave(16000, 1), _wave(9600, 2)
    batch = np.stack([long, np.pad(short, (0, len(long) - len(short)))])
    lengths = mx.array([len(long), len(short)])
    emb, mask = tower(mx.array(batch), lengths)
    emb, mask = np.array(emb), np.array(mask)
    assert not np.isnan(emb[mask]).any()

    for i, w in enumerate((long, short)):
        solo, solo_mask = tower(mx.array(w)[None])
        solo, solo_mask = np.array(solo)[0], np.array(solo_mask)[0]
        n = int(solo_mask.sum())
        assert int(mask[i].sum()) == n
        assert _cos(emb[i, :n], solo[:n]).min() > 0.9999


# --------------------------------------------------------------------------------------
# weight loading
# --------------------------------------------------------------------------------------
def test_map_key():
    assert audio._map_key("sound_encoder.encoder.layers.0.conv.norm.weight") == "encoder.layers.0.conv.norm.weight"
    assert audio._map_key("sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked") is None
    assert audio._map_key("sound_projection.linear1.weight") == "projection.linear1.weight"
    assert audio._map_key("backbone.layers.0.mixer.weight") is None
    assert audio._map_key("vision_model.x") is None


def test_convert_layouts():
    w4 = mx.zeros((8, 1, 3, 3), dtype=mx.bfloat16)
    out = audio._convert("encoder.subsampling.layers.0.weight", w4)
    assert out.shape == (8, 3, 3, 1) and out.dtype == mx.float32
    w3 = mx.zeros((64, 32, 1))
    assert audio._convert("encoder.layers.0.conv.pointwise_conv1.weight", w3).shape == (64, 1, 32)
    fb = mx.zeros((1, 16, 257))
    assert audio._convert("encoder.feature_extractor.featurizer.fb", fb).shape == (257, 16)
    fb2 = mx.zeros((1, 257, 16))
    assert audio._convert("encoder.feature_extractor.featurizer.fb", fb2).shape == (257, 16)
    assert audio._convert("projection.linear1.weight", mx.zeros((4, 5))).shape == (4, 5)


def _to_checkpoint(tower):
    """Inverse of _map_key/_convert: write the tower's params in the HF checkpoint layout."""
    out = {}
    for path, w in tree_flatten(tower.parameters()):
        if path.startswith("encoder."):
            key = "sound_encoder." + path
        else:
            key = "sound_" + path
        if path.startswith("encoder.subsampling.layers.") and path.endswith(".weight") and w.ndim == 4:
            w = w.transpose(0, 3, 1, 2)
        elif ".conv." in path and path.endswith(".weight") and w.ndim == 3:
            w = w.transpose(0, 2, 1)
        elif path.endswith("featurizer.fb"):
            w = w.transpose(1, 0)[None]
        out[key] = w
    return out


def test_load_audio_tower_round_trip(tower, tmp_path):
    ckpt = _to_checkpoint(tower)
    ckpt["sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked"] = mx.array([7])
    ckpt["backbone.embeddings.weight"] = mx.zeros((4, 4))
    half = {k: v for i, (k, v) in enumerate(sorted(ckpt.items())) if i % 2 == 0}
    rest = {k: v for k, v in ckpt.items() if k not in half}
    mx.save_safetensors(str(tmp_path / "a.safetensors"), half)
    mx.save_safetensors(str(tmp_path / "b.safetensors"), rest)

    loaded = audio.load_audio_tower(
        [str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")], audio.AudioConfig(**TINY)
    )
    w = mx.array(_wave(16000))[None]
    a, _ = tower(w)
    b, _ = loaded(w)
    np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-5, atol=1e-5)


def test_load_audio_tower_is_strict(tower, tmp_path):
    ckpt = _to_checkpoint(tower)
    ckpt.pop("sound_projection.linear2.weight")
    mx.save_safetensors(str(tmp_path / "s.safetensors"), ckpt)
    with pytest.raises(ValueError):
        audio.load_audio_tower(str(tmp_path / "s.safetensors"), audio.AudioConfig(**TINY))
