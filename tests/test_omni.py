"""Unit tests for src/omni.py: the splice, EVS pruning, the LM wrapper, shard lookup,
EOS handling, prompt construction and the --dry-run CLI. No weights are loaded; MLX runs
on whatever device is available (CPU on Linux CI)."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")

from src import omni  # noqa: E402
from src.processing import OmniImageProcessor, OmniProcessor  # noqa: E402

CTX = 18


# --------------------------------------------------------------------------------------
# splice_embeddings
# --------------------------------------------------------------------------------------
class TestSplice:
    def test_scatters_in_order(self):
        ids = np.array([[1, CTX, 2, CTX, CTX, 3]])
        base = mx.zeros((1, 6, 4))
        tower = mx.array(np.arange(12, dtype=np.float32).reshape(1, 3, 4))
        out = np.array(omni.splice_embeddings(base, ids, CTX, tower))
        assert out.shape == (1, 6, 4)
        np.testing.assert_array_equal(out[0, [0, 2, 5]], 0)
        np.testing.assert_array_equal(out[0, 1], [0, 1, 2, 3])
        np.testing.assert_array_equal(out[0, 3], [4, 5, 6, 7])
        np.testing.assert_array_equal(out[0, 4], [8, 9, 10, 11])

    def test_keeps_text_embeddings(self):
        ids = np.array([[5, CTX, 6]])
        base = mx.array(np.full((1, 3, 2), 7.0, np.float32))
        out = np.array(omni.splice_embeddings(base, ids, CTX, mx.ones((1, 2))))
        np.testing.assert_array_equal(out[0], [[7, 7], [1, 1], [7, 7]])

    def test_casts_to_embedding_dtype(self):
        ids = np.array([[CTX]])
        base = mx.zeros((1, 1, 2), dtype=mx.bfloat16)
        out = omni.splice_embeddings(base, ids, CTX, mx.ones((1, 1, 2), dtype=mx.float32))
        assert out.dtype == mx.bfloat16

    def test_flattens_grouped_tower_output(self):
        # video: (n_groups, n_tok, C) -> flattened across groups
        ids = np.array([[CTX] * 4])
        tower = mx.array(np.arange(8, dtype=np.float32).reshape(2, 2, 2))
        out = np.array(omni.splice_embeddings(mx.zeros((1, 4, 2)), ids, CTX, tower))
        np.testing.assert_array_equal(out[0].reshape(-1), np.arange(8))

    def test_no_context_tokens(self):
        with pytest.raises(ValueError, match="No context tokens"):
            omni.splice_embeddings(mx.zeros((1, 3, 2)), np.array([[1, 2, 3]]), CTX, mx.zeros((1, 2)))

    def test_count_mismatch(self):
        with pytest.raises(ValueError, match=r"context token count \(2\) != tower embedding count \(3\)"):
            omni.splice_embeddings(
                mx.zeros((1, 3, 2)), np.array([[CTX, CTX, 1]]), CTX, mx.zeros((3, 2))
            )


# --------------------------------------------------------------------------------------
# apply_evs
# --------------------------------------------------------------------------------------
class TestApplyEVS:
    def test_prunes_static_video_to_first_group(self, capsys):
        n_groups, n_tok, c = 3, 4, 8
        frame = np.random.default_rng(0).standard_normal((1, n_tok, c)).astype(np.float32)
        video = mx.array(np.repeat(frame, n_groups, axis=0))
        ids = np.array([[7] + [CTX] * (n_groups * n_tok) + [9]])
        embeds = mx.array(np.arange(ids.shape[1] * 2, dtype=np.float32).reshape(1, -1, 2))

        out_embeds, out_ids = omni.apply_evs(embeds, ids, video, CTX, pruning_rate=0.9)
        # budget = max(4, int(12 * 0.1)) = 4 -> only the first group survives
        assert out_ids.tolist() == [[7] + [CTX] * n_tok + [9]]
        assert out_embeds.shape == (1, n_tok + 2, 2)
        np.testing.assert_array_equal(np.array(out_embeds)[0, -1], np.array(embeds)[0, -1])
        assert "[evs] retained 4/12" in capsys.readouterr().out

    def test_non_square_grid_is_left_alone(self):
        video = mx.zeros((2, 6, 4))
        ids = np.array([[CTX] * 12])
        embeds = mx.zeros((1, 12, 4))
        out_embeds, out_ids = omni.apply_evs(embeds, ids, video, CTX, 0.7)
        assert out_embeds is embeds and out_ids is ids


# --------------------------------------------------------------------------------------
# _make_omni_lm
# --------------------------------------------------------------------------------------
class _FakeBackbone:
    def __init__(self):
        self.embeddings = lambda inputs: mx.full((1, inputs.shape[-1], 2), -1.0)
        self.seen = None

    def __call__(self, inputs, cache=None):
        self.seen = self.embeddings(inputs)
        return self.seen


class _FakeLM:
    def __init__(self, fail=False):
        self.backbone = _FakeBackbone()
        self.fail = fail
        self.extra = "passthrough"

    def lm_head(self, x):
        if self.fail:
            raise RuntimeError("boom")
        return x * 10

    def __call__(self, inputs, cache=None):
        return ("plain", inputs)

    def make_cache(self):
        return ["cache"]


class TestOmniLM:
    def test_without_embeddings_calls_base(self):
        lm = omni._make_omni_lm(_FakeLM())
        assert lm(mx.array([1, 2]))[0] == "plain"

    def test_injects_embeddings_and_restores(self):
        base = _FakeLM()
        real = base.backbone.embeddings
        lm = omni._make_omni_lm(base)
        emb = mx.ones((1, 2, 2))
        out = lm(mx.array([[1, 2]]), input_embeddings=emb)
        np.testing.assert_array_equal(np.array(out), 10.0)
        assert base.backbone.embeddings is real

    def test_restores_embeddings_on_error(self):
        base = _FakeLM(fail=True)
        real = base.backbone.embeddings
        lm = omni._make_omni_lm(base)
        with pytest.raises(RuntimeError):
            lm(mx.array([[1]]), input_embeddings=mx.ones((1, 1, 2)))
        assert base.backbone.embeddings is real

    def test_attribute_passthrough(self):
        lm = omni._make_omni_lm(_FakeLM())
        assert lm.make_cache() == ["cache"]
        assert lm.extra == "passthrough"
        with pytest.raises(AttributeError):
            lm.does_not_exist

    def test_signature_advertises_input_embeddings(self):
        # mlx_lm gates on this parameter name existing on __call__
        import inspect

        lm = omni._make_omni_lm(_FakeLM())
        assert "input_embeddings" in inspect.signature(lm.__call__).parameters


def test_embed_tokens_uses_backbone_table():
    base = _FakeLM()
    base.backbone.embeddings = lambda ids: ids.astype(mx.float32)[..., None] * 2
    out = omni._embed_tokens(base, np.array([[1, 2, 3]], dtype=np.int64))
    np.testing.assert_array_equal(np.array(out)[0, :, 0], [2, 4, 6])


# --------------------------------------------------------------------------------------
# _shard_for_prefix / _eos_ids
# --------------------------------------------------------------------------------------
def _index(tmp_path, weight_map):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return str(tmp_path)


class TestShardForPrefix:
    def test_finds_single_shard(self, tmp_path):
        d = _index(
            tmp_path,
            {"vision_model.a": "s1.safetensors", "vision_model.b": "s1.safetensors", "backbone.x": "s2.safetensors"},
        )
        assert omni._shard_for_prefix(d, "vision_model.") == str(tmp_path / "s1.safetensors")

    def test_missing_prefix(self, tmp_path):
        d = _index(tmp_path, {"backbone.x": "s2.safetensors"})
        with pytest.raises(RuntimeError, match="no weights with prefix"):
            omni._shard_for_prefix(d, "sound_encoder.")

    def test_multiple_shards(self, tmp_path):
        d = _index(tmp_path, {"sound_encoder.a": "s1.safetensors", "sound_encoder.b": "s3.safetensors"})
        with pytest.raises(RuntimeError, match="spans multiple shards"):
            omni._shard_for_prefix(d, "sound_encoder.")


class TestEosIds:
    def test_tokenizer_and_generation_config_list(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 11]}))
        tok = types.SimpleNamespace(eos_token_id=11)
        assert omni._eos_ids(tok, str(tmp_path)) == {2, 11}

    def test_generation_config_int(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 5}))
        assert omni._eos_ids(types.SimpleNamespace(eos_token_id=None), str(tmp_path)) == {5}

    def test_fallback(self, tmp_path):
        assert omni._eos_ids(types.SimpleNamespace(), str(tmp_path)) == {11}


# --------------------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------------------
class TestBuildPrompt:
    def _proc(self, tok):
        return OmniProcessor(OmniImageProcessor(), tok)

    def test_text_only_is_chat_templated(self, fake_tokenizer):
        out = omni.build_prompt(self._proc(fake_tokenizer), "Hi", False, False, False)
        assert out == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"

    def test_modality_order(self, fake_tokenizer):
        omni.build_prompt(self._proc(fake_tokenizer), "Q", True, True, True)
        content = fake_tokenizer.chat_calls[-1][0]["content"]
        assert content == "<image>\nThis is a video:\n<video>\n<so_embedding>\nQ"


# --------------------------------------------------------------------------------------
# media loading
# --------------------------------------------------------------------------------------
class TestMedia:
    def test_nothing_requested(self):
        assert omni.load_media(None, None) == (None, None)

    def test_image_is_rgb(self, tmp_path):
        p = tmp_path / "x.png"
        Image.new("RGBA", (4, 3)).save(p)
        img, frames = omni.load_media(str(p), None)
        assert img.mode == "RGB" and img.size == (4, 3) and frames is None

    def test_video_without_decoder_has_a_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "decord", None)
        monkeypatch.setitem(sys.modules, "av", None)
        with pytest.raises(RuntimeError, match="install `decord` or `av`"):
            omni._read_video_frames("clip.mp4", 4)

    def test_video_with_fake_decord(self, monkeypatch):
        class _Frame:
            def __init__(self, i):
                self.i = i

            def asnumpy(self):
                return np.full((2, 3, 3), self.i, np.uint8)

        class _Reader:
            def __init__(self, path):
                self.n = 10

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                return _Frame(int(i))

        monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=_Reader))
        frames = omni._read_video_frames("clip.mp4", 4)
        # uniform sampling over 10 frames: linspace(0, 9, 4) rounded -> 0, 3, 6, 9
        assert [np.asarray(f)[0, 0, 0] for f in frames] == [0, 3, 6, 9]


# --------------------------------------------------------------------------------------
# CLI --dry-run (processor only, no weights)
# --------------------------------------------------------------------------------------
@pytest.fixture
def ref_dir(tmp_path, monkeypatch, fake_tokenizer):
    (tmp_path / "preprocessor_config.json").write_text("{}")
    real = OmniProcessor.from_pretrained.__func__

    def from_pretrained(cls, path, tokenizer=None):
        return real(cls, path, tokenizer=tokenizer or fake_tokenizer)

    monkeypatch.setattr(OmniProcessor, "from_pretrained", classmethod(from_pretrained))
    return tmp_path


class TestDryRun:
    def test_text(self, ref_dir, capsys):
        assert omni.main(["--prompt", "Hello", "--dry-run", "--ref-dir", str(ref_dir)]) == 0
        out = capsys.readouterr().out
        assert "[proc]" in out and "0 image/video ctx, 0 sound ctx" in out
        assert "expanded prompt" in out and "Hello" in out

    def test_image_and_audio(self, ref_dir, tmp_path, capsys):
        import wave

        img = tmp_path / "cat.png"
        Image.new("RGB", (448, 448), (200, 100, 50)).save(img)
        wav = tmp_path / "a.wav"
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(np.zeros(80000, np.int16).tobytes())

        rc = omni.main(
            ["--prompt", "What?", "--image", str(img), "--audio", str(wav), "--dry-run", "--ref-dir", str(ref_dir)]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "256 image/video ctx, 63 sound ctx" in out
        assert "pixel_values (1, 3, 512, 512), num_tokens=[256]" in out
        assert "...<image x256>..." in out
        assert "...<sound x63>..." in out

    def test_quiet_dry_run_returns_empty(self, ref_dir):
        assert omni.generate("hi", ref_dir=str(ref_dir), dry_run=True, verbose=False) == ""

    def test_prompt_is_required(self):
        with pytest.raises(SystemExit):
            omni.main([])
