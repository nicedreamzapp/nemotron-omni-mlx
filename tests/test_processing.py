"""Unit tests for src/processing.py. Pure numpy + PIL: no weights, no torch, no GPU.

The torch-side parity checks live in test_processing_parity.py; these cover the logic
itself (sizing rules, resize kernel, placeholder expansion, wav loading, EVS) so it is
exercised on every push.
"""
from __future__ import annotations

import json
import math
import sys
import types
import wave

import numpy as np
import pytest
from PIL import Image

from src.processing import (
    OmniImageProcessor,
    OmniProcessor,
    _compute_weights_aa,
    _cubic_filter,
    _image_size_wh,
    _image_to_uint8_hwc,
    _make_list_of_images,
    compute_evs_retention_mask,
    resize_bicubic_antialias,
)


def _rgb(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), mode="RGB")


def _write_wav(path, data: np.ndarray, sr: int, width: int, channels: int = 1):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(sr)
        f.writeframes(data.tobytes())


# --------------------------------------------------------------------------------------
# resize kernel
# --------------------------------------------------------------------------------------
class TestCubicFilter:
    def test_interpolating_kernel_values(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, -1.0, -2.0], dtype=np.float32)
        np.testing.assert_allclose(_cubic_filter(x), [1, 0, 0, 0, 0, 0], atol=1e-7)

    def test_symmetric(self):
        x = np.linspace(-2.5, 2.5, 51).astype(np.float32)
        np.testing.assert_allclose(_cubic_filter(x), _cubic_filter(-x), atol=1e-7)

    def test_uses_pillow_coefficient(self):
        # a = -0.5: k(0.5) = (a+2)/8 - (a+3)/4 + 1 = 0.5625. With a = -0.75 it would be 0.59375.
        np.testing.assert_allclose(_cubic_filter(np.array([0.5], np.float32)), [0.5625])

    def test_preserves_dtype(self):
        assert _cubic_filter(np.zeros(3, np.float32)).dtype == np.float32


class TestResampleWeights:
    @pytest.mark.parametrize("n_in,n_out", [(10, 10), (8, 16), (16, 8), (7, 3), (3, 11), (100, 37)])
    def test_rows_sum_to_one(self, n_in, n_out):
        w = _compute_weights_aa(n_in, n_out)
        assert w.shape == (n_out, n_in)
        assert w.dtype == np.float32
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)

    def test_same_size_is_identity(self):
        np.testing.assert_allclose(_compute_weights_aa(12, 12), np.eye(12), atol=1e-7)

    def test_antialias_widens_support_only_when_downscaling(self):
        down = _compute_weights_aa(32, 8)
        up = _compute_weights_aa(8, 32)
        # bicubic touches at most 4 taps per output when upscaling; AA downscale by 4x uses ~16
        assert (np.count_nonzero(up, axis=1) <= 4).all()
        assert np.count_nonzero(down, axis=1).max() > 4
        no_aa = _compute_weights_aa(32, 8, antialias=False)
        assert (np.count_nonzero(no_aa, axis=1) <= 4).all()


class TestResize:
    def test_shape_and_dtype(self):
        img = np.random.default_rng(0).random((3, 20, 30)).astype(np.float32)
        out = resize_bicubic_antialias(img, 13, 47)
        assert out.shape == (3, 13, 47)
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize("out_hw", [(5, 5), (64, 40), (17, 91)])
    def test_constant_image_stays_constant(self, out_hw):
        img = np.full((3, 32, 24), 123.0, np.float32)
        np.testing.assert_allclose(resize_bicubic_antialias(img, *out_hw), 123.0, atol=1e-3)

    def test_noop_when_size_unchanged(self):
        img = np.random.default_rng(1).random((3, 9, 11)).astype(np.float32)
        np.testing.assert_array_equal(resize_bicubic_antialias(img, 9, 11), img)

    def test_channels_are_independent(self):
        rng = np.random.default_rng(2)
        img = rng.random((3, 16, 16)).astype(np.float32)
        out = resize_bicubic_antialias(img, 8, 24)
        for c in range(3):
            single = resize_bicubic_antialias(img[c : c + 1], 8, 24)
            np.testing.assert_allclose(out[c], single[0], atol=1e-6)


# --------------------------------------------------------------------------------------
# image processor: sizing rules
# --------------------------------------------------------------------------------------
class TestImageSizing:
    def setup_method(self):
        self.p = OmniImageProcessor()

    @pytest.mark.parametrize(
        "size_wh",
        [(448, 448), (1024, 768), (300, 700), (64, 64), (4000, 3000), (10, 5000), (5000, 10), (1, 1)],
    )
    def test_invariants(self, size_wh):
        budget = self.p.max_num_patches
        w, h = self.p._compute_target_patches(size_wh, budget)
        assert w % 2 == 0 and h % 2 == 0, "patch grid must be divisible by the 2x pixel shuffle"
        assert w * h <= budget
        assert w * h >= self.p.min_num_patches, "small images are upscaled to min_num_patches"

    def test_small_square_is_upscaled_to_min_patches(self):
        # 448 / 16 = 28 patches per side -> 784 < 1024, so it is scaled up to a 32x32 grid
        assert self.p._compute_target_patches((448, 448), self.p.max_num_patches) == (32, 32)

    def test_large_image_is_capped_and_keeps_aspect(self):
        w, h = self.p._compute_target_patches((4000, 3000), self.p.max_num_patches)
        assert w * h <= self.p.max_num_patches
        assert abs(w / h - 4 / 3) < 0.05

    def test_odd_grid_rounds_down_when_rounding_up_breaks_budget(self):
        # 133 x 100 fits the 13312 budget, 134 x 100 does not -> 132
        assert self.p._compute_target_patches((4000, 3000), 13312) == (132, 100)

    @pytest.mark.parametrize("size_wh", [(320, 180), (180, 320), (448, 448), (1920, 1080), (50, 900)])
    def test_video_sizing(self, size_wh):
        w, h = self.p._compute_target_patches_video(size_wh)
        assert w % 2 == 0 and h % 2 == 0
        # rounding each side to the nearest patch can land slightly over the target
        # (same as vLLM's _compute_aspect_preserving_size), but never far from it
        assert abs(w * h - self.p.video_target_num_patches) / self.p.video_target_num_patches < 0.1
        if size_wh[0] >= size_wh[1]:
            assert w >= h
        else:
            assert h >= w

    def test_video_square_grid_without_aspect(self):
        p = OmniImageProcessor(video_maintain_aspect_ratio=False, video_target_num_patches=1000)
        # int(sqrt(1000)) = 31 -> rounded down to the even 30
        assert p._compute_target_patches_video((640, 360)) == (30, 30)

    def test_video_transposed_aspect_transposes_grid(self):
        assert self.p._compute_target_patches_video((320, 180)) == tuple(
            reversed(self.p._compute_target_patches_video((180, 320)))
        )


class TestImageProcessorCall:
    def setup_method(self):
        self.p = OmniImageProcessor()

    def test_single_image_output(self):
        out = self.p(_rgb(448, 448))
        assert out["pixel_values"].shape == (1, 3, 512, 512)
        assert out["pixel_values"].dtype == np.float32
        assert out["num_patches"] == [1]
        assert out["num_tokens"] == [32 * 32 // 4]
        assert out["imgs_sizes"] == [(512, 512)]

    def test_normalization(self):
        # a solid colour at the target size skips the resize, so values are exact
        arr = np.full((512, 512, 3), 255, np.uint8)
        pv = self.p(arr)["pixel_values"][0]
        expect = (1.0 - np.array(self.p.norm_mean)) / np.array(self.p.norm_std)
        np.testing.assert_allclose(pv[:, 0, 0], expect, rtol=1e-6)

    def test_mixed_sizes_return_a_list(self):
        out = self.p([_rgb(448, 448), _rgb(300, 700)])
        assert isinstance(out["pixel_values"], list)
        assert [t.shape[1:] for t in out["pixel_values"]] == [s for s in out["imgs_sizes"]]
        assert len(out["num_tokens"]) == 2

    def test_same_sizes_are_stacked(self):
        out = self.p([_rgb(448, 448, 0), _rgb(448, 448, 1)])
        assert isinstance(out["pixel_values"], np.ndarray)
        assert out["pixel_values"].shape[0] == 2

    def test_tokens_match_grid(self):
        out = self.p(_rgb(1024, 768))
        h, w = out["imgs_sizes"][0]
        assert out["num_tokens"][0] == (h // 16) * (w // 16) // 4

    def test_video_mode(self):
        frames = [_rgb(320, 180, i) for i in range(3)]
        out = self.p(frames, is_video=True)
        assert out["pixel_values"].shape == (3, 3, 24 * 16, 42 * 16)
        assert out["num_tokens"] == [42 * 24 // 4] * 3

    def test_from_pretrained_ignores_hf_only_keys(self, tmp_path):
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "image_processor_type": "Whatever",
                    "auto_map": {"AutoImageProcessor": "x.y"},
                    "patch_size": 16,
                    "max_num_patches": 2048,
                    "some_future_key": True,
                }
            )
        )
        p = OmniImageProcessor.from_pretrained(str(tmp_path))
        assert p.max_num_patches == 2048
        assert p.patch_size == 16


class TestImageHelpers:
    def test_make_list_variants(self):
        img = _rgb(4, 4)
        arr3 = np.zeros((4, 5, 3), np.uint8)
        arr4 = np.zeros((2, 4, 5, 3), np.uint8)
        assert _make_list_of_images(img) == [img]
        assert len(_make_list_of_images(arr3)) == 1
        assert len(_make_list_of_images(arr4)) == 2
        assert len(_make_list_of_images([img, [arr3, arr4]])) == 4

    def test_make_list_rejects_bad_input(self):
        with pytest.raises(ValueError, match="ndim"):
            _make_list_of_images(np.zeros((4, 4)))
        with pytest.raises(ValueError, match="Unsupported image input type"):
            _make_list_of_images("cat.png")

    def test_size_is_width_height(self):
        assert _image_size_wh(_rgb(7, 3)) == (7, 3)
        assert _image_size_wh(np.zeros((3, 7, 3))) == (7, 3)
        with pytest.raises(ValueError, match="HWC"):
            _image_size_wh(np.zeros((3, 7)))

    def test_non_rgb_pil_is_converted(self):
        rgba = Image.new("RGBA", (5, 4), (10, 20, 30, 40))
        arr = _image_to_uint8_hwc(rgba)
        assert arr.shape == (4, 5, 3)
        assert tuple(arr[0, 0]) == (10, 20, 30)
        gray = Image.new("L", (5, 4), 77)
        assert _image_to_uint8_hwc(gray).shape == (4, 5, 3)


# --------------------------------------------------------------------------------------
# EVS
# --------------------------------------------------------------------------------------
class TestEVS:
    def test_first_frame_always_kept_and_count(self):
        rng = np.random.default_rng(0)
        T, H, W, C = 4, 4, 4, 8
        mask = compute_evs_retention_mask(rng.standard_normal((T, H * W, C)), (T, H, W), q=0.7)
        assert mask.shape == (T * H * W,)
        assert mask.dtype == bool
        assert mask[: H * W].all()
        assert mask.sum() == max(H * W, int(T * H * W * 0.3))

    def test_static_video_keeps_only_first_frame(self):
        frame = np.random.default_rng(1).standard_normal((1, 9, 4))
        embeds = np.repeat(frame, 5, axis=0)
        mask = compute_evs_retention_mask(embeds, (5, 3, 3), q=0.9)
        assert mask[:9].all() and not mask[9:].any()

    def test_changed_positions_are_kept_first(self):
        rng = np.random.default_rng(2)
        embeds = np.repeat(rng.standard_normal((1, 4, 6)), 3, axis=0)
        embeds[2, 1] = -embeds[1, 1]  # one token flips direction in the last frame
        mask = compute_evs_retention_mask(embeds, (3, 2, 2), q=0.5)
        # budget is max(4, int(12 * 0.5)) = 6 -> first frame (4) + the two most changed
        assert mask.sum() == 6
        assert mask[4 * 2 + 1], "the changed token must be retained"

    def test_q_zero_keeps_everything(self):
        embeds = np.random.default_rng(3).standard_normal((3, 4, 5))
        assert compute_evs_retention_mask(embeds, (3, 2, 2), q=0.0).all()

    def test_flat_input_layout_is_accepted(self):
        embeds = np.random.default_rng(4).standard_normal((3, 4, 5))
        a = compute_evs_retention_mask(embeds, (3, 2, 2))
        b = compute_evs_retention_mask(embeds.reshape(12, 5), (3, 2, 2))
        np.testing.assert_array_equal(a, b)

    def test_zero_vectors_do_not_produce_nan(self):
        embeds = np.zeros((3, 4, 5), np.float32)
        mask = compute_evs_retention_mask(embeds, (3, 2, 2), q=0.5)
        assert mask[:4].all()


# --------------------------------------------------------------------------------------
# audio: token count + loading
# --------------------------------------------------------------------------------------
class TestAudioTokenCount:
    def test_five_seconds_is_63_tokens(self, processor):
        # 5 s @ 16 kHz -> 1 + 80000 // 160 = 501 mel frames -> three stride-2 convs -> 63
        assert processor._estimate_audio_num_embeddings(80000) == 63

    @pytest.mark.parametrize("n", [0, 1, 159, 160, 161, 1279, 1280, 16000, 80001, 1_000_000])
    def test_matches_conv_arithmetic(self, processor, n):
        L = 1 + n // 160
        for _ in range(3):
            L = (L + 2 * 1 - 3) // 2 + 1  # torch conv out length, k=3, s=2, p=1
        assert processor._estimate_audio_num_embeddings(n) == L

    def test_monotonic(self, processor):
        counts = [processor._estimate_audio_num_embeddings(n) for n in range(0, 40000, 97)]
        assert counts == sorted(counts)

    def test_subsampling_factor_from_config(self, fake_tokenizer):
        p4 = OmniProcessor(OmniImageProcessor(), fake_tokenizer, audio_subsampling_factor=4)
        p8 = OmniProcessor(OmniImageProcessor(), fake_tokenizer, audio_subsampling_factor=8)
        assert p4._estimate_audio_num_embeddings(80000) == 126
        assert p8._estimate_audio_num_embeddings(80000) == 63


class TestWavLoading:
    def test_pcm16_mono(self, tmp_path):
        data = (np.sin(np.linspace(0, 20, 1600)) * 20000).astype(np.int16)
        _write_wav(tmp_path / "a.wav", data, 16000, 2)
        got = OmniProcessor._load_audio_stdlib(str(tmp_path / "a.wav"), 16000)
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, data / 32768.0, atol=1e-7)

    def test_pcm8_is_unsigned(self, tmp_path):
        data = np.array([0, 128, 255], np.uint8)
        _write_wav(tmp_path / "a.wav", data, 16000, 1)
        got = OmniProcessor._load_audio_stdlib(str(tmp_path / "a.wav"), 16000)
        np.testing.assert_allclose(got, [-1.0, 0.0, 127 / 128], atol=1e-7)

    def test_pcm32(self, tmp_path):
        data = np.array([0, 2**30, -(2**31)], np.int32)
        _write_wav(tmp_path / "a.wav", data, 16000, 4)
        got = OmniProcessor._load_audio_stdlib(str(tmp_path / "a.wav"), 16000)
        np.testing.assert_allclose(got, [0.0, 0.5, -1.0], atol=1e-7)

    def test_stereo_is_averaged(self, tmp_path):
        data = np.array([[1000, 3000], [-2000, 0]], np.int16)
        _write_wav(tmp_path / "a.wav", data, 16000, 2, channels=2)
        got = OmniProcessor._load_audio_stdlib(str(tmp_path / "a.wav"), 16000)
        np.testing.assert_allclose(got, [2000 / 32768, -1000 / 32768], atol=1e-7)

    def test_unsupported_width(self, tmp_path):
        _write_wav(tmp_path / "a.wav", np.zeros(6, np.uint8), 16000, 3)
        with pytest.raises(ValueError, match="sample width"):
            OmniProcessor._load_audio_stdlib(str(tmp_path / "a.wav"), 16000)

    def test_resample(self, tmp_path):
        pytest.importorskip("scipy")
        _write_wav(tmp_path / "a.wav", np.zeros(800, np.int16), 8000, 2)
        got = OmniProcessor._load_audio_stdlib(str(tmp_path / "a.wav"), 16000)
        assert got.shape == (1600,)

    def test_falls_back_to_stdlib_without_librosa_or_soundfile(self, processor, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "librosa", None)
        monkeypatch.setitem(sys.modules, "soundfile", None)
        data = np.arange(-50, 50, dtype=np.int16) * 300
        _write_wav(tmp_path / "a.wav", data, 16000, 2)
        got = processor._load_audio(str(tmp_path / "a.wav"), 16000)
        np.testing.assert_allclose(got, data / 32768.0, atol=1e-7)

    def test_prefers_librosa(self, processor, monkeypatch):
        fake = types.ModuleType("librosa")
        fake.load = lambda path, sr, mono: (np.ones(5, np.float32), sr)
        monkeypatch.setitem(sys.modules, "librosa", fake)
        np.testing.assert_array_equal(processor._load_audio("x.wav", 16000), np.ones(5))


class TestProcessAudio:
    def test_array_inputs(self, processor):
        clips, n = processor._process_audio([np.zeros(80000, np.float32), np.zeros((1, 16000), np.float32)])
        assert [c.shape for c in clips] == [(80000,), (16000,)]
        assert n == [63, processor._estimate_audio_num_embeddings(16000)]

    def test_single_array_and_minimum_one_token(self, processor):
        clips, n = processor._process_audio(np.zeros(1, np.float32))
        assert len(clips) == 1 and n[0] >= 1

    def test_path_input(self, processor, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "librosa", None)
        monkeypatch.setitem(sys.modules, "soundfile", None)
        _write_wav(tmp_path / "a.wav", np.zeros(16000, np.int16), 16000, 2)
        clips, n = processor._process_audio(str(tmp_path / "a.wav"))
        assert clips[0].shape == (16000,)


# --------------------------------------------------------------------------------------
# the processor: placeholder expansion + tokenization
# --------------------------------------------------------------------------------------
class TestProcessorExpansion:
    def test_text_only(self, processor):
        out = processor(text="hi")
        assert out["input_ids"].shape == (1, 2)
        assert out["input_ids"].dtype == np.int32
        assert out["attention_mask"].tolist() == [[1, 1]]
        assert "pixel_values" not in out and "sound_clips" not in out

    def test_image_expansion(self, processor):
        out = processor(text="<image>\nWhat?", images=_rgb(448, 448))
        n = out["num_tokens"][0]
        assert out["expanded_text"][0] == "<img>" + "<image>" * n + "</img>\nWhat?"
        ids = out["input_ids"][0]
        assert int((ids == processor.image_token_id).sum()) == n == 256
        start = list(ids).index(19)
        assert ids[start + n + 1] == 20

    def test_multi_image_each_gets_its_own_count(self, processor):
        out = processor(text="<image> and <image>", images=[_rgb(448, 448), _rgb(1024, 768)])
        a, b = out["num_tokens"]
        assert a != b
        assert out["expanded_text"][0] == (
            "<img>" + "<image>" * a + "</img> and <img>" + "<image>" * b + "</img>"
        )

    def test_audio_expansion(self, processor):
        out = processor(text="<so_embedding>\nTranscribe.", audio=np.zeros(80000, np.float32))
        assert out["expanded_text"][0] == "<so_start>" + "<so_embedding>" * 63 + "<so_end>\nTranscribe."
        assert int((out["input_ids"] == processor.audio_token_id).sum()) == 63
        assert out["sound_clips"][0].shape == (80000,)

    def test_audio_placeholder_without_clip_gets_one_token(self, processor):
        out = processor(text="<so_embedding><so_embedding>", audio=np.zeros(80000, np.float32))
        assert int((out["input_ids"] == processor.audio_token_id).sum()) == 63 + 1

    def test_image_plus_audio(self, processor):
        out = processor(
            text="<image>\n<so_embedding>\nMatch?",
            images=_rgb(448, 448),
            audio=np.zeros(16000, np.float32),
        )
        ids = out["input_ids"]
        assert int((ids == processor.image_token_id).sum()) == 256
        assert int((ids == processor.audio_token_id).sum()) == processor._estimate_audio_num_embeddings(16000)
        assert "pixel_values" in out and "sound_clips" in out

    def _video(self, n=5):
        return [_rgb(320, 180, i) for i in range(n)]

    def test_video_expansion_groups_frames_in_pairs(self, processor):
        out = processor(text="<video>\nWhat happens?", videos=self._video(5))
        text = out["expanded_text"][0]
        tpt = out["num_tokens"][0]
        assert text.startswith("Frame 1 and frame 2: <img>")
        assert "\nFrame 3 and frame 4: <img>" in text
        assert "\nFrame 5: <img>" in text
        assert "<video>" not in text
        # ceil(5 / 2) = 3 tubelets
        assert int((out["input_ids"] == processor.image_token_id).sum()) == 3 * tpt
        assert out["pixel_values_videos"].shape[0] == 5
        assert "pixel_values" not in out

    def test_video_timestamps_from_fps(self, processor):
        meta = types.SimpleNamespace(fps=2.0, frames_indices=None)
        out = processor(text="<video>", videos=self._video(3), video_metadata=meta)
        text = out["expanded_text"][0]
        assert "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.50 seconds: " in text
        assert "Frame 3 sampled at 1.00 seconds: " in text

    def test_video_timestamps_from_frame_indices(self, processor):
        meta = types.SimpleNamespace(fps=10.0, frames_indices=[0, 15, 30])
        out = processor(text="<video>", videos=self._video(3), video_metadata=meta)
        text = out["expanded_text"][0]
        assert "frame 2 sampled at 1.50 seconds" in text
        assert "Frame 3 sampled at 3.00 seconds" in text

    def test_video_rejects_batches(self, processor):
        with pytest.raises(AssertionError, match="batch size"):
            processor(text=["<video>", "<video>"], videos=self._video(2))

    def test_custom_temporal_patch(self, fake_tokenizer):
        p = OmniProcessor(OmniImageProcessor(), fake_tokenizer, video_temporal_patch_dim=3)
        out = p(text="<video>", videos=self._video(4))
        assert out["expanded_text"][0].count("<img>") == math.ceil(4 / 3)

    def test_decode_passthrough(self, processor):
        out = processor(text="hello")
        assert processor.decode(out["input_ids"][0].tolist()) == "hello"
        assert processor.batch_decode(out["input_ids"].tolist()) == ["hello"]


class TestFromPretrained:
    def test_reads_sound_and_video_config(self, tmp_path, fake_tokenizer):
        (tmp_path / "preprocessor_config.json").write_text(json.dumps({"patch_size": 16}))
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "video_temporal_patch_size": 4,
                    "video_pruning_rate": 0.7,
                    "sound_config": {"sampling_rate": 8000, "subsampling_factor": 4},
                }
            )
        )
        p = OmniProcessor.from_pretrained(str(tmp_path), tokenizer=fake_tokenizer)
        assert p.video_temporal_patch_dim == 4
        assert p.audio_sampling_rate == 8000
        assert p.audio_subsampling_factor == 4
        assert p.config["video_pruning_rate"] == 0.7
        assert p.image_token_id == 18

    def test_defaults_without_config_json(self, tmp_path, fake_tokenizer):
        (tmp_path / "preprocessor_config.json").write_text("{}")
        p = OmniProcessor.from_pretrained(str(tmp_path), tokenizer=fake_tokenizer)
        assert p.video_temporal_patch_dim == 2
        assert p.audio_sampling_rate == 16000
        assert p.audio_subsampling_factor == 8
        assert p.config == {}

    def test_missing_preprocessor_config(self, tmp_path, fake_tokenizer):
        with pytest.raises(FileNotFoundError):
            OmniProcessor.from_pretrained(str(tmp_path), tokenizer=fake_tokenizer)
