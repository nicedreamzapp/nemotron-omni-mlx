# Parity test: PyTorch reference (transformers ParakeetEncoder + NVIDIA
# SoundProjection, CPU fp32) vs the MLX port in src/audio.py, using the real
# checkpoint weights (BF16 shard 1, upcast to fp32).
#
# Run: ~/.local/mlx-server/bin/python tests/test_audio_parity.py

import importlib.util
import json
import math
import os
import sys
import wave
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARD = os.path.join(ROOT, "weights-bf16", "model-00001-of-00017.safetensors")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "audio_5s_16k.npy")
FIXTURE_WAV = os.path.join(ROOT, "tests", "fixtures", "audio_5s_16k.wav")

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "reference"))

EPSILON = 1e-5
LOG_ZERO_GUARD_VALUE = 2.0 ** -24
SR = 16000
DURATION = 5.0


# ---------------------------------------------------------------------------
# Fixture: deterministic 5 s / 16 kHz waveform (sine sweeps + tone + noise)
# ---------------------------------------------------------------------------
def make_fixture():
    if os.path.exists(FIXTURE):
        return np.load(FIXTURE)
    t = np.arange(int(SR * DURATION), dtype=np.float64) / SR
    # linear chirps: f(t) = f0 + (f1 - f0) * t / T, phase = 2*pi*(f0*t + k/2 t^2)
    def chirp(f0, f1, amp):
        k = (f1 - f0) / DURATION
        return amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))

    rng = np.random.default_rng(2026)
    wav = (
        chirp(100.0, 4000.0, 0.45)
        + chirp(6000.0, 500.0, 0.25)
        + 0.2 * np.sin(2 * np.pi * 440.0 * t)
        + 0.02 * rng.standard_normal(t.shape)
    )
    wav = (0.9 * wav / np.max(np.abs(wav))).astype(np.float32)
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    np.save(FIXTURE, wav)
    with wave.open(FIXTURE_WAV, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((wav * 32767.0).astype(np.int16).tobytes())
    return wav


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def torch_reference(wav, sound_cfg_dict, llm_hidden_size):
    import torch
    from safetensors.torch import safe_open

    # NVIDIA reference wrappers (imports transformers ParakeetEncoder)
    spec = importlib.util.spec_from_file_location(
        "nvidia_audio_model", os.path.join(ROOT, "reference", "audio_model.py")
    )
    am = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(am)

    sound_cfg = SimpleNamespace(**sound_cfg_dict)
    encoder = am.SoundEncoder(config=sound_cfg)
    projection = am.SoundProjection(
        sound_hidden_size=sound_cfg.hidden_size,
        projection_hidden_size=sound_cfg.projection_hidden_size,
        llm_hidden_size=llm_hidden_size,
        bias=sound_cfg.projection_bias,
    )

    # load checkpoint weights (bf16 -> fp32)
    enc_sd, proj_sd, featurizer = {}, {}, {}
    with safe_open(SHARD, framework="pt") as f:
        for key in f.keys():
            if key.startswith("sound_encoder.encoder.feature_extractor.featurizer."):
                featurizer[key.rsplit(".", 1)[-1]] = f.get_tensor(key).to(torch.float32)
            elif key.startswith("sound_encoder.encoder."):
                enc_sd[key[len("sound_encoder.encoder."):]] = f.get_tensor(key).to(torch.float32)
            elif key.startswith("sound_projection."):
                proj_sd[key[len("sound_projection."):]] = f.get_tensor(key).to(torch.float32)

    missing, unexpected = encoder.encoder.load_state_dict(enc_sd, strict=False)
    assert not missing, f"missing encoder keys: {missing}"
    assert not unexpected, f"unexpected encoder keys: {unexpected}"
    missing, unexpected = projection.load_state_dict(proj_sd, strict=True)

    encoder = encoder.float().eval()
    projection = projection.float().eval()
    encoder.encoder.set_attn_implementation("eager")

    # ---- frontend: exact ParakeetFeatureExtractor ops, filters from checkpoint
    fb = featurizer["fb"].squeeze(0)  # (n_mels, n_freq)
    window = featurizer["window"]  # (win_length,)
    n_fft, hop, win_length, preemphasis = 512, 160, 400, 0.97

    x = torch.from_numpy(wav).to(torch.float32)[None, :]  # (1, L)
    audio_lengths = torch.tensor([x.shape[1]])
    x = torch.cat([x[:, :1], x[:, 1:] - preemphasis * x[:, :-1]], dim=1)

    stft = torch.stft(
        x, n_fft, hop_length=hop, win_length=win_length, window=window,
        return_complex=True, pad_mode="constant",
    )
    magnitudes = torch.view_as_real(stft)
    magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)).pow(2)
    mel = (fb @ magnitudes)
    mel = torch.log(mel + LOG_ZERO_GUARD_VALUE).permute(0, 2, 1)  # (1, T, n_mels)

    feat_lengths = torch.floor_divide(audio_lengths + n_fft // 2 * 2 - n_fft, hop)
    attention_mask = torch.arange(mel.shape[1])[None, :] < feat_lengths[:, None]

    mask = attention_mask.unsqueeze(-1)
    masked = mel * mask
    mean = (masked.sum(dim=1) / feat_lengths.unsqueeze(-1)).unsqueeze(1)
    variance = ((masked - mean) ** 2 * mask).sum(dim=1) / (feat_lengths - 1).unsqueeze(-1)
    std = torch.sqrt(variance).unsqueeze(1)
    features = ((mel - mean) / (std + EPSILON)) * mask

    # frontend verification vs analytic hann + slaney mel filters
    from transformers.audio_utils import mel_filter_bank
    analytic_fb = torch.from_numpy(
        mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1, num_mel_filters=sound_cfg_dict["num_mel_bins"],
            min_frequency=0.0, max_frequency=SR / 2, sampling_rate=SR,
            norm="slaney", mel_scale="slaney",
        ).T
    ).to(torch.float32)  # (n_mels, n_freq)
    analytic_window = torch.hann_window(win_length, periodic=False)
    frontend_report = {
        "fb_max_abs_diff_vs_analytic": (fb - analytic_fb).abs().max().item(),
        "window_max_abs_diff_vs_hann": (window - analytic_window).abs().max().item(),
    }

    # ---- encoder + projection
    with torch.no_grad():
        out = encoder.encoder(input_features=features, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        projected = projection(hidden)
    out_mask = out.attention_mask
    return (
        features.numpy(),
        attention_mask.numpy(),
        hidden.numpy(),
        projected.numpy(),
        out_mask.numpy(),
        frontend_report,
    )


# ---------------------------------------------------------------------------
# MLX side
# ---------------------------------------------------------------------------
def mlx_port(wav, sound_cfg_dict, llm_hidden_size):
    import mlx.core as mx
    from audio import AudioConfig, load_audio_tower

    config = AudioConfig.from_sound_config(sound_cfg_dict, llm_hidden_size=llm_hidden_size)
    model = load_audio_tower(SHARD, config)

    waveform = mx.array(wav)[None, :]
    features, mask = model.featurize(waveform)
    hidden, out_mask = model.encoder(features, mask)
    projected = model.projection(hidden)
    mx.eval(features, mask, hidden, projected, out_mask)
    return (
        np.array(features),
        np.array(mask),
        np.array(hidden),
        np.array(projected),
        np.array(out_mask),
    )


def cos_sim_per_frame(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-12)


def main():
    wav = make_fixture()
    cfg = json.load(open(os.path.join(ROOT, "reference", "config.json")))
    sound_cfg = cfg["sound_config"]
    llm_hidden = cfg["llm_config"]["hidden_size"]

    print("== PyTorch reference (CPU fp32) ==")
    t_feat, t_mask, t_hidden, t_proj, t_out_mask, frontend_report = torch_reference(
        wav, sound_cfg, llm_hidden
    )
    print("features:", t_feat.shape, "encoder out:", t_hidden.shape, "projected:", t_proj.shape)
    print("frontend buffer check:", frontend_report)

    print("== MLX port ==")
    m_feat, m_mask, m_hidden, m_proj, m_out_mask = mlx_port(wav, sound_cfg, llm_hidden)
    print("features:", m_feat.shape, "encoder out:", m_hidden.shape, "projected:", m_proj.shape)

    # --- sequence lengths / masks
    assert t_feat.shape == m_feat.shape, (t_feat.shape, m_feat.shape)
    assert t_proj.shape == m_proj.shape, (t_proj.shape, m_proj.shape)
    assert (t_mask.astype(bool) == m_mask.astype(bool)).all(), "mel attention masks differ"
    assert (t_out_mask.astype(bool) == m_out_mask.astype(bool)).all(), "output masks differ"
    n_valid = int(t_out_mask.astype(bool)[0].sum())
    print(f"sequence length: {t_proj.shape[1]} frames, {n_valid} valid")

    # --- frontend parity (same checkpoint filters both sides)
    feat_diff = np.abs(t_feat - m_feat).max()
    print(f"frontend max |diff|: {feat_diff:.3e}")

    # --- encoder parity (valid frames)
    v = slice(0, n_valid)
    enc_cos = cos_sim_per_frame(t_hidden[0, v], m_hidden[0, v])
    print(f"encoder cos-sim: min {enc_cos.min():.8f} mean {enc_cos.mean():.8f}")

    # --- final projected parity
    proj_cos = cos_sim_per_frame(t_proj[0, v], m_proj[0, v])
    proj_diff = np.abs(t_proj[0, v] - m_proj[0, v]).max()
    denom = np.abs(t_proj[0, v]).max()
    print(f"projected cos-sim: min {proj_cos.min():.8f} mean {proj_cos.mean():.8f}")
    print(f"projected max |diff|: {proj_diff:.3e} (ref max |val| {denom:.3f})")

    assert feat_diff < 1e-3, f"frontend mismatch: {feat_diff}"
    assert proj_cos.min() > 0.999, f"cos-sim too low: {proj_cos.min()}"
    print("PARITY PASS")
    return {
        "frontend_max_diff": float(feat_diff),
        "encoder_cos_min": float(enc_cos.min()),
        "proj_cos_min": float(proj_cos.min()),
        "proj_cos_mean": float(proj_cos.mean()),
        "frontend_report": frontend_report,
    }


def batched_parity():
    """Padded MLX batch (5 s + 3 s) vs per-item unpadded torch runs.

    Exercises the attention / conv / subsampling mask paths. Note the torch
    reference itself cannot run the padded batch: HF masks with -inf, so
    fully-padded query rows go NaN after softmax and poison later layers
    through k/v. The MLX port masks with -1e9 (identical for valid rows) and
    stays finite, so its batched valid frames must match unpadded references.
    """
    import mlx.core as mx
    from audio import AudioConfig, load_audio_tower

    wav = make_fixture()
    cfg = json.load(open(os.path.join(ROOT, "reference", "config.json")))
    sound_cfg, llm_hidden = cfg["sound_config"], cfg["llm_config"]["hidden_size"]

    short_len = SR * 3
    items = [wav, wav[:short_len]]
    batch = np.stack([wav, np.pad(wav[:short_len], (0, len(wav) - short_len))])
    lengths = np.array([len(wav), short_len], dtype=np.int32)

    # --- MLX: one padded batch
    config = AudioConfig.from_sound_config(sound_cfg, llm_hidden_size=llm_hidden)
    model = load_audio_tower(SHARD, config)
    m_proj, m_out_mask = model(mx.array(batch), mx.array(lengths))
    mx.eval(m_proj, m_out_mask)
    m_proj, m_out_mask = np.array(m_proj), np.array(m_out_mask)
    assert not np.isnan(m_proj[m_out_mask.astype(bool)]).any(), "NaN in valid MLX frames"

    # --- torch: each item separately, unpadded (the reference NaNs on padding)
    worst = 1.0
    for b, item in enumerate(items):
        _, _, _, t_proj, t_out_mask, _ = torch_reference(item, sound_cfg, llm_hidden)
        n_valid = int(t_out_mask.astype(bool)[0].sum())
        assert n_valid == int(m_out_mask.astype(bool)[b].sum()), "valid frame counts differ"
        cs = cos_sim_per_frame(t_proj[0, :n_valid], m_proj[b, :n_valid])
        assert not np.isnan(cs).any(), f"NaN cos-sim in batched item {b}"
        worst = min(worst, float(cs.min()))
        print(f"batched item {b}: {n_valid} valid frames, cos-sim min {cs.min():.8f}")
    assert worst > 0.999, f"batched cos-sim too low: {worst}"
    print("BATCHED PARITY PASS")
    return worst


def test_audio_parity():
    main()


def test_audio_parity_batched():
    batched_parity()


if __name__ == "__main__":
    main()
    print()
    batched_parity()
