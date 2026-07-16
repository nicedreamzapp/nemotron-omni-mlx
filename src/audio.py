# Copyright (c) 2026. MLX port of the Nemotron-3-Nano-Omni audio tower.
#
# Pure-MLX (no torch) implementation of:
#   * the Parakeet log-mel frontend (NeMo FilterbankFeatures, whose learned
#     `fb` / `window` buffers ship in the checkpoint under
#     `sound_encoder.encoder.feature_extractor.featurizer.*`)
#   * transformers' `ParakeetEncoder` (Fast Conformer: Conv2D subsampling,
#     Transformer-XL relative-position attention, conv module w/ BatchNorm)
#   * NVIDIA's `SoundProjection` MLP (RMSNorm -> linear -> ReLU^2 -> linear)
#
# Reference implementations:
#   transformers/models/parakeet/modeling_parakeet.py (v5.5.0)
#   transformers/models/parakeet/feature_extraction_parakeet.py (v5.5.0)
#   reference/audio_model.py (NVIDIA)
#
# Everything runs in float32 and is inference-only (dropout / layerdrop are
# omitted; BatchNorm always uses running statistics).

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

EPSILON = 1e-5  # frontend normalization epsilon
LOG_ZERO_GUARD_VALUE = 2.0 ** -24


@dataclass
class AudioConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    intermediate_size: int = 4096
    conv_kernel_size: int = 9
    convolution_bias: bool = False
    attention_bias: bool = False
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    num_mel_bins: int = 128
    scale_input: bool = False
    max_position_embeddings: int = 5000
    # frontend (ParakeetFeatureExtractor defaults)
    sampling_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 512
    win_length: int = 400
    preemphasis: float = 0.97
    # projection
    projection_hidden_size: int = 4096
    projection_bias: bool = False
    llm_hidden_size: int = 2688

    @classmethod
    def from_sound_config(cls, sound_config: dict, llm_hidden_size: int = 2688):
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in sound_config.items() if k in known}
        kwargs["llm_hidden_size"] = llm_hidden_size
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Frontend: preemphasis -> STFT -> mel filterbank -> log -> per-utt norm
# Mirrors ParakeetFeatureExtractor.__call__ / _torch_extract_fbank_features.
# ---------------------------------------------------------------------------
class ParakeetFeaturizer(nn.Module):
    """Log-mel frontend. `fb` (n_freq, n_mels) and `window` (win_length,)
    are loaded from the checkpoint (NeMo stores fb as (1, n_freq, n_mels))."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        n_freq = config.n_fft // 2 + 1
        # placeholders; overwritten by checkpoint weights
        self.fb = mx.zeros((n_freq, config.num_mel_bins))
        self.window = mx.zeros((config.win_length,))

    def __call__(self, waveform: mx.array, audio_lengths: mx.array | None = None):
        """waveform: (B, L) float32. Returns (features (B, T, n_mels),
        attention_mask (B, T) bool).

        Runs on the CPU stream: Metal's fp32 matmul/FFT use fast reductions
        (~7e-4 relative error) which is too lossy for the log-mel frontend;
        the CPU path matches PyTorch to ~1e-6 and this stage is tiny compute.
        """
        with mx.stream(mx.cpu):
            return self._compute(waveform, audio_lengths)

    def _compute(self, waveform: mx.array, audio_lengths: mx.array | None = None):
        cfg = self.config
        batch_size, num_samples = waveform.shape
        if audio_lengths is None:
            audio_lengths = mx.full((batch_size,), num_samples, dtype=mx.int32)

        # zero out padding, then preemphasis y[t] = x[t] - c * x[t-1]
        timemask = mx.arange(num_samples)[None, :] < audio_lengths[:, None]
        x = mx.where(timemask, waveform, 0.0)
        if cfg.preemphasis is not None and cfg.preemphasis != 0.0:
            x = mx.concatenate([x[:, :1], x[:, 1:] - cfg.preemphasis * x[:, :-1]], axis=1)
            x = mx.where(timemask, x, 0.0)

        # STFT, matching torch.stft(center=True, pad_mode="constant")
        pad = cfg.n_fft // 2
        x = mx.pad(x, [(0, 0), (pad, pad)])
        num_frames = 1 + (x.shape[1] - cfg.n_fft) // cfg.hop_length
        idx = mx.arange(num_frames)[:, None] * cfg.hop_length + mx.arange(cfg.n_fft)[None, :]
        frames = x[:, idx]  # (B, T, n_fft)

        # window: hann(win_length, periodic=False) centered inside n_fft
        left = (cfg.n_fft - cfg.win_length) // 2
        window = mx.pad(self.window, [(left, cfg.n_fft - cfg.win_length - left)])
        spec = mx.fft.rfft(frames * window[None, None, :], axis=-1)
        power = spec.real ** 2 + spec.imag ** 2  # (B, T, n_freq)

        mel = power @ self.fb  # (B, T, n_mels)
        mel = mx.log(mel + LOG_ZERO_GUARD_VALUE)

        # valid frame count (same formula as the HF feature extractor)
        feat_lengths = (audio_lengths + 2 * pad - cfg.n_fft) // cfg.hop_length
        attention_mask = mx.arange(num_frames)[None, :] < feat_lengths[:, None]

        # per-utterance, per-mel-bin normalization over valid frames
        maskf = attention_mask[..., None].astype(mel.dtype)
        lengths_f = feat_lengths[:, None].astype(mel.dtype)
        masked = mel * maskf
        mean = masked.sum(axis=1) / lengths_f  # (B, n_mels)
        var = (((masked - mean[:, None, :]) ** 2) * maskf).sum(axis=1) / (lengths_f - 1)
        std = mx.sqrt(var)
        mel = (mel - mean[:, None, :]) / (std[:, None, :] + EPSILON)
        mel = mel * maskf
        return mel, attention_mask


# ---------------------------------------------------------------------------
# Encoder building blocks
# ---------------------------------------------------------------------------
class BatchNormEval(nn.Module):
    """Inference-only BatchNorm over the last axis using running statistics."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def __call__(self, x):
        return (x - self.running_mean) * mx.rsqrt(self.running_var + self.eps) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)

    def __call__(self, x):
        return self.linear2(nn.silu(self.linear1(x)))


class ConvolutionModule(nn.Module):
    """Conformer convolution module (GLU pointwise, depthwise, BN, SiLU, pointwise)."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, bias=config.convolution_bias)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=config.convolution_bias
        )
        self.norm = BatchNormEval(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, bias=config.convolution_bias)

    def __call__(self, x, pad_mask=None):
        # x: (B, T, C); pad_mask: (B, T) bool — True where valid
        x = self.pointwise_conv1(x)
        a, b = mx.split(x, 2, axis=-1)
        x = a * mx.sigmoid(b)  # GLU over channel dim
        if pad_mask is not None:
            # masked_fill semantics (NaN-safe, unlike multiplication)
            x = mx.where(pad_mask[..., None], x, 0.0)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = nn.silu(x)
        x = self.pointwise_conv2(x)
        return x


class RelPositionAttention(nn.Module):
    """Multi-head attention with Transformer-XL relative positional encoding.
    See ParakeetEncoderAttention (transformers v5.5.0)."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        bias = config.attention_bias
        hs = config.hidden_size
        self.q_proj = nn.Linear(hs, hs, bias=bias)
        self.k_proj = nn.Linear(hs, hs, bias=bias)
        self.v_proj = nn.Linear(hs, hs, bias=bias)
        self.o_proj = nn.Linear(hs, hs, bias=bias)
        self.relative_k_proj = nn.Linear(hs, hs, bias=False)
        self.bias_u = mx.zeros((self.num_heads, self.head_dim))
        self.bias_v = mx.zeros((self.num_heads, self.head_dim))

    @staticmethod
    def _rel_shift(scores):
        # scores: (B, H, T, P) with P = 2T-1
        b, h, q, p = scores.shape
        scores = mx.pad(scores, [(0, 0), (0, 0), (0, 0), (1, 0)])
        scores = scores.reshape(b, h, p + 1, q)[:, :, 1:]
        return scores.reshape(b, h, q, p)

    def __call__(self, x, pos_emb, attention_mask=None):
        # x: (B, T, C); pos_emb: (B, 2T-1, C); attention_mask: (B, 1, T, T) bool
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q_u = q + self.bias_u[None, :, None, :]
        q_v = q + self.bias_v[None, :, None, :]

        rel_k = self.relative_k_proj(pos_emb).reshape(b, -1, self.num_heads, self.head_dim)
        # terms (b) and (d)
        matrix_bd = q_v @ rel_k.transpose(0, 2, 3, 1)  # (B, H, T, 2T-1)
        matrix_bd = self._rel_shift(matrix_bd)[..., :t] * self.scaling
        if attention_mask is not None:
            # HF uses -inf here, which turns fully-padded query rows into NaN
            # after softmax and poisons subsequent layers through k/v (HF's own
            # batched inference does this). A large finite negative is exactly
            # equivalent for valid rows (exp underflows to 0) but keeps padded
            # rows finite so they never leak NaN into valid frames.
            matrix_bd = mx.where(attention_mask, matrix_bd, -1e9)

        scores = (q_u @ k.transpose(0, 1, 3, 2)) * self.scaling + matrix_bd
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = weights @ v  # (B, H, T, hd)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self.o_proj(out)


class ConformerBlock(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.feed_forward1 = FeedForward(config)
        self.self_attn = RelPositionAttention(config)
        self.conv = ConvolutionModule(config)
        self.feed_forward2 = FeedForward(config)
        hs = config.hidden_size
        self.norm_feed_forward1 = nn.LayerNorm(hs)
        self.norm_self_att = nn.LayerNorm(hs)
        self.norm_conv = nn.LayerNorm(hs)
        self.norm_feed_forward2 = nn.LayerNorm(hs)
        self.norm_out = nn.LayerNorm(hs)

    def __call__(self, x, pos_emb, attention_mask=None, pad_mask=None):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, attention_mask)
        x = x + self.conv(self.norm_conv(x), pad_mask)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class SubsamplingConv2D(nn.Module):
    """8x time subsampling: full conv + 2x (depthwise + pointwise), stride 2 each."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        k = config.subsampling_conv_kernel_size
        s = config.subsampling_conv_stride
        c = config.subsampling_conv_channels
        p = (k - 1) // 2
        self.kernel_size, self.stride, self.padding = k, s, p
        num_stages = int(math.log2(config.subsampling_factor))
        layers = [nn.Conv2d(1, c, kernel_size=k, stride=s, padding=p), nn.ReLU()]
        for _ in range(num_stages - 1):
            layers.append(nn.Conv2d(c, c, kernel_size=k, stride=s, padding=p, groups=c))
            layers.append(nn.Conv2d(c, c, kernel_size=1))
            layers.append(nn.ReLU())
        self.layers = layers
        out_length = config.num_mel_bins // (s ** num_stages)
        self.linear = nn.Linear(c * out_length, config.hidden_size, bias=True)

    def __call__(self, input_features, attention_mask=None):
        # input_features: (B, T, n_mels) -> NHWC (B, T, n_mels, 1)
        x = input_features[..., None]
        lengths = attention_mask.sum(-1) if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and attention_mask is not None:
                if layer.weight.shape[1] > 1:  # strided conv (kernel > 1 here implies stride 2)
                    lengths = (lengths + 2 * self.padding - self.kernel_size) // self.stride + 1
                mask = mx.arange(x.shape[1])[None, :] < lengths[:, None]
                x = x * mask[:, :, None, None].astype(x.dtype)
        # (B, T', F', C) -> (B, T', C, F') -> (B, T', C*F') to match torch layout
        b, t, f, c = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(b, t, c * f)
        return self.linear(x)


class RelPositionalEncoding(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        # underscore prefix: computed buffer, not a checkpoint parameter
        self._inv_freq = 1.0 / (
            10000.0 ** (mx.arange(0, config.hidden_size, 2).astype(mx.float32) / config.hidden_size)
        )

    def __call__(self, t: int, batch_size: int, dtype=mx.float32):
        if t > self.max_position_embeddings:
            raise ValueError(f"sequence length {t} > max_position_embeddings")
        positions = mx.arange(t - 1, -t, -1).astype(mx.float32)
        freqs = positions[:, None] * self._inv_freq[None, :]  # (2T-1, C/2)
        # interleave sin and cos: [sin0, cos0, sin1, cos1, ...]
        pos = mx.stack([mx.sin(freqs), mx.cos(freqs)], axis=-1).reshape(freqs.shape[0], -1)
        pos = mx.broadcast_to(pos[None], (batch_size, *pos.shape))
        return pos.astype(dtype)


class ParakeetEncoderMLX(nn.Module):
    """MLX port of transformers.ParakeetEncoder + the checkpoint featurizer."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = FeaturizerContainer(config)
        self.subsampling = SubsamplingConv2D(config)
        self.encode_positions = RelPositionalEncoding(config)
        self.layers = [ConformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0

    def _subsampled_lengths(self, lengths):
        k = self.config.subsampling_conv_kernel_size
        s = self.config.subsampling_conv_stride
        add_pad = (k - 1) // 2 * 2 - k
        out = lengths.astype(mx.float32)
        for _ in range(int(math.log2(self.config.subsampling_factor))):
            out = mx.floor((out + add_pad) / s + 1.0)
        return out.astype(mx.int32)

    def __call__(self, input_features, attention_mask=None):
        # input_features: (B, T, n_mels); attention_mask: (B, T) bool
        x = self.subsampling(input_features, attention_mask) * self.input_scale
        b, t, _ = x.shape
        pos_emb = self.encode_positions(t, b, x.dtype)

        mask_4d = pad_mask = out_mask = None
        if attention_mask is not None:
            out_lengths = self._subsampled_lengths(attention_mask.sum(-1))
            out_mask = mx.arange(t)[None, :] < out_lengths[:, None]  # (B, T')
            m = out_mask[:, None, :] & out_mask[:, :, None]  # (B, T', T')
            mask_4d = m[:, None]  # (B, 1, T', T')
            # conv-module padding mask: positions where any query attends
            pad_mask = mx.any(m, axis=1)  # (B, T') == out_mask for this mask shape

        for layer in self.layers:
            x = layer(x, pos_emb, mask_4d, pad_mask)
        return x, out_mask


class FeaturizerContainer(nn.Module):
    """Matches checkpoint path `...feature_extractor.featurizer.{fb,window}`."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.featurizer = ParakeetFeaturizer(config)

    def __call__(self, waveform, audio_lengths=None):
        return self.featurizer(waveform, audio_lengths)


class RMSNormFP32(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x):
        dtype = x.dtype
        x = x.astype(mx.float32)
        var = mx.mean(x * x, axis=-1, keepdims=True)
        return (self.weight.astype(mx.float32) * x * mx.rsqrt(var + self.eps)).astype(dtype)


class SoundProjectionMLX(nn.Module):
    """RMSNorm -> linear1 -> ReLU^2 -> linear2 (NVIDIA SoundProjection)."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.norm = RMSNormFP32(config.hidden_size)
        self.linear1 = nn.Linear(config.hidden_size, config.projection_hidden_size, bias=config.projection_bias)
        self.linear2 = nn.Linear(config.projection_hidden_size, config.llm_hidden_size, bias=config.projection_bias)

    def __call__(self, x):
        x = self.linear1(self.norm(x))
        x = mx.maximum(x, 0.0) ** 2
        return self.linear2(x)


class AudioTower(nn.Module):
    """Full audio path: waveform -> log-mel -> Conformer -> projection."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.encoder = ParakeetEncoderMLX(config)
        self.projection = SoundProjectionMLX(config)

    def featurize(self, waveform, audio_lengths=None):
        return self.encoder.feature_extractor(waveform, audio_lengths)

    def encode_features(self, input_features, attention_mask=None):
        hidden, out_mask = self.encoder(input_features, attention_mask)
        return self.projection(hidden), out_mask

    def __call__(self, waveform, audio_lengths=None):
        """waveform: (B, L) float32 @ 16 kHz. Returns (embeddings
        (B, T', llm_hidden), out_mask (B, T'))."""
        features, mask = self.featurize(waveform, audio_lengths)
        return self.encode_features(features, mask)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def _map_key(key: str):
    """Map a checkpoint key to the AudioTower parameter path (or None to skip)."""
    if key.startswith("sound_encoder.encoder."):
        sub = key[len("sound_encoder.encoder."):]
        if sub.endswith("num_batches_tracked"):
            return None
        return "encoder." + sub
    if key.startswith("sound_projection."):
        return "projection." + key[len("sound_projection."):]
    return None


def _convert(path: str, w: mx.array) -> mx.array:
    w = w.astype(mx.float32)
    if path.startswith("encoder.subsampling.layers.") and path.endswith(".weight") and w.ndim == 4:
        return w.transpose(0, 2, 3, 1)  # torch OIHW -> mlx OHWI
    if ".conv." in path and path.endswith(".weight") and w.ndim == 3:
        return w.transpose(0, 2, 1)  # torch OIK -> mlx OKI
    if path.endswith("featurizer.fb"):
        w = w.squeeze()  # NeMo stores (1, n_freq, n_mels)
        if w.shape[0] < w.shape[1]:  # (n_mels, n_freq) -> (n_freq, n_mels)
            w = w.transpose(1, 0)
        return w
    return w


def load_audio_tower(shard_paths, config: AudioConfig | None = None) -> AudioTower:
    """Build an AudioTower and load `sound_encoder.*` / `sound_projection.*`
    weights from one or more safetensors shards."""
    if config is None:
        config = AudioConfig()
    model = AudioTower(config)
    if isinstance(shard_paths, str):
        shard_paths = [shard_paths]
    weights = []
    for shard in shard_paths:
        for key, value in mx.load(shard).items():
            path = _map_key(key)
            if path is not None:
                weights.append((path, _convert(path, value)))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model
