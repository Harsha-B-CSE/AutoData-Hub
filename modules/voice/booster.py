"""EBU-style loudness normalization and zero-clipping true-peak limiting."""

from __future__ import annotations

from math import ceil
import warnings
from typing import Any, Dict

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import firwin, resample_poly, sosfilt, upfirdn

from modules.voice.audio_loader import downmix_to_mono


_EPSILON = np.finfo(np.float64).eps


def _highpass_sos(sample_rate: int, frequency: float, q: float) -> np.ndarray:
    omega = 2.0 * np.pi * min(frequency, sample_rate * 0.45) / sample_rate
    cosine = np.cos(omega)
    alpha = np.sin(omega) / (2.0 * q)
    b0 = (1.0 + cosine) / 2.0
    b1 = -(1.0 + cosine)
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha
    return np.asarray([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _high_shelf_sos(
    sample_rate: int,
    frequency: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * min(frequency, sample_rate * 0.45) / sample_rate
    cosine = np.cos(omega)
    alpha = np.sin(omega) / (2.0 * q)
    root_amplitude = np.sqrt(amplitude)
    b0 = amplitude * (
        (amplitude + 1.0)
        + (amplitude - 1.0) * cosine
        + 2.0 * root_amplitude * alpha
    )
    b1 = -2.0 * amplitude * (
        (amplitude - 1.0) + (amplitude + 1.0) * cosine
    )
    b2 = amplitude * (
        (amplitude + 1.0)
        + (amplitude - 1.0) * cosine
        - 2.0 * root_amplitude * alpha
    )
    a0 = (
        (amplitude + 1.0)
        - (amplitude - 1.0) * cosine
        + 2.0 * root_amplitude * alpha
    )
    a1 = 2.0 * (
        (amplitude - 1.0) - (amplitude + 1.0) * cosine
    )
    a2 = (
        (amplitude + 1.0)
        - (amplitude - 1.0) * cosine
        - 2.0 * root_amplitude * alpha
    )
    return np.asarray([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _k_weight(audio_signal: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = downmix_to_mono(audio_signal).astype(np.float64, copy=False)
    if audio.size < 3:
        return audio
    # BS.1770 K-weighting: pre-filter shelf followed by the RLB high-pass.
    shelf = _high_shelf_sos(
        sample_rate,
        frequency=1_681.974450955533,
        gain_db=3.999843853973347,
        q=0.7071752369554196,
    )
    highpass = _highpass_sos(
        sample_rate,
        frequency=38.13547087602444,
        q=0.5003270373238773,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        return sosfilt(highpass, sosfilt(shelf, audio))


def estimate_integrated_lufs(
    audio_signal: np.ndarray,
    sample_rate: int = 16_000,
) -> float:
    """Estimate mono EBU R128 integrated loudness with absolute/relative gates."""

    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    weighted = _k_weight(audio_signal, sample_rate)
    if weighted.size == 0 or not np.any(weighted):
        return -120.0
    block_size = max(1, int(round(0.400 * sample_rate)))
    hop_size = max(1, int(round(0.100 * sample_rate)))
    if weighted.size < block_size:
        block_powers = np.asarray([np.mean(np.square(weighted))])
    else:
        starts = range(0, weighted.size - block_size + 1, hop_size)
        block_powers = np.asarray(
            [np.mean(np.square(weighted[start : start + block_size])) for start in starts],
            dtype=np.float64,
        )
    block_loudness = -0.691 + 10.0 * np.log10(block_powers + _EPSILON)
    absolute_mask = block_loudness > -70.0
    if not np.any(absolute_mask):
        return -120.0
    absolute_power = block_powers[absolute_mask]
    relative_gate = (
        -0.691 + 10.0 * np.log10(float(np.mean(absolute_power)) + _EPSILON) - 10.0
    )
    final_mask = absolute_mask & (block_loudness > max(-70.0, relative_gate))
    gated_power = block_powers[final_mask]
    if gated_power.size == 0:
        gated_power = absolute_power
    return float(-0.691 + 10.0 * np.log10(float(np.mean(gated_power)) + _EPSILON))


def calculate_true_peak_dbfs(
    audio_signal: np.ndarray,
    sample_rate: int = 16_000,
    *,
    oversample_factor: int = 4,
) -> float:
    """Measure inter-sample true peak through band-limited oversampling."""

    del sample_rate
    audio = downmix_to_mono(audio_signal)
    if audio.size == 0 or not np.any(audio):
        return -120.0
    factor = max(1, int(oversample_factor))
    oversampled = (
        resample_poly(audio, factor, 1, window=("kaiser", 8.6))
        if factor > 1
        else audio
    )
    peak = float(np.max(np.abs(oversampled)))
    return float(20.0 * np.log10(max(peak, 1e-12)))


def _future_peak_envelope(values: np.ndarray, lookahead_samples: int) -> np.ndarray:
    """Return max(abs(values[i:i+lookahead+1])) in linear native code."""

    size = max(1, int(lookahead_samples) + 1)
    return maximum_filter1d(
        np.abs(values),
        size=size,
        origin=-(size // 2),
        mode="nearest",
    )


def apply_true_peak_limiter(
    audio_signal: np.ndarray,
    sr: int = 16_000,
    *,
    target_peak_dbfs: float = -0.5,
    lookahead_ms: float = 1.0,
) -> np.ndarray:
    """Apply a 4x oversampled true-peak limiter with immediate attack."""

    audio = downmix_to_mono(audio_signal)
    if audio.size == 0:
        return audio.copy()
    oversample_factor = 4
    oversampled = resample_poly(
        audio, oversample_factor, 1, window=("kaiser", 8.6)
    ).astype(np.float64, copy=False)
    ceiling = 10.0 ** (float(target_peak_dbfs) / 20.0)
    lookahead = max(
        1,
        int(round(float(lookahead_ms) * sr * oversample_factor / 1_000.0)),
    )
    envelope = _future_peak_envelope(oversampled, lookahead)
    required_gain = np.minimum(1.0, ceiling / (envelope + _EPSILON))
    release_samples = max(1, int(round(0.050 * sr * oversample_factor)))
    release_size = release_samples + 1
    held_gain = minimum_filter1d(
        required_gain,
        size=release_size,
        origin=(release_size - 1) // 2,
        mode="nearest",
    )
    limited_oversampled = oversampled * np.minimum(required_gain, held_gain)
    limited = resample_poly(
        limited_oversampled,
        1,
        oversample_factor,
        window=("kaiser", 8.6),
    )
    limited = np.asarray(limited[: audio.size], dtype=np.float32)
    if limited.size < audio.size:
        limited = np.pad(limited, (0, audio.size - limited.size))

    measured_peak = 10.0 ** (
        calculate_true_peak_dbfs(limited, sr) / 20.0
    )
    if measured_peak > ceiling:
        limited *= ceiling / measured_peak
    return np.clip(limited, -ceiling, ceiling).astype(np.float32, copy=False)


class StreamingTruePeakLimiter:
    """Streaming true-peak detection with bounded lookahead and a 50 ms hold."""

    # Same Kaiser reconstruction as the meter, with a finer phase grid.
    _up = 8
    _fir = _up * firwin(20 * _up + 1, 1 / _up, window=("kaiser", 8.6))

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        target_peak_dbfs: float = -0.5,
        lookahead_ms: float = 1.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.target_peak_dbfs = float(target_peak_dbfs)
        self.lookahead_ms = float(lookahead_ms)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        # One hop must span the FIR tail: only adjacent hops can overlap.
        lookahead = max(1, round(self.sample_rate * self.lookahead_ms / 1_000))
        self._hop = lookahead * ceil((self._fir.size - 1) / (self._up * lookahead))
        self._guard_samples = max(
            2 * self._hop, round(self.sample_rate * (self.lookahead_ms + 3.0) / 1_000)
        )
        self._hold = ceil(0.050 * self.sample_rate / self._hop)
        phase_bound = max(np.abs(self._fir[k::self._up]).sum() for k in range(self._up))
        # Allow for a Nyquist peak halfway between detector phases, plus one PCM16 LSB.
        target = 10 ** (self.target_peak_dbfs / 20) * np.cos(np.pi / (2 * self._up))
        self._ceiling = max(0.0, target - phase_bound / 32767)
        self.safe_input_peak = self._ceiling / phase_bound
        self.reset()

    @property
    def latency_ms(self) -> float:
        return self._guard_samples * 1_000.0 / self.sample_rate

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._ready = np.empty(0, dtype=np.float32)
        self._left_seam = 0.0
        self._peak_history = np.zeros(self._hold, dtype=np.float64)

    def _limit_hops(self, hops: np.ndarray) -> np.ndarray:
        if np.max(np.abs(hops)) <= self.safe_input_peak:
            # The FIR phase L1 bound certifies even a cut to silence, for any gains in [0, 1].
            peaks = np.zeros(len(hops) - 1)
            peaks[0] = self._left_seam
            self._left_seam = 0.0
        else:
            interp = upfirdn(self._fir, hops, up=self._up, axis=1)
            solo = np.max(np.abs(interp), axis=1)
            tails = interp[:-1, self._up * self._hop:]
            heads = interp[1:, :tails.shape[1]]
            # With independent gains in [0, G], the worst join is at a corner:
            # G * max(|tail|, |head|, |tail + head|). Both hops share this bound.
            seams = np.maximum.reduce(
                (np.max(np.abs(tails), axis=1), np.max(np.abs(heads), axis=1),
                 np.max(np.abs(tails + heads), axis=1))
            )
            peaks = np.maximum.reduce(
                (solo[:-1], solo[1:], seams, np.r_[self._left_seam, seams[:-1]])
            )
            self._left_seam = seams[-1]

        history = np.concatenate((self._peak_history, peaks))
        held = maximum_filter1d(
            history, size=self._hold + 1, origin=self._hold // 2, mode="nearest"
        )[self._hold:]
        gains = np.minimum(1.0, self._ceiling / np.maximum(held, _EPSILON))
        self._peak_history = history[-self._hold:].copy()
        return (hops[:-1] * gains[:, None]).astype(np.float32).ravel()

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        self._buffer = np.concatenate((self._buffer, downmix_to_mono(audio_chunk)))
        emit = max(0, self._buffer.size + self._ready.size - self._guard_samples)
        n = self._buffer.size // self._hop
        if n >= 2:
            hops = self._buffer[:n * self._hop].reshape(n, self._hop)
            self._ready = np.concatenate((self._ready, self._limit_hops(hops)))
            self._buffer = self._buffer[(n - 1) * self._hop:].copy()
        output = self._ready[:emit]
        self._ready = self._ready[emit:].copy()
        return output

    def flush(self) -> np.ndarray:
        output = self.process(np.zeros(self._guard_samples, dtype=np.float32))
        self.reset()
        return output


def analyze_audio_levels(
    audio_signal: np.ndarray,
    sample_rate: int = 16_000,
) -> Dict[str, Any]:
    audio = downmix_to_mono(audio_signal)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    return {
        "lufs": estimate_integrated_lufs(audio, sample_rate),
        "rms": rms,
        "rms_dbfs": float(20.0 * np.log10(max(rms, 1e-12))),
        "true_peak_dbfs": calculate_true_peak_dbfs(audio, sample_rate),
    }


def apply_quality_booster(
    audio_signal: np.ndarray,
    sr: int = 16_000,
    boost_db: float = 3.0,
    target_peak_dbfs: float = -0.5,
    *,
    normalize_loudness: bool = True,
    apply_limiter: bool = True,
) -> np.ndarray:
    """Normalize toward -14 LUFS, apply tanh drive, and enforce true peak."""

    audio = downmix_to_mono(audio_signal)
    if audio.size == 0 or not np.any(audio):
        return audio.copy()
    current_lufs = estimate_integrated_lufs(audio, sr) if normalize_loudness else -14.0
    normalization_db = (
        float(np.clip(-14.0 - current_lufs, -24.0, 24.0))
        if normalize_loudness
        else 0.0
    )
    total_gain_db = normalization_db + float(np.clip(boost_db, 0.0, 12.0))
    if abs(total_gain_db) <= 1e-12 and not apply_limiter:
        return audio.copy()
    gain = 10.0 ** (total_gain_db / 20.0)
    saturated = np.tanh(gain * audio.astype(np.float64, copy=False))
    if not apply_limiter:
        return saturated.astype(np.float32)
    return apply_true_peak_limiter(
        saturated.astype(np.float32),
        sr,
        target_peak_dbfs=target_peak_dbfs,
        lookahead_ms=1.0,
    )
