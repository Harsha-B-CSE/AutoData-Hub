"""Reference and non-reference telemetry for enhanced speech audio."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.signal import istft, stft

from modules.voice.audio_loader import downmix_to_mono, resample_audio
from modules.voice.booster import calculate_true_peak_dbfs, estimate_integrated_lufs


_EPSILON = 1e-10


def _aligned_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = downmix_to_mono(first).astype(np.float64, copy=False)
    right = downmix_to_mono(second).astype(np.float64, copy=False)
    length = min(left.size, right.size)
    if length == 0:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    return left[:length], right[:length]


def compute_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute energy SNR in decibels with epsilon-stabilized powers."""

    clean, residual = _aligned_pair(signal, noise)
    signal_power = float(np.mean(np.square(clean, dtype=np.float64)))
    noise_power = float(np.mean(np.square(residual, dtype=np.float64)))
    ratio = max(signal_power, _EPSILON) / max(noise_power, _EPSILON)
    return float(10.0 * np.log10(ratio))


def compute_sisdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """Compute scale-invariant SDR after mandatory zero-mean centering."""

    target, estimate = _aligned_pair(reference, estimation)
    target = target - float(np.mean(target))
    estimate = estimate - float(np.mean(estimate))
    target_energy = max(float(np.dot(target, target)), _EPSILON)
    scale = float(np.dot(estimate, target)) / target_energy
    projected = scale * target
    residual = estimate - projected
    projected_energy = max(float(np.dot(projected, projected)), _EPSILON)
    residual_energy = max(float(np.dot(residual, residual)), _EPSILON)
    return float(
        10.0
        * np.log10(
            projected_energy / residual_energy
        )
    )


def _estimate_clean_reference(audio_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a conservative stationary-noise reference for blind metrics."""

    audio = downmix_to_mono(audio_signal)
    if audio.size < 32:
        return audio.copy(), np.zeros_like(audio)
    nperseg = min(512, max(32, 2 ** int(np.floor(np.log2(audio.size)))))
    noverlap = min(nperseg // 2, nperseg - 1)
    _, _, spectrum = stft(
        audio,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    power = np.square(np.abs(spectrum), dtype=np.float64)
    noise_power = np.quantile(power, 0.20, axis=1, keepdims=True)
    speech_probability = np.maximum(power - noise_power, 0.0) / (
        power + _EPSILON
    )
    gain = 0.08 + 0.92 * np.sqrt(speech_probability)
    _, clean = istft(
        spectrum * gain,
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    clean = np.asarray(clean[: audio.size], dtype=np.float32)
    if clean.size < audio.size:
        clean = np.pad(clean, (0, audio.size - clean.size))
    noise = audio - clean
    return clean, noise


def _native_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
) -> float:
    from pesq import pesq  # type: ignore

    target_rate = 8_000 if int(sample_rate) == 8_000 else 16_000
    mode = "nb" if target_rate == 8_000 else "wb"
    if int(sample_rate) != target_rate:
        reference = resample_audio(reference, int(sample_rate), target_rate)
        degraded = resample_audio(degraded, int(sample_rate), target_rate)
    reference, degraded = _aligned_pair(reference, degraded)
    if not np.any(reference) or not np.any(degraded):
        raise ValueError("PESQ is undefined for silent audio.")
    return float(pesq(target_rate, reference, degraded, mode))


def _native_stoi(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
) -> float:
    from pystoi import stoi  # type: ignore

    reference, degraded = _aligned_pair(reference, degraded)
    if reference.size < int(0.5 * sample_rate):
        raise ValueError("STOI requires at least 0.5 seconds of audio.")
    if not np.any(reference) or not np.any(degraded):
        raise ValueError("STOI is undefined for silent audio.")
    return float(stoi(reference, degraded, int(sample_rate), extended=False))


def _stoi_surrogate(reference: np.ndarray, degraded: np.ndarray) -> float:
    reference, degraded = _aligned_pair(reference, degraded)
    reference = reference - float(np.mean(reference))
    degraded = degraded - float(np.mean(degraded))
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(degraded))
    if denominator <= _EPSILON:
        return 1.0 if np.allclose(reference, degraded) else 0.0
    correlation = float(np.dot(reference, degraded) / denominator)
    return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))


def evaluate_audio_telemetry(
    audio_signal: np.ndarray,
    sample_rate: int = 16_000,
) -> Dict[str, Any]:
    """Compute blind speech-quality telemetry for a mono waveform.

    PESQ and STOI are intrusive metrics.  With no clean reference in this API,
    a conservative stationary-noise estimate is used as the reference.  The
    returned method fields make that limitation explicit and distinguish an
    installed native PESQ implementation from the bounded heuristic fallback.
    """

    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    audio = downmix_to_mono(audio_signal)
    centered_audio = (
        audio - float(np.mean(audio, dtype=np.float64))
        if audio.size
        else audio.copy()
    ).astype(np.float32, copy=False)
    clean_estimate, noise_estimate = _estimate_clean_reference(centered_audio)
    snr_db = compute_snr_db(clean_estimate, noise_estimate)
    si_sdr_db = compute_sisdr(clean_estimate, centered_audio)
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0

    try:
        pesq_score = _native_pesq(clean_estimate, centered_audio, sample_rate)
        pesq_method = "pesq"
    except Exception:
        quality_curve = 1.0 + 3.5 / (1.0 + np.exp(-(snr_db - 5.0) / 7.0))
        pesq_score = float(np.clip(quality_curve - 1.5 * clipping_ratio, 1.0, 4.5))
        pesq_method = "pesq_heuristic_surrogate"

    try:
        stoi_index = _native_stoi(clean_estimate, centered_audio, sample_rate)
        stoi_method = "pystoi"
    except Exception:
        stoi_index = _stoi_surrogate(clean_estimate, centered_audio)
        stoi_method = "stoi_correlation_surrogate"

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    peak_amplitude = float(np.max(np.abs(audio))) if audio.size else 0.0
    crest_factor_db = (
        float(20.0 * np.log10((peak_amplitude + 1e-12) / (rms + 1e-12)))
        if peak_amplitude > 0
        else 0.0
    )
    return {
        "snr_db": round(float(snr_db), 4),
        "si_sdr_db": round(float(si_sdr_db), 4),
        "pesq_score": round(float(pesq_score), 4),
        "pesq_method": pesq_method,
        "pesq_metric": pesq_method,
        "stoi_index": round(float(np.clip(stoi_index, 0.0, 1.0)), 4),
        "stoi_method": stoi_method,
        "clipping_ratio": clipping_ratio,
        "clipping_ratio_pct": round(clipping_ratio * 100.0, 6),
        "peak_amplitude": peak_amplitude,
        "true_peak_dbfs": calculate_true_peak_dbfs(audio, sample_rate),
        "rms_dbfs": round(rms_dbfs, 4),
        "lufs_est": round(estimate_integrated_lufs(audio, sample_rate), 4),
        "crest_factor_db": round(crest_factor_db, 4),
        "duration_seconds": float(audio.size / sample_rate),
        "sample_rate": sample_rate,
        "sample_count": int(audio.size),
        "reference_type": "stationary_noise_estimate",
    }
