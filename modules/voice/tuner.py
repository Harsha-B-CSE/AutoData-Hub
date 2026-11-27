"""Spectral diagnostics, acoustic suggestions, and parametric voice EQ."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import sosfilt, welch

from modules.voice.audio_loader import downmix_to_mono
from modules.voice.booster import estimate_integrated_lufs


def _peaking_sos(
    sample_rate: int,
    frequency: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    frequency = min(float(frequency), sample_rate * 0.45)
    amplitude = 10.0 ** (float(gain_db) / 40.0)
    omega = 2.0 * np.pi * frequency / sample_rate
    alpha = np.sin(omega) / (2.0 * q)
    cosine = np.cos(omega)
    b0 = 1.0 + alpha * amplitude
    b1 = -2.0 * cosine
    b2 = 1.0 - alpha * amplitude
    a0 = 1.0 + alpha / amplitude
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha / amplitude
    return np.asarray([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _shelf_sos(
    sample_rate: int,
    frequency: float,
    gain_db: float,
    *,
    high: bool,
) -> np.ndarray:
    frequency = min(float(frequency), sample_rate * 0.45)
    amplitude = 10.0 ** (float(gain_db) / 40.0)
    omega = 2.0 * np.pi * frequency / sample_rate
    cosine = np.cos(omega)
    alpha = np.sin(omega) / np.sqrt(2.0)
    root_amplitude = np.sqrt(amplitude)
    if high:
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
        a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine)
        a2 = (
            (amplitude + 1.0)
            - (amplitude - 1.0) * cosine
            - 2.0 * root_amplitude * alpha
        )
    else:
        b0 = amplitude * (
            (amplitude + 1.0)
            - (amplitude - 1.0) * cosine
            + 2.0 * root_amplitude * alpha
        )
        b1 = 2.0 * amplitude * (
            (amplitude - 1.0) - (amplitude + 1.0) * cosine
        )
        b2 = amplitude * (
            (amplitude + 1.0)
            - (amplitude - 1.0) * cosine
            - 2.0 * root_amplitude * alpha
        )
        a0 = (
            (amplitude + 1.0)
            + (amplitude - 1.0) * cosine
            + 2.0 * root_amplitude * alpha
        )
        a1 = -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cosine)
        a2 = (
            (amplitude + 1.0)
            + (amplitude - 1.0) * cosine
            - 2.0 * root_amplitude * alpha
        )
    return np.asarray([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _band_ratio(
    frequencies: np.ndarray,
    power_density: np.ndarray,
    low_hz: float,
    high_hz: float,
    total_power: float,
) -> float:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if np.count_nonzero(mask) < 2 or total_power <= 0:
        return 0.0
    band_power = float(trapezoid(power_density[mask], frequencies[mask]))
    return float(np.clip(band_power / total_power, 0.0, 1.0))


def analyze_audio_spectrum(
    audio_signal: np.ndarray,
    sr: int = 16_000,
) -> Dict[str, Any]:
    """Return loudness, crest factor, spectral ratios, and tuning advice."""

    sr = int(sr)
    if sr <= 0:
        raise ValueError("sr must be positive.")
    audio = downmix_to_mono(audio_signal)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    crest_factor_db = (
        float(20.0 * np.log10((peak + 1e-12) / (rms + 1e-12)))
        if peak > 0
        else 0.0
    )
    lufs_est = estimate_integrated_lufs(audio, sr)
    if audio.size < 8:
        frequencies = np.asarray([0.0, sr / 2.0])
        power_density = np.zeros(2)
    else:
        nperseg = min(4_096, max(8, 2 ** int(np.floor(np.log2(audio.size)))))
        frequencies, power_density = welch(
            audio,
            fs=sr,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
        )
    total_mask = (frequencies >= 20.0) & (frequencies <= sr / 2.0)
    total_power = (
        float(trapezoid(power_density[total_mask], frequencies[total_mask]))
        if np.count_nonzero(total_mask) >= 2
        else 0.0
    )
    band_energies = {
        "sub_bass": _band_ratio(frequencies, power_density, 20.0, 150.0, total_power),
        "mids": _band_ratio(frequencies, power_density, 250.0, 2_500.0, total_power),
        "highs": _band_ratio(
            frequencies,
            power_density,
            3_000.0,
            min(16_000.0, sr / 2.0),
            total_power,
        ),
    }

    suggestions: List[str] = []
    if band_energies["mids"] > 0.55:
        suggestions.append("Mids heavy: dip 350Hz")
    if band_energies["sub_bass"] > 0.28:
        suggestions.append("Sub-bass heavy: reduce 80Hz")
    if band_energies["highs"] > 0.30:
        suggestions.append("Highs bright: soften 6kHz air")
    elif sr >= 16_000 and band_energies["highs"] < 0.025 and peak > 0:
        suggestions.append("Highs recessed: add gentle 6kHz air")
    if crest_factor_db < 6.0 and peak > 0:
        suggestions.append("Low Crest Factor: increase transient attack")
    elif crest_factor_db > 18.0:
        suggestions.append("High Crest Factor: apply gentle peak control")
    if lufs_est < -20.0 and peak > 0:
        suggestions.append("Low integrated loudness: enable Quality Booster")
    if not suggestions:
        suggestions.append("Spectrum balanced: use subtle EQ adjustments")

    return {
        "lufs_est": round(float(lufs_est), 4),
        "crest_factor_db": round(float(crest_factor_db), 4),
        "band_energies": band_energies,
        "suggestions": suggestions,
    }


def apply_parametric_eq(
    audio_signal: np.ndarray,
    sr: int = 16_000,
    low_gain_db: float = 0.0,
    mid_gain_db: float = 0.0,
    high_gain_db: float = 0.0,
) -> np.ndarray:
    """Apply a five-section EQ controlled by low, mid, and high macro gains."""

    sr = int(sr)
    if sr <= 0:
        raise ValueError("sr must be positive.")
    audio = downmix_to_mono(audio_signal)
    if audio.size == 0:
        return audio.copy()
    low_gain_db = float(np.clip(low_gain_db, -12.0, 12.0))
    mid_gain_db = float(np.clip(mid_gain_db, -12.0, 12.0))
    high_gain_db = float(np.clip(high_gain_db, -12.0, 12.0))
    sections = (
        _shelf_sos(sr, 90.0, low_gain_db * 0.65, high=False),
        _peaking_sos(sr, 180.0, low_gain_db * 0.35, q=0.8),
        _peaking_sos(sr, 500.0, mid_gain_db * 0.45, q=1.0),
        _peaking_sos(sr, 1_800.0, mid_gain_db * 0.55, q=1.0),
        _shelf_sos(sr, 6_000.0, high_gain_db, high=True),
    )
    equalized = audio.astype(np.float64, copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        for section in sections:
            equalized = sosfilt(section, equalized)
    equalized = np.nan_to_num(equalized, nan=0.0, posinf=0.0, neginf=0.0)
    return equalized.astype(np.float32)
