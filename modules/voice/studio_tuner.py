"""Linked-channel mastering DSP with stable SOS filter implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple
import warnings

import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d


BROADCAST_PRESET = "🎙️ Broadcast Vocal Polish"
PODCAST_PRESET = "📻 Warm Podcast Warmth"
AIR_PRESET = "✨ Air & Crystal Clarity"
CUSTOM_PRESET = "🎛️ Custom Studio Manual Mode"
STUDIO_PRESETS = (BROADCAST_PRESET, PODCAST_PRESET, AIR_PRESET, CUSTOM_PRESET)
_EPSILON = 1e-10


def _require_finite(**parameters: float) -> None:
    for name, value in parameters.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")


def _as_samples_channels(
    audio: np.ndarray,
    channel_axis: int | None = None,
) -> tuple[np.ndarray, str]:
    array = np.nan_to_num(
        np.asarray(audio, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0
    )
    if array.ndim == 1:
        return array[:, np.newaxis], "mono"
    if array.ndim != 2:
        raise ValueError("Audio must be mono or a two-dimensional channel array.")
    if channel_axis is not None:
        normalized_axis = int(channel_axis) % 2
        if normalized_axis == 0:
            return np.ascontiguousarray(array.T), "channels_first"
        return np.ascontiguousarray(array), "samples_first"
    first_is_channels = array.shape[0] <= 32 and array.shape[1] > array.shape[0]
    second_is_channels = array.shape[1] <= 32 and array.shape[0] > array.shape[1]
    if first_is_channels and not second_is_channels:
        return np.ascontiguousarray(array.T), "channels_first"
    if second_is_channels or array.shape[1] <= 32:
        return np.ascontiguousarray(array), "samples_first"
    if array.shape[0] <= 32:
        return np.ascontiguousarray(array.T), "channels_first"
    raise ValueError("Could not infer the channel axis for studio audio.")


def _restore_layout(audio: np.ndarray, layout: str) -> np.ndarray:
    restored = np.asarray(audio, dtype=np.float32)
    if layout == "mono":
        return np.ascontiguousarray(restored[:, 0])
    if layout == "channels_first":
        return np.ascontiguousarray(restored.T)
    return np.ascontiguousarray(restored)


def _stable_sos_filter(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        return signal.sosfilt(np.asarray(sos, dtype=np.float64), audio, axis=0)


def _bounded_frequency(frequency: float, sample_rate: int) -> float:
    if not np.isfinite(frequency):
        raise ValueError("Filter frequencies must be finite.")
    return float(np.clip(frequency, 10.0, sample_rate * 0.45))


def _peaking_sos(
    sample_rate: int,
    frequency: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    frequency = _bounded_frequency(frequency, sample_rate)
    q = float(np.clip(q, 0.1, 20.0))
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
    frequency = _bounded_frequency(frequency, sample_rate)
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


def apply_highpass_filter(
    audio: np.ndarray,
    sr: int = 16_000,
    cutoff: float = 80.0,
    *,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Apply a fourth-order Butterworth high-pass to every linked channel."""

    _require_finite(sr=float(sr), cutoff=float(cutoff))
    if sr <= 0 or not 0.0 < cutoff < sr / 2.0:
        raise ValueError("cutoff must be between 0 Hz and Nyquist.")
    samples, layout = _as_samples_channels(audio, channel_axis)
    sos = signal.butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return _restore_layout(_stable_sos_filter(sos, samples), layout)


def apply_parametric_biquad_eq(
    audio: np.ndarray,
    sr: int = 16_000,
    low_shelf_gain: float = 0.0,
    mid_bell_gain: float = 0.0,
    mid_bell_freq: float = 2_500.0,
    mid_bell_q: float = 1.2,
    high_shelf_gain: float = 0.0,
    *,
    low_mid_gain: float = 0.0,
    low_mid_freq: float = 400.0,
    low_mid_q: float = 1.0,
    presence_gain: float = 0.0,
    presence_freq: float = 5_000.0,
    presence_q: float = 1.0,
    low_shelf_freq: float = 150.0,
    high_shelf_freq: float = 10_000.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Apply five serial parametric bands represented as stable SOS biquads."""

    if sr <= 0:
        raise ValueError("sr must be positive.")
    _require_finite(
        sr=float(sr),
        low_shelf_gain=float(low_shelf_gain),
        mid_bell_gain=float(mid_bell_gain),
        mid_bell_freq=float(mid_bell_freq),
        mid_bell_q=float(mid_bell_q),
        high_shelf_gain=float(high_shelf_gain),
        low_mid_gain=float(low_mid_gain),
        low_mid_freq=float(low_mid_freq),
        low_mid_q=float(low_mid_q),
        presence_gain=float(presence_gain),
        presence_freq=float(presence_freq),
        presence_q=float(presence_q),
        low_shelf_freq=float(low_shelf_freq),
        high_shelf_freq=float(high_shelf_freq),
    )
    samples, layout = _as_samples_channels(audio, channel_axis)
    bands = (
        (_shelf_sos(sr, low_shelf_freq, low_shelf_gain, high=False), low_shelf_gain),
        (_peaking_sos(sr, low_mid_freq, low_mid_gain, low_mid_q), low_mid_gain),
        (_peaking_sos(sr, mid_bell_freq, mid_bell_gain, mid_bell_q), mid_bell_gain),
        (_peaking_sos(sr, presence_freq, presence_gain, presence_q), presence_gain),
        (_shelf_sos(sr, high_shelf_freq, high_shelf_gain, high=True), high_shelf_gain),
    )
    processed = samples.astype(np.float64, copy=True)
    for sos, gain_db in bands:
        if abs(float(gain_db)) > 1e-9:
            processed = _stable_sos_filter(sos, processed)
    return _restore_layout(processed, layout)


def _linked_envelope(
    detector: np.ndarray,
    sample_rate: int,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    _require_finite(attack_ms=float(attack_ms), release_ms=float(release_ms))
    if attack_ms <= 0.0 or release_ms <= 0.0:
        raise ValueError("Envelope attack and release must be positive.")
    attack = np.exp(-1.0 / max(1.0, sample_rate * attack_ms / 1_000.0))
    release = np.exp(-1.0 / max(1.0, sample_rate * release_ms / 1_000.0))
    envelope = np.empty(detector.size, dtype=np.float64)
    previous = 0.0
    for index, value in enumerate(detector):
        coefficient = attack if value > previous else release
        previous = coefficient * previous + (1.0 - coefficient) * float(value)
        envelope[index] = previous
    return envelope


def apply_deesser(
    audio: np.ndarray,
    sr: int = 16_000,
    threshold_db: float = -20.0,
    amount: float = 0.5,
    *,
    attack_ms: float = 1.0,
    release_ms: float = 65.0,
    knee_db: float = 4.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Dynamically attenuate only the 5-9 kHz sibilance component."""

    if sr <= 0:
        raise ValueError("sr must be positive.")
    _require_finite(
        sr=float(sr),
        threshold_db=float(threshold_db),
        amount=float(amount),
        attack_ms=float(attack_ms),
        release_ms=float(release_ms),
        knee_db=float(knee_db),
    )
    samples, layout = _as_samples_channels(audio, channel_axis)
    if samples.size == 0 or amount <= 0.0:
        return _restore_layout(samples, layout)
    nyquist = sr / 2.0
    high_hz = min(9_000.0, nyquist * 0.98)
    if high_hz <= 5_500.0:
        return _restore_layout(samples, layout)
    low_hz = 5_000.0
    if low_hz <= 0.0 or low_hz >= high_hz:
        return _restore_layout(samples, layout)
    sidechain_sos = signal.butter(
        2, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos"
    )
    sibilance = _stable_sos_filter(sidechain_sos, samples)
    detector = np.max(np.abs(sibilance), axis=1)
    envelope = _linked_envelope(detector, sr, attack_ms, release_ms)
    envelope_db = 20.0 * np.log10(np.maximum(envelope, _EPSILON))
    over_db = envelope_db - float(threshold_db)
    knee_db = max(float(knee_db), 0.1)
    soft_over = np.where(
        over_db <= -knee_db / 2.0,
        0.0,
        np.where(
            over_db >= knee_db / 2.0,
            over_db,
            np.square(over_db + knee_db / 2.0) / (2.0 * knee_db),
        ),
    )
    reduction_db = np.minimum(12.0, np.clip(float(amount), 0.0, 1.0) * soft_over)
    gain = 10.0 ** (-reduction_db / 20.0)
    processed = samples - sibilance + sibilance * gain[:, np.newaxis]
    return _restore_layout(processed, layout)


def apply_soft_knee_compressor(
    audio: np.ndarray,
    sr: int = 16_000,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    knee_db: float = 6.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
    makeup_gain_db: float = 0.0,
    *,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Apply a stereo-linked feed-forward soft-knee VCA compressor."""

    _require_finite(
        sr=float(sr),
        threshold_db=float(threshold_db),
        ratio=float(ratio),
        knee_db=float(knee_db),
        attack_ms=float(attack_ms),
        release_ms=float(release_ms),
        makeup_gain_db=float(makeup_gain_db),
    )
    if sr <= 0 or ratio < 1.0:
        raise ValueError("sr must be positive and ratio must be at least 1:1.")
    samples, layout = _as_samples_channels(audio, channel_axis)
    if samples.size == 0 or (ratio == 1.0 and abs(makeup_gain_db) <= 1e-12):
        return _restore_layout(samples, layout)
    detector = np.max(np.abs(samples), axis=1)
    envelope = _linked_envelope(detector, sr, attack_ms, release_ms)
    level_db = 20.0 * np.log10(np.maximum(envelope, _EPSILON))
    offset = level_db - float(threshold_db)
    knee_db = max(float(knee_db), 0.0)
    slope = 1.0 / float(ratio) - 1.0
    if knee_db > 0.0:
        gain_reduction_db = np.where(
            offset <= -knee_db / 2.0,
            0.0,
            np.where(
                offset >= knee_db / 2.0,
                slope * offset,
                slope * np.square(offset + knee_db / 2.0) / (2.0 * knee_db),
            ),
        )
    else:
        gain_reduction_db = np.where(offset > 0.0, slope * offset, 0.0)
    gain = 10.0 ** ((gain_reduction_db + float(makeup_gain_db)) / 20.0)
    return _restore_layout(samples * gain[:, np.newaxis], layout)


def apply_harmonic_saturation(
    audio: np.ndarray,
    saturation_type: str = "tube",
    drive: float = 0.2,
    *,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Generate oversampled even/odd harmonics with tube or tape curves."""

    samples, layout = _as_samples_channels(audio, channel_axis)
    _require_finite(drive=float(drive))
    drive = float(np.clip(drive, 0.0, 1.0))
    if samples.size == 0 or drive <= 0.0:
        return _restore_layout(samples, layout)
    oversampled = signal.resample_poly(
        samples, 4, 1, axis=0, window=("kaiser", 8.6)
    ).astype(np.float64, copy=False)
    driven = oversampled * (1.0 + 2.0 * drive)
    saturation_type = str(saturation_type).strip().lower()
    if saturation_type == "tube":
        bounded = np.tanh(driven)
        nonlinear = np.tanh(driven + 0.15 * drive * np.square(bounded))
        nonlinear -= np.mean(nonlinear, axis=0, keepdims=True)
    elif saturation_type == "tape":
        scale = 1.0 + 4.0 * drive
        nonlinear = np.arctan(scale * driven) / np.arctan(scale)
    else:
        raise ValueError("saturation_type must be 'tube' or 'tape'.")
    nonlinear /= 1.0 + 0.35 * drive
    nonlinear_base = signal.resample_poly(
        nonlinear, 1, 4, axis=0, window=("kaiser", 8.6)
    )[: samples.shape[0]]
    if nonlinear_base.shape[0] < samples.shape[0]:
        nonlinear_base = np.pad(
            nonlinear_base,
            ((0, samples.shape[0] - nonlinear_base.shape[0]), (0, 0)),
        )
    restored = (1.0 - drive) * samples + drive * nonlinear_base
    return _restore_layout(restored, layout)


apply_harmonic_exciter = apply_harmonic_saturation


def apply_linked_true_peak_limiter(
    audio: np.ndarray,
    sr: int = 16_000,
    target_peak_dbfs: float = -0.1,
    lookahead_ms: float = 1.0,
    *,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Apply one linked gain envelope using a four-times true-peak detector."""

    _require_finite(
        sr=float(sr),
        target_peak_dbfs=float(target_peak_dbfs),
        lookahead_ms=float(lookahead_ms),
    )
    if sr <= 0:
        raise ValueError("sr must be positive.")
    samples, layout = _as_samples_channels(audio, channel_axis)
    if samples.size == 0:
        return _restore_layout(samples, layout)
    factor = 8
    oversampled = signal.resample_poly(
        samples, factor, 1, axis=0, window=("kaiser", 8.6)
    ).astype(np.float64, copy=False)
    detector = np.max(np.abs(oversampled), axis=1)
    lookahead = max(1, int(round(lookahead_ms * sr * factor / 1_000.0)))
    window_size = lookahead + 1
    envelope = maximum_filter1d(
        detector,
        size=window_size,
        origin=-(window_size // 2),
        mode="nearest",
    )
    ceiling = 10.0 ** (float(target_peak_dbfs) / 20.0)
    required_gain = np.minimum(1.0, ceiling / np.maximum(envelope, _EPSILON))
    release_size = max(2, int(round(0.050 * sr * factor))) + 1
    held_gain = minimum_filter1d(
        required_gain,
        size=release_size,
        origin=(release_size - 1) // 2,
        mode="nearest",
    )
    oversampled_gain = np.minimum(required_gain, held_gain)
    needed = samples.shape[0] * factor
    if oversampled_gain.size < needed:
        oversampled_gain = np.pad(
            oversampled_gain,
            (0, needed - oversampled_gain.size),
            mode="edge",
        )
    base_gain = np.min(oversampled_gain[:needed].reshape(-1, factor), axis=1)
    restored = samples.astype(np.float64, copy=False) * base_gain[:, np.newaxis]
    validation_factor = 16
    measured = signal.resample_poly(
        restored,
        validation_factor,
        1,
        axis=0,
        window=("kaiser", 8.6),
    )
    measured_peak = float(np.max(np.abs(measured))) if measured.size else 0.0
    # A 0.15 dB reconstruction margin protects higher-order true-peak probes.
    safety_ceiling = 10.0 ** ((float(target_peak_dbfs) - 0.15) / 20.0)
    if measured_peak > safety_ceiling:
        restored *= safety_ceiling / measured_peak
    return _restore_layout(restored, layout)


def _preset_parameters(preset: str, custom_params: Dict[str, Any] | None) -> Dict[str, Any]:
    presets: Dict[str, Dict[str, Any]] = {
        BROADCAST_PRESET: {
            "highpass_cutoff": 80.0,
            "low_gain": 0.5,
            "mid_gain": 2.0,
            "mid_freq": 2_800.0,
            "mid_q": 1.2,
            "presence_gain": 1.0,
            "high_gain": 1.5,
            "high_freq": 10_000.0,
            "deess_enabled": True,
            "deess_thresh": -18.0,
            "deess_amount": 0.6,
            "comp_enabled": True,
            "comp_threshold": -18.0,
            "comp_ratio": 2.5,
            "comp_knee": 6.0,
            "comp_attack": 8.0,
            "comp_release": 100.0,
            "comp_makeup": 1.0,
            "sat_type": "tube",
            "sat_drive": 0.15,
            "limiter_ceiling": -0.1,
        },
        PODCAST_PRESET: {
            "highpass_cutoff": 65.0,
            "low_gain": 3.0,
            "low_mid_gain": -1.0,
            "low_mid_freq": 400.0,
            "mid_gain": 0.5,
            "mid_freq": 1_800.0,
            "high_gain": 0.5,
            "deess_enabled": False,
            "comp_enabled": True,
            "comp_threshold": -20.0,
            "comp_ratio": 3.0,
            "comp_knee": 8.0,
            "comp_attack": 15.0,
            "comp_release": 160.0,
            "comp_makeup": 1.5,
            "sat_type": "tape",
            "sat_drive": 0.25,
            "limiter_ceiling": -0.1,
        },
        AIR_PRESET: {
            "highpass_cutoff": 70.0,
            "low_gain": -1.0,
            "mid_gain": 1.0,
            "mid_freq": 3_500.0,
            "mid_q": 1.1,
            "presence_gain": 1.0,
            "high_gain": 3.5,
            "high_freq": 12_000.0,
            "deess_enabled": True,
            "deess_thresh": -22.0,
            "deess_amount": 0.8,
            "deess_attack": 0.5,
            "deess_release": 55.0,
            "comp_enabled": True,
            "comp_threshold": -16.0,
            "comp_ratio": 2.0,
            "comp_knee": 5.0,
            "comp_attack": 6.0,
            "comp_release": 90.0,
            "comp_makeup": 0.5,
            "sat_type": "tube",
            "sat_drive": 0.06,
            "limiter_ceiling": -0.1,
        },
        CUSTOM_PRESET: {
            "highpass_cutoff": 80.0,
            "low_gain": 0.0,
            "low_mid_gain": 0.0,
            "low_mid_freq": 400.0,
            "mid_gain": 0.0,
            "mid_freq": 2_500.0,
            "mid_q": 1.2,
            "presence_gain": 0.0,
            "presence_freq": 5_000.0,
            "high_gain": 0.0,
            "high_freq": 10_000.0,
            "deess_enabled": True,
            "deess_thresh": -20.0,
            "deess_amount": 0.5,
            "deess_knee": 4.0,
            "comp_enabled": True,
            "comp_threshold": -18.0,
            "comp_ratio": 3.0,
            "comp_knee": 6.0,
            "comp_attack": 10.0,
            "comp_release": 120.0,
            "comp_makeup": 0.0,
            "sat_type": "tube",
            "sat_drive": 0.0,
            "limiter_ceiling": -0.1,
        },
    }
    if preset not in presets:
        raise ValueError(f"Unknown studio mastering preset: {preset}")
    parameters = deepcopy(presets[preset])
    if preset == CUSTOM_PRESET and custom_params:
        parameters.update(custom_params)
    for key, value in parameters.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, (bool, np.bool_)
        ):
            if not np.isfinite(value):
                raise ValueError(f"Studio parameter {key!r} must be finite.")
    if str(parameters.get("sat_type", "tube")).lower() not in {"tube", "tape"}:
        raise ValueError("sat_type must be 'tube' or 'tape'.")
    return parameters


def _master_levels(
    audio: np.ndarray,
    sample_rate: int,
    channel_axis: int | None = None,
) -> Dict[str, float]:
    samples, _ = _as_samples_channels(audio, channel_axis)
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    oversampled = (
        signal.resample_poly(samples, 16, 1, axis=0, window=("kaiser", 8.6))
        if samples.size
        else samples
    )
    peak = float(np.max(np.abs(oversampled))) if oversampled.size else 0.0
    return {
        "rms_dbfs": float(20.0 * np.log10(max(rms, _EPSILON))),
        "true_peak_dbfs": float(20.0 * np.log10(max(peak, _EPSILON))),
    }


def apply_studio_mastering_chain(
    audio: np.ndarray,
    sr: int = 16_000,
    preset: str = BROADCAST_PRESET,
    custom_params: Dict[str, Any] | None = None,
    *,
    channel_axis: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute the complete linked-channel mastering chain and return telemetry."""

    parameters = _preset_parameters(preset, custom_params)
    original = np.asarray(audio, dtype=np.float32)
    out = original.copy()
    stages: list[str] = []
    processing_warnings: list[str] = []

    cutoff = float(parameters.get("highpass_cutoff", 0.0))
    if cutoff > 0.0:
        out = apply_highpass_filter(
            out, sr=sr, cutoff=cutoff, channel_axis=channel_axis
        )
        stages.append("highpass_filter")

    out = apply_parametric_biquad_eq(
        out,
        sr=sr,
        low_shelf_gain=float(parameters.get("low_gain", 0.0)),
        mid_bell_gain=float(parameters.get("mid_gain", 0.0)),
        mid_bell_freq=float(parameters.get("mid_freq", 2_500.0)),
        mid_bell_q=float(parameters.get("mid_q", 1.2)),
        high_shelf_gain=float(parameters.get("high_gain", 0.0)),
        low_mid_gain=float(parameters.get("low_mid_gain", 0.0)),
        low_mid_freq=float(parameters.get("low_mid_freq", 400.0)),
        low_mid_q=float(parameters.get("low_mid_q", 1.0)),
        presence_gain=float(parameters.get("presence_gain", 0.0)),
        presence_freq=float(parameters.get("presence_freq", 5_000.0)),
        presence_q=float(parameters.get("presence_q", 1.0)),
        low_shelf_freq=float(parameters.get("low_shelf_freq", 150.0)),
        high_shelf_freq=float(parameters.get("high_freq", 10_000.0)),
        channel_axis=channel_axis,
    )
    stages.append("five_band_parametric_eq")

    if bool(parameters.get("deess_enabled", True)):
        effective_deess_high = min(9_000.0, sr / 2.0 * 0.98)
        if effective_deess_high <= 5_500.0:
            processing_warnings.append(
                "De-esser bypassed: sample rate cannot represent the 5-9 kHz band."
            )
            stages.append("dynamic_deesser_bypassed_nyquist")
        else:
            out = apply_deesser(
                out,
                sr=sr,
                threshold_db=float(parameters.get("deess_thresh", -20.0)),
                amount=float(parameters.get("deess_amount", 0.5)),
                attack_ms=float(parameters.get("deess_attack", 1.0)),
                release_ms=float(parameters.get("deess_release", 65.0)),
                knee_db=float(parameters.get("deess_knee", 4.0)),
                channel_axis=channel_axis,
            )
            stages.append("dynamic_deesser")
            if effective_deess_high < 9_000.0:
                processing_warnings.append(
                    f"De-esser upper band limited to {effective_deess_high:.0f} Hz by the sample rate."
                )

    if bool(parameters.get("comp_enabled", True)):
        out = apply_soft_knee_compressor(
            out,
            sr=sr,
            threshold_db=float(parameters.get("comp_threshold", -18.0)),
            ratio=float(parameters.get("comp_ratio", 3.0)),
            knee_db=float(parameters.get("comp_knee", 6.0)),
            attack_ms=float(parameters.get("comp_attack", 10.0)),
            release_ms=float(parameters.get("comp_release", 120.0)),
            makeup_gain_db=float(parameters.get("comp_makeup", 0.0)),
            channel_axis=channel_axis,
        )
        stages.append("soft_knee_vca_compressor")

    saturation_drive = float(parameters.get("sat_drive", 0.0))
    if saturation_drive > 0.0:
        out = apply_harmonic_saturation(
            out,
            saturation_type=str(parameters.get("sat_type", "tube")),
            drive=saturation_drive,
            channel_axis=channel_axis,
        )
        stages.append("harmonic_exciter")

    limiter_ceiling = float(parameters.get("limiter_ceiling", -0.1))
    out = apply_linked_true_peak_limiter(
        out,
        sr=sr,
        target_peak_dbfs=limiter_ceiling,
        lookahead_ms=1.0,
        channel_axis=channel_axis,
    )
    stages.append("true_peak_lookahead_limiter")

    samples, _ = _as_samples_channels(original, channel_axis)
    requested_high_shelf = float(parameters.get("high_freq", 10_000.0))
    effective_high_shelf = _bounded_frequency(requested_high_shelf, sr)
    if effective_high_shelf < requested_high_shelf:
        processing_warnings.append(
            f"High shelf limited to {effective_high_shelf:.0f} Hz by the {sr} Hz sample rate."
        )
    metadata = {
        "status": "SUCCESS",
        "preset_applied": preset,
        "stages": stages,
        "parameters": parameters,
        "input_levels": _master_levels(original, sr, channel_axis),
        "output_levels": _master_levels(out, sr, channel_axis),
        "channel_count": int(samples.shape[1]),
        "phase_hygiene": "linked controls and identical SOS paths across channels",
        "effective_high_shelf_hz": effective_high_shelf,
        "warnings": processing_warnings,
    }
    return np.asarray(out, dtype=np.float32), metadata
