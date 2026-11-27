"""Mastering-chain DSP, stability, and linked-channel regression tests."""

from __future__ import annotations

import inspect

import numpy as np
from scipy.signal import resample_poly

import modules.voice.studio_tuner as studio_module
from modules.voice.studio_tuner import (
    AIR_PRESET,
    BROADCAST_PRESET,
    CUSTOM_PRESET,
    PODCAST_PRESET,
    STUDIO_PRESETS,
    apply_deesser,
    apply_harmonic_saturation,
    apply_highpass_filter,
    apply_linked_true_peak_limiter,
    apply_parametric_biquad_eq,
    apply_soft_knee_compressor,
    apply_studio_mastering_chain,
)


def _tone(frequency: float, sample_rate: int, seconds: float, amplitude: float) -> np.ndarray:
    indices = np.arange(int(sample_rate * seconds), dtype=np.float64)
    return (amplitude * np.sin(2.0 * np.pi * frequency * indices / sample_rate)).astype(
        np.float32
    )


def _frequency_amplitude(audio: np.ndarray, sample_rate: int, frequency: float) -> float:
    signal = np.asarray(audio, dtype=np.float64).reshape(-1)
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    index = int(np.argmin(np.abs(frequencies - frequency)))
    return float(2.0 * np.abs(spectrum[index]) / max(np.sum(window), 1e-12))


def _true_peak_dbfs(audio: np.ndarray) -> float:
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    oversampled = resample_poly(samples, 32, 1, axis=0, window=("kaiser", 8.6))
    peak = float(np.max(np.abs(oversampled))) if oversampled.size else 0.0
    return float(20.0 * np.log10(max(peak, 1e-10)))


def test_all_biquad_iir_paths_use_sosfilt_not_direct_form_filtering():
    source = inspect.getsource(studio_module)
    assert "signal.sosfilt" in source
    assert "signal.lfilter(" not in source
    assert "output=\"sos\"" in source


def test_fourth_order_highpass_removes_rumble_without_erasing_voice_band():
    sample_rate = 48_000
    source = _tone(30.0, sample_rate, 1.0, 0.4) + _tone(
        1_000.0, sample_rate, 1.0, 0.2
    )

    filtered = apply_highpass_filter(source, sample_rate, cutoff=80.0)

    rumble_ratio = _frequency_amplitude(filtered, sample_rate, 30.0) / _frequency_amplitude(
        source, sample_rate, 30.0
    )
    voice_ratio = _frequency_amplitude(
        filtered, sample_rate, 1_000.0
    ) / _frequency_amplitude(source, sample_rate, 1_000.0)
    assert rumble_ratio < 0.05
    assert voice_ratio > 0.95


def test_five_band_high_q_eq_is_finite_and_phase_matched_across_channels():
    rng = np.random.default_rng(42)
    mono = (rng.normal(0.0, 0.05, 24_000)).astype(np.float32)
    stereo = np.column_stack((mono, mono))

    equalized = apply_parametric_biquad_eq(
        stereo,
        sr=48_000,
        low_shelf_gain=9.0,
        low_mid_gain=-10.0,
        mid_bell_gain=12.0,
        mid_bell_freq=2_750.0,
        mid_bell_q=20.0,
        presence_gain=-9.0,
        high_shelf_gain=8.0,
        high_shelf_freq=12_000.0,
    )

    assert equalized.shape == stereo.shape
    assert np.all(np.isfinite(equalized))
    assert float(np.max(np.abs(equalized))) < 10.0
    assert np.allclose(equalized[:, 0], equalized[:, 1], atol=1e-7)


def test_dynamic_deesser_reduces_sibilance_more_than_speech_fundamental():
    sample_rate = 48_000
    source = _tone(1_000.0, sample_rate, 1.0, 0.15) + _tone(
        7_000.0, sample_rate, 1.0, 0.35
    )

    processed = apply_deesser(
        source,
        sample_rate,
        threshold_db=-34.0,
        amount=1.0,
        attack_ms=0.5,
        release_ms=40.0,
    )

    sibilance_ratio = _frequency_amplitude(
        processed, sample_rate, 7_000.0
    ) / _frequency_amplitude(source, sample_rate, 7_000.0)
    fundamental_ratio = _frequency_amplitude(
        processed, sample_rate, 1_000.0
    ) / _frequency_amplitude(source, sample_rate, 1_000.0)
    assert sibilance_ratio < 0.65
    assert fundamental_ratio > 0.90


def test_soft_knee_vca_compressor_reduces_loud_section_and_links_stereo():
    sample_rate = 48_000
    quiet = _tone(1_000.0, sample_rate, 0.5, 0.03)
    loud = _tone(1_000.0, sample_rate, 0.5, 0.9)
    mono = np.concatenate((quiet, loud))
    stereo = np.column_stack((mono, mono))

    compressed = apply_soft_knee_compressor(
        stereo,
        sample_rate,
        threshold_db=-18.0,
        ratio=4.0,
        knee_db=8.0,
        attack_ms=1.0,
        release_ms=80.0,
    )

    quiet_input_rms = float(np.sqrt(np.mean(np.square(quiet))))
    quiet_output_rms = float(np.sqrt(np.mean(np.square(compressed[: quiet.size, 0]))))
    loud_input_rms = float(np.sqrt(np.mean(np.square(loud))))
    loud_output_rms = float(np.sqrt(np.mean(np.square(compressed[quiet.size :, 0]))))
    assert quiet_output_rms / quiet_input_rms > 0.85
    assert loud_output_rms / loud_input_rms < 0.65
    assert np.allclose(compressed[:, 0], compressed[:, 1], atol=1e-7)


def test_harmonic_exciter_generates_expected_tube_and_tape_overtones():
    sample_rate = 48_000
    source = _tone(1_000.0, sample_rate, 1.0, 0.35)

    tube = apply_harmonic_saturation(source, "tube", drive=0.5)
    tape = apply_harmonic_saturation(source, "tape", drive=0.5)

    source_second = _frequency_amplitude(source, sample_rate, 2_000.0)
    source_third = _frequency_amplitude(source, sample_rate, 3_000.0)
    assert _frequency_amplitude(tube, sample_rate, 2_000.0) > source_second + 1e-3
    assert _frequency_amplitude(tape, sample_rate, 3_000.0) > source_third + 1e-3
    assert np.all(np.isfinite(tube))
    assert np.all(np.isfinite(tape))


def test_linked_true_peak_limiter_preserves_stereo_relationship_and_ceiling():
    sample_rate = 48_000
    left = _tone(997.0, sample_rate, 0.5, 1.3)
    stereo = np.column_stack((left, -left))

    limited = apply_linked_true_peak_limiter(
        stereo, sample_rate, target_peak_dbfs=-0.1, lookahead_ms=1.0
    )

    assert limited.shape == stereo.shape
    assert np.allclose(limited[:, 0], -limited[:, 1], atol=1e-6)
    assert _true_peak_dbfs(limited) <= -0.09


def test_limiter_is_transparent_when_signal_is_already_below_ceiling():
    sample_rate = 16_000
    source = _tone(7_000.0, sample_rate, 0.5, 0.1)

    limited = apply_linked_true_peak_limiter(
        source, sample_rate, target_peak_dbfs=-0.1
    )

    assert np.allclose(limited, source, atol=1e-7)


def test_tube_saturation_is_continuous_at_zero_and_monotonic_at_full_drive():
    sample_rate = 48_000
    source = _tone(1_000.0, sample_rate, 0.25, 0.5)
    near_bypass = apply_harmonic_saturation(source, "tube", drive=1e-5)
    relative_error = float(
        np.sqrt(np.mean(np.square(near_bypass - source)))
        / np.sqrt(np.mean(np.square(source)))
    )
    ramp = np.linspace(-0.8, 0.8, 24_000, dtype=np.float32)
    saturated_ramp = apply_harmonic_saturation(ramp, "tube", drive=1.0)

    assert relative_error < 1e-3
    assert float(np.min(np.diff(saturated_ramp[200:-200]))) > -1e-4

    near_nyquist = _tone(7_000.0, 16_000, 0.25, 0.3)
    near_nyquist_wet = apply_harmonic_saturation(
        near_nyquist, "tube", drive=1e-8
    )
    nyquist_error = float(
        np.sqrt(np.mean(np.square(near_nyquist_wet - near_nyquist)))
        / np.sqrt(np.mean(np.square(near_nyquist)))
    )
    assert nyquist_error < 1e-6


def test_low_rate_preset_reports_band_limit_and_deesser_bypass():
    source = _tone(1_000.0, 8_000, 0.25, 0.2)

    _, metadata = apply_studio_mastering_chain(source, 8_000, AIR_PRESET)

    assert "dynamic_deesser_bypassed_nyquist" in metadata["stages"]
    assert any("De-esser bypassed" in warning for warning in metadata["warnings"])
    assert any("High shelf limited" in warning for warning in metadata["warnings"])

    source_11025 = _tone(1_000.0, 11_025, 0.25, 0.2)
    _, metadata_11025 = apply_studio_mastering_chain(
        source_11025, 11_025, AIR_PRESET
    )
    assert "dynamic_deesser_bypassed_nyquist" in metadata_11025["stages"]


def test_non_finite_custom_parameters_are_rejected():
    source = _tone(440.0, 16_000, 0.1, 0.2)

    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        apply_studio_mastering_chain(
            source,
            16_000,
            CUSTOM_PRESET,
            custom_params={"limiter_ceiling": np.nan},
        )
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        apply_deesser(source, 16_000, amount=np.nan)
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        apply_soft_knee_compressor(source, 16_000, ratio=np.nan)
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        apply_harmonic_saturation(source, "tube", drive=np.nan)


def test_short_channel_first_buffers_preserve_layout_and_linked_polarity():
    channel = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
    channel_first = np.vstack((channel, -channel))

    limited = apply_linked_true_peak_limiter(channel_first, 16_000)

    assert limited.shape == channel_first.shape
    assert np.allclose(limited[0], -limited[1], atol=1e-6)

    short_channel = channel[:8]
    sixteen_channels = np.vstack(
        [short_channel if index % 2 == 0 else -short_channel for index in range(16)]
    )
    limited_sixteen = apply_linked_true_peak_limiter(
        sixteen_channels, 16_000, channel_axis=0
    )
    assert limited_sixteen.shape == sixteen_channels.shape
    assert np.allclose(limited_sixteen[0], -limited_sixteen[1], atol=1e-6)

    mastered_sixteen, metadata = apply_studio_mastering_chain(
        sixteen_channels,
        16_000,
        CUSTOM_PRESET,
        custom_params={"deess_enabled": False, "comp_enabled": False},
        channel_axis=0,
    )
    assert mastered_sixteen.shape == sixteen_channels.shape
    assert metadata["channel_count"] == 16


def test_all_studio_presets_are_finite_audited_and_true_peak_safe():
    sample_rate = 48_000
    source = _tone(220.0, sample_rate, 0.5, 0.4) + _tone(
        6_500.0, sample_rate, 0.5, 0.15
    )
    stereo = np.column_stack((source, source))

    for preset in STUDIO_PRESETS:
        mastered, metadata = apply_studio_mastering_chain(
            stereo, sample_rate, preset=preset
        )
        assert mastered.shape == stereo.shape
        assert mastered.dtype == np.float32
        assert np.all(np.isfinite(mastered))
        assert np.allclose(mastered[:, 0], mastered[:, 1], atol=1e-6)
        assert _true_peak_dbfs(mastered) <= -0.09
        assert metadata["status"] == "SUCCESS"
        assert metadata["preset_applied"] == preset
        assert metadata["channel_count"] == 2
        assert metadata["stages"][-1] == "true_peak_lookahead_limiter"

    _, broadcast = apply_studio_mastering_chain(stereo, sample_rate, BROADCAST_PRESET)
    _, podcast = apply_studio_mastering_chain(stereo, sample_rate, PODCAST_PRESET)
    _, air = apply_studio_mastering_chain(stereo, sample_rate, AIR_PRESET)
    assert broadcast["parameters"]["highpass_cutoff"] == 80.0
    assert broadcast["parameters"]["sat_type"] == "tube"
    assert podcast["parameters"]["low_gain"] == 3.0
    assert podcast["parameters"]["comp_ratio"] == 3.0
    assert podcast["parameters"]["sat_type"] == "tape"
    assert air["parameters"]["high_gain"] == 3.5
    assert air["parameters"]["comp_ratio"] == 2.0


def test_custom_manual_mode_honors_stage_bypasses_and_parameters():
    source = _tone(440.0, 16_000, 0.25, 0.3)
    mastered, metadata = apply_studio_mastering_chain(
        source,
        16_000,
        CUSTOM_PRESET,
        custom_params={
            "highpass_cutoff": 100.0,
            "mid_gain": 4.0,
            "mid_freq": 2_000.0,
            "deess_enabled": False,
            "comp_enabled": False,
            "sat_type": "tape",
            "sat_drive": 0.3,
        },
    )

    assert mastered.shape == source.shape
    assert "dynamic_deesser" not in metadata["stages"]
    assert "soft_knee_vca_compressor" not in metadata["stages"]
    assert "harmonic_exciter" in metadata["stages"]
    assert metadata["parameters"]["mid_gain"] == 4.0
    assert metadata["parameters"]["sat_type"] == "tape"
