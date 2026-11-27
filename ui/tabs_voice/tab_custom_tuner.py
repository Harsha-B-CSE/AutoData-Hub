"""Smart acoustic analysis and interactive macro EQ controls."""

from __future__ import annotations

import numpy as np
import streamlit as st

from modules.voice.studio_tuner import (
    CUSTOM_PRESET,
    STUDIO_PRESETS,
    apply_studio_mastering_chain,
)
from modules.voice.tuner import analyze_audio_spectrum, apply_parametric_eq
from ui.tabs_voice.common import player_wav
from ui.utils_ux import render_tab_guidance_header


def render_tab_custom_tuner(
    audio_signal: np.ndarray,
    sample_rate: int = 16_000,
    *,
    persist_for_export: bool = True,
) -> None:
    """Render diagnostic suggestions and a live three-band macro EQ."""
    render_tab_guidance_header(
        title="Smart Studio Tuner",
        objective="Shape vocal tone, tame sharp consonants, and even out speech levels.",
        methodology=(
            "Use the band-energy and loudness readings to choose a preset or tune five EQ bands, "
            "a 5-9 kHz de-esser, and soft-knee compression. Saturation adds harmonics before "
            "lookahead true-peak limiting."
        ),
        deliverable="A playable master with a true-peak reading. Tuning enhanced audio also updates the Export Studio track.",
    )
    if not persist_for_export:
        st.info("The tuner is previewing decoded audio. Run Voice Enhancement to use the tuned track in Export Studio.")
    audio = np.asarray(audio_signal, dtype=np.float32).reshape(-1)
    analysis = analyze_audio_spectrum(audio, sample_rate)
    with st.container(border=True):
        st.markdown("#### AI Acoustic Suggestions")
        for suggestion in analysis["suggestions"]:
            st.markdown(f"- {suggestion}")
        sub, mids, highs, loudness = st.columns(4)
        sub.metric(
            "Sub-bass Energy",
            f"{analysis['band_energies']['sub_bass'] * 100:.1f}%",
        )
        mids.metric(
            "Mid Energy",
            f"{analysis['band_energies']['mids'] * 100:.1f}%",
        )
        highs.metric(
            "High Energy",
            f"{analysis['band_energies']['highs'] * 100:.1f}%",
        )
        loudness.metric("Estimated LUFS", f"{analysis['lufs_est']:.1f}")

    preset = st.selectbox(
        "Studio Mastering Preset",
        options=list(STUDIO_PRESETS),
        key="voice_studio_preset",
    )
    st.markdown("#### Interactive 3-Band EQ")
    low_column, mid_column, high_column = st.columns(3)
    with low_column:
        low_gain = st.slider(
            "Low Bass (20-250 Hz)",
            min_value=-12.0,
            max_value=12.0,
            value=0.0,
            step=0.5,
            key="voice_tuner_low_db",
        )
    with mid_column:
        mid_gain = st.slider(
            "Mid Range (250-2.5 kHz)",
            min_value=-12.0,
            max_value=12.0,
            value=0.0,
            step=0.5,
            key="voice_tuner_mid_db",
        )
    with high_column:
        high_gain = st.slider(
            "High Air (2.5-16 kHz)",
            min_value=-12.0,
            max_value=12.0,
            value=0.0,
            step=0.5,
            key="voice_tuner_high_db",
        )

    custom_params = None
    source_for_mastering = audio
    if preset == CUSTOM_PRESET:
        with st.expander("Custom Studio Dynamics & Harmonics", expanded=True):
            eq_tab, dynamics_tab, harmonics_tab = st.tabs(
                ["Five-Band EQ", "Dynamics", "Harmonics & Ceiling"]
            )
            maximum_frequency = float(max(500, min(16_000, sample_rate * 0.45)))
            with eq_tab:
                render_tab_guidance_header(
                    title="Five-Band EQ",
                    objective="Remove low-frequency rumble and adjust vocal body, presence, and air.",
                    methodology=(
                        "Enable Low-Cut for rumble. Low Bass and High Air set the shelf gains; Mid Range sets "
                        "the central bell gain. Adjust the band frequencies here; higher Q narrows a bell."
                    ),
                    deliverable="The tonal balance sent into the dynamics stage; listen to the master below the custom controls.",
                )
                eq_a, eq_b, eq_c = st.columns(3)
            with eq_a:
                highpass_enabled = st.toggle(
                    "Enable Low-Cut Filter",
                    value=True,
                    key="voice_studio_highpass_enabled",
                )
                highpass_cutoff = st.slider(
                    "Low-Cut Frequency (Hz)",
                    min_value=20.0,
                    max_value=200.0,
                    value=80.0,
                    step=5.0,
                    key="voice_studio_highpass",
                )
                low_shelf_frequency = st.slider(
                    "Low-Shelf Frequency (Hz)",
                    min_value=50.0,
                    max_value=300.0,
                    value=150.0,
                    step=5.0,
                    key="voice_studio_low_shelf_freq",
                )
                low_mid_gain = st.slider(
                    "Low-Mid Bell Gain (dB)",
                    min_value=-12.0,
                    max_value=12.0,
                    value=0.0,
                    step=0.5,
                    key="voice_studio_low_mid_gain",
                )
                low_mid_frequency = st.slider(
                    "Low-Mid Frequency (Hz)",
                    min_value=150.0,
                    max_value=1_200.0,
                    value=400.0,
                    step=25.0,
                    key="voice_studio_low_mid_freq",
                )
                low_mid_q = st.slider(
                    "Low-Mid Bell Q",
                    min_value=0.2,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="voice_studio_low_mid_q",
                )
            with eq_b:
                mid_frequency = st.slider(
                    "Mid Bell Frequency (Hz)",
                    min_value=200.0,
                    max_value=float(min(8_000, sample_rate // 2 - 100)),
                    value=float(min(2_500, sample_rate // 2 - 100)),
                    step=50.0,
                    key="voice_studio_mid_freq",
                )
                mid_q = st.slider(
                    "Mid Bell Q",
                    min_value=0.2,
                    max_value=10.0,
                    value=1.2,
                    step=0.1,
                    key="voice_studio_mid_q",
                )
                presence_gain = st.slider(
                    "Presence Bell Gain (dB)",
                    min_value=-12.0,
                    max_value=12.0,
                    value=0.0,
                    step=0.5,
                    key="voice_studio_presence_gain",
                )
                presence_frequency = st.slider(
                    "Presence Frequency (Hz)",
                    min_value=1_500.0,
                    max_value=maximum_frequency,
                    value=float(min(5_000, maximum_frequency)),
                    step=50.0,
                    key="voice_studio_presence_freq",
                )
            with eq_c:
                high_shelf_frequency = st.slider(
                    "High-Shelf Frequency (Hz)",
                    min_value=2_500.0,
                    max_value=maximum_frequency,
                    value=float(min(10_000, maximum_frequency)),
                    step=100.0,
                    key="voice_studio_high_shelf_freq",
                )
                presence_q = st.slider(
                    "Presence Bell Q",
                    min_value=0.2,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="voice_studio_presence_q",
                )

            with dynamics_tab:
                render_tab_guidance_header(
                    title="Vocal Dynamics",
                    objective="Soften sibilant consonants and reduce level swings without flattening the performance.",
                    methodology=(
                        "Set the de-esser threshold and amount for sharp S sounds. Set the compressor "
                        "threshold and ratio for loud passages; attack and release control how quickly gain changes."
                    ),
                    deliverable="A level-controlled vocal feeding the harmonics stage. Toggle each processor to hear its contribution.",
                )
                dynamics_a, dynamics_b, dynamics_c = st.columns(3)
            with dynamics_a:
                deess_enabled = st.toggle(
                    "Enable Dynamic De-Esser",
                    value=True,
                    key="voice_studio_deess_enabled",
                )
                deess_threshold = st.slider(
                    "De-Esser Threshold (dBFS)",
                    min_value=-40.0,
                    max_value=-6.0,
                    value=-20.0,
                    step=1.0,
                    key="voice_studio_deess_threshold",
                )
                deess_amount = st.slider(
                    "De-Esser Amount",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="voice_studio_deess_amount",
                )
                deess_attack = st.slider(
                    "De-Esser Attack (ms)",
                    min_value=0.1,
                    max_value=20.0,
                    value=1.0,
                    step=0.1,
                    key="voice_studio_deess_attack",
                )
                deess_release = st.slider(
                    "De-Esser Release (ms)",
                    min_value=20.0,
                    max_value=300.0,
                    value=65.0,
                    step=5.0,
                    key="voice_studio_deess_release",
                )
                deess_knee = st.slider(
                    "De-Esser Knee (dB)",
                    min_value=0.5,
                    max_value=12.0,
                    value=4.0,
                    step=0.5,
                    key="voice_studio_deess_knee",
                )
            with dynamics_b:
                compressor_enabled = st.toggle(
                    "Enable VCA Compressor",
                    value=True,
                    key="voice_studio_comp_enabled",
                )
                compressor_ratio = st.slider(
                    "Compressor Ratio",
                    min_value=1.0,
                    max_value=8.0,
                    value=3.0,
                    step=0.5,
                    key="voice_studio_comp_ratio",
                )
                compressor_threshold = st.slider(
                    "Compressor Threshold (dBFS)",
                    min_value=-40.0,
                    max_value=0.0,
                    value=-18.0,
                    step=1.0,
                    key="voice_studio_comp_threshold",
                )
                compressor_knee = st.slider(
                    "Compressor Knee (dB)",
                    min_value=0.0,
                    max_value=18.0,
                    value=6.0,
                    step=0.5,
                    key="voice_studio_comp_knee",
                )
            with dynamics_c:
                compressor_attack = st.slider(
                    "Compressor Attack (ms)",
                    min_value=0.1,
                    max_value=100.0,
                    value=10.0,
                    step=0.5,
                    key="voice_studio_comp_attack",
                )
                compressor_release = st.slider(
                    "Compressor Release (ms)",
                    min_value=20.0,
                    max_value=500.0,
                    value=120.0,
                    step=5.0,
                    key="voice_studio_comp_release",
                )
                compressor_makeup = st.slider(
                    "Compressor Makeup (dB)",
                    min_value=0.0,
                    max_value=12.0,
                    value=0.0,
                    step=0.5,
                    key="voice_studio_comp_makeup",
                )

            with harmonics_tab:
                render_tab_guidance_header(
                    title="Harmonics & Ceiling",
                    objective="Add harmonic color while leaving headroom for the loudest syllables.",
                    methodology=(
                        "Choose tube or tape and raise Drive from zero to hear saturation. Set the "
                        "true-peak ceiling in dBTP; a more negative ceiling leaves more headroom."
                    ),
                    deliverable="A limited master to audition with its true-peak reading. Export applies a separate codec-specific ceiling.",
                )
                harmonics_a, harmonics_b = st.columns(2)
            with harmonics_a:
                saturation_type = st.selectbox(
                    "Harmonic Character",
                    options=["tube", "tape"],
                    key="voice_studio_sat_type",
                )
                saturation_drive = st.slider(
                    "Harmonic Drive",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key="voice_studio_sat_drive",
                )
            with harmonics_b:
                limiter_ceiling = st.slider(
                    "True-Peak Ceiling (dBTP)",
                    min_value=-2.0,
                    max_value=-0.1,
                    value=-0.1,
                    step=0.1,
                    key="voice_studio_limiter_ceiling",
                )
            custom_params = {
                "highpass_cutoff": highpass_cutoff if highpass_enabled else 0.0,
                "low_gain": low_gain,
                "low_shelf_freq": low_shelf_frequency,
                "low_mid_gain": low_mid_gain,
                "low_mid_freq": low_mid_frequency,
                "low_mid_q": low_mid_q,
                "mid_gain": mid_gain,
                "mid_freq": mid_frequency,
                "mid_q": mid_q,
                "presence_gain": presence_gain,
                "presence_freq": presence_frequency,
                "presence_q": presence_q,
                "high_gain": high_gain,
                "high_freq": high_shelf_frequency,
                "deess_enabled": deess_enabled,
                "deess_thresh": deess_threshold,
                "deess_amount": deess_amount,
                "deess_attack": deess_attack,
                "deess_release": deess_release,
                "deess_knee": deess_knee,
                "comp_enabled": compressor_enabled,
                "comp_threshold": compressor_threshold,
                "comp_ratio": compressor_ratio,
                "comp_knee": compressor_knee,
                "comp_attack": compressor_attack,
                "comp_release": compressor_release,
                "comp_makeup": compressor_makeup,
                "sat_type": saturation_type,
                "sat_drive": saturation_drive,
                "limiter_ceiling": limiter_ceiling,
            }
    else:
        source_for_mastering = apply_parametric_eq(
            audio,
            sample_rate,
            low_gain_db=low_gain,
            mid_gain_db=mid_gain,
            high_gain_db=high_gain,
        )

    tuned, mastering = apply_studio_mastering_chain(
        source_for_mastering,
        sr=sample_rate,
        preset=preset,
        custom_params=custom_params,
    )
    if persist_for_export:
        st.session_state["voice_tuned_audio"] = tuned
        st.session_state["voice_tuned_sample_rate"] = int(sample_rate)
    st.audio(player_wav(tuned, sample_rate), format="audio/wav")
    st.caption(
        f"{mastering['preset_applied']} · {' → '.join(mastering['stages'])} · "
        f"True Peak {mastering['output_levels']['true_peak_dbfs']:.2f} dBFS"
    )
    for warning in mastering["warnings"]:
        st.warning(warning)


render_custom_tuner_tab = render_tab_custom_tuner
