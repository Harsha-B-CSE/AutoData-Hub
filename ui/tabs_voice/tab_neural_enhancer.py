"""Dual-tier enhancement controls and A/B listening view."""

from __future__ import annotations

import numpy as np
import streamlit as st

from core.schema_inference import ExecutionStatus
from modules.voice.pipeline_voice import enhance_voice
from ui.tabs_voice.common import cached_device_context, player_wav
from ui.utils_ux import render_tab_guidance_header


TIER_1_LABEL = "⚡ Tier-1: Fast Spectral Gate"
TIER_2_LABEL = "🚀 Tier-2: DeepFilterNet3 / MP-SENet"


def _invalidate_enhanced_output() -> None:
    for key in (
        "voice_enhanced_audio",
        "voice_enhanced_sample_rate",
        "voice_transform_result",
        "voice_module_result",
        "voice_tuned_audio",
        "voice_tuned_sample_rate",
    ):
        st.session_state.pop(key, None)


def render_tab_neural_enhancer(
    raw_audio: np.ndarray,
    sample_rate: int = 16_000,
    source_name: str = "audio.wav",
    audio_source=None,
) -> None:
    """Render enhancement controls and side-by-side raw/enhanced players."""
    render_tab_guidance_header(
        title="Neural Enhancer & Booster",
        objective="Suppress background noise and normalize speech to broadcast loudness.",
        methodology=(
            "Runs Tier-1 spectral gating or Tier-2 DeepFilterNet3 + 48 kHz harmonic "
            "extension, followed by EBU R128 boosting and -0.5 dBFS true-peak limiting."
        ),
        deliverable="A/B enhanced 48 kHz speech waveform ready for Smart Tuner or Export.",
    )
    control_a, control_b, control_c = st.columns([1.6, 1, 1])
    with control_a:
        processing_label = st.radio(
            "Processing Mode",
            options=[TIER_1_LABEL, TIER_2_LABEL],
            horizontal=False,
            key="voice_processing_mode",
            on_change=_invalidate_enhanced_output,
        )
    with control_b:
        attenuation_db = st.slider(
            "Denoising Attenuation (dB)",
            min_value=0,
            max_value=30,
            value=20,
            step=1,
            key="voice_attenuation_db",
            on_change=_invalidate_enhanced_output,
        )
    with control_c:
        boost_db = st.slider(
            "Quality Booster Gain (dB)",
            min_value=0.0,
            max_value=12.0,
            value=3.0,
            step=0.5,
            key="voice_boost_db",
            on_change=_invalidate_enhanced_output,
        )
        st.success("True-Peak Limiter: -0.5 dBFS")
    st.caption(
        "Tier-2 loads local end-to-end waveform ONNX/TorchScript exports. "
        "Raw training checkpoints or feature/state graphs are rejected and audited as fallbacks."
    )

    if st.button(
        "Run Voice Enhancement",
        type="primary",
        width="stretch",
        key="voice_run_enhancement",
    ):
        tier = "tier2" if processing_label == TIER_2_LABEL else "tier1"
        _invalidate_enhanced_output()
        with st.spinner("Enhancing speech audio..."):
            result = enhance_voice(
                (
                    audio_source
                    if audio_source is not None
                    else np.asarray(raw_audio, dtype=np.float32)
                ),
                sample_rate,
                processing_mode=tier,
                mode="file_offline",
                attenuation_db=float(attenuation_db),
                boost_db=float(boost_db),
                device_ctx=cached_device_context(),
                dataset_version=source_name,
            )
        st.session_state["voice_transform_result"] = result
        st.session_state["voice_module_result"] = result.module_result
        st.session_state["voice_execution_mode"] = "file_offline"
        if result.transformed_audio is not None:
            st.session_state["voice_enhanced_audio"] = result.transformed_audio
            st.session_state["voice_enhanced_sample_rate"] = result.sample_rate
        if result.status == ExecutionStatus.FAILED:
            st.error(result.error_message or "Voice enhancement failed.")
        else:
            st.rerun()

    result = st.session_state.get("voice_transform_result")
    if result is not None and result.status == ExecutionStatus.PARTIAL:
        denoiser_reason = result.metadata.get("denoiser_fallback_reason")
        bandwidth_reason = result.metadata.get("bandwidth_fallback_reason")
        if denoiser_reason:
            st.warning(
                f"DeepFilterNet3 could not run ({denoiser_reason}); "
                "Tier-1 spectral gating produced the denoised stage."
            )
        if bandwidth_reason:
            st.info(
                f"MP-SENet / VoiceFixer could not run ({bandwidth_reason}); "
                "phase-aligned harmonic reconstruction produced the 48 kHz stage."
            )
    elif result is not None and result.status == ExecutionStatus.SUCCESS:
        st.success(
            f"Enhancement completed in {result.execution_time_ms:.1f} ms "
            f"at {result.sample_rate / 1000:g} kHz."
        )

    enhanced = st.session_state.get("voice_enhanced_audio")
    enhanced_sr = int(st.session_state.get("voice_enhanced_sample_rate", sample_rate))
    raw_column, clean_column = st.columns(2)
    with raw_column:
        st.markdown("#### Original Raw")
        st.audio(player_wav(raw_audio, sample_rate), format="audio/wav")
        st.caption(f"Mono · {sample_rate / 1000:g} kHz")
    with clean_column:
        st.markdown("#### Clean Boosted Output")
        if enhanced is None:
            st.info("Run enhancement to create the clean A/B output.")
        else:
            st.audio(player_wav(enhanced, enhanced_sr), format="audio/wav")
            true_peak = (
                result.module_result.metrics.get("booster_true_peak_dbfs")
                if result is not None and result.module_result is not None
                else None
            )
            peak_text = (
                f" · True Peak {float(true_peak):.2f} dBFS"
                if true_peak is not None
                else ""
            )
            st.caption(f"Mono · {enhanced_sr / 1000:g} kHz{peak_text}")


render_neural_enhancer_tab = render_tab_neural_enhancer
