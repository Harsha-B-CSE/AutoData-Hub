"""Master layout for the Streamlit voice enhancement workspace."""

from __future__ import annotations

import streamlit as st

from modules.voice.audio_loader import load_audio_with_metadata
from ui.tabs_voice.common import audio_fingerprint, uploaded_audio_name
from ui.tabs_voice.tab_custom_tuner import render_tab_custom_tuner
from ui.tabs_voice.tab_neural_enhancer import render_tab_neural_enhancer
from ui.tabs_voice.tab_telemetry import render_tab_telemetry
from ui.tabs_voice.tab_voice_export import render_tab_voice_export


def render_voice_app(uploaded_file) -> None:
    """Route between static file processing and live WebRTC enhancement."""

    with st.sidebar:
        input_source = st.radio(
            "Voice Input Source",
            options=[
                "📁 Audio File Upload",
                "🎤 Live Microphone Stream (WebRTC)",
            ],
            horizontal=False,
            key="voice_input_source",
        )
    if input_source == "🎤 Live Microphone Stream (WebRTC)":
        from ui.tabs_voice.tab_live_webrtc import render_tab_live_webrtc

        st.session_state["voice_execution_mode"] = "live_stream"
        render_tab_live_webrtc()
        return

    st.session_state["voice_execution_mode"] = "file_offline"
    if uploaded_file is None:
        st.info("Upload a WAV, MP3, FLAC, OGG, or M4A file from the sidebar.")
        return

    try:
        fingerprint = audio_fingerprint(uploaded_file)
        source_name = uploaded_audio_name(uploaded_file)
        if st.session_state.get("voice_source_fingerprint") != fingerprint:
            loaded = load_audio_with_metadata(uploaded_file, target_sr=16_000)
            st.session_state["voice_source_fingerprint"] = fingerprint
            st.session_state["voice_raw_audio"] = loaded.waveform
            st.session_state["voice_raw_sample_rate"] = loaded.sample_rate
            st.session_state["voice_source_metadata"] = {
                "filename": loaded.filename,
                "decoder": loaded.decoder,
                "source_sample_rate": loaded.source_sample_rate,
                "source_channels": loaded.source_channels,
            }
            for key in (
                "voice_enhanced_audio",
                "voice_enhanced_sample_rate",
                "voice_transform_result",
                "voice_module_result",
                "voice_tuned_audio",
                "voice_tuned_sample_rate",
            ):
                st.session_state.pop(key, None)
    except (TypeError, ValueError) as exc:
        st.error(f"Audio validation failed: {exc}")
        return

    raw_audio = st.session_state["voice_raw_audio"]
    raw_sample_rate = int(st.session_state["voice_raw_sample_rate"])
    source_metadata = st.session_state["voice_source_metadata"]
    duration = raw_audio.size / max(raw_sample_rate, 1)
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Duration", f"{duration:.2f} s")
    metric_b.metric("Source Rate", f"{source_metadata['source_sample_rate'] / 1000:g} kHz")
    metric_c.metric("Channels", source_metadata["source_channels"])
    metric_d.metric("Decoder", str(source_metadata["decoder"]).upper())

    tab_telemetry, tab_enhance, tab_tuner, tab_export = st.tabs(
        [
            "🔍 Telemetry",
            "🧠 Neural Enhancer & Booster",
            "🎛️ Smart Tuner",
            "📄 Export Studio",
        ]
    )
    with tab_enhance:
        render_tab_neural_enhancer(
            raw_audio,
            raw_sample_rate,
            source_name,
            audio_source=uploaded_file,
        )
    with tab_tuner:
        tuner_audio = st.session_state.get("voice_enhanced_audio", raw_audio)
        tuner_rate = int(
            st.session_state.get("voice_enhanced_sample_rate", raw_sample_rate)
        )
        render_tab_custom_tuner(
            tuner_audio,
            tuner_rate,
            persist_for_export="voice_enhanced_audio" in st.session_state,
        )
    with tab_telemetry:
        if "voice_tuned_audio" in st.session_state:
            metrics_audio = st.session_state["voice_tuned_audio"]
            metrics_rate = int(st.session_state["voice_tuned_sample_rate"])
            metrics_source = "Smart Tuner output"
            recompute_metrics = True
        elif "voice_enhanced_audio" in st.session_state:
            metrics_audio = st.session_state["voice_enhanced_audio"]
            metrics_rate = int(st.session_state["voice_enhanced_sample_rate"])
            metrics_source = "Clean boosted output"
            recompute_metrics = False
        else:
            metrics_audio = raw_audio
            metrics_rate = raw_sample_rate
            metrics_source = "Raw decoded input"
            recompute_metrics = True
        render_tab_telemetry(
            raw_audio,
            raw_sample_rate,
            st.session_state.get("voice_module_result"),
            mode=st.session_state.get("voice_execution_mode", "file_offline"),
            metrics_audio_signal=metrics_audio,
            metrics_sample_rate=metrics_rate,
            metrics_source=metrics_source,
            recompute_metrics=recompute_metrics,
        )
    with tab_export:
        export_audio = st.session_state.get(
            "voice_tuned_audio",
            st.session_state.get("voice_enhanced_audio", raw_audio),
        )
        export_rate = int(
            st.session_state.get(
                "voice_tuned_sample_rate",
                st.session_state.get("voice_enhanced_sample_rate", raw_sample_rate),
            )
        )
        render_tab_voice_export(
            export_audio,
            export_rate,
            source_name,
            is_enhanced=(
                "voice_tuned_audio" in st.session_state
                or "voice_enhanced_audio" in st.session_state
            ),
        )
