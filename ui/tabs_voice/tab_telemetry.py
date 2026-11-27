"""Hardware and speech-quality telemetry presentation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.signal import spectrogram
import streamlit as st

from core.schema_inference import ModuleResult
from modules.voice.audio_metrics import evaluate_audio_telemetry
from ui.tabs_voice.common import cached_device_context, player_wav
from ui.utils_ux import render_tab_guidance_header


def _hardware_metrics(module_result: ModuleResult | None) -> dict:
    if module_result is not None:
        metrics = module_result.metrics
        if "device" in metrics:
            return metrics
    return cached_device_context()


def _quality_metrics(
    audio_signal: np.ndarray,
    sample_rate: int,
    module_result: ModuleResult | None,
    *,
    recompute: bool = False,
) -> dict:
    if (
        not recompute
        and module_result is not None
        and "snr_db" in module_result.metrics
    ):
        return module_result.metrics
    return evaluate_audio_telemetry(audio_signal, sample_rate)


def render_tab_telemetry(
    audio_signal: np.ndarray,
    sample_rate: int = 16_000,
    module_result: ModuleResult | None = None,
    mode: str = "file_offline",
    metrics_audio_signal: np.ndarray | None = None,
    metrics_sample_rate: int | None = None,
    metrics_source: str = "Raw input",
    recompute_metrics: bool = False,
) -> None:
    """Render hardware status, metric cards, waveform, and an audio player."""
    render_tab_guidance_header(
        title="Voice Telemetry & Diagnostics",
        objective="Measure speech clarity, loudness, and hardware acceleration status.",
        methodology=(
            "Computes SNR, SI-SDR, estimated PESQ/STOI, crest factor, time-domain "
            "waveform, and FFT spectrogram."
        ),
        deliverable="Quantitative acoustic scorecard and visual spectrogram inspection.",
    )
    audio = np.asarray(audio_signal, dtype=np.float32).reshape(-1)
    hardware = _hardware_metrics(module_result)
    effective_mode = (
        str(module_result.metrics.get("mode", mode))
        if module_result is not None
        else mode
    )
    if effective_mode == "live_stream":
        badge_text = "🎤 Live Stream Latency Guard Active"
    elif hardware.get("device") == "cuda":
        badge_text = "⚡ CUDA GPU | Mode: Offline Full-Precision"
    else:
        badge_text = "💻 CPU Host | Mode: Offline Full-Precision"
    st.markdown(
        f"<div style='display:inline-block;padding:.45rem .8rem;border-radius:999px;"
        f"background:#172033;border:1px solid #334155;font-weight:700'>{badge_text}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Execution provider: {hardware.get('execution_provider', 'unavailable')} · "
        f"VRAM allocated: {float(hardware.get('vram_allocated_mb', 0.0)):.1f} MB"
    )

    metrics_audio = (
        audio
        if metrics_audio_signal is None
        else np.asarray(metrics_audio_signal, dtype=np.float32).reshape(-1)
    )
    metrics_rate = int(metrics_sample_rate or sample_rate)
    metrics = _quality_metrics(
        metrics_audio,
        metrics_rate,
        module_result,
        recompute=recompute_metrics,
    )
    st.caption(f"Telemetry signal: {metrics_source}")
    snr, sisdr, pesq, stoi, crest = st.columns(5)
    snr.metric("SNR (dB)", f"{float(metrics.get('snr_db', 0.0)):.2f}")
    sisdr.metric("SI-SDR (dB)", f"{float(metrics.get('si_sdr_db', 0.0)):.2f}")
    pesq.metric("Estimated PESQ", f"{float(metrics.get('pesq_score', 1.0)):.2f}")
    stoi.metric("STOI Index", f"{float(metrics.get('stoi_index', 0.0)):.3f}")
    crest.metric(
        "Crest Factor (dB)",
        f"{float(metrics.get('crest_factor_db', 0.0)):.2f}",
    )
    st.caption(
        f"PESQ source: {metrics.get('pesq_method', 'pesq_heuristic_surrogate')} · "
        f"STOI source: {metrics.get('stoi_method', 'stoi_correlation_surrogate')}"
    )

    st.subheader("Complete Raw Input Waveform")
    if audio.size:
        stride = max(1, audio.size // 8_000)
        samples = audio[::stride]
        seconds = np.arange(0, audio.size, stride, dtype=np.float64) / sample_rate
        figure = go.Figure(
            go.Scattergl(
                x=seconds,
                y=samples,
                mode="lines",
                line={"color": "#8B5CF6", "width": 1},
                hovertemplate="%{x:.3f} s<br>%{y:.4f}<extra></extra>",
            )
        )
        figure.update_layout(
            height=280,
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            yaxis_range=[-1.05, 1.05],
            template="plotly_dark",
        )
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
        st.audio(player_wav(audio, sample_rate), format="audio/wav")
        if audio.size >= 32:
            nperseg = min(1_024, max(32, 2 ** int(np.floor(np.log2(audio.size)))))
            frequencies, times, magnitude = spectrogram(
                audio,
                fs=sample_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=(3 * nperseg) // 4,
                scaling="spectrum",
                mode="magnitude",
            )
            time_stride = max(1, times.size // 500)
            magnitude_db = 20.0 * np.log10(magnitude[:, ::time_stride] + 1e-8)
            spectrogram_figure = go.Figure(
                go.Heatmap(
                    x=times[::time_stride],
                    y=frequencies,
                    z=magnitude_db,
                    colorscale="Magma",
                    colorbar={"title": "dB"},
                    hovertemplate="%{x:.3f} s<br>%{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
                )
            )
            spectrogram_figure.update_layout(
                title="FFT Spectrogram Preview",
                height=340,
                margin={"l": 20, "r": 20, "t": 45, "b": 20},
                xaxis_title="Time (seconds)",
                yaxis_title="Frequency (Hz)",
                template="plotly_dark",
            )
            st.plotly_chart(
                spectrogram_figure,
                width="stretch",
                config={"displaylogo": False},
            )
    else:
        st.info("The decoded audio stream is empty.")


render_telemetry_tab = render_tab_telemetry
