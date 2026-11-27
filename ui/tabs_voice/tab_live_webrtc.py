"""Low-latency WebRTC microphone enhancement for the voice workspace."""

from __future__ import annotations

import threading
from time import perf_counter_ns
from typing import Any, Dict

import numpy as np
import streamlit as st

from modules.voice.booster import StreamingTruePeakLimiter, apply_quality_booster
from modules.voice.denoiser_sota import StreamingSpectralGate
from ui.utils_ux import render_tab_guidance_header

try:
    import av
    from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = None
except ImportError as exc:  # keeps static file processing available without the extra
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = str(exc)

    class AudioProcessorBase:  # type: ignore[no-redef]
        pass

    WebRtcMode = None  # type: ignore[assignment]
    webrtc_streamer = None  # type: ignore[assignment]


def _audio_frame_to_mono_float32(frame: av.AudioFrame) -> tuple[np.ndarray, int]:
    array = np.asarray(frame.to_ndarray())
    channel_count = max(1, len(frame.layout.channels))
    if np.issubdtype(array.dtype, np.integer):
        limits = np.iinfo(array.dtype)
        if limits.min == 0:
            midpoint = float(limits.max + 1) / 2.0
            normalized = (array.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(limits.min), limits.max + 1))
            normalized = array.astype(np.float32) / scale
    else:
        normalized = np.nan_to_num(
            array.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0
        )

    if frame.format.is_planar:
        channels = normalized.reshape(channel_count, -1)
    else:
        interleaved = normalized.reshape(-1)
        usable = interleaved.size - (interleaved.size % channel_count)
        channels = interleaved[:usable].reshape(-1, channel_count).T
    mono = np.mean(channels, axis=0, dtype=np.float64).astype(np.float32)
    sample_rate = int(frame.sample_rate or 48_000)
    return np.ascontiguousarray(mono), sample_rate


def _mono_float32_to_audio_frame(
    audio: np.ndarray,
    sample_rate: int,
    source_frame: av.AudioFrame | None = None,
) -> av.AudioFrame:
    mono = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = np.round(mono * 32767.0).astype(np.int16).reshape(1, -1)
    output = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    output.sample_rate = int(sample_rate)
    if source_frame is not None:
        output.pts = source_frame.pts
        if source_frame.time_base is not None:
            output.time_base = source_frame.time_base
    return output


class LiveAudioDenoiseProcessor(AudioProcessorBase):
    """Process browser microphone frames through the guarded live DSP path."""

    _MISS_LIMIT = 3
    _RECOVERY_FRAMES = 32

    def __init__(self, attenuation_db: float = 12.0, boost_db: float = 0.0):
        self.attenuation_db = float(np.clip(attenuation_db, 0.0, 30.0))
        self.boost_db = float(np.clip(boost_db, 0.0, 6.0))
        self._lock = threading.Lock()
        self._processing_lock = threading.Lock()
        self._frames_processed = 0
        self._frames_failed = 0
        self._deadline_misses = 0
        self._miss_streak = 0
        self._guard_remaining = 0
        self._degraded_frames = 0
        self._last_degraded = False
        self._last_residual: float | None = None
        self._last_latency_ms = 0.0
        self._maximum_latency_ms = 0.0
        self._last_sample_rate = 0
        self._last_frame_duration_ms = 0.0
        self._sample_rate = 0
        self._spectral_gate: StreamingSpectralGate | None = None
        self._true_peak_limiter: StreamingTruePeakLimiter | None = None

    def update_settings(self, attenuation_db: float, boost_db: float) -> None:
        with self._lock:
            self.attenuation_db = float(np.clip(attenuation_db, 0.0, 30.0))
            self.boost_db = float(np.clip(boost_db, 0.0, 6.0))

    def _ensure_state(self, sample_rate: int) -> None:
        if sample_rate == self._sample_rate:
            return
        self._sample_rate = sample_rate
        self._spectral_gate = StreamingSpectralGate(
            sample_rate=sample_rate,
            attenuation_db=self.attenuation_db,
        )
        self._true_peak_limiter = StreamingTruePeakLimiter(
            sample_rate=sample_rate,
            target_peak_dbfs=-0.5,
            lookahead_ms=1.0,
        )
        self._last_residual = None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        started = perf_counter_ns()
        if not frame.samples:
            return frame
        sample_rate = int(frame.sample_rate or 48_000)
        frame_duration_ms = frame.samples * 1_000.0 / sample_rate
        budget_ns = min(20_000_000, frame.samples * 1_000_000_000 // sample_rate)
        with self._processing_lock:
            degraded = self._guard_remaining > 0
            failed = False
            try:
                mono, _ = _audio_frame_to_mono_float32(frame)
                self._ensure_state(sample_rate)
                assert self._spectral_gate is not None
                assert self._true_peak_limiter is not None
                if degraded:
                    # A memoryless soft knee fits the limiter's certified cheap path.
                    safe = np.float32(self._true_peak_limiter.safe_input_peak)
                    enhanced = safe * np.tanh(np.clip(mono, -1.0, 1.0) / safe)
                else:
                    with self._lock:
                        self._spectral_gate.attenuation_db = self.attenuation_db
                        boost_db = self.boost_db
                    denoised = self._spectral_gate.process(mono)
                    enhanced = apply_quality_booster(
                        denoised, sr=sample_rate, boost_db=boost_db,
                        normalize_loudness=False, apply_limiter=False,
                    )
                if degraded != self._last_degraded and self._last_residual is not None:
                    # Decay the wet/dry offset over 1 ms without replaying the last sample.
                    n = min(mono.size, max(1, sample_rate // 1_000))
                    jump = self._last_residual - (enhanced[0] - mono[0])
                    enhanced[:n] += jump * np.linspace(1, 0, n, dtype=np.float32)
                self._last_residual = float(enhanced[-1] - mono[-1])
                limited = self._true_peak_limiter.process(enhanced)
                if limited.size < frame.samples:  # fixed lookahead, only at stream startup
                    limited = np.pad(limited, (frame.samples - limited.size, 0))
                output = _mono_float32_to_audio_frame(limited, sample_rate, frame)
            except Exception:
                # Fail closed. Do not send unbounded input or replay stale pending samples.
                failed = True
                self._sample_rate = 0
                output = _mono_float32_to_audio_frame(
                    np.zeros(frame.samples, dtype=np.float32), sample_rate, frame
                )
            with self._lock:
                elapsed_ns = perf_counter_ns() - started
                elapsed_ms = elapsed_ns / 1_000_000
                missed = elapsed_ns >= budget_ns
                self._deadline_misses += missed
                self._frames_failed += failed
                if degraded:
                    self._guard_remaining -= 1
                    self._degraded_frames += 1
                else:
                    self._miss_streak = self._miss_streak + 1 if missed else 0
                    if self._miss_streak >= self._MISS_LIMIT or failed:
                        self._guard_remaining = self._RECOVERY_FRAMES
                self._last_degraded = degraded
                self._frames_processed += 1
                self._last_latency_ms = elapsed_ms
                self._maximum_latency_ms = max(self._maximum_latency_ms, elapsed_ms)
                self._last_sample_rate = sample_rate
                self._last_frame_duration_ms = frame_duration_ms
            return output

    @property
    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "frames_processed": self._frames_processed,
                "frames_failed": self._frames_failed,
                "deadline_misses": self._deadline_misses,
                "degraded_frames": self._degraded_frames,
                "degraded_remaining": self._guard_remaining,
                "last_degraded": self._last_degraded,
                "last_latency_ms": self._last_latency_ms,
                "maximum_latency_ms": self._maximum_latency_ms,
                "sample_rate": self._last_sample_rate,
                "frame_duration_ms": self._last_frame_duration_ms,
                "latency_budget_ms": (
                    min(20.0, self._last_frame_duration_ms)
                    if self._frames_processed
                    else 0.0
                ),
                "latency_budget_met": (
                    self._last_latency_ms < min(20.0, self._last_frame_duration_ms)
                    if self._frames_processed
                    else None
                ),
            }


def render_tab_live_webrtc() -> None:
    """Render browser microphone capture and processed audio sendback."""
    render_tab_guidance_header(
        title="Live Microphone Stream (WebRTC)",
        objective="Keep DSP compute below the frame duration, capped at 20 ms.",
        methodology=(
            "Captures 10-20 ms browser PCM av.AudioFrame buffers via streamlit-webrtc, "
            "applying spectral gating and a lookahead -0.5 dBTP ceiling. Repeated "
            "compute misses temporarily bypass the gate and boost, never peak protection."
        ),
        deliverable="Real-time cleaned return audio stream and live DSP budget telemetry.",
    )
    st.subheader("🎤 Real-Time Microphone Enhancement")
    st.caption(
        "Browser audio is downmixed to mono and processed in 10-20 ms frames with "
        "spectral gating, soft-knee loudness control, and a -0.5 dBTP ceiling."
    )
    if not WEBRTC_AVAILABLE:
        st.error(
            "Live microphone support is unavailable. Install av and streamlit-webrtc "
            f"from requirements.txt. ({WEBRTC_IMPORT_ERROR})"
        )
        return

    control_a, control_b = st.columns(2)
    attenuation_db = control_a.slider(
        "Live Noise Attenuation (dB)",
        min_value=0.0,
        max_value=30.0,
        value=12.0,
        step=1.0,
        key="voice_live_attenuation_db",
    )
    boost_db = control_b.slider(
        "Live Output Boost (dB)",
        min_value=0.0,
        max_value=6.0,
        value=0.0,
        step=0.5,
        key="voice_live_boost_db",
    )

    try:
        context = webrtc_streamer(
            key="autodata-voice-live-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": False, "audio": True},
            audio_processor_factory=lambda: LiveAudioDenoiseProcessor(
                attenuation_db=attenuation_db,
                boost_db=boost_db,
            ),
            async_processing=False,
            sendback_audio=True,
            audio_html_attrs={"autoPlay": True, "controls": True},
        )
    except (AttributeError, RuntimeError) as exc:
        st.warning(
            "WebRTC capture requires a live Streamlit server session; the static "
            f"test renderer cannot open a peer connection. ({type(exc).__name__})"
        )
        return
    st.info(
        "Select START, grant microphone permission, and monitor the processed "
        "return audio. Headphones prevent acoustic feedback."
    )
    processor = context.audio_processor
    if processor is not None:
        processor.update_settings(attenuation_db, boost_db)

        @st.fragment(run_every=1.0)
        def render_live_telemetry() -> None:
            telemetry = processor.telemetry
            if not telemetry["frames_processed"]:
                st.caption("Waiting for the first audio frame.")
                return
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Frames", telemetry["frames_processed"])
            metric_b.metric(
                "Last DSP Compute", f"{telemetry['last_latency_ms']:.1f} ms"
            )
            metric_c.metric("DSP Deadline Misses", telemetry["deadline_misses"])
            st.caption(
                f"Last mode: {'guarded' if telemetry['last_degraded'] else 'full'}. "
                f"Budget: < {telemetry['latency_budget_ms']:g} ms; "
                f"guarded frames: {telemetry['degraded_frames']}; muted on error: "
                f"{telemetry['frames_failed']}. DSP compute is not round-trip latency: "
                "it excludes lookahead, capture, network, jitter buffer, and playback."
            )

        render_live_telemetry()
