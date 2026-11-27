"""Orchestration for static and streaming dual-tier voice enhancement."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
from scipy.signal import firwin

from core.schema_inference import (
    ExecutionStatus,
    ModuleResult,
    TransformationAction,
    TransformResult,
)
from modules.voice.audio_loader import (
    downmix_to_mono,
    float32_to_pcm_bytes,
    load_audio_with_metadata,
    pcm_bytes_to_float32,
)
from modules.voice.audio_metrics import evaluate_audio_telemetry
from modules.voice.bandwidth_extension import (
    apply_voicefixer_sota,
)
from modules.voice.booster import (
    StreamingTruePeakLimiter,
    analyze_audio_levels,
    apply_quality_booster,
    apply_true_peak_limiter,
)
from modules.voice.denoiser_sota import (
    StreamingSpectralGate,
    apply_deepfilternet3_sota,
    apply_spectral_gating,
    get_deepfilternet3_status,
)
from modules.voice.device_manager import hardware_telemetry, resolve_execution_device
from modules.voice.tuner import apply_parametric_eq


BASELINE_SAMPLE_RATE = 16_000
BANDWIDTH_EXTENSION_SAMPLE_RATE = 48_000


@dataclass
class VoiceTransformResult(TransformResult):
    """Audio-specialized TransformResult that leaves DataFrame data unset."""

    transformed_audio: np.ndarray | None = None
    sample_rate: int = BASELINE_SAMPLE_RATE
    step_queue: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    execution_timing: Dict[str, float] = field(default_factory=dict)
    module_result: ModuleResult | None = None


AudioTransformResult = VoiceTransformResult


def _tier_name(processing_mode: str | int) -> str:
    normalized = str(processing_mode).strip().lower().replace("_", "-")
    if normalized in {"2", "tier2", "tier-2", "neural", "sota"} or "tier-2" in normalized:
        return "tier2"
    if normalized in {"1", "tier1", "tier-1", "classical", "spectral"} or "tier-1" in normalized:
        return "tier1"
    raise ValueError("processing_mode must select Tier-1 or Tier-2.")


def _execution_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "offline": "file_offline",
        "file": "file_offline",
        "file_offline": "file_offline",
        "live": "live_stream",
        "stream": "live_stream",
        "live_stream": "live_stream",
    }
    if normalized not in aliases:
        raise ValueError("mode must be 'file_offline' or 'live_stream'.")
    return aliases[normalized]


def _step(
    queue: list[dict[str, Any]],
    timing: dict[str, float],
    name: str,
    started: float,
    **details: Any,
) -> None:
    elapsed = (time.perf_counter() - started) * 1_000.0
    timing[name] = elapsed
    queue.append({"step": name, "elapsed_ms": elapsed, "status": "complete", **details})


def enhance_voice(
    audio_source: Any,
    sample_rate: int | None = None,
    *,
    processing_mode: str | int = "tier1",
    mode: str = "file_offline",
    attenuation_db: float = 20.0,
    enable_bandwidth_extension: bool = False,
    boost_db: float = 3.0,
    low_gain_db: float = 0.0,
    mid_gain_db: float = 0.0,
    high_gain_db: float = 0.0,
    device_ctx: dict | None = None,
    dataset_version: str = "audio_upload",
) -> VoiceTransformResult:
    """Decode and enhance a static audio source without importing Streamlit."""

    total_started = time.perf_counter()
    queue: list[dict[str, Any]] = []
    timing: dict[str, float] = {}
    audit_log: list[str] = []
    actions: list[TransformationAction] = []
    context: dict[str, Any] = {}
    try:
        requested_tier = _tier_name(processing_mode)
        execution_mode = _execution_mode(mode)
        effective_tier = "tier2" if execution_mode == "live_stream" else requested_tier
        attenuation_db = float(np.clip(float(attenuation_db), 0.0, 30.0))
        boost_db = float(np.clip(float(boost_db), 0.0, 12.0))

        step_started = time.perf_counter()
        context = dict(device_ctx or resolve_execution_device())
        if execution_mode == "live_stream":
            context["require_onnx"] = True
        serializable_hardware = hardware_telemetry(context)
        _step(
            queue,
            timing,
            "hardware_resolution",
            step_started,
            device=serializable_hardware["device"],
            execution_provider=serializable_hardware["execution_provider"],
            mode=execution_mode,
        )

        step_started = time.perf_counter()
        if (
            isinstance(audio_source, tuple)
            and len(audio_source) == 2
            and isinstance(audio_source[0], np.ndarray)
        ):
            audio_source, tuple_rate = audio_source
            sample_rate = int(tuple_rate)
        loaded = load_audio_with_metadata(
            audio_source,
            BASELINE_SAMPLE_RATE,
            sample_rate=sample_rate,
        )
        raw_audio = loaded.waveform
        _step(
            queue,
            timing,
            "ingestion",
            step_started,
            decoder=loaded.decoder,
            source_sample_rate=loaded.source_sample_rate,
            source_channels=loaded.source_channels,
        )
        audit_log.append(
            f"Decoded {loaded.filename} via {loaded.decoder}; downmixed to mono and resampled to 16000 Hz."
        )
        actions.append(
            TransformationAction(
                step_id=1,
                module="voice",
                action_type="decode_downmix_resample",
                target_columns=[],
                parameters={
                    "source_sample_rate": loaded.source_sample_rate,
                    "source_channels": loaded.source_channels,
                    "target_sample_rate": BASELINE_SAMPLE_RATE,
                    "decoder": loaded.decoder,
                },
            )
        )

        step_started = time.perf_counter()
        fallback_reasons: list[str] = []
        fallback_reason = None
        denoiser_fallback_reason = None
        bandwidth_fallback_reason = None
        if effective_tier == "tier2":
            neural_result = apply_deepfilternet3_sota(
                raw_audio,
                BASELINE_SAMPLE_RATE,
                attenuation_db=attenuation_db,
                device_ctx=context,
            )
            enhanced = np.asarray(neural_result["audio"], dtype=np.float32)
            tier_used = str(neural_result["tier_used"])
            runtime = str(neural_result["runtime"])
            fallback_reason = neural_result.get("fallback_reason")
            denoiser_fallback_reason = fallback_reason
            if fallback_reason:
                fallback_reasons.append(f"DeepFilterNet3:{fallback_reason}")
        else:
            enhanced = apply_spectral_gating(
                raw_audio,
                BASELINE_SAMPLE_RATE,
                attenuation_db=attenuation_db,
            )
            tier_used = "tier1"
            runtime = "scipy_stft"
        _step(
            queue,
            timing,
            "denoising",
            step_started,
            tier_requested=requested_tier,
            tier_effective=effective_tier,
            tier_used=tier_used,
            runtime=runtime,
            fallback_reason=fallback_reason,
        )
        actions.append(
            TransformationAction(
                step_id=2,
                module="voice",
                action_type=(
                    "deepfilternet3_enhancement" if tier_used == "tier2" else "spectral_gate"
                ),
                target_columns=[],
                parameters={
                    "attenuation_db": attenuation_db,
                    "tier_requested": requested_tier,
                    "tier_effective": effective_tier,
                    "tier_used": tier_used,
                    "runtime": runtime,
                    "fallback_reason": fallback_reason,
                },
            )
        )
        if fallback_reason:
            audit_log.append(
                f"Tier-2 fallback engaged ({fallback_reason}); Tier-1 spectral gating completed."
            )
        else:
            audit_log.append(f"{tier_used} denoising completed via {runtime}.")

        output_sample_rate = BASELINE_SAMPLE_RATE
        if execution_mode == "file_offline":
            step_started = time.perf_counter()
            enhanced = apply_parametric_eq(
                enhanced,
                BASELINE_SAMPLE_RATE,
                low_gain_db=low_gain_db,
                mid_gain_db=mid_gain_db,
                high_gain_db=high_gain_db,
            )
            _step(
                queue,
                timing,
                "parametric_eq",
                step_started,
                sections=5,
                low_gain_db=float(low_gain_db),
                mid_gain_db=float(mid_gain_db),
                high_gain_db=float(high_gain_db),
            )
            actions.append(
                TransformationAction(
                    step_id=3,
                    module="voice",
                    action_type="five_band_parametric_eq",
                    target_columns=[],
                    parameters={
                        "low_gain_db": float(low_gain_db),
                        "mid_gain_db": float(mid_gain_db),
                        "high_gain_db": float(high_gain_db),
                        "sections": 5,
                    },
                )
            )

            step_started = time.perf_counter()
            bandwidth_result = apply_voicefixer_sota(
                enhanced,
                BASELINE_SAMPLE_RATE,
                BANDWIDTH_EXTENSION_SAMPLE_RATE,
                device_ctx=context,
            )
            enhanced = np.asarray(bandwidth_result["audio"], dtype=np.float32)
            bandwidth_runtime = str(bandwidth_result["runtime"])
            bandwidth_fallback = bandwidth_result.get("fallback_reason")
            bandwidth_fallback_reason = bandwidth_fallback
            if bandwidth_fallback:
                fallback_reasons.append(
                    f"MP-SENet/VoiceFixer:{bandwidth_fallback}"
                )
            output_sample_rate = BANDWIDTH_EXTENSION_SAMPLE_RATE
            _step(
                queue,
                timing,
                "bandwidth_extension",
                step_started,
                source_sample_rate=BASELINE_SAMPLE_RATE,
                target_sample_rate=output_sample_rate,
                engine=bandwidth_runtime,
                fallback_reason=bandwidth_fallback,
            )
            actions.append(
                TransformationAction(
                    step_id=4,
                    module="voice",
                    action_type="bandwidth_extension",
                    target_columns=[],
                    parameters={
                        "source_sample_rate": BASELINE_SAMPLE_RATE,
                        "target_sample_rate": output_sample_rate,
                        "phase_alignment": True,
                        "runtime": bandwidth_runtime,
                        "fallback_reason": bandwidth_fallback,
                    },
                )
            )
            if bandwidth_fallback:
                audit_log.append(
                    "Neural bandwidth extension fallback engaged "
                    f"({bandwidth_fallback}); phase-aligned harmonic extension completed."
                )
            else:
                audit_log.append(
                    f"Bandwidth extension completed via {bandwidth_runtime} at 48000 Hz."
                )

            step_started = time.perf_counter()
            levels_before_boost = analyze_audio_levels(enhanced, output_sample_rate)
            enhanced = apply_quality_booster(
                enhanced,
                output_sample_rate,
                boost_db=boost_db,
                target_peak_dbfs=-0.5,
            )
            quality_levels = analyze_audio_levels(enhanced, output_sample_rate)
            _step(
                queue,
                timing,
                "quality_booster",
                step_started,
                target_lufs=-14.0,
                boost_db=boost_db,
                target_peak_dbfs=-0.5,
                input_lufs=levels_before_boost["lufs"],
                output_lufs=quality_levels["lufs"],
                output_true_peak_dbfs=quality_levels["true_peak_dbfs"],
            )
            actions.append(
                TransformationAction(
                    step_id=5,
                    module="voice",
                    action_type="ebu_r128_quality_booster",
                    target_columns=[],
                    parameters={
                        "target_lufs": -14.0,
                        "boost_db": boost_db,
                        "target_peak_dbfs": -0.5,
                        "saturation": "tanh",
                    },
                )
            )
            audit_log.append(
                "Applied EBU R128 normalization, tanh saturation, and -0.5 dBFS true-peak limiting."
            )
        else:
            step_started = time.perf_counter()
            enhanced = apply_true_peak_limiter(
                enhanced,
                BASELINE_SAMPLE_RATE,
                target_peak_dbfs=-0.5,
                lookahead_ms=1.0,
            )
            quality_levels = analyze_audio_levels(enhanced, BASELINE_SAMPLE_RATE)
            _step(
                queue,
                timing,
                "true_peak_limiter",
                step_started,
                lookahead_ms=1.0,
                target_peak_dbfs=-0.5,
                multiband_eq=False,
                bandwidth_extension=False,
            )
            actions.append(
                TransformationAction(
                    step_id=3,
                    module="voice",
                    action_type="live_true_peak_limiter",
                    target_columns=[],
                    parameters={"lookahead_ms": 1.0, "target_peak_dbfs": -0.5},
                )
            )
            audit_log.append(
                "Live latency guard applied the 1 ms true-peak limiter; EQ and bandwidth extension were bypassed."
            )

        fallback_reason = "; ".join(fallback_reasons) or None

        step_started = time.perf_counter()
        raw_metrics = evaluate_audio_telemetry(raw_audio, BASELINE_SAMPLE_RATE)
        enhanced_metrics = evaluate_audio_telemetry(enhanced, output_sample_rate)
        _step(queue, timing, "telemetry", step_started)

        total_ms = (time.perf_counter() - total_started) * 1_000.0
        timing["total"] = total_ms
        metrics: dict[str, Any] = {
            **enhanced_metrics,
            **serializable_hardware,
            **{f"booster_{key}": value for key, value in quality_levels.items()},
            "mode": execution_mode,
            "latency_guard_active": execution_mode == "live_stream",
            "latency_budget_ms": 20.0 if execution_mode == "live_stream" else None,
            "tier_requested": requested_tier,
            "tier_effective": effective_tier,
            "tier_used": tier_used,
            "runtime": runtime,
            "fallback_reason": fallback_reason,
            "denoiser_fallback_reason": denoiser_fallback_reason,
            "bandwidth_fallback_reason": bandwidth_fallback_reason,
            "attenuation_db": attenuation_db,
            "multiband_eq": execution_mode == "file_offline",
            "bandwidth_extension": execution_mode == "file_offline",
            "bandwidth_extension_requested": bool(enable_bandwidth_extension),
            "boost_db": boost_db if execution_mode == "file_offline" else 0.0,
            "input_sample_rate": loaded.source_sample_rate,
            "output_sample_rate": output_sample_rate,
            "input_channels": loaded.source_channels,
            "decoder": loaded.decoder,
            "execution_time_ms": total_ms,
            "raw": raw_metrics,
            "enhanced": enhanced_metrics,
        }
        issues = []
        status = ExecutionStatus.SUCCESS
        if fallback_reason:
            status = ExecutionStatus.PARTIAL
            if denoiser_fallback_reason:
                issues.append(
                    "DeepFilterNet3 unavailable; Tier-1 spectral fallback: "
                    f"{denoiser_fallback_reason}"
                )
            if bandwidth_fallback_reason:
                issues.append(
                    "MP-SENet/VoiceFixer unavailable; harmonic extension fallback: "
                    f"{bandwidth_fallback_reason}"
                )
        module_result = ModuleResult(
            status=status,
            module_name="voice_enhancement",
            dataset_version=dataset_version,
            metrics=metrics,
            issues=issues,
            artifacts={
                "step_queue": queue,
                "execution_timing_ms": timing,
            },
        )
        return VoiceTransformResult(
            status=status,
            transformed_df=None,
            transformed_audio=np.ascontiguousarray(enhanced, dtype=np.float32),
            sample_rate=output_sample_rate,
            applied_actions=actions,
            step_queue=queue,
            audit_log=audit_log,
            error_message=None,
            execution_time_ms=total_ms,
            execution_timing=timing,
            module_result=module_result,
            metadata={
                "filename": loaded.filename,
                "hardware": serializable_hardware,
                "metrics": metrics,
                "mode": execution_mode,
                "tier_requested": requested_tier,
                "tier_effective": effective_tier,
                "tier_used": tier_used,
                "fallback_reason": fallback_reason,
                "denoiser_fallback_reason": denoiser_fallback_reason,
                "bandwidth_fallback_reason": bandwidth_fallback_reason,
            },
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - total_started) * 1_000.0
        timing["total"] = total_ms
        error_message = f"Voice enhancement failed: {exc}"
        telemetry = hardware_telemetry(context) if context else {}
        module_result = ModuleResult.failure(
            module_name="voice_enhancement",
            dataset_version=dataset_version,
            error_message=error_message,
            metrics={**telemetry, "execution_time_ms": total_ms},
            artifacts={"step_queue": queue, "execution_timing_ms": timing},
        )
        return VoiceTransformResult(
            status=ExecutionStatus.FAILED,
            transformed_df=None,
            transformed_audio=None,
            sample_rate=BASELINE_SAMPLE_RATE,
            applied_actions=actions,
            step_queue=queue,
            audit_log=audit_log,
            error_message=error_message,
            execution_time_ms=total_ms,
            execution_timing=timing,
            module_result=module_result,
            metadata={"hardware": telemetry},
        )


def run_voice_pipeline(*args: Any, **kwargs: Any) -> VoiceTransformResult:
    """Public orchestration alias."""

    return enhance_voice(*args, **kwargs)


def process_voice_audio(*args: Any, **kwargs: Any) -> VoiceTransformResult:
    """Public orchestration alias for callers that use process terminology."""

    return enhance_voice(*args, **kwargs)


class VoiceEnhancementPipeline:
    """Reusable configuration wrapper for static voice enhancement."""

    def __init__(
        self,
        *,
        processing_mode: str | int = "tier1",
        mode: str = "file_offline",
        attenuation_db: float = 20.0,
        enable_bandwidth_extension: bool = False,
        boost_db: float = 3.0,
        low_gain_db: float = 0.0,
        mid_gain_db: float = 0.0,
        high_gain_db: float = 0.0,
        device_ctx: dict | None = None,
    ) -> None:
        self.processing_mode = processing_mode
        self.mode = _execution_mode(mode)
        self.attenuation_db = attenuation_db
        self.enable_bandwidth_extension = enable_bandwidth_extension
        self.boost_db = boost_db
        self.low_gain_db = low_gain_db
        self.mid_gain_db = mid_gain_db
        self.high_gain_db = high_gain_db
        self.device_ctx = dict(device_ctx or resolve_execution_device())

    def process(
        self,
        audio_source: Any,
        sample_rate: int | None = None,
        *,
        dataset_version: str = "audio_upload",
    ) -> VoiceTransformResult:
        return enhance_voice(
            audio_source,
            sample_rate,
            processing_mode=self.processing_mode,
            mode=self.mode,
            attenuation_db=self.attenuation_db,
            enable_bandwidth_extension=self.enable_bandwidth_extension,
            boost_db=self.boost_db,
            low_gain_db=self.low_gain_db,
            mid_gain_db=self.mid_gain_db,
            high_gain_db=self.high_gain_db,
            device_ctx=self.device_ctx,
            dataset_version=dataset_version,
        )


class _StreamingResampler:
    """Stateful band-limited rational polyphase resampler."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Streaming sample rates must be positive.")
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        divisor = gcd(self.source_rate, self.target_rate)
        self.up = self.target_rate // divisor
        self.down = self.source_rate // divisor
        if self.source_rate == self.target_rate:
            self._filter = np.ones(1, dtype=np.float64)
            self._center = 0
            self.reset()
            return
        maximum_rate = max(self.up, self.down)
        half_length = 10 * maximum_rate
        self._filter = firwin(
            2 * half_length + 1,
            1.0 / maximum_rate,
            window=("kaiser", 5.0),
        ).astype(np.float64)
        self._filter *= self.up
        self._center = half_length
        self.reset()

    def reset(self) -> None:
        self.input_samples = 0
        self.output_samples = 0
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return -((-numerator) // denominator)

    def _produce(self, *, flush: bool) -> np.ndarray:
        desired_total = int(
            round(self.input_samples * self.target_rate / self.source_rate)
        )
        output: list[float] = []
        filter_end = self._filter.size - 1
        while self.output_samples < desired_total:
            filter_time = self.output_samples * self.down + self._center
            maximum_input = filter_time // self.up
            if not flush and maximum_input >= self.input_samples:
                break
            minimum_input = self._ceil_div(filter_time - filter_end, self.up)
            first = max(0, minimum_input, self._buffer_start)
            last = min(maximum_input, self.input_samples - 1)
            if first <= last:
                input_indices = np.arange(first, last + 1, dtype=np.int64)
                filter_indices = filter_time - input_indices * self.up
                buffer_indices = input_indices - self._buffer_start
                value = float(
                    np.dot(
                        self._buffer[buffer_indices].astype(np.float64, copy=False),
                        self._filter[filter_indices],
                    )
                )
            else:
                value = 0.0
            output.append(value)
            self.output_samples += 1

        if not flush:
            next_time = self.output_samples * self.down + self._center
            earliest_needed = max(
                0,
                self._ceil_div(next_time - filter_end, self.up),
            )
            trim = min(
                max(0, earliest_needed - self._buffer_start),
                self._buffer.size,
            )
            if trim:
                self._buffer = self._buffer[trim:]
                self._buffer_start += int(trim)
        return np.asarray(output, dtype=np.float32)

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        chunk = downmix_to_mono(audio_chunk)
        if chunk.size == 0:
            return chunk
        if self.source_rate == self.target_rate:
            self.input_samples += int(chunk.size)
            self.output_samples += int(chunk.size)
            return chunk.copy()
        self._buffer = np.concatenate((self._buffer, chunk))
        self.input_samples += int(chunk.size)
        return self._produce(flush=False)

    def flush(self) -> np.ndarray:
        if self.source_rate == self.target_rate:
            return np.empty(0, dtype=np.float32)
        output = self._produce(flush=True)
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = self.input_samples
        return output


class PCMStreamEnhancer:
    """Stateful low-latency enhancer for interleaved integer PCM chunks."""

    def __init__(
        self,
        *,
        input_sample_rate: int = BASELINE_SAMPLE_RATE,
        channels: int = 1,
        sample_width: int = 2,
        output_bit_depth: int | None = None,
        processing_mode: str | int = "tier1",
        mode: str = "live_stream",
        attenuation_db: float = 20.0,
        enable_bandwidth_extension: bool = False,
        device_ctx: dict | None = None,
    ) -> None:
        self.input_sample_rate = int(input_sample_rate)
        self.channels = int(channels)
        self.sample_width = int(sample_width)
        self.output_bit_depth = int(output_bit_depth or sample_width * 8)
        self.mode = _execution_mode(mode)
        if self.mode != "live_stream":
            raise ValueError("PCMStreamEnhancer supports mode='live_stream' only.")
        self.requested_tier = _tier_name(processing_mode)
        self.processing_mode = "tier2"
        self.attenuation_db = float(np.clip(attenuation_db, 0.0, 30.0))
        self.enable_bandwidth_extension = False
        self.output_sample_rate = BASELINE_SAMPLE_RATE
        self.device_ctx = dict(device_ctx or resolve_execution_device())
        self.device_ctx["require_onnx"] = True
        self._spectral_gate = StreamingSpectralGate(
            BASELINE_SAMPLE_RATE, self.attenuation_db
        )
        self._stream_limiter = StreamingTruePeakLimiter(
            BASELINE_SAMPLE_RATE,
            target_peak_dbfs=-0.5,
            lookahead_ms=1.0,
        )
        self._resampler = _StreamingResampler(
            self.input_sample_rate, BASELINE_SAMPLE_RATE
        )
        self._pcm_remainder = b""
        self._neural_status = get_deepfilternet3_status(self.device_ctx)
        self._neural_ready = self._neural_status["ready"]
        self._neural_context = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._chunks_processed = 0
        self._samples_processed = 0
        self._processing_time_ms = 0.0
        self._maximum_chunk_latency_ms = 0.0
        self._last_chunk_latency_ms = 0.0
        self.last_runtime = "uninitialized"
        self.last_fallback_reason: str | None = None

    def reset(self) -> None:
        with self._lock:
            self._spectral_gate.reset()
            self._stream_limiter.reset()
            self._resampler.reset()
            self._pcm_remainder = b""
            self._neural_context = np.empty(0, dtype=np.float32)
            self._chunks_processed = 0
            self._samples_processed = 0
            self._processing_time_ms = 0.0
            self._maximum_chunk_latency_ms = 0.0
            self._last_chunk_latency_ms = 0.0
            self.last_runtime = "uninitialized"
            self.last_fallback_reason = None

    def _enhance_baseline_unlocked(self, baseline: np.ndarray) -> np.ndarray:
        if baseline.size == 0:
            return np.empty(0, dtype=np.float32)
        fallback_reasons: list[str] = []
        if self._neural_ready:
            context_size = BASELINE_SAMPLE_RATE // 2
            combined = np.concatenate((self._neural_context, baseline))
            neural = apply_deepfilternet3_sota(
                combined,
                BASELINE_SAMPLE_RATE,
                attenuation_db=self.attenuation_db,
                device_ctx=self.device_ctx,
            )
            enhanced = np.asarray(neural["audio"], dtype=np.float32)[-baseline.size :]
            self._neural_context = combined[-context_size:]
            self.last_runtime = str(neural["runtime"])
            if neural.get("fallback_reason"):
                fallback_reasons.append(
                    f"DeepFilterNet3:{neural['fallback_reason']}"
                )
        else:
            enhanced = self._spectral_gate.process(baseline)
            self.last_runtime = "streaming_spectral_gate"
            neural_reason = (
                "onnx_weights_missing"
                if not self._neural_status["weights_available"]
                else "onnxruntime_unavailable"
            )
            fallback_reasons.append(f"DeepFilterNet3:{neural_reason}")
        enhanced = self._stream_limiter.process(enhanced)
        self.last_fallback_reason = "; ".join(fallback_reasons) or None
        return enhanced

    def _enhance_float_unlocked(self, audio_chunk: np.ndarray) -> np.ndarray:
        baseline = self._resampler.process(audio_chunk)
        return self._enhance_baseline_unlocked(baseline)

    def _record_chunk(self, sample_count: int, started: float) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        self._chunks_processed += 1
        self._samples_processed += int(sample_count)
        self._processing_time_ms += elapsed_ms
        self._last_chunk_latency_ms = elapsed_ms
        self._maximum_chunk_latency_ms = max(
            self._maximum_chunk_latency_ms, elapsed_ms
        )

    def process_float_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        with self._lock:
            chunk = downmix_to_mono(audio_chunk)
            result = self._enhance_float_unlocked(chunk)
            self._record_chunk(chunk.size, started)
            return result

    def process_pcm_chunk(self, pcm_chunk: bytes) -> bytes:
        started = time.perf_counter()
        with self._lock:
            combined = self._pcm_remainder + bytes(pcm_chunk)
            frame_bytes = self.sample_width * self.channels
            complete_size = len(combined) - (len(combined) % frame_bytes)
            self._pcm_remainder = combined[complete_size:]
            if complete_size == 0:
                return b""
            audio = pcm_bytes_to_float32(
                combined[:complete_size],
                sample_width=self.sample_width,
                channels=self.channels,
            )
            enhanced = self._enhance_float_unlocked(audio)
            self._record_chunk(audio.size, started)
            return float32_to_pcm_bytes(enhanced, bit_depth=self.output_bit_depth)

    def process(self, pcm_chunk: bytes) -> bytes:
        return self.process_pcm_chunk(pcm_chunk)

    def process_stream(self, chunks: Iterable[bytes]) -> Iterator[bytes]:
        for chunk in chunks:
            if chunk:
                output = self.process_pcm_chunk(chunk)
                if output:
                    yield output
        tail = self.flush()
        if tail:
            yield tail

    def flush_float(self) -> np.ndarray:
        """Flush deferred resampler output after the final float chunk."""

        with self._lock:
            pending = self._resampler.flush()
            processed = self._enhance_baseline_unlocked(pending)
            tail = self._stream_limiter.flush()
            return np.concatenate((processed, tail))

    def flush(self) -> bytes:
        """Finish a PCM stream and return its deferred output bytes."""

        with self._lock:
            if self._pcm_remainder:
                raise ValueError("PCM stream ended with an incomplete channel frame.")
            pending = self._resampler.flush()
            processed = self._enhance_baseline_unlocked(pending)
            tail = self._stream_limiter.flush()
            enhanced = np.concatenate((processed, tail))
            return float32_to_pcm_bytes(
                enhanced, bit_depth=self.output_bit_depth
            )

    @property
    def module_result(self) -> ModuleResult:
        fallback = self.last_fallback_reason
        status = ExecutionStatus.PARTIAL if fallback else ExecutionStatus.SUCCESS
        return ModuleResult(
            status=status,
            module_name="voice_stream_enhancement",
            dataset_version="pcm_stream",
            metrics={
                **hardware_telemetry(self.device_ctx),
                "mode": self.mode,
                "latency_guard_active": True,
                "latency_budget_ms": 20.0,
                "tier_requested": self.requested_tier,
                "tier_effective": self.processing_mode,
                "runtime": self.last_runtime,
                "fallback_reason": fallback,
                "chunks_processed": self._chunks_processed,
                "samples_processed": self._samples_processed,
                "processing_time_ms": self._processing_time_ms,
                "last_chunk_latency_ms": self._last_chunk_latency_ms,
                "maximum_chunk_latency_ms": self._maximum_chunk_latency_ms,
                "latency_budget_met": self._maximum_chunk_latency_ms < 20.0,
                "limiter_lookahead_ms": 1.0,
                "pipeline_latency_ms": self._stream_limiter.latency_ms,
                "multiband_eq": False,
                "bandwidth_extension": False,
                "input_sample_rate": self.input_sample_rate,
                "output_sample_rate": self.output_sample_rate,
            },
            issues=([f"Tier-2 stream fallback: {fallback}"] if fallback else []),
            artifacts={},
        )


StreamingVoiceEnhancer = PCMStreamEnhancer
