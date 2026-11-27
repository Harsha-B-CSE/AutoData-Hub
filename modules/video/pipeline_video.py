"""Master video processing orchestrator returning standardized dataclass contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List

import numpy as np

from core.schema_inference import (
    ExecutionStatus,
    ModuleResult,
    TransformationAction,
    TransformResult,
)
from modules.image.color_engine import ColorTriplet
from modules.video.color_engine_video import (
    DEFAULT_EMA_ALPHA,
    apply_temporal_color_balance,
)
from modules.video.keyframe_extractor import (
    MAX_MEMORY_BUDGET_BYTES,
    extract_keyframes,
)
from modules.video.scene_detector import (
    DEFAULT_SCENE_CUT_THRESHOLD,
    detect_scenes,
)
from modules.video.temporal_blur import analyze_video_blur


@dataclass
class VideoTransformResult(TransformResult):
    """Video-specialized TransformResult preserving standard core dataclass fields."""

    transformed_frames: List[np.ndarray] = field(default_factory=list)
    fps: float = 30.0
    execution_time_ms: float = 0.0
    module_result: ModuleResult | None = None


def process_video_pipeline(
    video_source: Any,
    *,
    sample_every_n_frames: int = 1,
    max_keyframes: int = 120,
    max_dimension: int | None = 1080,
    memory_budget_bytes: int = MAX_MEMORY_BUDGET_BYTES,
    scene_cut_threshold: float = DEFAULT_SCENE_CUT_THRESHOLD,
    sharpness_threshold: float = 100.0,
    motion_threshold: float = 2.0,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    white_balance_method: str = "gray_world",
    apply_grading: bool = False,
    lift: ColorTriplet = 0.0,
    gamma: ColorTriplet = 1.0,
    gain: ColorTriplet = 1.0,
    saturation: float = 1.0,
    preset: str = "Custom",
    dataset_version: str = "video_upload",
) -> VideoTransformResult:
    """Run streamed extraction, scene cuts, blur triage, and EMA color grading."""
    started = time.perf_counter()
    audit_log: List[str] = []
    actions: List[TransformationAction] = []

    try:
        extracted = extract_keyframes(
            video_source,
            sample_every_n_frames=sample_every_n_frames,
            max_keyframes=max_keyframes,
            max_dimension=max_dimension,
            memory_budget_bytes=memory_budget_bytes,
            reserve_processing=True,
        )
        decoded_at = time.perf_counter()
        keyframes = extracted["keyframes"]
        fps = float(extracted["fps"])
        audit_log.append(
            f"[VIDEO] Extracted {extracted['keyframe_count']} keyframes from "
            f"{extracted['source_name']} ({extracted['memory_mb']} MiB retained originals)."
        )
        reason = extracted["sampling_stop_reason"]
        issues = []
        if reason != "end_of_stream":
            why = {
                "memory_budget": "the working memory budget was reached",
                "max_keyframes": "the requested keyframe limit was reached",
                "early_end_of_stream": "the stream ended before the reported frame count",
                "decode_error": "the decoder could not retrieve the next sample",
            }[reason]
            issues.append(
                f"Partial sampling: {why}. Metrics and exports cover only the "
                f"{len(keyframes)} sampled frames through source frame {keyframes[-1].frame_index}."
            )
            audit_log.extend(issues)
        actions.append(
            TransformationAction(
                step_id=1,
                module="video",
                action_type="streamed_keyframe_extraction",
                target_columns=[],
                parameters={
                    "sample_every_n_frames": int(sample_every_n_frames),
                    "keyframe_count": int(extracted["keyframe_count"]),
                    "max_keyframes": int(max_keyframes),
                    "memory_budget_bytes": extracted["memory_budget_bytes"],
                    "sampling_stop_reason": reason,
                },
            )
        )

        scene_info = detect_scenes(
            keyframes,
            fps=fps,
            correlation_threshold=scene_cut_threshold,
        )
        scenes_at = time.perf_counter()
        audit_log.append(
            f"[VIDEO] Detected {len(scene_info['scene_cuts'])} scene cuts "
            f"(d_corr < {scene_cut_threshold:.2f})."
        )
        actions.append(
            TransformationAction(
                step_id=2,
                module="video",
                action_type="hsv_pearson_scene_detection",
                target_columns=[],
                parameters={
                    "threshold": float(scene_cut_threshold),
                    "scene_cut_indices": list(scene_info["scene_cut_indices"]),
                },
            )
        )

        blur_info = analyze_video_blur(
            keyframes,
            sharpness_threshold=sharpness_threshold,
            motion_threshold=motion_threshold,
        )
        blur_at = time.perf_counter()
        audit_log.append(
            f"[VIDEO] Blur classification: {blur_info['sharp_frames']} sharp, "
            f"{blur_info['motion_blur_frames']} motion blur, "
            f"{blur_info['defocus_blur_frames']} defocus blur."
        )
        actions.append(
            TransformationAction(
                step_id=3,
                module="video",
                action_type="temporal_blur_classification",
                target_columns=[],
                parameters={
                    "sharpness_threshold": float(sharpness_threshold),
                    "motion_threshold": float(motion_threshold),
                    "motion_blur_frames": int(blur_info["motion_blur_frames"]),
                    "defocus_blur_frames": int(blur_info["defocus_blur_frames"]),
                },
            )
        )

        color_info = apply_temporal_color_balance(
            keyframes,
            alpha=ema_alpha,
            method=white_balance_method,
            scene_cut_indices=scene_info["scene_cut_indices"],
            apply_grading=apply_grading,
            lift=lift,
            gamma=gamma,
            gain=gain,
            saturation=saturation,
            preset=preset,
        )
        audit_log.append(
            f"[VIDEO] Stabilized color balance via EMA (alpha={ema_alpha:.2f}, "
            f"method={white_balance_method}, preset={preset})."
        )
        actions.append(
            TransformationAction(
                step_id=4,
                module="video",
                action_type="ema_temporal_color_balance",
                target_columns=[],
                parameters={
                    "alpha": float(ema_alpha),
                    "method": str(white_balance_method),
                    "preset": str(preset),
                },
            )
        )

        finished = time.perf_counter()
        elapsed_ms = (finished - started) * 1000.0
        metrics: Dict[str, Any] = {
            "source_name": extracted["source_name"],
            "fps": fps,
            "width": extracted["width"],
            "height": extracted["height"],
            "analysis_width": extracted["analysis_width"],
            "analysis_height": extracted["analysis_height"],
            "keyframe_count": extracted["keyframe_count"],
            "sample_every_n_frames": int(sample_every_n_frames),
            "reported_frame_count": extracted["reported_frame_count"],
            "frames_scanned": extracted["frames_scanned"],
            "last_sampled_frame_index": keyframes[-1].frame_index,
            "sampling_stop_reason": reason,
            "sampling_complete": not issues,
            "source_buffer_bytes": extracted["source_buffer_bytes"],
            "retained_original_bytes": extracted["memory_bytes"],
            "retained_output_bytes": sum(frame.nbytes for frame in color_info["frames"]),
            "estimated_peak_bytes": extracted["estimated_peak_bytes"],
            "memory_mb": round(extracted["estimated_peak_bytes"] / (1024 * 1024), 3),
            "memory_budget_bytes": extracted["memory_budget_bytes"],
            "within_memory_budget": extracted["within_memory_budget"],
            "scene_cut_count": len(scene_info["scene_cuts"]),
            "scene_count": scene_info["scene_count"],
            "scene_cut_indices": scene_info["scene_cut_indices"],
            "min_scene_correlation": scene_info["min_correlation"],
            "sharp_frames": blur_info["sharp_frames"],
            "motion_blur_frames": blur_info["motion_blur_frames"],
            "defocus_blur_frames": blur_info["defocus_blur_frames"],
            "mean_laplacian_variance": blur_info["mean_laplacian_variance"],
            "mean_flow_magnitude": blur_info["mean_flow_magnitude"],
            "ema_alpha": float(ema_alpha),
            "white_balance_method": str(white_balance_method),
            "color_preset": str(preset),
            "execution_time_ms": elapsed_ms,
            "stage_times_ms": {
                "stage_and_decode": (decoded_at - started) * 1000,
                "scene_detection": (scenes_at - decoded_at) * 1000,
                "blur_analysis": (blur_at - scenes_at) * 1000,
                "color_balance": (finished - blur_at) * 1000,
            },
        }
        artifacts: Dict[str, Any] = {
            "keyframes": keyframes,
            "scene_detection": scene_info,
            "blur_analysis": blur_info,
            "gain_history": color_info["gain_history"],
        }
        status = ExecutionStatus.PARTIAL if issues else ExecutionStatus.SUCCESS
        module_result = ModuleResult(
            status=status,
            module_name="video_processing",
            dataset_version=dataset_version,
            metrics=metrics,
            artifacts=artifacts,
            issues=issues,
        )
        return VideoTransformResult(
            status=status,
            transformed_df=None,
            transformed_frames=color_info["frames"],
            fps=fps,
            applied_actions=actions,
            audit_log=audit_log,
            error_message=None,
            execution_time_ms=elapsed_ms,
            module_result=module_result,
            metadata={
                "metrics": metrics,
                "scene_cuts": scene_info["scene_cuts"],
                "blur_summary": {
                    "sharp": blur_info["sharp_frames"],
                    "motion_blur": blur_info["motion_blur_frames"],
                    "defocus_blur": blur_info["defocus_blur_frames"],
                },
            },
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        error_msg = f"Video processing failed: {exc}"
        module_result = ModuleResult.failure(
            module_name="video_processing",
            dataset_version=dataset_version,
            error_message=error_msg,
            metrics={"execution_time_ms": elapsed_ms},
        )
        return VideoTransformResult(
            status=ExecutionStatus.FAILED,
            transformed_df=None,
            transformed_frames=[],
            applied_actions=actions,
            audit_log=audit_log,
            error_message=error_msg,
            execution_time_ms=elapsed_ms,
            module_result=module_result,
            metadata={},
        )
