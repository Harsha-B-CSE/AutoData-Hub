"""Dual-metric temporal blur classification using Laplacian variance & Farneback optical flow."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from modules.image.sharpness_evaluator import evaluate_sharpness_laplacian


def _to_gray_uint8(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("A non-empty frame is required for optical flow.")
    if frame_bgr.ndim == 2:
        gray = frame_bgr
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] in (3, 4):
        gray = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported frame channel layout for blur classification.")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def compute_farneback_flow_magnitude(
    prev_frame_bgr: np.ndarray,
    curr_frame_bgr: np.ndarray,
    *,
    max_flow_dim: int = 320,
) -> Dict[str, float]:
    """Compute dense Farneback optical flow magnitude statistics between frames."""
    prev_gray = _to_gray_uint8(prev_frame_bgr)
    curr_gray = _to_gray_uint8(curr_frame_bgr)
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(
            curr_gray,
            (prev_gray.shape[1], prev_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    height, width = prev_gray.shape[:2]
    longest = max(height, width)
    scale = 1.0
    if longest > max_flow_dim > 0:
        scale = float(max_flow_dim) / float(longest)
        target_size = (
            max(8, int(round(width * scale))),
            max(8, int(round(height * scale))),
        )
        prev_small = cv2.resize(prev_gray, target_size, interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_gray, target_size, interpolation=cv2.INTER_AREA)
    else:
        prev_small = prev_gray
        curr_small = curr_gray

    flow = cv2.calcOpticalFlowFarneback(
        prev_small,
        curr_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude = np.hypot(flow[..., 0], flow[..., 1]) / max(scale, 1e-6)
    return {
        "mean_flow_magnitude": float(np.mean(magnitude)),
        "p95_flow_magnitude": float(np.percentile(magnitude, 95)),
        "max_flow_magnitude": float(np.max(magnitude)),
    }


def classify_temporal_blur(
    prev_frame_bgr: np.ndarray | None,
    curr_frame_bgr: np.ndarray,
    *,
    sharpness_threshold: float = 100.0,
    motion_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Distinguish dynamic motion blur from static defocus blur."""
    sharpness = evaluate_sharpness_laplacian(
        curr_frame_bgr, threshold=sharpness_threshold
    )
    lap_var = float(sharpness["laplacian_variance"])
    is_blurry = bool(sharpness["is_blurry"])

    if prev_frame_bgr is not None:
        flow_stats = compute_farneback_flow_magnitude(prev_frame_bgr, curr_frame_bgr)
    else:
        flow_stats = {
            "mean_flow_magnitude": 0.0,
            "p95_flow_magnitude": 0.0,
            "max_flow_magnitude": 0.0,
        }

    mean_flow = float(flow_stats["mean_flow_magnitude"])
    if not is_blurry:
        blur_type = "sharp"
    elif mean_flow >= float(motion_threshold):
        blur_type = "motion_blur"
    else:
        blur_type = "defocus_blur"

    return {
        "laplacian_variance": lap_var,
        "tenengrad_score": float(sharpness["tenengrad_score"]),
        "mean_flow_magnitude": mean_flow,
        "p95_flow_magnitude": float(flow_stats["p95_flow_magnitude"]),
        "is_blurry": is_blurry,
        "is_motion_blur": blur_type == "motion_blur",
        "is_defocus_blur": blur_type == "defocus_blur",
        "blur_type": blur_type,
        "sharpness_threshold": float(sharpness_threshold),
        "motion_threshold": float(motion_threshold),
    }


def analyze_video_blur(
    frames: Sequence[Any],
    *,
    sharpness_threshold: float = 100.0,
    motion_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Classify every frame in a sequence as sharp, motion_blur, or defocus_blur."""
    if not frames:
        raise ValueError("At least one frame is required for temporal blur analysis.")

    per_frame: List[Dict[str, Any]] = []
    prev_bgr: np.ndarray | None = None
    counts = {"sharp": 0, "motion_blur": 0, "defocus_blur": 0}

    for seq_idx, item in enumerate(frames):
        curr_bgr = item.image_bgr if hasattr(item, "image_bgr") else item
        frame_index = int(getattr(item, "frame_index", seq_idx))
        metrics = classify_temporal_blur(
            prev_bgr,
            curr_bgr,
            sharpness_threshold=sharpness_threshold,
            motion_threshold=motion_threshold,
        )
        metrics["frame_index"] = frame_index
        counts[metrics["blur_type"]] += 1
        per_frame.append(metrics)
        prev_bgr = curr_bgr

    return {
        "frames_analyzed": len(per_frame),
        "sharp_frames": counts["sharp"],
        "motion_blur_frames": counts["motion_blur"],
        "defocus_blur_frames": counts["defocus_blur"],
        "mean_laplacian_variance": float(
            np.mean([entry["laplacian_variance"] for entry in per_frame])
        ),
        "mean_flow_magnitude": float(
            np.mean([entry["mean_flow_magnitude"] for entry in per_frame])
        ),
        "per_frame": per_frame,
    }
