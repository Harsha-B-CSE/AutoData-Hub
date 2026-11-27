"""Scene cut detection using Pearson correlation on 3D HSV histograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_SCENE_CUT_THRESHOLD = 0.65


@dataclass(frozen=True)
class SceneCut:
    """Detected transition between consecutive analyzed video frames."""

    frame_index: int
    timestamp_sec: float
    correlation: float


def compute_hsv_histogram(
    frame_bgr: np.ndarray,
    bins: Tuple[int, int, int] = (8, 8, 8),
) -> np.ndarray:
    """Compute a normalized 3D HSV color histogram for a BGR frame."""
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("A non-empty BGR frame is required for histogram extraction.")
    bgr = frame_bgr[:, :, :3] if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4 else frame_bgr
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        list(bins),
        [0, 180, 0, 256, 0, 256],
    ).astype(np.float32)
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def compare_histograms_pearson(hist_prev: np.ndarray, hist_curr: np.ndarray) -> float:
    """Compute Pearson correlation d_corr(H_t, H_{t-1}) in [-1.0, 1.0]."""
    vec_a = np.asarray(hist_prev, dtype=np.float64).ravel()
    vec_b = np.asarray(hist_curr, dtype=np.float64).ravel()
    if vec_a.size == 0 or vec_a.size != vec_b.size:
        raise ValueError("Histograms must be non-empty and have matching dimensions.")

    centered_a = vec_a - np.mean(vec_a)
    centered_b = vec_b - np.mean(vec_b)
    norm_a = float(np.linalg.norm(centered_a))
    norm_b = float(np.linalg.norm(centered_b))
    if norm_a <= 1e-12 and norm_b <= 1e-12:
        return 1.0
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    corr = float(np.dot(centered_a, centered_b) / (norm_a * norm_b))
    return float(np.clip(corr, -1.0, 1.0))


def detect_scenes(
    frames: Sequence[Any],
    *,
    fps: float = 30.0,
    correlation_threshold: float = DEFAULT_SCENE_CUT_THRESHOLD,
    bins: Tuple[int, int, int] = (8, 8, 8),
) -> Dict[str, Any]:
    """Detect scene cuts where d_corr(H_t, H_{t-1}) < correlation_threshold."""
    if not frames:
        raise ValueError("At least one frame is required for scene detection.")
    threshold = float(correlation_threshold)
    safe_fps = float(fps) if fps and fps > 0 else 30.0

    cuts: List[SceneCut] = []
    correlations: List[float] = []
    prev_hist: np.ndarray | None = None

    for seq_idx, item in enumerate(frames):
        if hasattr(item, "image_bgr"):
            frame_bgr = item.image_bgr
            frame_index = int(getattr(item, "frame_index", seq_idx))
            timestamp_sec = float(getattr(item, "timestamp_sec", frame_index / safe_fps))
        else:
            frame_bgr = item
            frame_index = seq_idx
            timestamp_sec = float(seq_idx) / safe_fps

        curr_hist = compute_hsv_histogram(frame_bgr, bins=bins)
        if prev_hist is not None:
            corr = compare_histograms_pearson(prev_hist, curr_hist)
            correlations.append(corr)
            if corr < threshold:
                cuts.append(
                    SceneCut(
                        frame_index=frame_index,
                        timestamp_sec=round(timestamp_sec, 4),
                        correlation=round(corr, 4),
                    )
                )
        prev_hist = curr_hist

    return {
        "threshold": threshold,
        "frames_analyzed": len(frames),
        "scene_cuts": cuts,
        "scene_cut_indices": [cut.frame_index for cut in cuts],
        "scene_count": len(cuts) + 1,
        "correlations": correlations,
        "min_correlation": min(correlations) if correlations else 1.0,
        "mean_correlation": float(np.mean(correlations)) if correlations else 1.0,
    }
