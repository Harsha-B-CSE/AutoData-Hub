"""In-memory sharpness metrics for the Streamlit image workflow."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from modules.image.profiler import compute_laplacian_variance


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("A non-empty image is required.")
    if image_bgr.ndim == 2:
        return image_bgr
    if image_bgr.ndim != 3:
        raise ValueError("Expected a grayscale, BGR, or BGRA image array.")
    if image_bgr.shape[2] == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Expected one, three, or four image channels.")


def evaluate_sharpness_laplacian(
    image_bgr: np.ndarray,
    threshold: float = 100.0,
) -> Dict[str, Any]:
    """Return focus telemetry and threshold classification for a BGR/BGRA image."""
    if threshold < 0:
        raise ValueError("Sharpness threshold must be non-negative.")
    gray = _to_gray(image_bgr)
    laplacian_variance = float(compute_laplacian_variance(image_bgr))
    tenengrad_score = evaluate_tenengrad(image_bgr)
    mean_luminance = float(np.mean(gray))
    is_blurry = laplacian_variance < threshold
    return {
        "laplacian_variance": laplacian_variance,
        "tenengrad_score": tenengrad_score,
        "mean_luminance": mean_luminance,
        "threshold": float(threshold),
        "is_blurry": bool(is_blurry),
        "status": "BLURRY" if is_blurry else "SHARP",
    }


def evaluate_tenengrad(image_bgr: np.ndarray) -> float:
    """Return mean Sobel gradient magnitude (Tenengrad focus score)."""
    gray = _to_gray(image_bgr)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gradient_x, gradient_y)))
