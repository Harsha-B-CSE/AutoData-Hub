"""Classical image denoising operators."""

from __future__ import annotations

import cv2
import numpy as np


def _preserve_alpha(image_bgr: np.ndarray, operation) -> np.ndarray:
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        filtered = operation(image_bgr[:, :, :3])
        return np.concatenate([filtered, image_bgr[:, :, 3:4]], axis=2)
    return operation(image_bgr)


def apply_bilateral_denoise(
    image_bgr: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving bilateral denoising."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("A non-empty image is required.")
    if d <= 0 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("Bilateral parameters must be positive.")
    return _preserve_alpha(
        image_bgr,
        lambda image: cv2.bilateralFilter(
            image,
            d=int(d),
            sigmaColor=float(sigma_color),
            sigmaSpace=float(sigma_space),
        ),
    )


def apply_gaussian_denoise(
    image_bgr: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian smoothing while preserving an existing alpha channel."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("A non-empty image is required.")
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if sigma < 0:
        raise ValueError("Gaussian sigma must be non-negative.")
    return _preserve_alpha(
        image_bgr,
        lambda image: cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            float(sigma),
        ),
    )
