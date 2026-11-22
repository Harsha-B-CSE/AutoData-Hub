"""Validated aspect-preserving image resizing adapter."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from modules.image.transforms import resize_image


def letterbox_resize(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    fill_color: Tuple[int, int, int] = (0, 0, 0),
    *,
    interpolation: str = "lanczos",
    target_width: int | None = None,
    target_height: int | None = None,
    pad_color: Tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Resize to an exact canvas while preserving the source aspect ratio."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("A non-empty image is required.")
    if target_width is None or target_height is None:
        target_width, target_height = target_size
    if pad_color is not None:
        fill_color = pad_color
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive.")
    if target_width > 16_384 or target_height > 16_384:
        raise ValueError("Target dimensions cannot exceed 16,384 pixels.")
    return resize_image(
        image_bgr,
        target_width=target_width,
        target_height=target_height,
        interpolation=interpolation,
        preserve_aspect_ratio=True,
        pad_color=fill_color,
    )
