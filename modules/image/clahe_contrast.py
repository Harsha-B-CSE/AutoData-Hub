"""Luminance-only CLAHE contrast enhancement."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from modules.image.adaptive import apply_lab_clahe


def apply_clahe_contrast(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to LAB luminance while preserving color and alpha channels."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("A non-empty image is required.")
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive.")
    if len(tile_grid_size) != 2 or any(value <= 0 for value in tile_grid_size):
        raise ValueError("tile_grid_size must contain two positive integers.")
    return apply_lab_clahe(
        image_bgr,
        clip_limit=float(clip_limit),
        tile_grid_size=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
