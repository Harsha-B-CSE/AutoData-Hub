"""Image and video color balancing and ASC CDL Lift-Gamma-Gain grading."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import cv2
import numpy as np


_EPSILON = 1e-6

ColorTriplet = Union[float, Sequence[float]]

COLOR_GRADING_PRESETS: Dict[str, Dict[str, object]] = {
    "Custom": {
        "lift": (0.0, 0.0, 0.0),
        "gamma": (1.0, 1.0, 1.0),
        "gain": (1.0, 1.0, 1.0),
        "saturation": 1.0,
    },
    "🎬 Teal & Orange": {
        # Cool teal shadows (higher B/G lift, negative R lift) and warm orange highlights.
        "lift": (0.04, 0.02, -0.02),
        "gamma": (0.96, 1.00, 1.04),
        "gain": (0.90, 1.00, 1.12),
        "saturation": 1.18,
    },
    "📻 Vintage Film": {
        # Lifted warm shadows, gentle highlight rolloff, slightly muted filmic saturation.
        "lift": (0.02, 0.04, 0.06),
        "gamma": (0.95, 1.00, 1.05),
        "gain": (0.92, 0.98, 1.04),
        "saturation": 0.82,
    },
    "❄️ Cold Cyberpunk": {
        # Deep cool cyan shadows with punchy contrast and heightened saturation.
        "lift": (0.06, 0.03, -0.02),
        "gamma": (1.06, 1.00, 0.90),
        "gain": (1.14, 1.02, 0.94),
        "saturation": 1.28,
    },
}


def _split_bgr_alpha(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("A non-empty BGR/BGRA image is required.")
    if image_bgr.ndim == 2:
        bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        bgr = image_bgr
        alpha = None
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = image_bgr[:, :, :3]
        alpha = image_bgr[:, :, 3]
    else:
        raise ValueError("Color engine expects 2D grayscale, 3-channel BGR, or 4-channel BGRA image.")
    return bgr, alpha


def _to_unit_float(bgr: np.ndarray) -> np.ndarray:
    if np.issubdtype(bgr.dtype, np.integer):
        max_val = float(np.iinfo(bgr.dtype).max)
        return np.clip(bgr.astype(np.float32) / max_val, 0.0, 1.0)
    arr = np.nan_to_num(bgr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.max(arr, initial=0.0)) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _restore_dtype(unit_bgr: np.ndarray, target_dtype: np.dtype, alpha: np.ndarray | None) -> np.ndarray:
    clipped = np.clip(unit_bgr, 0.0, 1.0)
    if np.issubdtype(target_dtype, np.integer):
        max_val = float(np.iinfo(target_dtype).max)
        out_bgr = np.round(clipped * max_val).astype(target_dtype)
    else:
        out_bgr = clipped.astype(target_dtype)
    if alpha is not None:
        return np.dstack([out_bgr, alpha])
    return out_bgr


def _as_bgr_vector(value: ColorTriplet, name: str) -> np.ndarray:
    if isinstance(value, (int, float, np.number)):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        return np.array([scalar, scalar, scalar], dtype=np.float32)
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"{name} must be a scalar or 3-element (B, G, R) sequence.")
    arr = np.asarray(seq, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} values must be finite.")
    return arr


def compute_white_balance_gains(
    image_bgr: np.ndarray,
    method: str = "gray_world",
) -> np.ndarray:
    """Compute per-channel BGR chromatic scaling gains in [0, inf)."""
    bgr, _ = _split_bgr_alpha(image_bgr)
    unit = _to_unit_float(bgr)
    normalized_method = str(method or "gray_world").strip().lower().replace("-", "_")

    if normalized_method in {"gray_world", "grayworld"}:
        channel_means = np.mean(unit, axis=(0, 1), dtype=np.float64)
        gray_mean = float(np.mean(channel_means))
        if gray_mean <= _EPSILON:
            return np.ones(3, dtype=np.float32)
        gains = np.where(
            channel_means > _EPSILON,
            gray_mean / np.maximum(channel_means, _EPSILON),
            1.0,
        )
        return np.clip(gains, 0.1, 10.0).astype(np.float32)

    if normalized_method in {"max_rgb", "maxrgb", "white_patch"}:
        channel_max = np.max(unit, axis=(0, 1)).astype(np.float64)
        if float(np.max(channel_max)) <= _EPSILON:
            return np.ones(3, dtype=np.float32)
        gains = np.where(
            channel_max > _EPSILON,
            1.0 / np.maximum(channel_max, _EPSILON),
            1.0,
        )
        return np.clip(gains, 0.1, 10.0).astype(np.float32)

    raise ValueError(f"Unsupported white balance method: {method}. Use 'gray_world' or 'max_rgb'.")


def apply_auto_white_balance(
    image_bgr: np.ndarray,
    method: str = "gray_world",
) -> np.ndarray:
    """Balance image color casts via Gray-World or Max-RGB chromatic scaling."""
    bgr, alpha = _split_bgr_alpha(image_bgr)
    gains = compute_white_balance_gains(bgr, method=method)
    unit = _to_unit_float(bgr)
    balanced = np.clip(unit * gains.reshape(1, 1, 3), 0.0, 1.0)
    return _restore_dtype(balanced, image_bgr.dtype, alpha)


def apply_color_grading(
    image_bgr: np.ndarray,
    lift: ColorTriplet = 0.0,
    gamma: ColorTriplet = 1.0,
    gain: ColorTriplet = 1.0,
    saturation: float = 1.0,
    preset: str = "Custom",
) -> np.ndarray:
    """Apply ASC CDL Lift-Gamma-Gain transfer and saturation grading.

    Transfer formula:
        I_out = clip(((I * (1 - L) + L) ** (1 / Gamma)) * G, 0, 1)
    """
    bgr, alpha = _split_bgr_alpha(image_bgr)
    preset_name = str(preset or "Custom")
    if preset_name not in COLOR_GRADING_PRESETS:
        raise ValueError(
            f"Unsupported color grading preset: {preset_name}. "
            f"Expected one of {list(COLOR_GRADING_PRESETS)}."
        )

    if preset_name != "Custom":
        cfg = COLOR_GRADING_PRESETS[preset_name]
        lift_vec = _as_bgr_vector(cfg["lift"], "lift")  # type: ignore[arg-type]
        gamma_vec = _as_bgr_vector(cfg["gamma"], "gamma")  # type: ignore[arg-type]
        gain_vec = _as_bgr_vector(cfg["gain"], "gain")  # type: ignore[arg-type]
        sat_val = float(cfg["saturation"])  # type: ignore[arg-type]
    else:
        lift_vec = _as_bgr_vector(lift, "lift")
        gamma_vec = _as_bgr_vector(gamma, "gamma")
        gain_vec = _as_bgr_vector(gain, "gain")
        sat_val = float(saturation)

    if not np.isfinite(sat_val) or sat_val < 0.0:
        raise ValueError("saturation must be finite and non-negative.")

    lift_vec = np.clip(lift_vec, -0.5, 0.5)
    gamma_vec = np.maximum(gamma_vec, 1e-3)
    gain_vec = np.maximum(gain_vec, 0.0)

    unit = _to_unit_float(bgr)
    lifted = np.clip(unit * (1.0 - lift_vec.reshape(1, 1, 3)) + lift_vec.reshape(1, 1, 3), 0.0, 1.0)
    graded = np.power(lifted, 1.0 / gamma_vec.reshape(1, 1, 3)) * gain_vec.reshape(1, 1, 3)
    graded = np.clip(graded, 0.0, 1.0)

    if abs(sat_val - 1.0) > 1e-6:
        # Rec.709 luminance weights in BGR order: B=0.0722, G=0.7152, R=0.2126
        luma = (
            graded[:, :, 0:1] * 0.0722
            + graded[:, :, 1:2] * 0.7152
            + graded[:, :, 2:3] * 0.2126
        )
        graded = np.clip(luma + sat_val * (graded - luma), 0.0, 1.0)

    return _restore_dtype(graded, image_bgr.dtype, alpha)
