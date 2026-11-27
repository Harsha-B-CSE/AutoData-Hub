"""Temporal color stabilization using Exponential Moving Average (EMA) gain smoothing."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

import numpy as np

from modules.image.color_engine import (
    ColorTriplet,
    apply_color_grading,
    compute_white_balance_gains,
)


DEFAULT_EMA_ALPHA = 0.15


class TemporalColorStabilizer:
    """Smooth per-frame white-balance gains via EMA (alpha=0.15) to eliminate flicker."""

    def __init__(
        self,
        alpha: float = DEFAULT_EMA_ALPHA,
        method: str = "gray_world",
    ) -> None:
        alpha_val = float(alpha)
        if not np.isfinite(alpha_val) or not (0.0 < alpha_val <= 1.0):
            raise ValueError("EMA smoothing factor alpha must lie in (0, 1].")
        self.alpha = alpha_val
        self.method = str(method or "gray_world")
        self._ema_gains: np.ndarray | None = None
        self.gain_history: List[List[float]] = []

    def reset(self) -> None:
        self._ema_gains = None
        self.gain_history.clear()

    def step(
        self,
        frame_bgr: np.ndarray,
        *,
        is_scene_cut: bool = False,
        apply_grading: bool = False,
        lift: ColorTriplet = 0.0,
        gamma: ColorTriplet = 1.0,
        gain: ColorTriplet = 1.0,
        saturation: float = 1.0,
        preset: str = "Custom",
    ) -> np.ndarray:
        """Compute smoothed gains g_t = alpha * g_raw + (1 - alpha) * g_{t-1} and apply."""
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("A non-empty frame is required.")

        raw_gains = compute_white_balance_gains(frame_bgr, method=self.method)
        if self._ema_gains is None or is_scene_cut:
            smoothed_gains = raw_gains.copy()
        else:
            smoothed_gains = (
                self.alpha * raw_gains + (1.0 - self.alpha) * self._ema_gains
            ).astype(np.float32)

        self._ema_gains = smoothed_gains
        self.gain_history.append([float(x) for x in smoothed_gains])

        if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            bgr = frame_bgr[:, :, :3]
            alpha_chan = frame_bgr[:, :, 3]
        else:
            bgr = frame_bgr
            alpha_chan = None

        unit = bgr.astype(np.float32) / 255.0
        balanced = np.clip(unit * smoothed_gains.reshape(1, 1, 3), 0.0, 1.0)
        out_bgr = np.round(balanced * 255.0).astype(np.uint8)
        if alpha_chan is not None:
            out_frame = np.dstack([out_bgr, alpha_chan])
        else:
            out_frame = out_bgr

        if apply_grading or preset != "Custom":
            out_frame = apply_color_grading(
                out_frame,
                lift=lift,
                gamma=gamma,
                gain=gain,
                saturation=saturation,
                preset=preset,
            )
        return out_frame


def apply_temporal_color_balance(
    frames: Sequence[Any],
    *,
    alpha: float = DEFAULT_EMA_ALPHA,
    method: str = "gray_world",
    scene_cut_indices: Sequence[int] | Set[int] = (),
    apply_grading: bool = False,
    lift: ColorTriplet = 0.0,
    gamma: ColorTriplet = 1.0,
    gain: ColorTriplet = 1.0,
    saturation: float = 1.0,
    preset: str = "Custom",
) -> Dict[str, Any]:
    """Apply EMA-smoothed white balance and optional color grading across frames."""
    if not frames:
        raise ValueError("At least one frame is required for temporal color stabilization.")
    cuts = set(int(idx) for idx in scene_cut_indices)
    stabilizer = TemporalColorStabilizer(alpha=alpha, method=method)
    stabilized_frames: List[np.ndarray] = []

    for seq_idx, item in enumerate(frames):
        frame_bgr = item.image_bgr if hasattr(item, "image_bgr") else item
        frame_index = int(getattr(item, "frame_index", seq_idx))
        stabilized = stabilizer.step(
            frame_bgr,
            is_scene_cut=frame_index in cuts,
            apply_grading=apply_grading,
            lift=lift,
            gamma=gamma,
            gain=gain,
            saturation=saturation,
            preset=preset,
        )
        stabilized_frames.append(stabilized)

    return {
        "alpha": stabilizer.alpha,
        "method": stabilizer.method,
        "preset": preset,
        "frames": stabilized_frames,
        "gain_history": stabilizer.gain_history,
        "frame_count": len(stabilized_frames),
    }
