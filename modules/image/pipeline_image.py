"""Per-image snapshots: raw upload -> restoration -> grade -> export."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from modules.image.color_engine import apply_auto_white_balance, apply_color_grading


@dataclass(frozen=True)
class GradeSettings:
    wb_method: str = "None"
    preset: str = "Custom"
    lift: float = 0.0
    gamma: float = 1.0
    gain: float = 1.0
    saturation: float = 1.0

    def apply(self, base: np.ndarray) -> np.ndarray:
        if self.wb_method != "None":
            base = apply_auto_white_balance(base, method=self.wb_method)
        return apply_color_grading(
            base, lift=self.lift, gamma=self.gamma, gain=self.gain,
            saturation=self.saturation, preset=self.preset,
        )


@dataclass(frozen=True, eq=False)
class ImagePipelineBuffer:
    raw_bgr: np.ndarray
    restored_bgr: np.ndarray | None = None
    graded_bgr: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    grade_settings: GradeSettings | None = None

    def __post_init__(self) -> None:
        for name in ("raw_bgr", "restored_bgr", "graded_bgr"):
            image = getattr(self, name)
            if image is None and name != "raw_bgr":
                continue
            if (not isinstance(image, np.ndarray) or not image.size or image.dtype.kind not in "uif"
                    or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (3, 4)))):
                raise ValueError(f"{name} must be a non-empty grayscale, BGR, or BGRA image.")
            # Share existing immutable planes between versions, never an external mutable view.
            owner = image
            while isinstance(owner, np.ndarray):
                owner = owner.base
            if not isinstance(owner, bytes):
                image = np.frombuffer(image.tobytes(), dtype=image.dtype).reshape(image.shape)
                object.__setattr__(self, name, image)
        object.__setattr__(self, "metadata", deepcopy(self.metadata))

    def get_base_for_color_grading(self) -> np.ndarray:
        base = self.restored_bgr if self.restored_bgr is not None else self.raw_bgr
        return base.copy()

    def get_final_output(self) -> np.ndarray:
        """Return the latest read-only plane without copying pixels on every rerun."""
        if self.graded_bgr is not None:
            return self.graded_bgr
        return self.restored_bgr if self.restored_bgr is not None else self.raw_bgr

    def with_restoration(self, restored: np.ndarray, metadata: dict | None = None) -> ImagePipelineBuffer:
        buf = replace(self, restored_bgr=restored, graded_bgr=None, metadata=metadata or {})
        if self.grade_settings is not None:
            graded = self.grade_settings.apply(buf.get_base_for_color_grading())
            return buf.with_grading(graded, self.grade_settings)
        return buf

    def with_grading(self, graded: np.ndarray, settings: GradeSettings) -> ImagePipelineBuffer:
        return replace(self, graded_bgr=graded, grade_settings=settings)

    def without_grading(self) -> ImagePipelineBuffer:
        return replace(self, graded_bgr=None, grade_settings=None)

    def reset_to_raw(self) -> ImagePipelineBuffer:
        return ImagePipelineBuffer(self.raw_bgr)


def bind_image_buffer(state: Any, buffer: ImagePipelineBuffer, *, selecting: bool = False) -> None:
    """Bind the active member; keep published UI outputs and downloads in sync."""
    state["image_buffer"] = buffer
    state["processed_image_bgr"] = buffer.get_final_output()
    if buffer.restored_bgr is None:
        state.pop("image_restoration_metadata", None)
    else:
        state["image_restoration_metadata"] = buffer.metadata
    state.pop("image_export_result", None)
    if selecting:
        state["image_sync_color_widgets"] = True
        state.pop("image_focus_metrics", None)
        return
    # Untouched ZIP members can be decoded on demand rather than retaining every raw plane.
    name = state["image_selected_name"]
    if buffer.restored_bgr is None and buffer.graded_bgr is None:
        state["image_buffers"].pop(name, None)
    else:
        state["image_buffers"][name] = buffer
    state.pop("image_export_zip_bytes", None)
