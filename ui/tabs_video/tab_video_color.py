"""Temporal EMA color stabilization and filmic grading tab."""

from __future__ import annotations

import cv2
import streamlit as st

from ui.utils_ux import render_tab_guidance_header


def _bgr_to_rgb(image_bgr):
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)


def render_tab_video_color(result) -> None:
    """Compare raw keyframes with EMA-stabilized and graded frames."""
    render_tab_guidance_header(
        title="Temporal Color Stabilization",
        objective="Correct white balance across shots without frame-to-frame flicker.",
        methodology=(
            "Exponential Moving Average (default alpha = 0.15) gain smoothing over sampled "
            "chromatic gains, reset on detected scene cuts, plus ASC CDL grading."
        ),
        deliverable="Temporally coherent color-graded frames ready for export.",
    )
    keyframes = result.module_result.artifacts["keyframes"]
    processed_frames = result.transformed_frames
    if not keyframes or not processed_frames:
        st.info("Run the video pipeline to inspect stabilized frames.")
        return

    if len(keyframes) > 1:
        frame_idx = st.slider(
            "Inspect Keyframe",
            min_value=0,
            max_value=len(keyframes) - 1,
            value=0,
            key="video_color_inspect_idx",
        )
    else:
        frame_idx = 0
    left, right = st.columns(2)
    left.markdown("**Raw Keyframe**")
    left.image(_bgr_to_rgb(keyframes[frame_idx].image_bgr), width="stretch")
    right.markdown("**EMA Stabilized & Graded**")
    right.image(_bgr_to_rgb(processed_frames[frame_idx]), width="stretch")
