"""Dual-metric temporal blur classification tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.utils_ux import render_tab_guidance_header


def render_tab_video_quality(result) -> None:
    """Display Laplacian variance and Farneback optical flow blur diagnostics."""
    render_tab_guidance_header(
        title="Temporal Blur Diagnostics",
        objective="Separate dynamic motion blur from static optical defocus blur.",
        methodology=(
            "Combines spatial Laplacian variance Var(∇²I) with dense Farneback "
            "optical flow magnitude M_flow across consecutive frames."
        ),
        deliverable="Frame-level classification table (sharp, motion_blur, defocus_blur).",
    )
    blur_info = result.module_result.artifacts["blur_analysis"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharp Frames", blur_info["sharp_frames"])
    c2.metric("Motion Blur", blur_info["motion_blur_frames"])
    c3.metric("Defocus Blur", blur_info["defocus_blur_frames"])
    c4.metric("Mean Flow", f"{blur_info['mean_flow_magnitude']:.2f} px")

    table = pd.DataFrame([
        {
            "Frame": item["frame_index"],
            "Classification": item["blur_type"],
            "Laplacian Var": round(item["laplacian_variance"], 2),
            "Mean Flow (px)": round(item["mean_flow_magnitude"], 3),
        }
        for item in blur_info["per_frame"]
    ])
    st.dataframe(table, width="stretch", hide_index=True)
