"""Video overview, streamed keyframe extraction, and scene cut detection tab."""

from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from ui.utils_ux import render_tab_guidance_header


def _bgr_to_rgb(image_bgr):
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)


def render_tab_video_overview(result) -> None:
    """Display extracted keyframes and HSV Pearson scene cut telemetry."""
    render_tab_guidance_header(
        title="Keyframes & Scene Cuts",
        objective="Sample frames within a working memory budget and locate shot boundaries in that sample.",
        methodology=(
            "Streamed cv2.VideoCapture frame sampling followed by 3D HSV "
            "histogram Pearson correlation (a cut is below the selected threshold, 0.65 by default)."
        ),
        deliverable="Memory-bounded keyframe strip and timestamped scene cut ledger.",
    )
    metrics = result.module_result.metrics
    artifacts = result.module_result.artifacts
    scene_info = artifacts["scene_detection"]
    keyframes = artifacts["keyframes"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Keyframes", metrics["keyframe_count"])
    c2.metric("Scene Cuts", metrics["scene_cut_count"])
    c3.metric("Est. Peak Workspace", f"{metrics['memory_mb']:.2f} MiB")
    c4.metric("Min d_corr", f"{metrics['min_scene_correlation']:.3f}")
    retained = metrics["retained_original_bytes"] + metrics["retained_output_bytes"]
    total = metrics["reported_frame_count"] or "unknown"
    st.caption(
        f"Stride {metrics['sample_every_n_frames']} | Last sampled source frame: "
        f"{metrics['last_sampled_frame_index']} (zero-based; reported frame count: {total}) | "
        f"Analysis: {metrics['analysis_width']} x {metrics['analysis_height']} | "
        f"Retained frame pairs: {retained / 2**20:.2f} MiB. "
        "Workspace is a planning estimate, not measured process RSS."
    )

    if scene_info["scene_cuts"]:
        cuts_df = pd.DataFrame([
            {
                "Frame Index": cut.frame_index,
                "Timestamp (s)": cut.timestamp_sec,
                "Pearson d_corr": cut.correlation,
            }
            for cut in scene_info["scene_cuts"]
        ])
        st.dataframe(cuts_df, width="stretch", hide_index=True)
    else:
        st.info("No hard scene cuts detected between the sampled frames.")

    preview_count = min(6, len(keyframes))
    cols = st.columns(preview_count)
    for idx in range(preview_count):
        kf = keyframes[idx]
        cols[idx].image(
            _bgr_to_rgb(kf.image_bgr),
            caption=f"#{kf.frame_index} ({kf.timestamp_sec:.2f}s)",
            width="stretch",
        )
