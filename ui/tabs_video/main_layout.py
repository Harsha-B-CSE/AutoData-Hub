"""Master layout for the Streamlit Video Processing workspace."""

from __future__ import annotations

import hashlib

import streamlit as st

from core.schema_inference import ExecutionStatus
from modules.image.color_engine import COLOR_GRADING_PRESETS
from modules.video.pipeline_video import process_video_pipeline
from ui.tabs_video.tab_video_color import render_tab_video_color
from ui.tabs_video.tab_video_export import render_tab_video_export
from ui.tabs_video.tab_video_overview import render_tab_video_overview
from ui.tabs_video.tab_video_quality import render_tab_video_quality


def render_video_app(uploaded_file) -> None:
    """Mount video configuration and the 4 video processing tabs."""
    if uploaded_file is None:
        st.info("Upload an MP4, AVI, MOV, MKV, or WEBM video from the sidebar.")
        return

    if isinstance(uploaded_file, dict):
        source = uploaded_file.get("source", uploaded_file.get("bytes", b""))
        source_name = uploaded_file.get("filename") or uploaded_file.get("name") or "video.mp4"
        upload_id = uploaded_file.get("upload_id")
    else:
        source = uploaded_file
        source_name = getattr(source, "name", "video.mp4")
        upload_id = None
    upload_id = upload_id or getattr(source, "file_id", None)
    if upload_id is None:
        # Legacy byte dictionaries have no Streamlit upload ID. Hash the
        # buffer directly, never concatenate it or consume a readable stream.
        if isinstance(source, (bytes, bytearray, memoryview)) and memoryview(source).c_contiguous:
            upload_id = hashlib.sha256(source).hexdigest()
        else:
            upload_id = id(source)
    fingerprint = (source_name, upload_id)
    if st.session_state.get("video_source_fingerprint") != fingerprint:
        st.session_state["video_source_fingerprint"] = fingerprint
        st.session_state.pop("video_transform_result", None)
        st.session_state.pop("video_color_inspect_idx", None)

    c1, c2, c3, c4 = st.columns(4)
    stride = c1.number_input("Frame Stride", min_value=1, max_value=60, value=5, step=1, key="video_stride")
    scene_thresh = c2.slider("Scene Cut d_corr", 0.2, 0.95, 0.65, 0.05, key="video_scene_thresh")
    ema_alpha = c3.slider("EMA Alpha (α)", 0.05, 0.50, 0.15, 0.05, key="video_ema_alpha")
    preset = c4.selectbox("Color Preset", options=list(COLOR_GRADING_PRESETS), key="video_color_preset")

    if st.button("Run Video Analysis & Color Pipeline", type="primary", width="stretch", key="video_run_pipeline"):
        st.session_state.pop("video_transform_result", None)
        st.session_state.pop("video_color_inspect_idx", None)
        with st.spinner("Decoding video stream and running temporal diagnostics..."):
            result = process_video_pipeline(
                uploaded_file,
                sample_every_n_frames=int(stride),
                scene_cut_threshold=float(scene_thresh),
                ema_alpha=float(ema_alpha),
                apply_grading=(preset != "Custom"),
                preset=preset,
                dataset_version=source_name,
            )
        st.session_state["video_transform_result"] = result

    result = st.session_state.get("video_transform_result")
    if result is None:
        st.info("Configure parameters above and click **Run Video Analysis & Color Pipeline**.")
        return
    if result.status == ExecutionStatus.FAILED:
        st.error(result.error_message or "Video processing failed.")
        return
    if result.status == ExecutionStatus.PARTIAL:
        for issue in result.module_result.issues:
            st.warning(issue)
        if result.module_result.metrics["sampling_stop_reason"] in ("max_keyframes", "memory_budget"):
            st.caption("Increase Frame Stride to sample a longer interval with fewer frames.")

    tab_overview, tab_quality, tab_color, tab_export = st.tabs([
        "🎬 Keyframes & Scene Cuts",
        "🔍 Temporal Blur",
        "🎨 EMA Color Grading",
        "📤 Export Studio",
    ])
    with tab_overview:
        render_tab_video_overview(result)
    with tab_quality:
        render_tab_video_quality(result)
    with tab_color:
        render_tab_video_color(result)
    with tab_export:
        render_tab_video_export(result, source_name)
