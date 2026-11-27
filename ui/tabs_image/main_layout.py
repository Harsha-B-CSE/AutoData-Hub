"""Master layout for the Streamlit Computer Vision workspace."""

from __future__ import annotations

import hashlib

import streamlit as st

from modules.image.pipeline_image import ImagePipelineBuffer, bind_image_buffer
from ui.tabs_image.common import decode_image_bytes, extract_image_items, uploaded_bytes, uploaded_name
from ui.tabs_image.tab_color_grading import render_tab_color
from ui.tabs_image.tab_export import render_tab_export
from ui.tabs_image.tab_restoration import render_tab_restoration
from ui.tabs_image.tab_sharpness import render_tab_sharpness


def render_image_app(uploaded_file) -> None:
    """Mount focus, enhancement, and export tabs for an uploaded image dataset."""
    state = st.session_state
    fingerprint = uploaded_file.get("upload_id") if isinstance(uploaded_file, dict) else None
    if fingerprint is None:
        fingerprint = (uploaded_name(uploaded_file), hashlib.sha256(uploaded_bytes(uploaded_file)).digest())
    if state.get("image_source_fingerprint") != fingerprint:
        state["image_source_fingerprint"] = fingerprint
        state["image_buffers"] = {}
        for key in ("image_buffer", "image_selected_name", "image_member_selector", "processed_image_bgr",
                    "image_restoration_metadata", "image_export_result", "image_export_zip_bytes", "image_source_error"):
            state.pop(key, None)
        try:
            state["image_source_items"] = extract_image_items(uploaded_file)
        except (TypeError, ValueError) as exc:
            state["image_source_items"] = []
            state["image_source_error"] = str(exc)
    if "image_source_error" in state:
        st.error(f"Image dataset validation failed: {state['image_source_error']}")
        return
    items = state["image_source_items"]

    names = [item["name"] for item in items]
    if len(names) > 1:
        selected_name = st.selectbox(
            "Select Image",
            options=names,
            key="image_member_selector",
        )
    else:
        selected_name = names[0]

    if selected_name != state.get("image_selected_name"):
        buffer = state["image_buffers"].get(selected_name)
        if buffer is None:
            selected_item = next(item for item in items if item["name"] == selected_name)
            buffer = ImagePipelineBuffer(decode_image_bytes(selected_item["bytes"]))
        state["image_selected_name"] = selected_name
        bind_image_buffer(state, buffer, selecting=True)

    original = state["image_buffer"].raw_bgr
    height, width = original.shape[:2]
    channels = 1 if original.ndim == 2 else original.shape[2]
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Images", len(items))
    metric2.metric("Selected", selected_name)
    metric3.metric("Resolution", f"{width}x{height}")
    metric4.metric("Channels", channels)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Focus & Sharpness",
        "🚀 Dual-Tier Restoration",
        "🎨 Color Balance & Grading",
        "📄 Export Studio",
    ])
    with tab1:
        render_tab_sharpness(uploaded_file)
    with tab2:
        render_tab_restoration(uploaded_file)
    with tab3:
        render_tab_color(uploaded_file)
    with tab4:
        render_tab_export(uploaded_file)
