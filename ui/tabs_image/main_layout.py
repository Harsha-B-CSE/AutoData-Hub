"""Master layout for the Streamlit Computer Vision workspace."""

from __future__ import annotations

import hashlib

import streamlit as st

from ui.tabs_image.common import decode_image_bytes, extract_image_items
from ui.tabs_image.tab_export import render_tab_export
from ui.tabs_image.tab_restoration import render_tab_restoration
from ui.tabs_image.tab_sharpness import render_tab_sharpness


def render_image_app(uploaded_file) -> None:
    """Mount focus, enhancement, and export tabs for an uploaded image dataset."""
    try:
        items = extract_image_items(uploaded_file)
    except (TypeError, ValueError) as exc:
        st.error(f"Image dataset validation failed: {exc}")
        return

    dataset_hash = hashlib.sha256()
    for item in items:
        dataset_hash.update(item["name"].encode("utf-8", errors="replace"))
        dataset_hash.update(item["bytes"])
    fingerprint = dataset_hash.hexdigest()

    if st.session_state.get("image_source_fingerprint") != fingerprint:
        st.session_state["image_source_fingerprint"] = fingerprint
        st.session_state["image_source_items"] = items
        st.session_state["processed_images_bgr"] = {}
        st.session_state.pop("processed_image_bgr", None)
        st.session_state.pop("image_original_bgr", None)
        st.session_state.pop("image_selected_name", None)
        st.session_state.pop("image_export_result", None)
        st.session_state.pop("image_export_zip_bytes", None)
    else:
        st.session_state["image_source_items"] = items

    names = [item["name"] for item in items]
    if len(names) > 1:
        selected_name = st.selectbox(
            "Select Image",
            options=names,
            key="image_member_selector",
        )
    else:
        selected_name = names[0]

    if selected_name != st.session_state.get("image_selected_name"):
        selected_item = next(item for item in items if item["name"] == selected_name)
        original = decode_image_bytes(selected_item["bytes"])
        st.session_state["image_selected_name"] = selected_name
        st.session_state["image_original_bgr"] = original
        existing_processed = st.session_state.get("processed_images_bgr", {}).get(
            selected_name
        )
        st.session_state["processed_image_bgr"] = (
            existing_processed.copy()
            if existing_processed is not None
            else original.copy()
        )
        st.session_state.pop("image_export_result", None)

    original = st.session_state["image_original_bgr"]
    height, width = original.shape[:2]
    channels = 1 if original.ndim == 2 else original.shape[2]
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Images", len(items))
    metric2.metric("Selected", selected_name)
    metric3.metric("Resolution", f"{width}x{height}")
    metric4.metric("Channels", channels)

    tab1, tab2, tab3 = st.tabs([
        "🔍 Focus & Sharpness",
        "🚀 Dual-Tier Restoration",
        "📄 Export Studio",
    ])
    with tab1:
        render_tab_sharpness(uploaded_file)
    with tab2:
        render_tab_restoration(uploaded_file)
    with tab3:
        render_tab_export(uploaded_file)
