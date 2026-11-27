"""Focus and sharpness profiling tab."""

from __future__ import annotations

import streamlit as st

from modules.image.sharpness_evaluator import (
    evaluate_sharpness_laplacian,
)
from ui.tabs_image.common import bgr_to_display
from ui.utils_ux import render_tab_guidance_header


def render_tab_sharpness(uploaded_file) -> None:
    """Render Laplacian variance and Tenengrad focus diagnostics."""
    render_tab_guidance_header(
        title="Focus & Sharpness Profiling",
        objective="Identify blur, motion softness, and low-light exposure issues.",
        methodology=(
            "Evaluates spatial Laplacian variance Var(∇²I), Sobel Tenengrad "
            "gradient energy, and mean luminance against user-tuned thresholds."
        ),
        deliverable="Automated sharpness verdict and restoration tier recommendation.",
    )
    buffer = st.session_state.get("image_buffer")
    if buffer is None:
        st.info("Select a decodable image to run sharpness evaluation.")
        return
    image_bgr = buffer.raw_bgr

    cutoff = st.slider(
        "Laplacian Variance Cutoff: var(∇²I)",
        min_value=10.0,
        max_value=500.0,
        value=100.0,
        step=1.0,
        key="image_sharpness_cutoff",
        help="Images below this threshold are classified as blurry.",
    )
    if "image_focus_metrics" not in st.session_state:
        st.session_state["image_focus_metrics"] = evaluate_sharpness_laplacian(image_bgr)
    telemetry = st.session_state["image_focus_metrics"]
    laplacian_variance = telemetry["laplacian_variance"]
    tenengrad_score = telemetry["tenengrad_score"]
    mean_luminance = telemetry["mean_luminance"]

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Laplacian Variance", f"{laplacian_variance:,.2f}")
    metric2.metric("Tenengrad Score", f"{tenengrad_score:,.2f}")
    metric3.metric("Mean Luminance", f"{mean_luminance:,.2f}")

    if laplacian_variance < cutoff:
        st.warning(
            f"BLURRY: Laplacian variance {laplacian_variance:.2f} is below "
            f"the selected cutoff {cutoff:.2f}."
        )
    else:
        st.success(
            f"SHARP: Laplacian variance {laplacian_variance:.2f} meets "
            f"the selected cutoff {cutoff:.2f}."
        )

    st.subheader("Auto-Degradation Alert")
    if mean_luminance < 60.0:
        st.warning(
            "Image is dark. Recommended restoration: Tier-1 LAB CLAHE with "
            "bilateral denoising."
        )
    elif laplacian_variance < cutoff:
        recommended = (
            "Restormer (High-Res Deblur)"
            if laplacian_variance < cutoff * 0.5
            else "NAFNet (Fast Denoise/Deblur)"
        )
        st.warning(f"Image is blurry. Recommended neural restoration: {recommended}.")
    else:
        st.success(
            "No severe focus or low-light degradation detected. Tier-1 classical "
            "enhancement is usually sufficient."
        )
    height, width = image_bgr.shape[:2]
    channels = 1 if image_bgr.ndim == 2 else image_bgr.shape[2]
    st.caption(f"Resolution: {width} × {height} | Channels: {channels}")
    st.image(
        bgr_to_display(image_bgr),
        caption=st.session_state.get("image_selected_name", "Original image"),
        width="stretch",
    )
