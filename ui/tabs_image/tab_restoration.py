"""Dual-tier classical and neural image restoration UI."""

from __future__ import annotations

import cv2
import streamlit as st

from modules.image.clahe_contrast import apply_clahe_contrast
from modules.image.denoiser import (
    apply_bilateral_denoise,
    apply_gaussian_denoise,
)
from modules.image.neural_restoration import (
    MODEL_REGISTRY,
    get_neural_model_status,
    run_neural_restoration,
)
from ui.tabs_image.common import bgr_to_display


TIER_CLASSICAL = "⚡ Tier-1: Classical Fast Filters (OpenCV)"
TIER_NEURAL = "🚀 Tier-2: SOTA Deep Learning Models (ONNX/PyTorch)"


def render_tab_restoration(uploaded_file) -> None:
    """Render classical and optional ONNX restoration controls."""
    original = st.session_state.get("image_original_bgr")
    if original is None:
        st.info("Select a decodable image to configure restoration.")
        return

    tier = st.radio(
        "Processing Tier",
        options=[TIER_CLASSICAL, TIER_NEURAL],
        key="image_processing_tier",
    )
    settings = {"tier": tier}

    if tier == TIER_CLASSICAL:
        st.caption("Fast local processing; no neural model weights are required.")
        apply_clahe = st.toggle(
            "Enable LAB-space CLAHE",
            value=True,
            key="restoration_enable_clahe",
        )
        control1, control2 = st.columns(2)
        clip_limit = control1.slider(
            "CLAHE Clip Limit",
            min_value=1.0,
            max_value=8.0,
            value=2.0,
            step=0.1,
            disabled=not apply_clahe,
            key="restoration_clahe_clip",
        )
        tile_grid_size = control2.slider(
            "CLAHE Tile Grid Size",
            min_value=2,
            max_value=16,
            value=8,
            step=1,
            disabled=not apply_clahe,
            key="restoration_clahe_grid",
        )
        denoise_method = st.radio(
            "Denoising",
            options=["None", "Bilateral Filter", "Gaussian Blur"],
            horizontal=True,
            key="restoration_denoise_method",
        )
        settings.update({
            "apply_clahe": apply_clahe,
            "clip_limit": clip_limit,
            "tile_grid_size": tile_grid_size,
            "denoise_method": denoise_method,
        })
        if denoise_method == "Bilateral Filter":
            parameter1, parameter2, parameter3 = st.columns(3)
            settings["diameter"] = parameter1.slider(
                "Bilateral Diameter",
                3,
                15,
                9,
                2,
                key="restoration_bilateral_diameter",
            )
            settings["sigma_color"] = parameter2.slider(
                "Sigma Color",
                10.0,
                150.0,
                75.0,
                5.0,
                key="restoration_sigma_color",
            )
            settings["sigma_space"] = parameter3.slider(
                "Sigma Space",
                10.0,
                150.0,
                75.0,
                5.0,
                key="restoration_sigma_space",
            )
        elif denoise_method == "Gaussian Blur":
            parameter1, parameter2 = st.columns(2)
            settings["gaussian_kernel"] = parameter1.slider(
                "Gaussian Kernel",
                3,
                15,
                5,
                2,
                key="restoration_gaussian_kernel",
            )
            settings["gaussian_sigma"] = parameter2.slider(
                "Gaussian Sigma",
                0.0,
                5.0,
                1.0,
                0.1,
                key="restoration_gaussian_sigma",
            )
    else:
        st.caption(
            "ONNX Runtime is optional. Missing runtime or local model weights "
            "automatically activates an audited OpenCV fallback."
        )
        model_name = st.selectbox(
            "Neural Restoration Model",
            options=list(MODEL_REGISTRY),
            format_func=lambda name: (
                f"{name} — {MODEL_REGISTRY[name]['recommendation_label']}"
            ),
            key="restoration_neural_model",
        )
        model_status = get_neural_model_status(model_name)
        st.caption(
            f"Task: {model_status['task']} | Native scale: "
            f"{model_status['native_scale']}×"
        )
        if model_status["fallback_only"]:
            st.info(
                f"{model_status['availability_note']} The audited OpenCV "
                "low-light fallback will be used."
            )
        elif not model_status["weights_available"]:
            st.warning(
                "Local ONNX weights are missing. Expected file: "
                f"{model_status['model_path']}. OpenCV fallback will be used."
            )
        elif not model_status["integrity_valid"]:
            st.error(
                "Local ONNX weights failed checksum verification. Neural execution "
                "is disabled and OpenCV fallback will be used."
            )
        elif not model_status["runtime_available"]:
            st.warning(
                "ONNX Runtime is unavailable. OpenCV fallback will be used."
            )
        parameter1, parameter2, parameter3 = st.columns(3)
        scale_options = [1, 2, 4]
        native_scale = model_status["native_scale"]
        scale_factor = parameter1.selectbox(
            "Final Output Scale",
            options=scale_options,
            index=scale_options.index(native_scale),
            format_func=lambda value: f"{value}×",
            key=(
                "restoration_scale_factor_"
                f"{list(MODEL_REGISTRY).index(model_name)}"
            ),
        )
        if scale_factor != native_scale:
            parameter1.warning(
                f"The model runs natively at {native_scale}× and will be "
                f"post-resized to {scale_factor}×."
            )
        blend_alpha = parameter2.slider(
            "Blend Alpha",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="restoration_blend_alpha",
            help="0 keeps the resized original; 1 uses the restored output.",
        )
        device = parameter3.selectbox(
            "Device",
            options=(
                ["cpu", "cuda"]
                if model_status["cuda_available"]
                else ["cpu"]
            ),
            key="restoration_device",
        )
        settings.update({
            "model_name": model_name,
            "scale_factor": scale_factor,
            "blend_alpha": blend_alpha,
            "device": device,
        })

    apply_col, reset_col = st.columns(2)
    apply_clicked = apply_col.button(
        "Run Restoration",
        width="stretch",
        key="image_run_restoration",
    )
    reset_clicked = reset_col.button(
        "Reset to Original",
        width="stretch",
        key="image_reset_restoration",
    )

    selected_name = st.session_state.get("image_selected_name", "image")
    if apply_clicked:
        with st.spinner("Running selected image restoration pipeline..."):
            if tier == TIER_CLASSICAL:
                processed = original.copy()
                if settings["denoise_method"] == "Bilateral Filter":
                    processed = apply_bilateral_denoise(
                        processed,
                        d=settings["diameter"],
                        sigma_color=settings["sigma_color"],
                        sigma_space=settings["sigma_space"],
                    )
                elif settings["denoise_method"] == "Gaussian Blur":
                    processed = apply_gaussian_denoise(
                        processed,
                        kernel_size=settings["gaussian_kernel"],
                        sigma=settings["gaussian_sigma"],
                    )
                if settings["apply_clahe"]:
                    processed = apply_clahe_contrast(
                        processed,
                        clip_limit=settings["clip_limit"],
                        tile_grid_size=(
                            settings["tile_grid_size"],
                            settings["tile_grid_size"],
                        ),
                    )
                result_metadata = {
                    "status": "success",
                    "runtime": "opencv",
                    "message": "Applied Tier-1 classical restoration.",
                }
            else:
                result_metadata = run_neural_restoration(
                    original,
                    model_name=settings["model_name"],
                    scale_factor=settings["scale_factor"],
                    device=settings["device"],
                )
                processed = result_metadata["image_bgr"]
                if settings["blend_alpha"] < 1.0:
                    original_for_blend = cv2.resize(
                        original,
                        (processed.shape[1], processed.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    if (
                        original_for_blend.ndim == 3
                        and processed.ndim == 3
                        and original_for_blend.shape[2] != processed.shape[2]
                    ):
                        original_for_blend = original_for_blend[:, :, :3]
                        processed = processed[:, :, :3]
                    processed = cv2.addWeighted(
                        processed,
                        settings["blend_alpha"],
                        original_for_blend,
                        1.0 - settings["blend_alpha"],
                        0.0,
                    )

            st.session_state["processed_image_bgr"] = processed
            st.session_state.setdefault("processed_images_bgr", {})[
                selected_name
            ] = processed.copy()
            st.session_state["image_restoration_metadata"] = result_metadata
            st.session_state.pop("image_export_result", None)

    if reset_clicked:
        st.session_state["processed_image_bgr"] = original.copy()
        st.session_state.setdefault("processed_images_bgr", {}).pop(
            selected_name, None
        )
        st.session_state.pop("image_restoration_metadata", None)
        st.session_state.pop("image_export_result", None)
        st.rerun()

    metadata = st.session_state.get("image_restoration_metadata")
    if metadata:
        if metadata.get("status") == "fallback":
            st.warning(metadata["message"])
        else:
            st.success(metadata["message"])
        st.caption(
            f"Runtime: {metadata.get('runtime', 'unknown')} | "
            f"Latency: {float(metadata.get('latency_ms', 0.0)):.1f} ms"
        )

    processed = st.session_state.get("processed_image_bgr", original)
    original_col, processed_col = st.columns(2)
    original_col.markdown("**Original Raw**")
    original_col.image(bgr_to_display(original), width="stretch")
    processed_col.markdown("**Processed / Enhanced Output**")
    processed_col.image(bgr_to_display(processed), width="stretch")
