"""Auto white balance and ASC CDL Lift-Gamma-Gain color grading tab."""

from __future__ import annotations

import streamlit as st

from modules.image.color_engine import COLOR_GRADING_PRESETS
from modules.image.pipeline_image import GradeSettings, bind_image_buffer
from ui.tabs_image.common import bgr_to_display
from ui.utils_ux import render_tab_guidance_header


def render_tab_color(uploaded_file) -> None:
    """Render auto white balance and ASC CDL Lift-Gamma-Gain color grading."""
    del uploaded_file
    render_tab_guidance_header(
        title="Color Balance & Grading",
        objective="Neutralize color casts and apply filmic tonal grading.",
        methodology=(
            "Gray-World or Max-RGB chromatic scaling paired with ASC CDL "
            "Lift-Gamma-Gain non-linear transfer and Rec.709 saturation control."
        ),
        deliverable="Color-balanced image persisted for Export Studio.",
    )
    buffer = st.session_state.get("image_buffer")
    if buffer is None:
        st.info("Select a decodable image to configure color grading.")
        return

    current = buffer.grade_settings or GradeSettings()
    sync = st.session_state.pop("image_sync_color_widgets", False)
    for param, value in vars(current).items():
        key = "image_color_" + ("sat" if param == "saturation" else param)
        if sync or key not in st.session_state:
            st.session_state[key] = value

    col_a, col_b = st.columns(2)
    wb_method = col_a.selectbox(
        "Auto White Balance",
        options=["None", "gray_world", "max_rgb"],
        format_func=lambda val: {
            "None": "Disabled",
            "gray_world": "Gray-World (μ_gray / μ_C)",
            "max_rgb": "Max-RGB (1.0 / max I_C)",
        }[val],
        key="image_color_wb_method",
    )
    preset = col_b.selectbox(
        "Color Grading Preset",
        options=list(COLOR_GRADING_PRESETS),
        key="image_color_preset",
    )

    lift = 0.0
    gamma = 1.0
    gain = 1.0
    saturation = 1.0
    if preset == "Custom":
        c1, c2, c3, c4 = st.columns(4)
        lift = c1.slider("Lift (Shadows)", -0.25, 0.25, step=0.01, key="image_color_lift")
        gamma = c2.slider("Gamma (Midtones)", 0.5, 2.0, step=0.05, key="image_color_gamma")
        gain = c3.slider("Gain (Highlights)", 0.5, 1.8, step=0.05, key="image_color_gain")
        saturation = c4.slider("Saturation", 0.0, 2.0, step=0.05, key="image_color_sat")

    apply_col, reset_col = st.columns(2)
    apply_clicked = apply_col.button(
        "Apply Color Pipeline",
        width="stretch",
        key="image_apply_color",
    )
    reset_clicked = reset_col.button(
        "Reset Color Grade",
        width="stretch",
        key="image_reset_color",
    )

    if reset_clicked:
        bind_image_buffer(st.session_state, buffer.without_grading())
        st.session_state["image_sync_color_widgets"] = True
        st.rerun()

    settings = GradeSettings(wb_method, preset, lift, gamma, gain, saturation)
    if settings != current or (apply_clicked and buffer.graded_bgr is None):
        base_img = st.session_state["image_buffer"].get_base_for_color_grading()
        buffer = buffer.with_grading(settings.apply(base_img), settings)
        bind_image_buffer(st.session_state, buffer)
    if apply_clicked:
        st.success(f"Applied color pipeline (WB={wb_method}, preset={preset}).")

    left, right = st.columns(2)
    restored = buffer.restored_bgr is not None
    left.markdown("**Restored Base**" if restored else "**Original Raw**")
    left.image(bgr_to_display(buffer.restored_bgr if restored else buffer.raw_bgr), width="stretch")
    right.markdown("**Color Graded Output**")
    right.image(bgr_to_display(buffer.get_final_output()), width="stretch")
