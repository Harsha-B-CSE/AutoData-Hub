"""
AutoData-Hub Tab 1: Data Health Audit.

Displays the composite quality score, missingness heatmap,
duplicate/outlier/type-error breakdowns, and the column-level schema profile.
"""

from __future__ import annotations
from typing import Optional

import pandas as pd
import streamlit as st

from ui.theme import (
    COLORS,
    PLOTLY_LAYOUT,
    badge,
    kpi_row,
    metric_card,
    render_plotly_artifact,
    section_header,
)
from ui.utils_ux import render_tab_guidance_header


def _quality_grade_color(score: float) -> str:
    if score >= 90:
        return COLORS["success"]
    elif score >= 70:
        return COLORS["warning"]
    return COLORS["danger"]


def _quality_grade_badge(score: float) -> str:
    if score >= 90:
        return badge("A — Excellent", "success")
    elif score >= 80:
        return badge("B — Good", "success")
    elif score >= 70:
        return badge("C — Fair", "warning")
    elif score >= 60:
        return badge("D — Poor", "warning")
    return badge("F — Critical", "danger")


def render_audit_tab(df: pd.DataFrame, module_result=None) -> None:
    """
    Renders the Health Audit tab.

    Parameters
    ----------
    df : pd.DataFrame
        The active dataset (any version).
    module_result : ModuleResult | None
        Populated ModuleResult from modules.health_audit.auditor if pipeline ran.
    """
    render_tab_guidance_header(
        title="Data Health Audit",
        objective="Quantify baseline dataset integrity before applying transformations.",
        methodology=(
            "Computes composite Quality Score Q across missingness, duplicate rows, "
            "IQR outliers, and mixed-type errors without mutating the source data."
        ),
        deliverable="Quality grade card, missingness heatmap, and column schema profile.",
    )
    if df is None or df.empty:
        st.info("Upload a dataset and run the Health Audit to see results here.")
        return

    flash_message = st.session_state.pop("health_audit_flash_message", None)
    if flash_message:
        st.success(flash_message)

    # ── Run Health Audit Button ──
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_audit = st.button(
            "🏥 Run Health Audit",
            key="btn_run_health_audit",
            width="stretch",
        )

    if run_audit:
        active_ver = st.session_state.get("active_version", "v0_raw")
        src_df = st.session_state["dataset_versions"].get(active_ver)
        if src_df is None:
            st.error("No active dataset loaded.")
        else:
            with st.spinner("Analyzing missingness, duplicate rows, data types, and computing Quality Score Q..."):
                try:
                    from modules.health_audit import audit_dataset_health
                    mr = audit_dataset_health(df=src_df, dataset_version=active_ver)
                    st.session_state["module_results"]["health_audit"] = mr
                    st.session_state.pop("pdf_bytes_cache", None)
                    st.session_state.pop("pdf_filename_cache", None)
                    st.session_state["health_audit_flash_message"] = (
                        f"Health Audit complete for {active_ver}."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Health Audit failed: {exc}")

    # ── Quality Score KPI Section ────────────────────────────────────────────
    metrics = module_result.metrics if module_result else {}
    q_score = metrics.get("quality_score", None)

    if q_score is not None:
        color = _quality_grade_color(q_score)
        grade_html = _quality_grade_badge(q_score)
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:1.5rem;
                        background:{COLORS['surface']};border:1px solid {COLORS['border']};
                        border-radius:12px;padding:1.2rem 1.8rem;margin-bottom:1.25rem;">
                <div style="text-align:center;min-width:110px;">
                    <div style="font-size:3.2rem;font-weight:800;color:{color};line-height:1;">
                        {q_score:.1f}
                    </div>
                    <div style="font-size:0.7rem;color:{COLORS['text_muted']};
                                text-transform:uppercase;letter-spacing:0.1em;">
                        Quality Score
                    </div>
                </div>
                <div style="width:1px;height:70px;background:{COLORS['border']};"></div>
                <div>
                    <div style="font-size:1rem;font-weight:600;margin-bottom:0.4rem;">
                        {grade_html}
                    </div>
                    <div style="font-size:0.78rem;color:{COLORS['text_muted']};">
                        Q = max(0, 100 − (w₁·M + w₂·D + w₃·O + w₄·T) × 100)
                    </div>
                    <div style="font-size:0.72rem;color:{COLORS['text_muted']};margin-top:0.2rem;">
                        w₁=0.35 (miss) · w₂=0.20 (dup) · w₃=0.25 (outlier) · w₄=0.20 (type)
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── KPI Row ──────────────────────────────────────────────────────────────
    rows, cols = df.shape
    null_total = int(df.isnull().sum().sum())
    null_pct = round(null_total / max(rows * cols, 1) * 100, 1)
    if "duplicate_count" in metrics:
        dup_count = int(metrics["duplicate_count"])
    else:
        try:
            dup_count = int(df.duplicated().sum())
        except TypeError:
            dup_count = 0
    outlier_count = int(metrics.get("outlier_count", 0))

    st.markdown(
        kpi_row([
            {"label": "Rows", "value": f"{rows:,}"},
            {"label": "Columns", "value": f"{cols:,}"},
            {"label": "Missing Cells", "value": f"{null_total:,}", "color": COLORS["warning"] if null_pct > 5 else ""},
            {"label": "Null %", "value": f"{null_pct}%", "color": COLORS["danger"] if null_pct > 20 else COLORS["warning"] if null_pct > 5 else ""},
            {"label": "Duplicates", "value": f"{dup_count:,}", "color": COLORS["warning"] if dup_count > 0 else ""},
            {"label": "Outliers (IQR)", "value": f"{outlier_count:,}"},
        ]),
        unsafe_allow_html=True,
    )

    # ── Missingness Heatmap ──────────────────────────────────────────────────
    st.markdown(section_header("Missingness Heatmap"), unsafe_allow_html=True)

    if "missingness_heatmap" in (metrics.get("artifacts") or {}):
        render_plotly_artifact(
            metrics["artifacts"]["missingness_heatmap"],
            key_prefix="audit_metrics_missingness",
        )
    elif module_result and hasattr(module_result, "artifacts") and "missingness_heatmap" in (module_result.artifacts or {}):
        render_plotly_artifact(
            module_result.artifacts["missingness_heatmap"],
            key_prefix="audit_missingness",
        )
    else:
        # Build inline heatmap if no precomputed artifact
        try:
            import plotly.graph_objects as go
            import numpy as np

            sample = df.head(300)
            null_matrix = sample.isnull().astype(int).T
            fig = go.Figure(
                go.Heatmap(
                    z=null_matrix.values,
                    x=list(range(len(sample))),
                    y=null_matrix.index.tolist(),
                    colorscale=[[0, "#0D3349"], [1, "#EF4444"]],
                    showscale=True,
                    colorbar=dict(
                        title="Missing",
                        tickvals=[0, 1],
                        ticktext=["Present", "Null"],
                        thickness=12,
                    ),
                    hovertemplate="Row %{x}<br>Col: %{y}<br>Missing: %{z}<extra></extra>",
                )
            )
            layout = dict(PLOTLY_LAYOUT)
            layout["title"] = f"Null Pattern (first {len(sample)} rows)"
            layout["height"] = max(200, min(500, len(df.columns) * 22))
            layout["yaxis"] = dict(tickfont=dict(size=10))
            fig.update_layout(**layout)
            render_plotly_artifact(fig, key_prefix="audit_inline_missingness")
        except Exception as exc:
            st.warning(f"Could not render heatmap: {exc}")

    # ── Per-Column Null Profile ───────────────────────────────────────────────
    st.markdown(section_header("Column-Level Null Profile"), unsafe_allow_html=True)

    null_by_col = df.isnull().sum()
    null_pct_by_col = (null_by_col / max(len(df), 1) * 100).round(2)
    column_metrics = metrics.get("columns", {})
    unique_values = []
    for column in df.columns:
        stored_unique = column_metrics.get(str(column), {}).get("unique_count")
        if stored_unique is not None:
            unique_values.append(int(stored_unique))
            continue
        try:
            unique_values.append(int(df[column].nunique()))
        except TypeError:
            unique_values.append(int(df[column].map(repr).nunique()))
    profile_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Null Count": null_by_col.values,
        "Null %": null_pct_by_col.values,
        "Unique Values": unique_values,
        "Sample": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
    }).sort_values("Null %", ascending=False).reset_index(drop=True)

    st.dataframe(
        profile_df,
        width="stretch",
        height=min(400, 35 * len(profile_df) + 38),
        hide_index=True,
    )

    # ── Feature Distribution Charts ──────────────────────────────────────────
    if module_result and hasattr(module_result, "artifacts"):
        distributions = (module_result.artifacts or {}).get("feature_distributions")
        if distributions:
            st.markdown(section_header("Feature Distributions"), unsafe_allow_html=True)
            render_plotly_artifact(
                distributions,
                key_prefix="audit_feature_distribution",
                columns=2,
            )

    # ── Audit Issues ─────────────────────────────────────────────────────────
    if module_result and module_result.issues:
        st.markdown(section_header("⚠️ Audit Issues"), unsafe_allow_html=True)
        for issue in module_result.issues:
            st.warning(issue)
