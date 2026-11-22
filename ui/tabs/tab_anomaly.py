"""
AutoData-Hub Tab 4: Ensemble Anomaly Detection.

Displays PCA 2D/3D scatter plots, majority voting controls,
detector comparison charts, and anomaly flagged records.
"""

from __future__ import annotations
import datetime
from html import escape
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from core.schema_inference import (
    ExecutionStatus,
    ModuleResult,
    TransformationAction,
    TransformResult,
)
from ui.theme import (
    COLORS,
    PLOTLY_LAYOUT,
    badge,
    kpi_row,
    render_plotly_artifact,
    section_header,
)


_THRESHOLD_OPTIONS = [
    "Adaptive 3-Sigma EVT (Recommended)",
    "Fixed Quantile Ratio (5%)",
]
_MITIGATION_OPTIONS = [
    "Quantile Capping (Winsorization)",
    "Row Deletion (Pruning)",
]


def _numeric_feature_columns(df: pd.DataFrame) -> list:
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
        and not str(column).endswith("_is_missing")
        and not str(column).startswith("anomaly_")
    ]


def _render_voting_controls() -> tuple[str, str, float, list, bool, bool]:
    """Renders anomaly detection configuration controls."""
    st.markdown(section_header("🔍 Ensemble Configuration"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        threshold_label = st.radio(
            "Threshold Mode",
            _THRESHOLD_OPTIONS,
            index=(
                0
                if st.session_state.get("cfg_threshold_mode", "adaptive_3sigma")
                == "adaptive_3sigma"
                else 1
            ),
            key="anomaly_threshold_mode",
        )
    with col2:
        mitigation_label = st.radio(
            "Mitigation Strategy",
            _MITIGATION_OPTIONS,
            index=(
                0
                if st.session_state.get("cfg_mitigation_strategy", "winsorize")
                == "winsorize"
                else 1
            ),
            key="anomaly_mitigation_strategy",
        )

    with st.expander("Detector Selection", expanded=False):
        algo_options = ["isolation_forest", "lof", "one_class_svm"]
        selected_algos = st.multiselect(
            "Active Detectors",
            options=algo_options,
            default=st.session_state.get("cfg_anomaly_algos", algo_options),
            key="anomaly_algo_multiselect",
            help="Select which algorithms participate in the ensemble.",
        )
        st.session_state["cfg_anomaly_algos"] = selected_algos

    threshold_mode = (
        "adaptive_3sigma"
        if threshold_label == _THRESHOLD_OPTIONS[0]
        else "fixed_quantile"
    )
    mitigation_strategy = (
        "winsorize" if mitigation_label == _MITIGATION_OPTIONS[0] else "prune"
    )
    st.session_state["cfg_threshold_mode"] = threshold_mode
    st.session_state["cfg_mitigation_strategy"] = mitigation_strategy

    contamination = st.slider(
        "Fixed quantile ratio",
        min_value=0.01,
        max_value=0.30,
        step=0.01,
        value=float(st.session_state.get("cfg_contamination", 0.05)),
        format="%.2f",
        disabled=threshold_mode == "adaptive_3sigma",
        help="Used only by Fixed Quantile Ratio mode.",
        key="anomaly_contamination",
    )
    st.session_state["cfg_contamination"] = contamination

    if mitigation_strategy == "prune":
        st.warning("Row deletion permanently removes records flagged by the ensemble.")
        pruning_confirmed = st.checkbox(
            "I understand that flagged rows will be permanently removed",
            key="anomaly_prune_acknowledgement",
        )
    else:
        pruning_confirmed = False

    run_anomaly = st.button(
        "🎯 Apply Anomaly Mitigation",
        key="btn_run_anomaly",
        width="stretch",
        disabled=mitigation_strategy == "prune" and not pruning_confirmed,
    )
    return (
        threshold_mode,
        mitigation_strategy,
        contamination,
        selected_algos,
        pruning_confirmed,
        run_anomaly,
    )


def _apply_anomaly_mitigation(
    annotated_df: pd.DataFrame,
    source_df: pd.DataFrame,
    strategy: str,
    step_id: int,
    selected_columns: Optional[list] = None,
    confirmed_bounds: Optional[dict] = None,
) -> tuple[pd.DataFrame, TransformationAction, dict]:
    """Apply winsorization or row pruning and return its replay action."""
    treated_df = annotated_df.copy(deep=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if strategy == "prune":
        removed_rows = int(treated_df.get("anomaly_flag", pd.Series(dtype=int)).sum())
        treated_df = (
            treated_df.loc[treated_df["anomaly_flag"].eq(0)]
            .copy()
            .reset_index(drop=True)
        )
        action = TransformationAction(
            step_id=step_id,
            module="anomaly",
            action_type="prune_anomalies",
            target_columns=["rows"],
            parameters={"rows_removed": removed_rows},
            code_snippet=(
                "df = df.loc[df['anomaly_flag'].eq(0)].copy().reset_index(drop=True)"
            ),
        )
        return treated_df, action, {
            "rows_removed": removed_rows,
            "values_capped": 0,
            "audit_entry": (
                f"[{timestamp}] [ANOMALY] Pruned {removed_rows} flagged rows."
            ),
        }

    bounds = {}
    values_capped = 0
    columns_to_cap = selected_columns or _numeric_feature_columns(source_df)
    for column in columns_to_cap:
        if confirmed_bounds is not None and column in confirmed_bounds:
            lower, upper = confirmed_bounds[column]
        else:
            lower = float(source_df[column].quantile(0.01))
            upper = float(source_df[column].quantile(0.99))
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        before = treated_df[column].copy()
        treated_df[column] = treated_df[column].clip(lower=lower, upper=upper)
        values_capped += int((before.ne(treated_df[column]) & before.notna()).sum())
        bounds[column] = (lower, upper)

    code_bounds = repr(bounds)
    action = TransformationAction(
        step_id=step_id,
        module="anomaly",
        action_type="winsorize_anomalies",
        target_columns=[str(column) for column in bounds],
        parameters={
            "quantile_lower": 0.01,
            "quantile_upper": 0.99,
            "values_capped": values_capped,
            "bounds": {str(column): value for column, value in bounds.items()},
        },
        code_snippet=(
            f"_anomaly_winsor_bounds = {code_bounds}\n"
            "for _anomaly_column, (_anomaly_lower, _anomaly_upper) in "
            "_anomaly_winsor_bounds.items():\n"
            "    df[_anomaly_column] = df[_anomaly_column].clip("
            "lower=_anomaly_lower, upper=_anomaly_upper)"
        ),
    )
    return treated_df, action, {
        "rows_removed": 0,
        "values_capped": values_capped,
        "audit_entry": (
            f"[{timestamp}] [ANOMALY] Winsorized {values_capped} values at Q01/Q99."
        ),
    }


def _build_winsor_preflight(source_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calculate immutable Q01/Q99 bounds and impact counts before approval."""
    rows = []
    bounds = {}
    for column in _numeric_feature_columns(source_df):
        lower = float(source_df[column].quantile(0.01))
        upper = float(source_df[column].quantile(0.99))
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        affected = int(
            ((source_df[column] < lower) | (source_df[column] > upper)).sum()
        )
        bounds[column] = (lower, upper)
        rows.append({
            "Column Name": str(column),
            "Lower Bound (Q0.01)": lower,
            "Upper Bound (Q0.99)": upper,
            "Affected Row Count": affected,
        })
    return pd.DataFrame(rows), bounds


def _source_fingerprint(df: pd.DataFrame) -> int:
    """Content-sensitive fingerprint used to reject stale preflight state."""
    if df is None:
        return 0
    value_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
    column_hash = hash(tuple(str(column) for column in df.columns))
    return value_hash ^ column_hash


def _build_committed_anomaly_results(
    detection_tr: TransformResult,
    detection_mr: ModuleResult,
    source_df: pd.DataFrame,
    strategy: str,
    step_id: int,
    selected_columns: Optional[list] = None,
    confirmed_bounds: Optional[dict] = None,
) -> tuple[TransformResult, ModuleResult, dict]:
    """Create immutable post-mitigation result contracts."""
    treated_df, mitigation_action, treatment = _apply_anomaly_mitigation(
        annotated_df=detection_tr.transformed_df,
        source_df=source_df,
        strategy=strategy,
        step_id=step_id,
        selected_columns=selected_columns,
        confirmed_bounds=confirmed_bounds,
    )
    treatment_log = detection_tr.audit_log + [treatment["audit_entry"]]
    committed_tr = TransformResult(
        status=detection_tr.status,
        transformed_df=treated_df,
        applied_actions=detection_tr.applied_actions + [mitigation_action],
        audit_log=treatment_log,
        error_message=detection_tr.error_message,
    )
    committed_metrics = dict(detection_mr.metrics)
    committed_metrics.update({
        "mitigation_strategy": strategy,
        "rows_removed": treatment["rows_removed"],
        "values_capped": treatment["values_capped"],
        "output_rows": len(treated_df),
    })
    committed_mr = ModuleResult(
        status=detection_mr.status,
        module_name=detection_mr.module_name,
        dataset_version="v3_anomaly",
        metrics=committed_metrics,
        issues=detection_mr.issues,
        artifacts=detection_mr.artifacts,
    )
    return committed_tr, committed_mr, treatment


def _commit_anomaly_state(
    committed_tr: TransformResult,
    committed_mr: ModuleResult,
    retained_actions: list,
    treatment: dict,
) -> None:
    """Atomically replace the committed v3 lineage after mitigation approval."""
    versions = st.session_state["dataset_versions"]
    versions.pop("v3_anomaly", None)
    versions["v3_anomaly"] = committed_tr.transformed_df.copy(deep=True)
    st.session_state["active_version"] = "v3_anomaly"
    st.session_state["_pending_active_version"] = "v3_anomaly"
    st.session_state["action_ledger"] = retained_actions + committed_tr.applied_actions
    st.session_state["audit_logs"] = [
        entry
        for entry in st.session_state.get("audit_logs", [])
        if "[ANOMALY]" not in str(entry).upper()
    ] + committed_tr.audit_log
    st.session_state.setdefault("module_results", {})["anomaly"] = committed_mr
    health_result = st.session_state["module_results"].get("health_audit")
    if health_result is not None and health_result.dataset_version == "v3_anomaly":
        st.session_state["module_results"].pop("health_audit", None)
    st.session_state.setdefault("transform_results", {})["anomaly"] = committed_tr
    st.session_state["anomaly_flash_message"] = (
        "Anomaly mitigation complete: "
        f"{int(committed_mr.metrics.get('n_anomalies', 0))} rows flagged, "
        f"{treatment['rows_removed']} removed, "
        f"{treatment['values_capped']} values capped."
    )
    for cache_key in (
        "pdf_bytes_cache",
        "pdf_filename_cache",
        "pipeline_bytes_cache",
        "pipeline_fingerprint_cache",
    ):
        st.session_state.pop(cache_key, None)
    st.session_state.pop("_anomaly_winsor_preflight", None)


def _render_winsor_preflight(config_signature: tuple) -> None:
    """Render a pending winsor impact review without mutating committed lineage."""
    pending = st.session_state.get("_anomaly_winsor_preflight")
    if not pending:
        return

    current_source = st.session_state.get("dataset_versions", {}).get("v2_imputed")
    if (
        current_source is None
        or _source_fingerprint(current_source) != pending["source_fingerprint"]
        or pending.get("config_signature") != config_signature
    ):
        st.session_state.pop("_anomaly_winsor_preflight", None)
        st.warning(
            "Pending winsorization was discarded because its source or configuration changed."
        )
        return

    st.markdown(section_header("⚖️ Winsorization Pre-flight Impact"), unsafe_allow_html=True)
    st.dataframe(pending["impact_table"], width="stretch", hide_index=True)
    selection_key = f"anomaly_winsor_columns_{pending['id']}"
    selected_columns = st.multiselect(
        "Select Columns to Winsorize",
        options=list(pending["bounds"]),
        default=[],
        key=selection_key,
        help="Only explicitly selected columns will be clipped to their reviewed Q01/Q99 bounds.",
    )

    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        confirm = st.button(
            "Confirm & Apply Winsorization",
            key=f"confirm_winsor_{pending['id']}",
            width="stretch",
        )
    with cancel_col:
        cancel = st.button(
            "Cancel Winsorization",
            key=f"cancel_winsor_{pending['id']}",
            width="stretch",
        )

    if cancel:
        st.session_state.pop("_anomaly_winsor_preflight", None)
        st.rerun()
    if confirm:
        if not selected_columns:
            st.warning("Select at least one column before confirming winsorization.")
            return
        committed_tr, committed_mr, treatment = _build_committed_anomaly_results(
            detection_tr=pending["detection_tr"],
            detection_mr=pending["detection_mr"],
            source_df=current_source,
            strategy="winsorize",
            step_id=pending["step_id"],
            selected_columns=selected_columns,
            confirmed_bounds=pending["bounds"],
        )
        _commit_anomaly_state(
            committed_tr,
            committed_mr,
            pending["retained_actions"],
            treatment,
        )
        st.rerun()


def _render_pca_scatter(module_result, transform_result, df: pd.DataFrame) -> None:
    """Renders PCA 2D and 3D scatter plots with anomaly overlay."""
    st.markdown(section_header("📍 PCA Anomaly Projection"), unsafe_allow_html=True)

    artifacts = module_result.artifacts if module_result else {}
    pca_2d = artifacts.get("pca_anomaly_scatter") if isinstance(artifacts, dict) else None

    if pca_2d is not None:
        tab_2d, tab_3d = st.tabs(["2D Projection", "3D Projection (if available)"])
        with tab_2d:
            render_plotly_artifact(pca_2d, key_prefix="anomaly_pca_2d")

        with tab_3d:
            # Try to build 3D if enough features
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import RobustScaler
                import numpy as np
                import plotly.graph_objects as go

                numeric_cols = [
                    c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_bool_dtype(df[c])
                    and not str(c).endswith("_is_missing")
                    and not str(c).startswith("anomaly_")
                ]

                if len(numeric_cols) >= 3 and len(df) >= 3:
                    X = df[numeric_cols].fillna(df[numeric_cols].median()).to_numpy()
                    X_scaled = RobustScaler().fit_transform(X)

                    # Check if anomaly columns exist
                    artifact_flags = (
                        module_result.artifacts.get("pre_mitigation_flags")
                        if module_result and isinstance(module_result.artifacts, dict)
                        else None
                    )
                    if artifact_flags is not None and len(artifact_flags) == len(df):
                        flags = np.asarray(artifact_flags)
                    elif "anomaly_flag" in df.columns:
                        flags = df["anomaly_flag"].to_numpy()
                    elif (
                        transform_result
                        and transform_result.transformed_df is not None
                        and len(transform_result.transformed_df) == len(df)
                    ):
                        flags = transform_result.transformed_df.get("anomaly_flag", pd.Series([0]*len(df))).to_numpy()
                    else:
                        flags = np.zeros(len(df))

                    pca = PCA(n_components=3, random_state=42)
                    components = pca.fit_transform(X_scaled)

                    normal_mask = flags == 0
                    anom_mask = flags == 1

                    fig_3d = go.Figure()
                    fig_3d.add_trace(go.Scatter3d(
                        x=components[normal_mask, 0].tolist(),
                        y=components[normal_mask, 1].tolist(),
                        z=components[normal_mask, 2].tolist(),
                        mode="markers",
                        name="Normal",
                        marker=dict(color="rgba(56,189,248,0.5)", size=4),
                    ))
                    fig_3d.add_trace(go.Scatter3d(
                        x=components[anom_mask, 0].tolist(),
                        y=components[anom_mask, 1].tolist(),
                        z=components[anom_mask, 2].tolist(),
                        mode="markers",
                        name="Anomaly",
                        marker=dict(
                            color="rgba(248,113,113,0.9)",
                            size=6,
                            symbol="x",
                            line=dict(width=1, color="white"),
                        ),
                    ))

                    layout = dict(PLOTLY_LAYOUT)
                    layout["title"] = (
                        f"PCA 3D Projection — Anomaly Overlay "
                        f"(PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
                        f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%, "
                        f"PC3={pca.explained_variance_ratio_[2]*100:.1f}%)"
                    )
                    layout["scene"] = dict(
                        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                        zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
                        xaxis=dict(gridcolor=COLORS["chart_grid"]),
                        yaxis=dict(gridcolor=COLORS["chart_grid"]),
                        zaxis=dict(gridcolor=COLORS["chart_grid"]),
                    )
                    layout["height"] = 500
                    fig_3d.update_layout(**layout)

                    render_plotly_artifact(fig_3d, key_prefix="anomaly_pca_3d")
                else:
                    st.info("Need at least 3 rows and 3 numeric features for 3D PCA projection.")
            except Exception as exc:
                st.warning(f"Could not render 3D PCA: {exc}")
    else:
        # Fallback: build inline if no precomputed artifact
        if "anomaly_flag" in df.columns:
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import RobustScaler
                import numpy as np
                import plotly.graph_objects as go

                numeric_cols = [
                    c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_bool_dtype(df[c])
                    and not str(c).endswith("_is_missing")
                    and not str(c).startswith("anomaly_")
                ]

                if len(numeric_cols) >= 2:
                    X = df[numeric_cols].fillna(df[numeric_cols].median()).to_numpy()
                    X_scaled = RobustScaler().fit_transform(X)
                    flags = df["anomaly_flag"].to_numpy()

                    pca = PCA(n_components=2, random_state=42)
                    components = pca.fit_transform(X_scaled)

                    normal_mask = flags == 0
                    anom_mask = flags == 1

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=components[normal_mask, 0].tolist(),
                        y=components[normal_mask, 1].tolist(),
                        mode="markers",
                        name="Normal",
                        marker=dict(color="rgba(56,189,248,0.5)", size=5),
                    ))
                    fig.add_trace(go.Scatter(
                        x=components[anom_mask, 0].tolist(),
                        y=components[anom_mask, 1].tolist(),
                        mode="markers",
                        name="Anomaly",
                        marker=dict(
                            color="rgba(248,113,113,0.9)",
                            size=8,
                            symbol="x",
                            line=dict(width=1, color="white"),
                        ),
                    ))

                    layout = dict(PLOTLY_LAYOUT)
                    layout["title"] = f"PCA 2D Projection — Anomaly Overlay (n={int(anom_mask.sum())} flagged)"
                    layout["xaxis"]["title"] = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)"
                    layout["yaxis"]["title"] = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
                    fig.update_layout(**layout)

                    render_plotly_artifact(fig, key_prefix="anomaly_inline_pca_2d")
                else:
                    st.info("Need at least 2 numeric features for PCA projection.")
            except Exception as exc:
                st.warning(f"Could not render PCA: {exc}")
        else:
            st.info("Run Anomaly Detection to see PCA projections.")


def _render_detector_charts(module_result) -> None:
    """Renders detector comparison and vote distribution charts."""
    artifacts = module_result.artifacts if module_result else {}

    if not artifacts:
        return

    st.markdown(section_header("📊 Detector Analysis"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        vote_dist = artifacts.get("detector_vote_distribution")
        if vote_dist is not None:
            render_plotly_artifact(vote_dist, key_prefix="anomaly_vote_distribution")

        score_dist = artifacts.get("anomaly_score_distribution")
        if score_dist is not None:
            render_plotly_artifact(score_dist, key_prefix="anomaly_score_distribution")

    with col2:
        det_comp = artifacts.get("detector_comparison")
        if det_comp is not None:
            render_plotly_artifact(det_comp, key_prefix="anomaly_detector_comparison")


def _render_detector_disagreement(module_result) -> None:
    """Render calibrated detector counts, vote consensus, and pairwise disagreement."""
    if not module_result:
        return

    metrics = module_result.metrics or {}
    summaries = metrics.get("detector_summaries", {})
    if not summaries:
        return

    st.markdown(section_header("🤝 Detector Disagreement"), unsafe_allow_html=True)
    model_columns = st.columns(3)
    model_specs = [
        ("isolation_forest", "N_IF"),
        ("lof", "N_LOF"),
        ("one_class_svm", "N_SVM"),
    ]
    for container, (model_name, label) in zip(model_columns, model_specs):
        summary = summaries.get(model_name, {})
        container.metric(
            label,
            int(summary.get("n_detected", 0)),
            help="Rows receiving a calibrated anomaly vote from this detector.",
        )

    n_rows = max(1, int(metrics.get("n_rows", 0)))
    consensus = metrics.get("vote_consensus_counts", {})
    consensus_table = pd.DataFrame([
        {
            "Consensus": "0 models",
            "Rows": int(consensus.get("zero_models", 0)),
            "Share": f"{int(consensus.get('zero_models', 0)) / n_rows * 100:.2f}%",
        },
        {
            "Consensus": "1 model",
            "Rows": int(consensus.get("one_model", 0)),
            "Share": f"{int(consensus.get('one_model', 0)) / n_rows * 100:.2f}%",
        },
        {
            "Consensus": "2 models (Majority)",
            "Rows": int(consensus.get("two_models", 0)),
            "Share": f"{int(consensus.get('two_models', 0)) / n_rows * 100:.2f}%",
        },
        {
            "Consensus": "3 models (Unanimous)",
            "Rows": int(consensus.get("three_models", 0)),
            "Share": f"{int(consensus.get('three_models', 0)) / n_rows * 100:.2f}%",
        },
    ])
    st.dataframe(consensus_table, width="stretch", hide_index=True)

    artifacts = module_result.artifacts or {}
    disagreement = artifacts.get("detector_disagreement_matrix")
    if isinstance(disagreement, pd.DataFrame) and not disagreement.empty:
        st.caption("Pairwise row-count disagreement matrix (symmetric XOR counts).")
        st.dataframe(disagreement, width="stretch")


def _render_anomaly_table(df: pd.DataFrame) -> None:
    """Renders the flagged anomaly records table."""
    if "anomaly_flag" not in df.columns:
        return

    st.markdown(section_header("🚩 Flagged Anomalies"), unsafe_allow_html=True)

    anom_df = df[df["anomaly_flag"] == 1].copy()
    if anom_df.empty:
        st.success("No anomalies detected with current configuration.")
        return

    # Show relevant columns: anomaly score, vote count, per-detector flags, and key features
    display_cols = ["anomaly_score_ensemble", "anomaly_vote_count"]
    for det in ["anomaly_isolation_forest", "anomaly_lof", "anomaly_one_class_svm"]:
        if det in anom_df.columns:
            display_cols.append(det)

    # Add a few high-variance numeric columns for context
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not str(c).startswith("anomaly_")
        and not str(c).endswith("_is_missing")
    ][:5]
    display_cols.extend(numeric_cols)

    st.dataframe(
        anom_df[display_cols].sort_values("anomaly_score_ensemble", ascending=False).head(50),
        width="stretch",
        height=300,
    )

    st.caption(f"Showing top 50 of {len(anom_df)} flagged anomalies (sorted by ensemble score).")


def _render_audit_log(transform_result) -> None:
    """Renders the anomaly detection audit log."""
    if transform_result and transform_result.audit_log:
        with st.expander("📋 Anomaly Detection Audit Log", expanded=False):
            for entry in transform_result.audit_log:
                st.markdown(
                    f'<p style="font-size:0.75rem;font-family:monospace;'
                    f'color:{COLORS["text_muted"]};margin:0;">'
                    f'{escape(str(entry))}</p>',
                    unsafe_allow_html=True,
                )


def render_anomaly_tab(
    df: pd.DataFrame,
    module_result=None,
    transform_result=None,
) -> None:
    """
    Renders the Ensemble Anomaly Detection tab (Tab 4).

    Parameters
    ----------
    df : pd.DataFrame
        The active dataset (v2_imputed or later).
    module_result : ModuleResult | None
        ModuleResult from modules.anomaly if pipeline ran.
    transform_result : TransformResult | None
        TransformResult containing the anomaly-annotated DataFrame (v3_anomaly).
    """
    if df is None or df.empty:
        st.info("Upload a dataset and run Imputation first, then run Anomaly Detection.")
        return

    flash_message = st.session_state.pop("anomaly_flash_message", None)
    if flash_message:
        st.success(flash_message)

    # Render controls
    (
        threshold_mode,
        mitigation_strategy,
        contamination,
        selected_algos,
        pruning_confirmed,
        run_anomaly,
    ) = _render_voting_controls()

    if not _numeric_feature_columns(df):
        st.warning("No continuous numeric columns available for anomaly scoring.")

    current_config_signature = (
        threshold_mode,
        mitigation_strategy,
        round(float(contamination), 6),
        tuple(selected_algos),
    )
    _render_winsor_preflight(current_config_signature)

    pending_result = st.session_state.get("_anomaly_winsor_preflight")
    display_module_result = (
        pending_result["detection_mr"] if pending_result else module_result
    )
    display_transform_result = (
        pending_result["detection_tr"] if pending_result else transform_result
    )
    if pending_result:
        st.info("Displaying pending detection results. No v3 changes are committed yet.")

    # Show metrics from previous run
    metrics = display_module_result.metrics if display_module_result else {}
    n_anomalies = int(metrics.get("n_anomalies", 0))
    anomaly_rate = float(metrics.get("anomaly_rate", 0.0))
    active_algos = metrics.get("active_algorithms", selected_algos)
    required_votes = int(metrics.get("required_votes", 0))
    active_model_count = int(metrics.get("active_model_count", len(active_algos)))
    result_threshold_mode = metrics.get("threshold_mode", threshold_mode)

    st.markdown(
        kpi_row([
            {"label": "Anomalies Found", "value": f"{n_anomalies:,}", "color": COLORS["danger"] if n_anomalies > 0 else COLORS["success"]},
            {"label": "Anomaly Rate", "value": f"{anomaly_rate * 100:.2f}%"},
            {
                "label": "Threshold",
                "value": (
                    "Adaptive 3σ"
                    if result_threshold_mode == "adaptive_3sigma"
                    else f"Fixed {float(metrics.get('contamination_param', contamination)) * 100:.0f}%"
                ),
            },
            {"label": "Consensus", "value": f"{required_votes}/{active_model_count}"},
        ]),
        unsafe_allow_html=True,
    )

    # Run anomaly detection on button click
    if run_anomaly:
        source_version = "v2_imputed"
        src_df = st.session_state["dataset_versions"].get(source_version)
        if src_df is None:
            st.error("Run imputation before anomaly detection.")
        else:
            if not selected_algos:
                st.error("Please select at least one detector.")
            else:
                with st.spinner("Fitting Isolation Forest, LOF, and OC-SVM ensemble voting models..."):
                    try:
                        from modules.anomaly.detector import (
                            detect_anomalies,
                        )

                        retained_actions = [
                            action
                            for action in st.session_state.get("action_ledger", [])
                            if getattr(action, "module", None) != "anomaly"
                        ]
                        step_id = max(
                            (getattr(action, "step_id", 0) for action in retained_actions),
                            default=0,
                        ) + 1

                        tr, mr = detect_anomalies(
                            df=src_df,
                            contamination=contamination,
                            algorithms=selected_algos,
                            dataset_version=source_version,
                            start_step_id=step_id,
                            threshold_mode=threshold_mode,
                            k_sigma=3.0,
                            min_vote_ratio=0.66,
                        )

                        if (
                            tr.status in {ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL}
                            and tr.transformed_df is not None
                        ):
                            if int(mr.metrics.get("active_model_count", 0)) == 0:
                                zero_metrics = dict(mr.metrics)
                                zero_metrics.update({
                                    "mitigation_strategy": "none",
                                    "rows_removed": 0,
                                    "values_capped": 0,
                                    "output_rows": len(tr.transformed_df),
                                })
                                zero_mr = ModuleResult(
                                    status=mr.status,
                                    module_name=mr.module_name,
                                    dataset_version="v3_anomaly",
                                    metrics=zero_metrics,
                                    issues=mr.issues,
                                    artifacts=mr.artifacts,
                                )
                                treatment = {
                                    "rows_removed": 0,
                                    "values_capped": 0,
                                }
                                _commit_anomaly_state(
                                    tr, zero_mr, retained_actions, treatment
                                )
                                st.rerun()
                            elif mitigation_strategy == "winsorize":
                                impact_table, bounds = _build_winsor_preflight(src_df)
                                if not bounds:
                                    no_op_metrics = dict(mr.metrics)
                                    no_op_metrics.update({
                                        "mitigation_strategy": "none",
                                        "rows_removed": 0,
                                        "values_capped": 0,
                                        "output_rows": len(tr.transformed_df),
                                    })
                                    no_op_mr = ModuleResult(
                                        status=mr.status,
                                        module_name=mr.module_name,
                                        dataset_version="v3_anomaly",
                                        metrics=no_op_metrics,
                                        issues=mr.issues + [
                                            "No finite winsorization bounds were available; detection annotations were committed without capping."
                                        ],
                                        artifacts=mr.artifacts,
                                    )
                                    _commit_anomaly_state(
                                        tr,
                                        no_op_mr,
                                        retained_actions,
                                        {"rows_removed": 0, "values_capped": 0},
                                    )
                                    st.rerun()
                                else:
                                    st.session_state["_anomaly_winsor_preflight"] = {
                                        "id": datetime.datetime.now(
                                            datetime.timezone.utc
                                        ).strftime("%Y%m%d%H%M%S%f"),
                                        "source_fingerprint": _source_fingerprint(src_df),
                                        "config_signature": current_config_signature,
                                        "detection_tr": tr,
                                        "detection_mr": mr,
                                        "retained_actions": retained_actions,
                                        "step_id": step_id + len(tr.applied_actions),
                                        "impact_table": impact_table,
                                        "bounds": bounds,
                                    }
                                    st.rerun()
                            else:
                                committed_tr, committed_mr, treatment = (
                                    _build_committed_anomaly_results(
                                        detection_tr=tr,
                                        detection_mr=mr,
                                        source_df=src_df,
                                        strategy="prune",
                                        step_id=step_id + len(tr.applied_actions),
                                    )
                                )
                                _commit_anomaly_state(
                                    committed_tr,
                                    committed_mr,
                                    retained_actions,
                                    treatment,
                                )
                                st.rerun()
                        else:
                            st.error(tr.error_message or "Anomaly detection returned no output.")

                    except Exception as exc:
                        st.error(f"Anomaly detection failed: {exc}")

    # Render PCA scatter plots
    _render_pca_scatter(display_module_result, display_transform_result, df)

    # Render detector analysis charts
    _render_detector_charts(display_module_result)
    _render_detector_disagreement(display_module_result)

    # Render anomaly table
    # Use the transformed df if available, otherwise current df
    flagged_artifact = (
        display_module_result.artifacts.get("flagged_records")
        if display_module_result and isinstance(display_module_result.artifacts, dict)
        else None
    )
    if isinstance(flagged_artifact, pd.DataFrame) and not flagged_artifact.empty:
        display_df = flagged_artifact
    else:
        display_df = (
            display_transform_result.transformed_df
            if display_transform_result and display_transform_result.transformed_df is not None
            else df
        )
    _render_anomaly_table(display_df)

    # Audit log
    _render_audit_log(display_transform_result)
