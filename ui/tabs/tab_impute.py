"""
AutoData-Hub Tab 3: Missing Value Imputation.

Displays imputation strategy controls, CV-RMSE benchmarking results,
per-column imputer recommendations, and post-imputation data preview.
"""

from __future__ import annotations
from html import escape
from typing import Optional

import pandas as pd
import streamlit as st

from core.schema_inference import ExecutionStatus, ModuleResult
from ui.theme import (
    COLORS,
    PLOTLY_LAYOUT,
    badge,
    kpi_row,
    render_plotly_artifact,
    section_header,
)


def _render_strategy_controls() -> tuple[str, dict, bool, bool, bool]:
    """Renders imputation strategy controls and returns selected config."""
    st.markdown(section_header("🧩 Imputation Strategy"), unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        strategy = st.selectbox(
            "Global Strategy",
            options=["auto_cv", "knn", "mice_ridge", "mice_rf", "median", "mean", "most_frequent"],
            index=["auto_cv", "knn", "mice_ridge", "mice_rf", "median", "mean", "most_frequent"].index(
                st.session_state.get("cfg_impute_strategy", "auto_cv")
            ),
            key="impute_global_strategy",
            help=(
                "**Auto CV**: Per-column cross-validation RMSE benchmark (recommended).\n\n"
                "**KNN**: K-Nearest Neighbors using all numeric features.\n\n"
                "**MICE Ridge**: Iterative imputation with Bayesian Ridge.\n\n"
                "**MICE RF**: Iterative imputation with Random Forest (slower, non-linear).\n\n"
                "**Median/Mean**: Univariate for numeric columns.\n\n"
                "**Most Frequent**: Mode imputation for categorical."
            ),
        )
        st.session_state["cfg_impute_strategy"] = strategy

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_impute = st.button("🔧 Run Imputation", key="btn_run_impute", width="stretch")

    with st.expander("Advanced Options", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            add_mnar = st.toggle(
                "Add MNAR indicator columns",
                value=st.session_state.get("cfg_mnar_indicators", True),
                key="impute_mnar_toggle",
                help="Creates binary {col}_is_missing columns for features with ≥15% missingness.",
            )
            st.session_state["cfg_mnar_indicators"] = add_mnar
        with col_b:
            drop_high = st.toggle(
                "Drop columns ≥95% missing",
                value=st.session_state.get("cfg_drop_high_missing", False),
                key="impute_drop_toggle",
                help="Removes features exceeding 95% missingness before imputation.",
            )
            st.session_state["cfg_drop_high_missing"] = drop_high

    # Per-column override (only visible when not auto_cv)
    col_strategies = {}
    if strategy != "auto_cv":
        st.markdown(section_header("Per-Column Overrides (Optional)"), unsafe_allow_html=True)
        # This will be populated when we have the dataframe

    return strategy, col_strategies, add_mnar, drop_high, run_impute


def _render_benchmark_results(bench_data: dict, df: pd.DataFrame) -> None:
    """Renders CV-RMSE benchmarking chart and recommendations."""
    st.markdown(section_header("📊 CV-RMSE Benchmark Results"), unsafe_allow_html=True)

    scores = bench_data.get("scores", {})
    recommendations = bench_data.get("recommendations", {})

    if not scores:
        st.info("Run imputation with **Auto CV** strategy to see cross-validation benchmarks.")
        return

    try:
        import plotly.graph_objects as go

        # Build comparison bar chart: per column, show RMSE for each strategy
        columns = list(scores.keys())
        if not columns:
            return

        strategies = ["mean", "median", "knn", "mice_ridge"]
        colors = {"mean": "#38BDF8", "median": "#818CF8", "knn": "#A78BFA", "mice_ridge": "#FB923C"}

        fig = go.Figure()

        for strat in strategies:
            vals = []
            for col in columns:
                val = scores.get(col, {}).get(strat, None)
                vals.append(val if val is not None else 0)
            fig.add_trace(go.Bar(
                name=strat.title(),
                x=columns,
                y=vals,
                marker_color=colors[strat],
                opacity=0.85,
                hovertemplate="Column: %{x}<br>Strategy: " + strat + "<br>RMSE: %{y:.4f}<extra></extra>",
            ))

        # Highlight recommended strategy per column
        for i, col in enumerate(columns):
            rec = recommendations.get(col, "")
            if rec and rec in strategies:
                idx = strategies.index(rec)
                fig.add_annotation(
                    x=col,
                    y=scores[col].get(rec, 0),
                    text="★",
                    showarrow=True,
                    arrowcolor=colors[rec],
                    arrowsize=1.5,
                    ax=0,
                    ay=-25,
                    font=dict(size=16, color=colors[rec]),
                )

        layout = dict(PLOTLY_LAYOUT)
        layout["title"] = "Cross-Validation RMSE by Strategy (★ = Recommended)"
        layout["xaxis"]["title"] = "Column"
        layout["yaxis"]["title"] = "RMSE (lower is better)"
        layout["barmode"] = "group"
        layout["height"] = max(350, len(columns) * 35)
        fig.update_layout(**layout)

        render_plotly_artifact(fig, key_prefix="imputation_benchmark")

    except Exception as exc:
        st.warning(f"Could not render benchmark chart: {exc}")

    # Recommendation table
    st.markdown(section_header("Per-Column Recommendations"), unsafe_allow_html=True)
    rec_rows = []
    for col in sorted(recommendations.keys()):
        rec = recommendations[col]
        col_scores = scores.get(col, {})
        best_rmse = col_scores.get(rec, "—")
        rec_rows.append({
            "Column": col,
            "Recommended": rec.upper(),
            "CV-RMSE": f"{best_rmse:.4f}" if isinstance(best_rmse, (int, float)) else "—",
            "Missing %": round(df[col].isna().sum() / len(df) * 100, 1),
        })

    if rec_rows:
        rec_df = pd.DataFrame(rec_rows)
        st.dataframe(rec_df, width="stretch", hide_index=True)


def _render_imputation_preview(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    transform_result,
) -> None:
    """Renders before/after preview of imputed columns."""
    st.markdown(section_header("👁️ Imputation Preview (Before → After)"), unsafe_allow_html=True)

    # Find columns that were actually imputed
    imputed_cols = [c for c in original_df.columns if original_df[c].isna().any() and c in imputed_df.columns]

    if not imputed_cols:
        st.success("No missing values to impute in the active dataset.")
        return

    # Show only columns that had missing values
    max_preview = 6
    preview_cols = imputed_cols[:max_preview]

    before_tab, after_tab, diff_tab = st.tabs(["🔴 Before", "🟢 After", "📋 Summary"])

    with before_tab:
        st.dataframe(original_df[preview_cols].head(20), width="stretch", hide_index=False)

    with after_tab:
        st.dataframe(imputed_df[preview_cols].head(20), width="stretch", hide_index=False)

    with diff_tab:
        summary_rows = []
        for col in imputed_cols:
            orig_nulls = int(original_df[col].isna().sum())
            if orig_nulls > 0:
                new_nulls = int(imputed_df[col].isna().sum())
                summary_rows.append({
                    "Column": col,
                    "Dtype": str(original_df[col].dtype),
                    "Nulls Before": orig_nulls,
                    "Nulls After": new_nulls,
                    "Filled": orig_nulls - new_nulls,
                })

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    # Show audit log
    if transform_result and transform_result.audit_log:
        with st.expander("📋 Imputation Audit Log", expanded=False):
            for entry in transform_result.audit_log:
                st.markdown(
                    f'<p style="font-size:0.75rem;font-family:monospace;'
                    f'color:{COLORS["text_muted"]};margin:0;">'
                    f'{escape(str(entry))}</p>',
                    unsafe_allow_html=True,
                )


def render_impute_tab(
    df: pd.DataFrame,
    module_result=None,
    transform_result=None,
    bench_result=None,
) -> None:
    """
    Renders the Missing Value Imputation tab (Tab 3).

    Parameters
    ----------
    df : pd.DataFrame
        The active dataset (v1_privacy or later).
    module_result : ModuleResult | None
        ModuleResult from modules.imputation if pipeline ran.
    transform_result : TransformResult | None
        TransformResult containing the imputed DataFrame (v2_imputed).
    bench_result : dict | None
        Benchmark results from MissingValueImputer.benchmark_dataset().
    """
    if df is None or df.empty:
        st.info("Upload a dataset and run the Privacy Firewall first, then run Imputation.")
        return

    flash_message = st.session_state.pop("imputation_flash_message", None)
    if flash_message:
        st.success(flash_message)

    existing_new_duplicates = int(
        (module_result.metrics if module_result else {}).get("new_duplicates", 0)
    )
    if existing_new_duplicates > 0:
        st.warning(
            f"Post-imputation audit found {existing_new_duplicates} newly duplicated rows."
        )

    # Render controls and get user config
    strategy, col_strategies, add_mnar, drop_high, run_impute = _render_strategy_controls()

    # Show metrics from previous run if available
    metrics = module_result.metrics if module_result else {}
    n_imputed = int(metrics.get("n_missing_imputed", 0))
    n_dropped = int(metrics.get("n_dropped_high_missing", 0))
    n_mnar = int(metrics.get("n_mnar_indicators_added", 0))

    st.markdown(
        kpi_row([
            {"label": "Values Imputed", "value": f"{n_imputed:,}", "color": COLORS["success"] if n_imputed > 0 else ""},
            {"label": "Columns Dropped", "value": str(n_dropped)},
            {"label": "MNAR Indicators", "value": str(n_mnar)},
            {"label": "Strategy Used", "value": metrics.get("strategy_used", strategy).upper()},
        ]),
        unsafe_allow_html=True,
    )

    # Run imputation on button click (event-gated)
    if run_impute:
        source_version = "v1_privacy"
        src_df = st.session_state["dataset_versions"].get(source_version)
        if src_df is None:
            st.error("Run the Privacy Firewall before imputation.")
        else:
            with st.spinner("Evaluating CV-RMSE across MICE, KNN, and Simple Imputers..."):
                try:
                    from modules.imputation.imputer import (
                        MissingValueImputer,
                        ImputationStrategy,
                    )

                    imputer = MissingValueImputer(
                        default_strategy=ImputationStrategy(strategy),
                        drop_high_missing=drop_high,
                    )

                    downstream_modules = {"imputation", "anomaly"}
                    retained_actions = [
                        action
                        for action in st.session_state.get("action_ledger", [])
                        if getattr(action, "module", None) not in downstream_modules
                    ]
                    step_id = max(
                        (getattr(action, "step_id", 0) for action in retained_actions),
                        default=0,
                    ) + 1

                    tr = imputer.impute(
                        df=src_df,
                        strategy=strategy,
                        column_strategies=col_strategies if col_strategies else None,
                        add_mnar_indicators=add_mnar,
                        dataset_version=source_version,
                        start_step_id=step_id,
                    )

                    if tr.status == ExecutionStatus.SUCCESS and tr.transformed_df is not None:
                        common_columns = [
                            col for col in src_df.columns if col in tr.transformed_df.columns
                        ]
                        n_imputed = sum(
                            int((src_df[col].isna() & tr.transformed_df[col].notna()).sum())
                            for col in common_columns
                        )
                        duplicate_metadata = tr.metadata or {}
                        new_duplicates = int(
                            duplicate_metadata.get("new_duplicates", 0)
                        )
                        imputation_result = ModuleResult(
                            status=ExecutionStatus.SUCCESS,
                            module_name="imputation",
                            dataset_version="v2_imputed",
                            metrics={
                                "n_missing_imputed": n_imputed,
                                "n_dropped_high_missing": sum(
                                    len(action.target_columns)
                                    for action in tr.applied_actions
                                    if action.action_type == "drop_high_missing"
                                ),
                                "n_mnar_indicators_added": sum(
                                    action.action_type == "mnar_indicator"
                                    for action in tr.applied_actions
                                ),
                                "strategy_used": strategy,
                                "benchmark_scores": duplicate_metadata.get(
                                    "benchmark_scores", {}
                                ),
                                "pre_imputation_duplicate_count": int(
                                    duplicate_metadata.get(
                                        "pre_imputation_duplicate_count", 0
                                    )
                                ),
                                "post_imputation_duplicate_count": int(
                                    duplicate_metadata.get(
                                        "post_imputation_duplicate_count", 0
                                    )
                                ),
                                "new_duplicates": new_duplicates,
                            },
                            issues=(
                                [
                                    f"Imputation introduced {new_duplicates} new exact duplicate rows."
                                ]
                                if new_duplicates > 0
                                else []
                            ),
                            artifacts={},
                        )

                        versions = st.session_state["dataset_versions"]
                        versions.pop("v2_imputed", None)
                        versions.pop("v3_anomaly", None)
                        versions["v2_imputed"] = tr.transformed_df.copy(deep=True)

                        module_results = st.session_state.setdefault("module_results", {})
                        transform_results = st.session_state.setdefault("transform_results", {})
                        for module_name in downstream_modules:
                            module_results.pop(module_name, None)
                            transform_results.pop(module_name, None)
                        health_result = module_results.get("health_audit")
                        if (
                            health_result is not None
                            and health_result.dataset_version
                            in {"v2_imputed", "v3_anomaly"}
                        ):
                            module_results.pop("health_audit", None)
                        module_results["imputation"] = imputation_result
                        transform_results["imputation"] = tr

                        st.session_state["active_version"] = "v2_imputed"
                        st.session_state["_pending_active_version"] = "v2_imputed"
                        st.session_state["action_ledger"] = retained_actions + tr.applied_actions
                        st.session_state["audit_logs"] = [
                            entry
                            for entry in st.session_state.get("audit_logs", [])
                            if not any(
                                tag in str(entry).upper()
                                for tag in ("[IMPUTATION]", "[ANOMALY]")
                            )
                        ] + tr.audit_log
                        st.session_state["imputation_flash_message"] = (
                            f"Imputation complete: {n_imputed} values filled."
                        )
                        st.session_state.pop("pdf_bytes_cache", None)
                        st.session_state.pop("pdf_filename_cache", None)
                        st.session_state.pop("pipeline_bytes_cache", None)
                        st.session_state.pop("pipeline_fingerprint_cache", None)
                        st.session_state.pop("_anomaly_winsor_preflight", None)
                        st.rerun()
                    else:
                        st.error(tr.error_message or "Imputation returned no output.")

                except Exception as exc:
                    st.error(f"Imputation failed: {exc}")

    # Render benchmark results if available
    if bench_result or (module_result and hasattr(module_result, "metrics") and "benchmark_scores" in module_result.metrics):
        bench = bench_result or module_result.metrics.get("benchmark_scores", {})
        if bench:
            _render_benchmark_results(bench, df)

    # Render preview if transform result available
    if transform_result and transform_result.transformed_df is not None:
        _render_imputation_preview(df, transform_result.transformed_df, transform_result)
    elif "v2_imputed" in st.session_state.get("dataset_versions", {}):
        # Also show preview if v2_imputed exists in session but we're viewing a different version
        pass

    # Show column-level missingness summary
    st.markdown(section_header("📋 Column Missingness Summary"), unsafe_allow_html=True)
    null_info = df.isnull().sum()
    missing_df = pd.DataFrame({
        "Column": null_info.index,
        "Null Count": null_info.values,
        "Null %": (null_info / len(df) * 100).round(2),
        "Dtype": [str(df[c].dtype) for c in null_info.index],
    }).sort_values("Null %", ascending=False).reset_index(drop=True)

    st.dataframe(missing_df, width="stretch", hide_index=True)
