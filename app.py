"""
AutoData-Hub — MLOps Data Preprocessing Engine
Master Streamlit Application Entry Point

Pipeline: Ingestion → Health Audit → Privacy Firewall → Imputation → Anomaly Ensemble → Export
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.theme import inject_theme, COLORS
from ui.sidebar import render_sidebar
from ui.tabs.tab_audit import render_audit_tab
from ui.tabs.tab_privacy import render_privacy_tab
from ui.tabs.tab_impute import render_impute_tab
from ui.tabs.tab_anomaly import render_anomaly_tab
from ui.tabs.tab_export import render_export_tab


TABULAR_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather", ".json",
    ".jsonl",
}
IMAGE_EXTENSIONS = {".zip", ".png", ".jpg", ".jpeg", ".webp"}


def detect_file_modality(file_name: str) -> str | None:
    """Classify a supported upload by its final filename extension."""
    extension = Path(str(file_name or "")).suffix.lower()
    if extension in TABULAR_EXTENSIONS:
        return "tabular"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    return None


# ---------------------------------------------------------------------------
# Page Configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoData-Hub",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "AutoData-Hub v1.0 — Privacy-First MLOps Data Preprocessing Engine",
    },
)


# ---------------------------------------------------------------------------
# Inject Dark Obsidian Theme
# ---------------------------------------------------------------------------
inject_theme()


# ---------------------------------------------------------------------------
# Cached Module Runners (Event-Gated Execution Prevention)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=4)
def _run_health_audit(df_hash: int, dataset_version: str) -> tuple:
    """Cached health audit runner. Returns (ModuleResult, None)."""
    from modules.health_audit import audit_dataset_health
    df = st.session_state["dataset_versions"][dataset_version]
    mr = audit_dataset_health(df=df, dataset_version=dataset_version)
    return mr


@st.cache_data(show_spinner=False, max_entries=4)
def _run_privacy_firewall(df_hash: int, dataset_version: str, mode: str) -> tuple:
    """Cached privacy firewall runner. Returns (TransformResult, ModuleResult)."""
    from modules.privacy.detector import PIIDetector
    from modules.privacy.masker import PIIMasker
    df = st.session_state["dataset_versions"][dataset_version]
    findings = PIIDetector().detect(df)
    masker = PIIMasker(session_salt=st.session_state.get("privacy_session_salt"))
    tr = masker.mask_dataframe(
        df,
        detections=findings,
        strategy_overrides={col: mode for col in findings},
        start_step_id=1,
    )
    from core.schema_inference import ModuleResult
    metrics = {
        "masking_mode": mode,
        "pii_columns_detected": len(findings),
        "pii_cells_detected": sum(item.match_count for item in findings.values()),
        "pii_cells_masked": sum(
            int(action.parameters.get("changed_cells", 0))
            for action in tr.applied_actions
        ),
        "pii_findings": {
            str(col): [item.pii_type.value]
            for col, item in findings.items()
        },
        "header_detections": {
            str(col): item.pii_type.value
            for col, item in findings.items()
            if item.detection_method in {"semantic_header", "hybrid"}
        },
        "detection_methods": {
            str(col): item.detection_method
            for col, item in findings.items()
        },
    }
    mr = ModuleResult(
        status=tr.status,
        module_name="privacy",
        dataset_version=dataset_version,
        metrics=metrics,
        issues=[],
        artifacts={},
    )
    return tr, mr


@st.cache_data(show_spinner=False, max_entries=4)
def _run_imputation(df_hash: int, dataset_version: str, strategy: str, add_mnar: bool, drop_high: bool) -> tuple:
    """Cached imputation runner. Returns (TransformResult, ModuleResult, benchmark_dict)."""
    from modules.imputation.imputer import MissingValueImputer, ImputationStrategy
    df = st.session_state["dataset_versions"][dataset_version]
    imputer = MissingValueImputer(
        default_strategy=ImputationStrategy(strategy),
        drop_high_missing=drop_high,
    )
    step_id = len(st.session_state.get("action_ledger", [])) + 1
    tr = imputer.impute(
        df=df,
        strategy=strategy,
        add_mnar_indicators=add_mnar,
        dataset_version=dataset_version,
        start_step_id=step_id,
    )
    bench = {}
    if strategy == "auto_cv":
        bench = imputer.benchmark_dataset(df)

    from core.schema_inference import ModuleResult, ExecutionStatus
    metrics = {
        "n_missing_imputed": sum(1 for a in tr.applied_actions if "impute" in a.action_type),
        "n_dropped_high_missing": sum(1 for a in tr.applied_actions if a.action_type == "drop_high_missing"),
        "n_mnar_indicators_added": sum(1 for a in tr.applied_actions if a.action_type == "mnar_indicator"),
        "strategy_used": strategy,
        "benchmark_scores": bench,
    }
    mr = ModuleResult(
        status=ExecutionStatus.SUCCESS,
        module_name="imputation",
        dataset_version=dataset_version,
        metrics=metrics,
        issues=[],
        artifacts={},
    )
    return tr, mr, bench


@st.cache_data(show_spinner=False, max_entries=4)
def _run_anomaly_detection(df_hash: int, dataset_version: str, contamination: float, vote_threshold: int, algorithms: list) -> tuple:
    """Cached anomaly detection runner. Returns (TransformResult, ModuleResult)."""
    from modules.anomaly.detector import detect_anomalies
    df = st.session_state["dataset_versions"][dataset_version]
    step_id = len(st.session_state.get("action_ledger", [])) + 1
    tr, mr = detect_anomalies(
        df=df,
        contamination=contamination,
        algorithms=algorithms,
        vote_threshold=vote_threshold,
        dataset_version=dataset_version,
        start_step_id=step_id,
    )
    return tr, mr


def _get_df_hash(df) -> int:
    """Generate a hash for DataFrame to use as cache key."""
    if df is None or df.empty:
        return 0
    # Use shape + column names + memory hash for cache invalidation
    return hash((df.shape, tuple(df.columns), int(df.memory_usage(deep=True).sum())))


# ---------------------------------------------------------------------------
# Main Application Logic
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main application entry point.
    Handles ingestion, version routing, and tab rendering.
    """

    # ── Sidebar: Ingestion + Version Selector + Module Config ───────────────
    render_sidebar(detect_file_modality)

    active_modality = st.session_state.get("active_modality", "tabular")
    if active_modality == "image":
        image_dataset = st.session_state.get("image_dataset")
        if image_dataset is None:
            _render_landing_page(active_modality="image")
        else:
            _render_image_workspace(image_dataset)
        return

    # ── Determine Active Dataset ─────────────────────────────────────────────
    if (
        "v0_raw" in st.session_state.get("dataset_versions", {})
        and st.session_state.get("dataset_versions")["v0_raw"] is not None
    ):
        versions = st.session_state["dataset_versions"]
    else:
        _render_landing_page(active_modality="tabular")
        return

    active_ver = st.session_state.get("active_version")
    if active_ver not in versions or versions[active_ver] is None:
        active_ver = "v0_raw"
        st.session_state["active_version"] = active_ver

    active_df = versions[active_ver].copy(deep=True)

    # ── Header Bar ───────────────────────────────────────────────────────────
    _render_header_bar(active_ver, active_df)

    # ── Route to Module Results ──────────────────────────────────────────────
    module_results = st.session_state.get("module_results", {})
    transform_results = st.session_state.get("transform_results", {})
    action_ledger = st.session_state.get("action_ledger", [])
    audit_logs = st.session_state.get("audit_logs", [])

    # ── Tab Rendering ────────────────────────────────────────────────────────
    tab_labels = [
        "🏥 1. Health Audit",
        "🔒 2. Privacy Firewall",
        "⚙️ 3. Imputation",
        "🔍 4. Anomaly Ensemble",
        "📤 5. Export",
    ]

    tabs = st.tabs(tab_labels)

    # Tab 1: Health Audit
    with tabs[0]:
        health_result = module_results.get("health_audit")
        if (
            health_result is not None
            and health_result.dataset_version != active_ver
        ):
            health_result = None
        render_audit_tab(active_df, module_result=health_result)

    # Tab 2: Privacy Firewall
    with tabs[1]:
        privacy_result = module_results.get("privacy")
        privacy_tr = transform_results.get("privacy")
        privacy_source = versions["v0_raw"].copy(deep=True)
        render_privacy_tab(
            privacy_source,
            module_result=privacy_result,
            transform_result=privacy_tr,
        )

    # Tab 3: Imputation
    with tabs[2]:
        impute_result = module_results.get("imputation")
        impute_tr = transform_results.get("imputation")
        impute_source = versions.get("v1_privacy", active_df).copy(deep=True)
        render_impute_tab(
            impute_source,
            module_result=impute_result,
            transform_result=impute_tr,
            bench_result={},
        )

    # Tab 4: Anomaly Ensemble
    with tabs[3]:
        anomaly_result = module_results.get("anomaly")
        anomaly_tr = transform_results.get("anomaly")
        anomaly_source = versions.get("v2_imputed", active_df).copy(deep=True)
        render_anomaly_tab(
            anomaly_source,
            module_result=anomaly_result,
            transform_result=anomaly_tr,
        )

    # Tab 5: Export
    with tabs[4]:
        render_export_tab(active_df, module_results, action_ledger, audit_logs)


def _render_image_workspace(image_dataset: dict) -> None:
    """Render the interactive Computer Vision pipeline for an image dataset."""
    from ui.tabs_image.main_layout import render_image_app

    st.title("🖼️ Computer Vision & Image Enhancement Engine")
    render_image_app(image_dataset)


def _render_landing_page(active_modality: str = "tabular") -> None:
    """Renders the landing page when no dataset is loaded."""
    if active_modality == "image":
        st.markdown(
            f"""
            <div style="text-align:center;padding:4rem 2rem;">
                <div style="font-size:3rem;font-weight:800;
                            background:linear-gradient(135deg,{COLORS['accent']},#8B5CF6);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;margin-bottom:1rem;">Image Workspace</div>
                <div style="font-size:1.1rem;color:{COLORS['text_muted']};">
                    Upload ZIP, PNG, JPG, JPEG, or WEBP from the sidebar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:3rem;font-weight:800;
                        background:linear-gradient(135deg,{COLORS['accent']},#8B5CF6);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;margin-bottom:1rem;">
                AutoData-Hub
            </div>
            <div style="font-size:1.2rem;color:{COLORS['text_muted']};margin-bottom:3rem;">
                Privacy-First MLOps Data Preprocessing Engine
            </div>
            <div style="max-width:600px;margin:0 auto;text-align:left;">
                <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                            border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                    <strong style="color:{COLORS['accent']};">📂 Step 1:</strong> Upload a dataset via the sidebar
                    (CSV, TSV, XLSX, Parquet, JSON, or SQL connection).
                </div>
                <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                            border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                    <strong style="color:{COLORS['success']};">🏥 Step 2:</strong> Run <strong>Health Audit</strong> (Tab 1)
                    — composite quality score Q, missingness heatmap, duplicate/outlier/type analysis.
                </div>
                <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                            border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                    <strong style="color:{COLORS['danger']};">🔒 Step 3:</strong> Run <strong>Privacy Firewall</strong> (Tab 2)
                    — PII detection + Redaction / Hashing / Anonymization with live preview.
                </div>
                <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                            border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                    <strong style="color:{COLORS['warning']};">⚙️ Step 4:</strong> Run <strong>Imputation</strong> (Tab 3)
                    — Auto-CV benchmarking across KNN / MICE / Median / Mean with MNAR safeguards.
                </div>
                <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                            border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                    <strong style="color:{COLORS['accent']};">🔍 Step 5:</strong> Run <strong>Anomaly Ensemble</strong> (Tab 4)
                    — Isolation Forest + LOF + One-Class SVM with majority voting (V ≥ 2/3).
                </div>
                <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                            border-radius:12px;padding:1.5rem;">
                    <strong style="color:{COLORS['text']};">📤 Step 6:</strong> Export (Tab 5)
                    — Clean CSV/Parquet, pipeline.py, and ReportLab audit_report.pdf with SHA-256 checksums.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header_bar(active_ver: str, df) -> None:
    """Renders the top header bar with version badge and key metrics."""
    rows, cols = df.shape
    null_pct = round(df.isnull().sum().sum() / max(rows * cols, 1) * 100, 1)

    version_badges = {
        "v0_raw": "🟣 v0_raw (Raw)",
        "v1_privacy": "🟢 v1_privacy (Sanitized)",
        "v2_imputed": "🟡 v2_imputed (Imputed)",
        "v3_anomaly": "🔴 v3_anomaly (Anomaly-Flagged)",
    }
    ver_label = version_badges.get(active_ver, f"⚪ {active_ver}")

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    background:{COLORS['surface']};border:1px solid {COLORS['border']};
                    border-radius:10px;padding:0.75rem 1.25rem;margin-bottom:1rem;">
            <div style="display:flex;align-items:center;gap:1rem;">
                <span style="font-size:0.85rem;font-weight:700;
                             background:linear-gradient(135deg,{COLORS['accent']},#8B5CF6);
                             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    AutoData-Hub
                </span>
                <span class="adh-badge adh-badge-accent">{ver_label}</span>
            </div>
            <div style="display:flex;gap:1.5rem;font-size:0.85rem;color:{COLORS['text_muted']};">
                <span>📊 {rows:,} rows × {cols:,} cols</span>
                <span>🕳️ {null_pct}% nulls</span>
                <span>💾 {round(df.memory_usage(deep=True).sum()/1e6, 1)} MB</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Event-Gated Module Execution (handled inside each tab via buttons)
# The tab modules themselves handle the button click execution logic.
# This ensures zero rerun-loop freezes per Rule #3.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
