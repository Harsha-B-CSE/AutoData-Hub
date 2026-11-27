"""AutoData-Hub Tab 5: Export & Versioning.

Displays pipeline action queue timeline, clean CSV/Excel/Parquet downloads,
ReportLab PDF report generation, and syntax-highlighted pipeline.py viewer.
"""

from __future__ import annotations

from html import escape
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import streamlit as st

from core.dataset_manager import DatasetManager
from ui.theme import COLORS, badge, section_header
from ui.utils_ux import render_tab_guidance_header


_PDF_REPORT_STYLE_VERSION = 2


# ==========================================
# Action Queue & Audit Log Components
# ==========================================

def _render_action_queue() -> None:
    """Renders the action ledger as a timeline."""
    st.markdown(section_header("📜 Action Queue (Ledger)"), unsafe_allow_html=True)

    action_ledger = st.session_state.get("action_ledger", [])
    if not action_ledger:
        st.info(
            "No actions logged yet. Run the pipeline "
            "(Health Audit → Privacy → Imputation → Anomaly) to populate this queue."
        )
        return

    sorted_actions = sorted(action_ledger, key=lambda a: getattr(a, "step_id", 0))

    for action in sorted_actions[:50]:
        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            badge_class = (
                "success" if action.step_id <= 10
                else ("accent" if action.step_id <= 20 else "warning")
            )
            st.markdown(badge(f"Step {action.step_id}", badge_class), unsafe_allow_html=True)

        with col2:
            st.markdown(f"**{action.module.upper()}**", unsafe_allow_html=True)
            st.caption(action.action_type.replace("_", " ").title())

        with col3:
            targets = getattr(action, "target_columns", [])
            st.markdown(
                f"**Target:** {', '.join(map(str, targets[:3]))}"
                f"{' ...' if len(targets) > 3 else ''}"
            )
            st.markdown(f"**Parameters:** {action.parameters or '—'}")

        st.markdown("---")

    if len(sorted_actions) > 50:
        st.caption(f"... and {len(sorted_actions) - 50} newer actions (truncated for UI).")


def _render_audit_log() -> None:
    """Renders combined audit log from all modules."""
    st.markdown(section_header("📋 Combined Audit Log"), unsafe_allow_html=True)

    audit_logs = st.session_state.get("audit_logs", [])
    if not audit_logs:
        st.info("No audit entries yet. Run pipeline modules to generate logs.")
        return

    for entry in reversed(audit_logs[-50:]):
        entry = str(entry)
        timestamp_part = ""
        if "[" in entry and "]" in entry:
            ts_part = entry.split("[", 1)[1].split("]", 1)[0]
            timestamp_part = f"[{ts_part}]"
            entry = entry.replace(f"[{ts_part}] ", "")

        icon = (
            "🔍" if "HEALTH" in entry.upper()
            else ("🔒" if "PRIVACY" in entry.upper()
            else ("⚙️" if "IMPUTATION" in entry.upper()
            else ("📊" if "ANOMALY" in entry.upper() else "📝")))
        )

        log_line = f"{icon} {entry}"
        if timestamp_part:
            log_line = f"{timestamp_part} — {log_line}"

        st.markdown(
            f'<div style="font-family: monospace; font-size: 0.9em;">'
            f'{escape(log_line)}</div>',
            unsafe_allow_html=True,
        )


# ==========================================
# Export & Download Components
# ==========================================

def _get_latest_transformed_df() -> Tuple[str, Optional[pd.DataFrame]]:
    """Resolve the highest available transformed dataset version."""
    versions = st.session_state.get("dataset_versions", {})
    for version_key in ("v3_anomaly", "v2_imputed", "v1_privacy", "v0_raw"):
        candidate = versions.get(version_key)
        if candidate is not None:
            return version_key, candidate
    return "v0_raw", None


def _render_download_buttons(
    df: Optional[pd.DataFrame] = None,
    version_key: Optional[str] = None,
) -> None:
    """Renders CSV, Excel, and Parquet download buttons with OS security policy fallbacks."""
    st.markdown(section_header("⬇️ Export Clean Dataset"), unsafe_allow_html=True)

    if df is None or version_key is None:
        version_key, df = _get_latest_transformed_df()

    if df is None or len(df.columns) == 0:
        st.warning("No dataset to export.")
        return

    file_prefix = f"cleaned_dataset_{version_key}"

    col1, col2, col3 = st.columns([1, 1, 1])

    # 1. CSV Download
    with col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download CSV",
            data=csv_bytes,
            file_name=f"{file_prefix}.csv",
            mime="text/csv",
            width="stretch",
            key="download_csv",
        )

    # 2. Excel Download (openpyxl)
    with col2:
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{file_prefix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                key="download_excel",
            )
        except Exception as exc:
            st.warning(f"Excel export unavailable: {exc}")

    # 3. Parquet Download (with PyArrow / Fastparquet Exception Guard)
    with col3:
        parquet_bytes = None
        try:
            parquet_bytes = df.to_parquet(index=False, engine="pyarrow")
        except Exception:
            try:
                parquet_bytes = df.to_parquet(index=False, engine="fastparquet")
            except Exception as exc:
                st.warning(f"Parquet export unavailable: {exc}")

        if parquet_bytes is not None:
            st.download_button(
                label="📦 Download Parquet",
                data=parquet_bytes,
                file_name=f"{file_prefix}.parquet",
                mime="application/octet-stream",
                width="stretch",
                key="download_parquet",
            )


# ==========================================
# Report & Pipeline Script Generators
# ==========================================

def _render_pdf_report_generator() -> None:
    """Renders PDF report generation trigger with browser download button."""
    st.markdown(section_header("📊 Generate Audit Report PDF"), unsafe_allow_html=True)

    if (
        "pdf_bytes_cache" in st.session_state
        and st.session_state.get("pdf_style_version") != _PDF_REPORT_STYLE_VERSION
    ):
        st.session_state.pop("pdf_bytes_cache", None)
        st.session_state.pop("pdf_filename_cache", None)

    module_results = st.session_state.get("module_results", {})
    action_ledger = st.session_state.get("action_ledger", [])
    audit_logs = st.session_state.get("audit_logs", [])

    try:
        import reportlab
        has_reportlab = True
    except ImportError:
        has_reportlab = False

    if not has_reportlab:
        st.error("ReportLab is required for PDF generation. Run: pip install reportlab")

    if not has_reportlab or not module_results:
        st.info("Run at least one pipeline module to enable PDF report generation.")
        return

    version_chain = [
        k for k in ["v0_raw", "v1_privacy", "v2_imputed", "v3_anomaly"]
        if k in st.session_state.get("dataset_versions", {})
    ]
    dataset_name = st.session_state.get("current_filename", "AutoData-Hub Dataset")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("📄 Generate PDF Report", key="btn_generate_pdf", width="stretch"):
            with st.spinner("Compiling PDF audit report & SHA-256 cryptographic checksum..."):
                try:
                    from modules.script_gen.exporter import export_audit_pdf

                    st.session_state.pop("pdf_bytes_cache", None)
                    st.session_state.pop("pdf_filename_cache", None)
                    report_payload = {
                        "dataset": dataset_name,
                        "versions": version_chain,
                        "modules": [
                            (name, result.dataset_version, result.status.value)
                            for name, result in module_results.items()
                        ],
                        "actions": [
                            action.to_dict() if hasattr(action, "to_dict") else str(action)
                            for action in action_ledger
                        ],
                        "logs": list(map(str, audit_logs)),
                    }
                    report_fingerprint = hashlib.sha256(
                        json.dumps(report_payload, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()
                    report_dir = (
                        DatasetManager().get_output_dir()
                        / f"report_{report_fingerprint[:12]}"
                    )
                    artifact = export_audit_pdf(
                        module_results=list(module_results.values()),
                        actions=action_ledger,
                        dataset_name=dataset_name,
                        version_chain=version_chain,
                        audit_logs=audit_logs,
                        output_dir=report_dir,
                    )

                    # Read binary bytes from generated file for Streamlit download button
                    file_path = getattr(artifact, "filepath", None) or getattr(artifact, "file_path", None)
                    if not file_path or not Path(file_path).is_file():
                        raise FileNotFoundError("Generated PDF artifact was not found on disk.")

                    st.session_state["pdf_bytes_cache"] = Path(file_path).read_bytes()
                    st.session_state["pdf_filename_cache"] = artifact.filename
                    st.session_state["pdf_style_version"] = _PDF_REPORT_STYLE_VERSION

                    st.success(f"✅ PDF generated — {artifact.filename} ({artifact.size_bytes} bytes).")
                    sha_val = getattr(artifact, "sha256", None)
                    if sha_val:
                        st.caption(f"SHA-256 Checksum: `{sha_val[:16]}...`")
                except Exception as exc:
                    st.error(f"PDF compilation failed: {exc}")

        # Render Download Button if PDF is compiled in session state
        if "pdf_bytes_cache" in st.session_state:
            st.download_button(
                label="⬇️ Click Here to Download PDF",
                data=st.session_state["pdf_bytes_cache"],
                file_name=st.session_state.get("pdf_filename_cache", "audit_report.pdf"),
                mime="application/pdf",
                width="stretch",
                key="btn_download_pdf_file",
            )

    with col2:
        st.markdown(f"**Status:** {len(module_results)} modules executed", unsafe_allow_html=True)
        st.markdown(f"**Actions:** {len(action_ledger)} logged", unsafe_allow_html=True)
        st.markdown(f"**Versions:** {len(version_chain)} active", unsafe_allow_html=True)


def _pipeline_fingerprint(action_ledger: list) -> str:
    """Identify the current dataset lineage and exact action queue."""
    serialized_actions = [
        action.to_dict() if hasattr(action, "to_dict") else str(action)
        for action in action_ledger
    ]
    payload = {
        "dataset": st.session_state.get("current_filename"),
        "active_version": st.session_state.get("active_version"),
        "versions": list(st.session_state.get("dataset_versions", {})),
        "actions": serialized_actions,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _render_pipeline_viewer() -> None:
    """Renders the generated pipeline.py script with a 1-click manual trigger."""
    st.markdown(section_header("⚙️ Generated Pipeline Script Viewer"), unsafe_allow_html=True)

    # Clean path resolution via DatasetManager
    output_dir = DatasetManager().get_output_dir()
    action_ledger = st.session_state.get("action_ledger", [])
    current_fingerprint = _pipeline_fingerprint(action_ledger)
    pipeline_path = output_dir / f"pipeline_{current_fingerprint[:12]}.py"

    # Manual trigger button: synthesize pipeline script from logged actions
    col_gen, _ = st.columns([1, 2])
    with col_gen:
        if st.button("⚡ Synthesize pipeline.py Code", key="btn_force_gen_script", width="stretch"):
            with st.spinner("Synthesizing pipeline.py code..."):
                try:
                    from modules.script_gen.exporter import export_pipeline_script
                    artifact = export_pipeline_script(
                        actions=action_ledger,
                        output_path=pipeline_path,
                        dataset_name=st.session_state.get(
                            "current_filename", "dataset"
                        ),
                        version_chain=list(
                            st.session_state.get("dataset_versions", {})
                        ),
                        input_csv_path=Path(
                            st.session_state.get("current_filename") or "input.csv"
                        ).name,
                    )
                    pipeline_bytes = Path(artifact.filepath).read_bytes()
                    st.session_state["pipeline_bytes_cache"] = pipeline_bytes
                    st.session_state["pipeline_fingerprint_cache"] = current_fingerprint
                    st.success("✅ pipeline.py generated successfully!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Script synthesis failed: {exc}")

    pipeline_bytes = st.session_state.get("pipeline_bytes_cache")
    cached_fingerprint = st.session_state.get("pipeline_fingerprint_cache")
    if pipeline_bytes is not None and cached_fingerprint == current_fingerprint:
        try:
            pipeline_content = pipeline_bytes.decode("utf-8")
            st.code(pipeline_content, language="python")
            st.caption(f"File: `pipeline.py` ({len(pipeline_bytes)} bytes)")
            st.download_button(
                label="⬇️ Download pipeline.py",
                data=pipeline_bytes,
                file_name="pipeline.py",
                mime="text/x-python",
                width="stretch",
                key="download_pipeline_script",
            )
        except Exception as exc:
            st.warning(f"Could not load pipeline viewer: {exc}")
    else:
        st.info(
            "No pipeline script has been generated for the current dataset and action queue. "
            "Click '⚡ Synthesize pipeline.py Code' above to create it."
        )


# ==========================================
# Main Entry Point
# ==========================================

def render_export_tab(
    df: pd.DataFrame,
    module_results: dict,
    action_ledger: list,
    audit_logs: list
) -> None:
    """Renders Export & Versioning Studio (Tab 5)."""
    render_tab_guidance_header(
        title="Export & Versioning Studio",
        objective="Export cleaned datasets, audit documentation, and reproducible code.",
        methodology=(
            "Compiles the immutable TransformationAction ledger into standalone "
            "pipeline.py scripts, ReportLab PDF executive reports, and Parquet/CSV/XLSX."
        ),
        deliverable="Downloadable cleaned dataset, audit_report.pdf, and pipeline.py.",
    )
    st.markdown("# 📤 Export & Versioning Studio", unsafe_allow_html=True)

    _render_action_queue()
    _render_audit_log()

    export_version, clean_df = _get_latest_transformed_df()
    if clean_df is not None:
        rows, columns = clean_df.shape
        st.info(
            f"Exporting active version: **{export_version}** "
            f"({rows:,} rows × {columns:,} cols)"
        )
        _render_download_buttons(clean_df, version_key=export_version)
    else:
        _render_download_buttons()

    _render_pdf_report_generator()
    _render_pipeline_viewer()
