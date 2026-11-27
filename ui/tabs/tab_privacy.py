"""
AutoData-Hub Tab 2: Privacy & PII Firewall.

Displays PII detection results, column-level sensitivity alerts,
interactive masking mode controls, and a live side-by-side preview
of original vs. sanitized values.
"""

from __future__ import annotations
from html import escape
from typing import Optional

import pandas as pd
import streamlit as st

from core.schema_inference import ExecutionStatus, ModuleResult
from ui.theme import COLORS, badge, kpi_row, section_header
from ui.utils_ux import render_tab_guidance_header


_PII_LABELS = {
    "email": ("Email Address", "danger"),
    "phone": ("Phone Number", "danger"),
    "ssn": ("SSN / Tax ID", "danger"),
    "credit_card": ("Credit Card", "danger"),
    "ip_address": ("IP Address", "warning"),
    "ipv4": ("IPv4 Address", "warning"),
    "ipv6": ("IPv6 Address", "warning"),
    "dob": ("Date of Birth", "danger"),
    "name": ("Person Name", "warning"),
    "address": ("Physical Address", "warning"),
    "generic_pii": ("Sensitive Data", "danger"),
}


def render_privacy_tab(
    df: pd.DataFrame,
    module_result=None,
    transform_result=None,
) -> None:
    """
    Renders the Privacy & PII Firewall tab.

    Parameters
    ----------
    df : pd.DataFrame
        Raw / pre-privacy dataset (v0_raw or v1_privacy for diff view).
    module_result : ModuleResult | None
        Privacy module result containing detection metrics.
    transform_result : TransformResult | None
        Privacy TransformResult containing sanitized DataFrame.
    """
    render_tab_guidance_header(
        title="Privacy & PII Firewall",
        objective="Find identifying values in `v0_raw` and mask them before imputation or modeling.",
        methodology=(
            "The scan checks value patterns and column names. Choose redaction to conceal the value, "
            "session-salted SHA-256 hashes to retain matching identifiers, or synthetic pseudonyms."
        ),
        deliverable="A `v1_privacy` dataset, a list of detected fields, and matching raw/masked samples for review.",
    )
    if df is None or df.empty:
        st.info("Upload a dataset to scan for PII and sensitive data.")
        return

    flash_message = st.session_state.pop("privacy_flash_message", None)
    if flash_message:
        st.success(flash_message)

    metrics = module_result.metrics if module_result else {}
    sanitized_df = (
        transform_result.transformed_df
        if transform_result and transform_result.transformed_df is not None
        else None
    )

    # ── KPI Row ──────────────────────────────────────────────────────────────
    n_pii_cols = int(metrics.get("pii_columns_detected", 0))
    n_pii_cells = int(metrics.get("pii_cells_masked", 0))
    masking_mode = str(metrics.get("masking_mode", st.session_state.get("cfg_privacy_mode", "—")))
    n_actions = len(getattr(transform_result, "applied_actions", None) or [])

    st.markdown(
        kpi_row([
            {"label": "PII Columns Detected", "value": str(n_pii_cols),
             "color": COLORS["danger"] if n_pii_cols > 0 else COLORS["success"]},
            {"label": "Cells Masked", "value": f"{n_pii_cells:,}"},
            {"label": "Masking Mode", "value": masking_mode.upper()},
            {"label": "Actions Applied", "value": str(n_actions)},
        ]),
        unsafe_allow_html=True,
    )

    # ── PII Alert Table ───────────────────────────────────────────────────────
    st.markdown(section_header("🔍 PII Detection Results"), unsafe_allow_html=True)

    pii_findings = metrics.get("pii_findings", {})  # {col: [pii_type, ...]}
    header_detections = metrics.get("header_detections", {})  # {col: pii_type}
    detection_methods = metrics.get("detection_methods", {})
    all_sensitive = set(list(pii_findings.keys()) + list(header_detections.keys()))
    ordered_sensitive = [column for column in df.columns if column in all_sensitive]
    ordered_sensitive.extend(
        column for column in all_sensitive if column not in ordered_sensitive
    )

    if not all_sensitive:
        if module_result:
            st.success("✅ No PII detected in this dataset.")
        else:
            st.info("Run the Privacy Firewall module to scan for PII.")
    else:
        rows_data = []
        for col in ordered_sensitive:
            pii_types = list(set(pii_findings.get(col, []) + [header_detections.get(col, "")]))
            pii_types = [p for p in pii_types if p]
            for pt in pii_types:
                label, variant = _PII_LABELS.get(pt, (pt.replace("_", " ").title(), "warning"))
                rows_data.append({
                    "Column": col,
                    "PII Type": label,
                    "Detection": badge(label, variant),
                    "Dtype": str(df[col].dtype) if col in df.columns else "—",
                    "Method": str(detection_methods.get(col, "detected")).replace("_", " ").title(),
                })

        # Render as HTML table with badge rendering
        if rows_data:
            header_cols = st.columns([2, 2, 2, 1, 2])
            for col_header, label in zip(header_cols, ["Column", "PII Type", "Detection", "Dtype", "Method"]):
                col_header.markdown(
                    f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.1em;color:{COLORS["text_muted"]};">{label}</div>',
                    unsafe_allow_html=True,
                )
            for row in rows_data:
                c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 2])
                c1.markdown(
                    f'<span style="font-family:monospace;">{escape(str(row["Column"]))}</span>',
                    unsafe_allow_html=True,
                )
                c2.markdown(row["PII Type"])
                c3.markdown(row["Detection"], unsafe_allow_html=True)
                c4.markdown(
                    f'<span style="color:{COLORS["text_muted"]}">{escape(str(row["Dtype"]))}</span>',
                    unsafe_allow_html=True,
                )
                c5.markdown(
                    f'<span style="font-family:monospace;font-size:0.8rem;">'
                    f'{escape(str(row["Method"]))}</span>',
                    unsafe_allow_html=True,
                )

    # ── Interactive Masking Controls ──────────────────────────────────────────
    st.markdown(section_header("🎛️ Masking Controls"), unsafe_allow_html=True)

    col_mode, col_run = st.columns([3, 1])
    with col_mode:
        mode = st.radio(
            "Masking Strategy",
            options=["redact", "hash", "anonymize"],
            horizontal=True,
            key="cfg_privacy_mode",
            help=(
                "**Redact**: Pattern-preserving replacement (XXX-XX-1234).\n\n"
                "**Hash**: Salted SHA-256 (preserves join-ability within session).\n\n"
                "**Anonymize**: Synthetic pseudonymous replacements."
            ),
        )
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_privacy = st.button("🔒 Run Privacy Scan", key="btn_run_privacy", width="stretch")

    if run_privacy:
        src_df = st.session_state["dataset_versions"].get("v0_raw")
        if src_df is None:
            st.error("The raw dataset version is unavailable.")
        else:
            with st.spinner("Scanning for Email, SSN, Credit Card (Luhn), and Phone patterns..."):
                try:
                    from modules.privacy.detector import PIIDetector

                    detector = PIIDetector()
                    findings = detector.detect(src_df)

                    with st.spinner("Applying salted SHA-256 hashing / redaction to selected columns..."):
                        from modules.privacy.masker import PIIMasker

                        masker = PIIMasker(
                            session_salt=st.session_state.get("privacy_session_salt")
                        )
                        st.session_state["privacy_session_salt"] = masker.session_salt

                        downstream_modules = {"privacy", "imputation", "anomaly"}
                        retained_actions = [
                            action
                            for action in st.session_state.get("action_ledger", [])
                            if getattr(action, "module", None) not in downstream_modules
                        ]
                        start_step_id = max(
                            (getattr(action, "step_id", 0) for action in retained_actions),
                            default=0,
                        ) + 1
                        tr = masker.mask_dataframe(
                            src_df,
                            detections=findings,
                            strategy_overrides={col: mode for col in findings},
                            start_step_id=start_step_id,
                        )

                    if tr.status == ExecutionStatus.SUCCESS and tr.transformed_df is not None:
                        changed_cells = sum(
                            int(action.parameters.get("changed_cells", 0))
                            for action in tr.applied_actions
                        )
                        pii_findings = {
                            col: [detection.pii_type.value]
                            for col, detection in findings.items()
                        }
                        header_detections = {
                            col: detection.pii_type.value
                            for col, detection in findings.items()
                            if detection.detection_method in {"semantic_header", "hybrid"}
                        }
                        privacy_result = ModuleResult(
                            status=ExecutionStatus.SUCCESS,
                            module_name="privacy",
                            dataset_version="v1_privacy",
                            metrics={
                                "pii_columns_detected": len(findings),
                                "pii_cells_detected": sum(
                                    detection.match_count for detection in findings.values()
                                ),
                                "pii_cells_masked": changed_cells,
                                "masking_mode": mode,
                                "pii_findings": pii_findings,
                                "header_detections": header_detections,
                                "detection_methods": {
                                    col: detection.detection_method
                                    for col, detection in findings.items()
                                },
                            },
                            issues=[],
                            artifacts={},
                        )

                        versions = st.session_state["dataset_versions"]
                        for version_id in ("v1_privacy", "v2_imputed", "v3_anomaly"):
                            versions.pop(version_id, None)
                        versions["v1_privacy"] = tr.transformed_df.copy(deep=True)

                        module_results = st.session_state.setdefault("module_results", {})
                        transform_results = st.session_state.setdefault("transform_results", {})
                        for module_name in downstream_modules:
                            module_results.pop(module_name, None)
                            transform_results.pop(module_name, None)
                        health_result = module_results.get("health_audit")
                        if (
                            health_result is not None
                            and health_result.dataset_version
                            in {"v1_privacy", "v2_imputed", "v3_anomaly"}
                        ):
                            module_results.pop("health_audit", None)
                        module_results["privacy"] = privacy_result
                        transform_results["privacy"] = tr

                        st.session_state["action_ledger"] = retained_actions + tr.applied_actions
                        st.session_state["audit_logs"] = [
                            entry
                            for entry in st.session_state.get("audit_logs", [])
                            if not any(
                                tag in entry.upper()
                                for tag in ("[PRIVACY]", "[IMPUTATION]", "[ANOMALY]")
                            )
                        ] + tr.audit_log
                        st.session_state["active_version"] = "v1_privacy"
                        st.session_state["_pending_active_version"] = "v1_privacy"
                        st.session_state["pipeline_ran"] = False
                        st.session_state.pop("pdf_bytes_cache", None)
                        st.session_state.pop("pdf_filename_cache", None)
                        st.session_state.pop("pipeline_bytes_cache", None)
                        st.session_state.pop("pipeline_fingerprint_cache", None)
                        st.session_state.pop("_anomaly_winsor_preflight", None)
                        st.session_state["privacy_flash_message"] = (
                            f"Privacy Firewall complete: {len(findings)} PII columns detected "
                            f"and {changed_cells} cells masked."
                        )
                        st.rerun()
                    else:
                        st.error(tr.error_message or "Privacy masking returned no output.")
                except Exception as exc:
                    st.error(f"Privacy scan failed: {exc}")

    # ── Side-by-Side Preview ──────────────────────────────────────────────────
    if sanitized_df is not None and all_sensitive:
        st.markdown(section_header("👁️ Before / After Preview"), unsafe_allow_html=True)
        preview_cols = [
            column
            for column in df.columns
            if column in all_sensitive and column in sanitized_df.columns
        ][:6]
        if preview_cols:
            before_tab, after_tab = st.tabs(["🔴 Original (Raw)", "🟢 Sanitized"])
            with before_tab:
                render_tab_guidance_header(
                    title="Original PII Values",
                    objective="Verify which source fields were flagged before masking. This view can expose raw identifiers.",
                    methodology=(
                        "Inspect the first 20 rows of up to six detected fields in `v0_raw`. "
                        "Match the row index and field name in Sanitized; do not share this raw preview."
                    ),
                    deliverable="An unmodified reference sample for checking the masking result.",
                )
                st.dataframe(df[preview_cols].head(20), width="stretch", hide_index=False)
            with after_tab:
                render_tab_guidance_header(
                    title="Sanitized PII Values",
                    objective="Check what the selected masking strategy actually changed.",
                    methodology=(
                        "Compare the same indexed cells in `v1_privacy` with Original (Raw). Check "
                        "redactions, hashes, or pseudonyms against your intended use before sharing."
                    ),
                    deliverable="A masked sample for review, not proof that every sensitive field was detected.",
                )
                st.dataframe(sanitized_df[preview_cols].head(20), width="stretch", hide_index=False)
    elif sanitized_df is not None:
        st.markdown(section_header("✅ Sanitized Dataset Preview"), unsafe_allow_html=True)
        st.dataframe(sanitized_df.head(20), width="stretch", hide_index=False)

    # ── Audit Log ─────────────────────────────────────────────────────────────
    if transform_result and transform_result.audit_log:
        with st.expander("📋 Privacy Audit Log", expanded=False):
            for entry in transform_result.audit_log:
                st.markdown(
                    f'<p style="font-size:0.75rem;font-family:monospace;'
                    f'color:{COLORS["text_muted"]};margin:0;">{escape(str(entry))}</p>',
                    unsafe_allow_html=True,
                )
