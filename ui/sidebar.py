"""
AutoData-Hub UI Sidebar.

Renders the left-panel control surface:
  - Branded logo & tagline header
  - File upload (CSV/TSV/XLSX/Parquet/JSON) or DB connection string
  - Dataset shape & version metadata
  - Dataset Version Selector with 1-click rollback
  - Pipeline trigger buttons (run all / step-by-step)
  - Module configuration expanders
"""

from __future__ import annotations

import hashlib
from typing import Callable, Optional
import streamlit as st
import pandas as pd

from core.dataset_manager import DatasetManager
from ui.theme import COLORS, badge, section_header
from ui.utils_ux import render_tab_guidance_header


TABULAR_UPLOAD_TYPES = [
    "csv", "tsv", "xlsx", "xls", "parquet", "feather", "json", "jsonl"
]
IMAGE_UPLOAD_TYPES = ["zip", "png", "jpg", "jpeg", "webp"]
VOICE_UPLOAD_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]
VIDEO_UPLOAD_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]


# ---------------------------------------------------------------------------
# Session State Initialiser
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialises all required st.session_state keys if not already set."""
    defaults = {
        "dataset_versions": {},        # {version_id: pd.DataFrame}
        "active_version": None,        # str — currently displayed version key
        "module_results": {},          # {module_name: ModuleResult}
        "transform_results": {},       # {module_name: TransformResult}
        "action_ledger": [],           # List[TransformationAction]
        "audit_logs": [],              # List[str]
        "pipeline_ran": False,
        "current_filename": None,
        "processed_upload_fingerprint": None,
        "uploader_had_file": False,
        "processed_image_fingerprint": None,
        "image_uploader_had_file": False,
        "image_dataset": None,
        "processed_voice_fingerprint": None,
        "voice_uploader_had_file": False,
        "voice_dataset": None,
        "processed_video_fingerprint": None,
        "video_uploader_had_file": False,
        "video_dataset": None,
        "active_modality": "tabular",
        "modality_selector": "tabular",
        "privacy_session_salt": None,
        # Module config
        "cfg_contamination": 0.05,
        "cfg_vote_threshold": 2,
        "cfg_threshold_mode": "adaptive_3sigma",
        "cfg_mitigation_strategy": "winsorize",
        "cfg_impute_strategy": "auto_cv",
        "cfg_privacy_mode": "redact",
        "cfg_mnar_indicators": True,
        "cfg_drop_high_missing": False,
        "cfg_anomaly_algos": ["isolation_forest", "lof", "one_class_svm"],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _clear_workspace_data() -> None:
    """Clear modality-specific data and all derived pipeline state."""
    st.cache_data.clear()
    st.session_state["dataset_versions"] = {}
    st.session_state["module_results"] = {}
    st.session_state["transform_results"] = {}
    st.session_state["action_ledger"] = []
    st.session_state["audit_logs"] = []
    st.session_state["active_version"] = None
    st.session_state["current_filename"] = None
    st.session_state["privacy_session_salt"] = None
    st.session_state["pipeline_ran"] = False
    st.session_state["image_dataset"] = None
    st.session_state["voice_dataset"] = None
    st.session_state["video_dataset"] = None
    st.session_state["processed_upload_fingerprint"] = None
    st.session_state["processed_image_fingerprint"] = None
    st.session_state["processed_voice_fingerprint"] = None
    st.session_state["processed_video_fingerprint"] = None
    st.session_state["uploader_had_file"] = False
    st.session_state["image_uploader_had_file"] = False
    st.session_state["voice_uploader_had_file"] = False
    st.session_state["video_uploader_had_file"] = False
    for key in (
        "pdf_bytes_cache",
        "pdf_filename_cache",
        "pipeline_bytes_cache",
        "pipeline_fingerprint_cache",
        "_anomaly_winsor_preflight",
        "_pending_active_version",
        "version_selector",
        "image_source_fingerprint",
        "image_source_items",
        "image_source_error",
        "image_buffer",
        "image_buffers",
        "image_focus_metrics",
        "image_sync_color_widgets",
        "image_selected_name",
        "processed_image_bgr",
        "image_restoration_metadata",
        "image_export_result",
        "image_export_options",
        "image_export_zip_bytes",
        "image_member_selector",
        "voice_source_fingerprint",
        "voice_raw_audio",
        "voice_raw_sample_rate",
        "voice_source_metadata",
        "voice_enhanced_audio",
        "voice_enhanced_sample_rate",
        "voice_transform_result",
        "voice_module_result",
        "voice_processing_mode",
        "voice_attenuation_db",
        "voice_bandwidth_extension",
        "voice_input_source",
        "voice_live_attenuation_db",
        "voice_live_boost_db",
        "voice_boost_db",
        "voice_execution_mode",
        "voice_tuned_audio",
        "voice_tuned_sample_rate",
        "voice_tuner_low_db",
        "voice_tuner_mid_db",
        "voice_tuner_high_db",
        "voice_studio_preset",
        "voice_studio_highpass",
        "voice_studio_highpass_enabled",
        "voice_studio_low_shelf_freq",
        "voice_studio_low_mid_gain",
        "voice_studio_low_mid_freq",
        "voice_studio_low_mid_q",
        "voice_studio_mid_freq",
        "voice_studio_mid_q",
        "voice_studio_presence_gain",
        "voice_studio_presence_freq",
        "voice_studio_presence_q",
        "voice_studio_high_shelf_freq",
        "voice_studio_deess_enabled",
        "voice_studio_deess_threshold",
        "voice_studio_deess_amount",
        "voice_studio_deess_attack",
        "voice_studio_deess_release",
        "voice_studio_deess_knee",
        "voice_studio_comp_enabled",
        "voice_studio_comp_ratio",
        "voice_studio_comp_threshold",
        "voice_studio_comp_knee",
        "voice_studio_comp_attack",
        "voice_studio_comp_release",
        "voice_studio_comp_makeup",
        "voice_studio_sat_type",
        "voice_studio_sat_drive",
        "voice_studio_limiter_ceiling",
        "voice_export_sample_rate",
        "voice_export_bit_depth",
        "voice_export_format",
        "video_source_fingerprint",
        "video_transform_result",
        "video_stride",
        "video_scene_thresh",
        "video_ema_alpha",
        "video_color_preset",
        "video_color_inspect_idx",
    ):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Ingestion Panel
# ---------------------------------------------------------------------------

def _activate_raw_dataset(df: pd.DataFrame, filename: str) -> None:
    """Store a newly ingested dataset as the session's raw source of truth."""
    _clear_workspace_data()

    dataset_manager = DatasetManager()
    dataset_manager.register_version("v0_raw", df)

    st.session_state["dataset_versions"] = {
        "v0_raw": dataset_manager.get_version("v0_raw")
    }
    st.session_state["current_filename"] = filename
    st.session_state["active_version"] = "v0_raw"
    st.session_state["active_modality"] = "tabular"
    st.session_state["privacy_session_salt"] = None


def _activate_image_dataset(payload: bytes, filename: str, upload_id: str) -> None:
    """Persist an image or ZIP upload without entering the tabular pipeline."""
    _clear_workspace_data()
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    st.session_state["image_dataset"] = {
        "upload_id": upload_id,
        "filename": filename,
        "extension": extension,
        "bytes": payload,
        "size_bytes": len(payload),
    }
    st.session_state["current_filename"] = filename
    st.session_state["active_modality"] = "image"


def _activate_voice_dataset(payload: bytes, filename: str) -> None:
    """Persist an audio upload without entering the tabular pipeline."""
    _clear_workspace_data()
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    st.session_state["voice_dataset"] = {
        "filename": filename,
        "extension": extension,
        "bytes": payload,
        "size_bytes": len(payload),
    }
    st.session_state["current_filename"] = filename
    st.session_state["active_modality"] = "voice"


def _activate_video_dataset(uploaded_file, upload_id: str) -> None:
    """Persist a video upload without entering the tabular pipeline."""
    _clear_workspace_data()
    filename = uploaded_file.name
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    st.session_state["video_dataset"] = {
        "filename": filename,
        "extension": extension,
        "upload_id": upload_id,
        "source": uploaded_file,
        "size_bytes": uploaded_file.size,
    }
    st.session_state["current_filename"] = filename
    st.session_state["active_modality"] = "video"


def render_sidebar_ingestion(
    detect_file_modality: Optional[Callable[[str], Optional[str]]] = None,
) -> None:
    """Render modality-aware upload controls and persist the selected source."""
    st.markdown(section_header("📂 Data Source"), unsafe_allow_html=True)

    selected_modality = st.radio(
        "Data Modality",
        options=["tabular", "image", "voice", "video"],
        format_func=lambda value: {
            "tabular": "📊 Tabular Datasets",
            "image": "🖼️ Image Datasets",
            "voice": "🎙️ Voice / Audio",
            "video": "📹 Video Media",
        }[value],
        horizontal=False,
        key="modality_selector",
    )
    if selected_modality != st.session_state.get("active_modality"):
        _clear_workspace_data()
        st.session_state.pop("file_uploader_tabular", None)
        st.session_state.pop("file_uploader_image", None)
        st.session_state.pop("file_uploader_voice", None)
        st.session_state.pop("file_uploader_video", None)
        st.session_state["active_modality"] = selected_modality
        st.rerun()

    if selected_modality == "image":
        uploaded = st.file_uploader(
            "Upload image dataset",
            type=IMAGE_UPLOAD_TYPES,
            key="file_uploader_image",
            label_visibility="collapsed",
            help="Supported: ZIP, PNG, JPG, JPEG, WEBP",
        )
        if uploaded is None:
            st.session_state["processed_image_fingerprint"] = None
            if st.session_state.get("image_uploader_had_file", False):
                _clear_workspace_data()
                st.session_state["active_modality"] = "image"
                st.rerun()
            return

        st.session_state["image_uploader_had_file"] = True
        detected = (
            detect_file_modality(uploaded.name)
            if detect_file_modality is not None
            else "image"
        )
        if detected != "image":
            st.error("The selected file is not a supported image dataset.")
            return
        upload_identity = f"image:{uploaded.file_id}:{uploaded.name}:{uploaded.size}"
        if st.session_state.get("processed_image_fingerprint") != upload_identity:
            with st.spinner(f"Loading {uploaded.name}..."):
                _activate_image_dataset(uploaded.getvalue(), uploaded.name, upload_identity)
                st.session_state["processed_image_fingerprint"] = upload_identity
                st.session_state["image_uploader_had_file"] = True
            st.rerun()
        return

    if selected_modality == "voice":
        uploaded = st.file_uploader(
            "Upload voice or audio file",
            type=VOICE_UPLOAD_TYPES,
            key="file_uploader_voice",
            label_visibility="collapsed",
            help="Supported: WAV, MP3, FLAC, OGG, M4A",
        )
        if uploaded is None:
            st.session_state["processed_voice_fingerprint"] = None
            if st.session_state.get("voice_uploader_had_file", False):
                _clear_workspace_data()
                st.session_state["active_modality"] = "voice"
                st.rerun()
            return

        st.session_state["voice_uploader_had_file"] = True
        detected = (
            detect_file_modality(uploaded.name)
            if detect_file_modality is not None
            else "voice"
        )
        if detected != "voice":
            st.error("The selected file is not a supported voice/audio asset.")
            return
        upload_identity = f"voice:{uploaded.file_id}:{uploaded.name}:{uploaded.size}"
        if st.session_state.get("processed_voice_fingerprint") != upload_identity:
            with st.spinner(f"Loading {uploaded.name}..."):
                _activate_voice_dataset(uploaded.getvalue(), uploaded.name)
                st.session_state["processed_voice_fingerprint"] = upload_identity
                st.session_state["voice_uploader_had_file"] = True
            st.rerun()
        return

    if selected_modality == "video":
        uploaded = st.file_uploader(
            "Upload video media",
            type=VIDEO_UPLOAD_TYPES,
            key="file_uploader_video",
            label_visibility="collapsed",
            help="Supported: MP4, AVI, MOV, MKV, WEBM",
        )
        if uploaded is None:
            st.session_state["processed_video_fingerprint"] = None
            if st.session_state.get("video_uploader_had_file", False):
                _clear_workspace_data()
                st.session_state["active_modality"] = "video"
                st.rerun()
            return

        st.session_state["video_uploader_had_file"] = True
        detected = (
            detect_file_modality(uploaded.name)
            if detect_file_modality is not None
            else "video"
        )
        if detected != "video":
            st.error("The selected file is not a supported video asset.")
            return
        upload_identity = f"video:{uploaded.file_id}:{uploaded.name}:{uploaded.size}"
        if st.session_state.get("processed_video_fingerprint") != upload_identity:
            with st.spinner(f"Loading {uploaded.name}..."):
                _activate_video_dataset(uploaded, upload_identity)
                st.session_state["processed_video_fingerprint"] = upload_identity
                st.session_state["video_uploader_had_file"] = True
            st.rerun()
        return

    upload_tab, db_tab = st.tabs(["File Upload", "Database"])
    with upload_tab:
        render_tab_guidance_header(
            title="File Upload",
            objective="Start a tabular audit from a local dataset.",
            methodology=(
                "Choose a CSV, TSV, Excel, Parquet, Feather, JSON, or JSONL file. "
                "Ingestion normalizes text and null markers and parses detected date columns."
            ),
            deliverable="A new `v0_raw` baseline for Health Audit. Loading a new source replaces prior versions.",
        )
        uploaded = st.file_uploader(
            "Upload tabular dataset",
            type=TABULAR_UPLOAD_TYPES,
            key="file_uploader_tabular",
            label_visibility="collapsed",
            help="Supported: CSV, TSV, XLSX, Parquet, Feather, JSON, JSONL",
        )
        if uploaded is None:
            st.session_state["processed_upload_fingerprint"] = None
            if st.session_state.get("uploader_had_file", False):
                _clear_workspace_data()
                st.session_state["active_modality"] = "tabular"
                st.rerun()
        else:
            payload = uploaded.getvalue()
            st.session_state["uploader_had_file"] = True
            detected = (
                detect_file_modality(uploaded.name)
                if detect_file_modality is not None
                else "tabular"
            )
            if detected != "tabular":
                st.error("The selected file is not a supported tabular dataset.")
            else:
                upload_identity = (
                    f"tabular:{uploaded.name}:{hashlib.sha256(payload).hexdigest()}"
                )
                if (
                    st.session_state.get("processed_upload_fingerprint")
                    != upload_identity
                ):
                    with st.spinner(f"Ingesting {uploaded.name}..."):
                        try:
                            from core.ingestion_engine import ingest_dataset

                            df_raw = ingest_dataset(
                                uploaded,
                                sanitize=True,
                                parse_dates=True,
                            )
                            _activate_raw_dataset(df_raw, uploaded.name)
                            st.session_state["processed_upload_fingerprint"] = (
                                upload_identity
                            )
                            st.session_state["uploader_had_file"] = True
                        except Exception as exc:
                            st.error(f"Dataset ingestion failed: {exc}")
                        else:
                            st.rerun()

    with db_tab:
        render_tab_guidance_header(
            title="Database",
            objective="Audit a table or a selected subset of a relational database.",
            methodology=(
                "Enter a SQLAlchemy URL for a read-only account and a table name or SELECT query. "
                "Select only needed rows and columns, then choose Connect & Load."
            ),
            deliverable="The returned rows become `v0_raw` in this workspace, replacing any previous dataset.",
        )
        conn_str = st.text_input(
            "SQLAlchemy Connection String",
            placeholder="postgresql+psycopg2://user:pass@host/db",
            key="db_conn_string",
        )
        table_name = st.text_input(
            "Table / Query",
            placeholder="my_table  or  SELECT * FROM ...",
            key="db_table",
        )
        if st.button("Connect & Load", key="btn_db_connect"):
            if conn_str and table_name:
                with st.spinner("Connecting to database..."):
                    try:
                        from core.ingestion_engine import DBConnectorConfig, ingest_dataset

                        is_query = table_name.strip().upper().startswith("SELECT")
                        df = ingest_dataset(
                            DBConnectorConfig(
                                dialect="custom",
                                custom_url=conn_str,
                            ),
                            db_table=None if is_query else table_name,
                            db_query=table_name if is_query else None,
                        )
                        source_name = (
                            "database_query" if is_query else table_name.strip()
                        )
                        _activate_raw_dataset(df, source_name)
                    except Exception as exc:
                        st.error(f"DB connection failed: {exc}")
                    else:
                        st.rerun()
            else:
                st.warning("Please provide both connection string and table/query.")


# ---------------------------------------------------------------------------
# Dataset Shape & Version Panel
# ---------------------------------------------------------------------------

def _render_version_panel() -> None:
    """Renders active dataset shape metrics and version selector with rollback."""
    versions: dict = st.session_state.get("dataset_versions", {})
    active: Optional[str] = st.session_state.get("active_version")

    if not versions:
        st.markdown(
            f'<p style="color:{COLORS["text_muted"]};font-size:0.8rem;">No dataset loaded yet.</p>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(section_header("🗂 Dataset Versions"), unsafe_allow_html=True)

    version_keys = list(versions.keys())
    pending_active = st.session_state.pop("_pending_active_version", None)
    if pending_active in versions:
        active = pending_active
        st.session_state["active_version"] = active
        st.session_state["version_selector"] = active
    elif st.session_state.get("version_selector") not in versions:
        st.session_state["version_selector"] = (
            active if active in versions else version_keys[-1]
        )

    selected = st.selectbox(
        "Active Version",
        options=version_keys,
        key="version_selector",
        label_visibility="collapsed",
    )

    if selected != active:
        st.session_state["active_version"] = selected
        st.rerun()

    df: pd.DataFrame = versions[selected].copy(deep=True)
    rows, cols = df.shape
    null_pct = round(df.isnull().sum().sum() / max(rows * cols, 1) * 100, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", f"{rows:,}")
        st.metric("Nulls", f"{null_pct}%")
    with col2:
        st.metric("Cols", f"{cols:,}")
        mem_mb = round(df.memory_usage(deep=True).sum() / 1e6, 2)
        st.metric("Memory", f"{mem_mb} MB")

    # Rollback button
    if len(version_keys) > 1:
        st.markdown("")
        if st.button("⏪ Rollback to v0_raw", key="btn_rollback", width="stretch"):
            raw_version = version_keys[0]
            st.session_state["dataset_versions"] = {
                raw_version: versions[raw_version].copy(deep=True)
            }
            health_result = st.session_state.get("module_results", {}).get(
                "health_audit"
            )
            st.session_state["module_results"] = (
                {"health_audit": health_result}
                if health_result is not None
                and health_result.dataset_version == raw_version
                else {}
            )
            st.session_state["transform_results"] = {}
            st.session_state["action_ledger"] = []
            st.session_state["audit_logs"] = []
            st.session_state["active_version"] = raw_version
            st.session_state["_pending_active_version"] = raw_version
            st.session_state.pop("_anomaly_winsor_preflight", None)
            for cache_key in (
                "pdf_bytes_cache",
                "pdf_filename_cache",
                "pipeline_bytes_cache",
                "pipeline_fingerprint_cache",
            ):
                st.session_state.pop(cache_key, None)
            st.rerun()

    # Version chain display
    chain_html = " → ".join(
        f'<span style="color:{COLORS["accent"]};font-weight:600;">{v}</span>'
        if v == selected else f'<span style="color:{COLORS["text_muted"]};">{v}</span>'
        for v in version_keys
    )
    st.markdown(
        f'<div style="font-size:0.72rem;margin-top:0.5rem;line-height:1.8;">{chain_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Public Render Entry Point
# ---------------------------------------------------------------------------

def _render_image_panel() -> None:
    asset = st.session_state.get("image_dataset")
    if not asset:
        st.markdown(
            f'<p style="color:{COLORS["text_muted"]};font-size:0.8rem;">'
            "No image dataset loaded yet.</p>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(section_header("🖼 Image Dataset"), unsafe_allow_html=True)
    st.caption(asset["filename"])
    col1, col2 = st.columns(2)
    col1.metric("Format", asset["extension"].upper())
    col2.metric("Size", f"{asset['size_bytes'] / (1024 * 1024):.2f} MB")


def _render_voice_panel() -> None:
    asset = st.session_state.get("voice_dataset")
    if not asset:
        st.markdown(
            f'<p style="color:{COLORS["text_muted"]};font-size:0.8rem;">'
            "No voice/audio file loaded yet.</p>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(section_header("🎙️ Voice / Audio"), unsafe_allow_html=True)
    st.caption(asset["filename"])
    col1, col2 = st.columns(2)
    col1.metric("Format", asset["extension"].upper())
    col2.metric("Size", f"{asset['size_bytes'] / (1024 * 1024):.2f} MB")


def _render_video_panel() -> None:
    asset = st.session_state.get("video_dataset")
    if not asset:
        st.markdown(
            f'<p style="color:{COLORS["text_muted"]};font-size:0.8rem;">'
            "No video media loaded yet.</p>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(section_header("📹 Video Media"), unsafe_allow_html=True)
    st.caption(asset["filename"])
    col1, col2 = st.columns(2)
    col1.metric("Format", asset["extension"].upper())
    col2.metric("Size", f"{asset['size_bytes'] / (1024 * 1024):.2f} MB")


def render_sidebar(
    detect_file_modality: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Renders the complete AutoData-Hub sidebar.

    Returns a newly ingested DataFrame if the user uploaded/connected
    a new data source, otherwise None.
    """
    _init_session_state()

    with st.sidebar:
        # ── Logo & Branding ──────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align:center;padding:1rem 0 0.5rem;">
                <div style="font-size:1.6rem;font-weight:800;
                            background:linear-gradient(135deg,{COLORS['accent']},#8B5CF6);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;letter-spacing:-0.02em;">
                    AutoData-Hub
                </div>
                <div style="font-size:0.72rem;color:{COLORS['text_muted']};
                            margin-top:0.2rem;letter-spacing:0.08em;">
                    MLOps Data Preprocessing Engine
                </div>
            </div>
            <hr style="border-color:{COLORS['border']};margin:0.75rem 0;">
            """,
            unsafe_allow_html=True,
        )

        # ── Ingestion Panel ───────────────────────────────────────────────
        render_sidebar_ingestion(detect_file_modality)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Version Panel ─────────────────────────────────────────────────
        if st.session_state.get("active_modality") == "tabular":
            _render_version_panel()
        elif st.session_state.get("active_modality") == "image":
            _render_image_panel()
        elif st.session_state.get("active_modality") == "voice":
            _render_voice_panel()
        else:
            _render_video_panel()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Footer ────────────────────────────────────────────────────────
        st.markdown(
            f'<p style="color:{COLORS["text_muted"]};font-size:0.65rem;'
            f'text-align:center;margin-top:1rem;">v1.0 · Privacy-First MLOps</p>',
            unsafe_allow_html=True,
        )

    return None
