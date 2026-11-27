"""Video telemetry JSON and processed frame archive export tab."""

from __future__ import annotations

from functools import partial
import json
import tempfile
import weakref
import zipfile

import cv2
import streamlit as st

from ui.utils_ux import render_tab_guidance_header


def _frame_archive(result_ref, manifest_bytes: bytes) -> bytes:
    result = result_ref()
    if result is None:
        raise ValueError("This analysis has been replaced; download the current result instead.")
    metrics = result.module_result.metrics
    available = (metrics["memory_budget_bytes"] - metrics["source_buffer_bytes"]
                 - metrics["retained_original_bytes"] - metrics["retained_output_bytes"])
    with tempfile.TemporaryFile() as buf:
        # PNGs are already compressed. Spool them one at a time rather than
        # keeping a growing in-memory archive beside the decoded frame pairs.
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("video_telemetry_manifest.json", manifest_bytes)
            for item, frame in zip(result.module_result.artifacts["keyframes"], result.transformed_frames, strict=True):
                ok, encoded = cv2.imencode(".png", frame)
                if not ok:
                    raise ValueError(f"Could not encode source frame {item.frame_index}.")
                zf.writestr(f"frames/frame_{item.frame_index:04d}.png", memoryview(encoded).cast("B"))
                del encoded
        size = buf.tell()
        if size > available:
            raise ValueError("Frame archive exceeds the remaining working memory budget.")
        buf.seek(0)
        return buf.read(size)


def render_tab_video_export(result, source_name: str) -> None:
    """Export video analysis manifest JSON and processed keyframe archive."""
    render_tab_guidance_header(
        title="Video Export Studio",
        objective="Export reproducible video telemetry and stabilized keyframes.",
        methodology=(
            "Packages TransformationAction audit logs, scene/blur metrics, and "
            "EMA-stabilized PNG samples by source frame index. The ZIP is built only on download."
        ),
        deliverable="Downloadable JSON run manifest and processed keyframes ZIP.",
    )
    metrics = result.module_result.metrics
    manifest = {
        "source_name": source_name,
        "status": result.status.value,
        "metrics": metrics,
        "audit_log": result.audit_log,
        "actions": [action.to_dict() for action in result.applied_actions],
    }
    manifest_bytes = json.dumps(manifest, indent=2, default=str).encode("utf-8")

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download Video Telemetry (JSON)",
        data=manifest_bytes,
        file_name="video_telemetry_manifest.json",
        mime="application/json",
        width="stretch",
        key="video_download_manifest",
    )

    col2.download_button(
        "Download Processed Frames (ZIP)",
        data=partial(_frame_archive, weakref.ref(result), manifest_bytes),
        file_name="video_processed_keyframes.zip",
        mime="application/zip",
        width="stretch",
        on_click="ignore",
        key="video_download_zip",
    )
