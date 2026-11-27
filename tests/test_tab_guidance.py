"""Check the first visible element of each Streamlit workspace and result tab."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from core.schema_inference import ModuleResult, TransformResult
from modules.voice.audio_loader import encode_wav_bytes
from modules.voice.studio_tuner import CUSTOM_PRESET


def _check_guide(guide, pane):
    assert guide.type == "expander", pane
    assert "Tab Capability Guide: " in guide.label, pane
    assert "Workflow & Methodology Guide" in guide.label, pane
    assert not guide.proto.expanded, pane
    headings = ("Objective & Scope", "Core Methodology", "Output Deliverable")
    for column, heading in zip(guide.columns, headings, strict=True):
        label, text = column.markdown[0].value.split("\n\n", 1)
        assert label.endswith(f"{heading}**") and text.strip(), pane


def _check_tabs(app, expected):
    assert not app.exception
    labels = [tab.label for tab in app.tabs]
    for name in expected:
        assert any(label.endswith(name) for label in labels), (name, labels)
    for tab in app.tabs:
        _check_guide(tab[0], tab.label)


def _media_layout(modality, name, payload):
    app = AppTest.from_string(
        f"from ui.tabs_{modality}.main_layout import render_{modality}_app\n"
        "import streamlit as st\n"
        f"render_{modality}_app(st.session_state.asset)",
        default_timeout=60,
    )
    app.session_state["asset"] = {"filename": name, "bytes": payload}
    return app


def test_tabular_guides_include_ingestion_and_result_previews():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60)
    raw = pd.DataFrame({"email": ["a@example.test", "b@example.test"], "value": [1.0, None], "units": [3, 7]})
    masked = raw.assign(email="[redacted]")
    filled = masked.fillna({"value": 1.0})
    app.session_state["dataset_versions"] = {"v0_raw": raw, "v1_privacy": masked, "v2_imputed": filled}
    app.session_state["transform_results"] = {
        "privacy": TransformResult.success(masked), "imputation": TransformResult.success(filled),
    }
    scatter = go.Figure(go.Scatter(x=[-1, 1], y=[0, 0], mode="markers"))
    app.session_state["module_results"] = {
        "privacy": ModuleResult.success("privacy", "v1_privacy", metrics={"pii_findings": {"email": ["email"]}}),
        "anomaly": ModuleResult.success("anomaly", "v2_imputed", artifacts={"pca_anomaly_scatter": scatter}),
    }
    _check_tabs(app.run(), (
        "File Upload", "Database", "Health Audit", "Privacy Firewall", "Imputation", "Anomaly Ensemble", "Export",
        "Original (Raw)", "Sanitized", "Before", "After", "Summary", "2D Projection", "3D Projection (if available)",
    ))


def test_raw_voice_and_custom_tuner_guides_precede_status_and_controls():
    audio = encode_wav_bytes(np.zeros(1600, dtype=np.float32), 16_000)
    app = _media_layout("voice", "voice.wav", audio)
    app.session_state["voice_studio_preset"] = CUSTOM_PRESET
    _check_tabs(app.run(), (
        "Telemetry", "Neural Enhancer & Booster", "Smart Tuner", "Export Studio",
        "Five-Band EQ", "Dynamics", "Harmonics & Ceiling",
    ))
    for tab in app.tabs:
        if tab.label.endswith(("Smart Tuner", "Export Studio")):
            assert "decoded" in tab.info[0].value.lower()


def test_image_guides():
    ok, image = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    app = _media_layout("image", "image.png", image.tobytes()).run()
    _check_tabs(app, ("Focus & Sharpness", "Dual-Tier Restoration", "Color Balance & Grading", "Export Studio"))


def test_video_result_guides(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (32, 32))
    try:
        assert writer.isOpened()
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    app = _media_layout("video", path.name, path.read_bytes()).run()
    app.button(key="video_run_pipeline").click().run()
    _check_tabs(app, ("Keyframes & Scene Cuts", "Temporal Blur", "EMA Color Grading", "Export Studio"))


def test_live_voice_guide_is_first_even_without_webrtc(monkeypatch):
    monkeypatch.setattr("ui.tabs_voice.tab_live_webrtc.WEBRTC_AVAILABLE", False)
    app = AppTest.from_string(
        "from ui.tabs_voice.tab_live_webrtc import render_tab_live_webrtc\nrender_tab_live_webrtc()"
    ).run()
    assert not app.exception
    _check_guide(app.main[0], "Live Microphone Stream")
