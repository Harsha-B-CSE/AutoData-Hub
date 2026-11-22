"""Streamlit modality-router regression tests."""

from pathlib import Path
import io
import zipfile

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from app import detect_file_modality


def test_detect_file_modality_supported_extensions():
    for filename in (
        "data.csv",
        "data.xlsx",
        "data.parquet",
        "data.json",
        "data.tsv",
        "data.feather",
        "data.jsonl",
    ):
        assert detect_file_modality(filename) == "tabular"

    for filename in (
        "images.zip",
        "photo.png",
        "photo.jpg",
        "photo.jpeg",
        "photo.WEBP",
    ):
        assert detect_file_modality(filename) == "image"

    assert detect_file_modality("unknown.bin") is None
    assert detect_file_modality("") is None


def test_modality_switch_isolates_image_and_tabular_session_state():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=60)
    modality = next(radio for radio in app.radio if radio.label == "Data Modality")
    assert modality.value == "tabular"

    modality.set_value("image").run(timeout=60)
    assert app.session_state["active_modality"] == "image"
    assert app.session_state["dataset_versions"] == {}

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(image, (16, 16), 10, (255, 255, 255), -1)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    image_uploader = next(
        uploader
        for uploader in app.get("file_uploader")
        if uploader.key == "file_uploader_image"
    )
    image_uploader.upload(
        "circle.png",
        buffer.tobytes(),
        "image/png",
    ).run(timeout=60)

    assert app.session_state["active_modality"] == "image"
    assert app.session_state["image_dataset"]["filename"] == "circle.png"
    assert app.session_state["dataset_versions"] == {}
    assert not any("Health Audit" in tab.label for tab in app.tabs)
    image_tabs = {tab.label for tab in app.tabs}
    assert {
        "🔍 Focus & Sharpness",
        "🚀 Dual-Tier Restoration",
        "📄 Export Studio",
    } <= image_tabs
    assert any(metric.label == "Laplacian Variance" for metric in app.metric)

    next(
        button
        for button in app.button
        if button.label == "Run Restoration"
    ).click().run(timeout=60)
    assert "processed_image_bgr" in app.session_state
    next(
        button
        for button in app.button
        if button.label == "Prepare Resized Image"
    ).click().run(timeout=60)
    assert any(
        item.label == "Download Processed Image"
        for item in app.get("download_button")
    )

    next(
        radio for radio in app.radio if radio.label == "Processing Tier"
    ).set_value(
        "🚀 Tier-2: SOTA Deep Learning Models (ONNX/PyTorch)"
    ).run(timeout=60)
    neural_selector = next(
        selector
        for selector in app.selectbox
        if selector.label == "Neural Restoration Model"
    )
    option_text = "\n".join(neural_selector.options)
    for expected_model in (
        "4x UltraSharp V2 (Experimental, Non-Commercial)",
        "DAT Nomos8k (Community x4)",
        "RealPLKSR (4x, MIT)",
    ):
        assert expected_model in option_text
    neural_selector.set_value("NAFNet (Fast Denoise/Deblur)").run(timeout=60)
    next(
        button for button in app.button if button.label == "Run Restoration"
    ).click().run(timeout=60)
    neural_metadata = app.session_state["image_restoration_metadata"]
    assert neural_metadata["status"] in {"success", "fallback"}
    if neural_metadata["status"] == "success":
        assert neural_metadata["runtime"] == "onnxruntime"
    else:
        assert neural_metadata["fallback_reason"] == "weights_missing"
        assert neural_metadata["runtime"] == "opencv_fallback"

    next(
        radio for radio in app.radio if radio.label == "Data Modality"
    ).set_value("tabular").run(timeout=60)
    assert app.session_state["active_modality"] == "tabular"
    assert app.session_state["image_dataset"] is None

    tabular_uploader = next(
        uploader
        for uploader in app.get("file_uploader")
        if uploader.key == "file_uploader_tabular"
    )
    tabular_uploader.upload(
        "records.csv",
        b"id,value\n1,10\n2,20\n",
        "text/csv",
    ).run(timeout=60)

    assert app.session_state["active_modality"] == "tabular"
    assert "v0_raw" in app.session_state["dataset_versions"]
    expected_tabs = {
        "🏥 1. Health Audit",
        "🔒 2. Privacy Firewall",
        "⚙️ 3. Imputation",
        "🔍 4. Anomaly Ensemble",
        "📤 5. Export",
    }
    assert expected_tabs <= {tab.label for tab in app.tabs}


def test_zip_image_dataset_preview_counts_supported_images():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("first.png", buffer.tobytes())
        archive.writestr("nested/second.png", buffer.tobytes())
        archive.writestr("notes.txt", "not an image")

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=60)
    next(
        radio for radio in app.radio if radio.label == "Data Modality"
    ).set_value("image").run(timeout=60)
    uploader = next(
        item
        for item in app.get("file_uploader")
        if item.key == "file_uploader_image"
    )
    uploader.upload(
        "images.zip",
        archive_buffer.getvalue(),
        "application/zip",
    ).run(timeout=60)

    assert not app.exception
    assert app.session_state["image_dataset"]["extension"] == "zip"
    image_count = next(
        metric for metric in app.metric if metric.label == "Images"
    )
    assert int(image_count.value) == 2
    next(
        button for button in app.button if button.label == "Build ZIP Export"
    ).click().run(timeout=60)
    assert any(
        item.label == "Download Processed ZIP"
        for item in app.get("download_button")
    )
