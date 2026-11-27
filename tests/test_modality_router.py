"""Streamlit modality-router regression tests."""

from pathlib import Path
import io
import wave
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

    for filename in (
        "speech.wav",
        "speech.mp3",
        "speech.flac",
        "speech.ogg",
        "speech.M4A",
    ):
        assert detect_file_modality(filename) == "voice"

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
    assert any("CPU Host Mode" in str(markdown.value) for markdown in app.markdown)

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


def test_voice_workspace_enhances_and_prepares_24bit_wav():
    sample_rate = 16_000
    samples = np.arange(1_600, dtype=np.float32)
    audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * samples / sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.round(audio * 32767.0).astype("<i2").tobytes())

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=60)
    next(
        radio for radio in app.radio if radio.label == "Data Modality"
    ).set_value("voice").run(timeout=60)
    uploader = next(
        item
        for item in app.get("file_uploader")
        if item.key == "file_uploader_voice"
    )
    uploader.upload("speech.wav", buffer.getvalue(), "audio/wav").run(timeout=60)

    assert not app.exception
    assert app.session_state["active_modality"] == "voice"
    assert app.session_state["voice_dataset"]["filename"] == "speech.wav"
    assert {
        "🔍 Telemetry",
        "🧠 Neural Enhancer & Booster",
        "🎛️ Smart Tuner",
        "📄 Export Studio",
    } <= {tab.label for tab in app.tabs}
    assert {
        "SNR (dB)",
        "SI-SDR (dB)",
        "Estimated PESQ",
        "STOI Index",
        "Crest Factor (dB)",
    } <= {metric.label for metric in app.metric}
    assert {
        "Quality Booster Gain (dB)",
        "Low Bass (20-250 Hz)",
        "Mid Range (250-2.5 kHz)",
        "High Air (2.5-16 kHz)",
    } <= {slider.label for slider in app.slider}
    assert any(
        "AI Acoustic Suggestions" in str(markdown.value)
        for markdown in app.markdown
    )
    studio_selector = next(
        selector
        for selector in app.selectbox
        if selector.label == "Studio Mastering Preset"
    )
    assert studio_selector.options == [
        "🎙️ Broadcast Vocal Polish",
        "📻 Warm Podcast Warmth",
        "✨ Air & Crystal Clarity",
        "🎛️ Custom Studio Manual Mode",
    ]
    studio_selector.set_value("🎛️ Custom Studio Manual Mode").run(timeout=60)
    assert {
        "Low-Mid Bell Gain (dB)",
        "Presence Bell Gain (dB)",
        "Low-Mid Bell Q",
        "De-Esser Knee (dB)",
        "De-Esser Attack (ms)",
        "Compressor Knee (dB)",
        "Compressor Makeup (dB)",
        "True-Peak Ceiling (dBTP)",
    } <= {slider.label for slider in app.slider}
    assert {
        "Enable Dynamic De-Esser",
        "Enable VCA Compressor",
        "Enable Low-Cut Filter",
    } <= {toggle.label for toggle in app.toggle}
    assert any(
        item.label == "Download Decoded Audio"
        for item in app.get("download_button")
    )

    next(
        button for button in app.button if button.label == "Run Voice Enhancement"
    ).click().run(timeout=60)
    assert not app.exception
    assert app.session_state["voice_enhanced_audio"].shape == (audio.size * 3,)
    assert app.session_state["voice_transform_result"].status.value == "partial"
    assert app.session_state["voice_module_result"].metrics["decoder"] == "wave"
    assert app.session_state["voice_module_result"].metrics["mode"] == "file_offline"
    assert app.session_state["voice_module_result"].metrics["booster_true_peak_dbfs"] <= -0.49

    next(
        selector
        for selector in app.selectbox
        if selector.label == "Multi-Format Export"
    ).set_value(".WAV (24-bit PCM)").run(timeout=60)
    downloads = app.get("download_button")
    assert any(item.label == "Download Enhanced Audio" for item in downloads)
