"""Unit tests for image sharpness and letterbox UI adapters."""

import cv2
import numpy as np
import pytest
import sys
from types import SimpleNamespace

import modules.image.neural_restoration as neural_restoration
from modules.image.clahe_contrast import apply_clahe_contrast
from modules.image.denoiser import apply_bilateral_denoise
from modules.image.neural_restoration import MODEL_REGISTRY, run_neural_restoration
from modules.image.resizer import letterbox_resize
from modules.image.sharpness_evaluator import (
    evaluate_sharpness_laplacian,
    evaluate_tenengrad,
)


def test_sharpness_metrics_distinguish_edges_from_flat_image():
    flat = np.zeros((64, 64, 3), dtype=np.uint8)
    edged = flat.copy()
    cv2.rectangle(edged, (12, 12), (52, 52), (255, 255, 255), 2)

    edged_metrics = evaluate_sharpness_laplacian(edged, threshold=100.0)
    flat_metrics = evaluate_sharpness_laplacian(flat, threshold=100.0)
    assert edged_metrics["laplacian_variance"] > flat_metrics["laplacian_variance"]
    assert edged_metrics["tenengrad_score"] > flat_metrics["tenengrad_score"]
    assert edged_metrics["mean_luminance"] > flat_metrics["mean_luminance"]
    assert flat_metrics["is_blurry"]
    assert flat_metrics["status"] == "BLURRY"
    assert evaluate_tenengrad(edged) > evaluate_tenengrad(flat)


def test_letterbox_resize_preserves_aspect_and_exact_canvas():
    source = np.full((50, 100, 3), (10, 20, 30), dtype=np.uint8)
    resized = letterbox_resize(
        source,
        target_size=(224, 224),
        fill_color=(0, 0, 0),
    )
    assert resized.shape == (224, 224, 3)
    nonzero_rows = np.where(resized.any(axis=(1, 2)))[0]
    assert len(nonzero_rows) in {111, 112}
    assert nonzero_rows[0] > 0
    assert nonzero_rows[-1] < 223


def test_classical_contrast_and_denoise_preserve_shape_and_dtype():
    rng = np.random.default_rng(7)
    source = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
    denoised = apply_bilateral_denoise(source)
    enhanced = apply_clahe_contrast(denoised)
    assert denoised.shape == source.shape
    assert enhanced.shape == source.shape
    assert enhanced.dtype == np.uint8


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_neural_models_use_explicit_fallback_when_weights_are_missing(
    model_name, monkeypatch, tmp_path
):
    monkeypatch.setattr(neural_restoration, "_model_root", lambda: tmp_path)
    source = np.zeros((12, 18, 3), dtype=np.uint8)
    result = run_neural_restoration(
        source,
        model_name=model_name,
        scale_factor=2,
        device="cpu",
    )
    assert result["status"] == "fallback"
    assert not result["neural_applied"]
    assert result["fallback_applied"]
    assert result["fallback_reason"] == "weights_missing"
    assert result["runtime"] == "opencv_fallback"
    assert result["image_bgr"].shape[:2] == (24, 36)


def test_neural_onnx_success_path_uses_inference_session(monkeypatch, tmp_path):
    class FakeSessionOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None

    class FakeSession:
        def __init__(self, path, sess_options, providers):
            self.providers = providers

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="input",
                    shape=[1, 3, None, None],
                    type="tensor(float)",
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name="output")]

        def run(self, output_names, feeds):
            return [feeds["input"]]

        def get_providers(self):
            return self.providers

    fake_runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    monkeypatch.setattr(neural_restoration, "_model_root", lambda: tmp_path)
    monkeypatch.setitem(
        neural_restoration.MODEL_REGISTRY["NAFNet (Fast Denoise/Deblur)"],
        "sha256",
        None,
    )
    (tmp_path / "nafnet.onnx").write_bytes(b"fake-model")

    source = np.zeros((8, 12, 3), dtype=np.uint8)
    source[:, :, 1] = 127
    result = run_neural_restoration(
        source,
        model_name="NAFNet (Fast Denoise/Deblur)",
        scale_factor=1,
        device="cpu",
    )
    assert result["status"] == "success"
    assert result["neural_applied"]
    assert not result["fallback_applied"]
    assert result["runtime"] == "onnxruntime"
    assert result["providers"] == ["CPUExecutionProvider"]
    assert result["image_bgr"].shape == source.shape


def test_fixed_x4_model_uses_overlap_tiling(monkeypatch, tmp_path):
    class FakeSessionOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None
        log_severity_level = None

    class FakeSession:
        def __init__(self, path, sess_options, providers):
            self.providers = providers

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="input",
                    shape=[1, 3, 4, 4],
                    type="tensor(float)",
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name="output")]

        def run(self, output_names, feeds):
            tensor = feeds["input"]
            return [np.repeat(np.repeat(tensor, 4, axis=2), 4, axis=3)]

        def get_providers(self):
            return self.providers

    fake_runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
    )
    model_name = "RealPLKSR (4x, MIT)"
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    monkeypatch.setattr(neural_restoration, "_model_root", lambda: tmp_path)
    monkeypatch.setitem(neural_restoration.MODEL_REGISTRY[model_name], "sha256", None)
    monkeypatch.setitem(
        neural_restoration.MODEL_REGISTRY[model_name], "tile_overlap", 1
    )
    (tmp_path / "realplksr_x4.onnx").write_bytes(b"fake-model")

    source = np.zeros((5, 6, 3), dtype=np.uint8)
    source[:, :, 2] = 200
    result = run_neural_restoration(
        source,
        model_name=model_name,
        scale_factor=4,
        device="cpu",
    )
    assert result["status"] == "success"
    assert result["tiled"]
    assert result["tile_overlap"] == 1
    assert result["native_scale"] == 4
    assert result["image_bgr"].shape == (20, 24, 3)


def test_model_metadata_exposes_license_and_experimental_flags():
    ultrasharp = neural_restoration.get_neural_model_status(
        "4x UltraSharp V2 (Experimental, Non-Commercial)"
    )
    dat = neural_restoration.get_neural_model_status(
        "DAT Nomos8k (Community x4)"
    )
    realplksr = neural_restoration.get_neural_model_status(
        "RealPLKSR (4x, MIT)"
    )
    assert not ultrasharp["commercial_use"]
    assert ultrasharp["experimental"]
    assert dat["experimental"]
    assert realplksr["license"] == "MIT"
    assert realplksr["native_scale"] == 4
    assert "SUPIR" not in neural_restoration.MODEL_REGISTRY
    for model_name in neural_restoration.MODEL_REGISTRY:
        status = neural_restoration.get_neural_model_status(model_name)
        assert status["recommendation_label"]
        assert status["recommended_for"]
        assert status["use_cases"]
        assert status["avoid_for"]
