"""Regression tests for color balancing, video processing core, and 4-modality UI."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app import detect_file_modality
from core.schema_inference import ExecutionStatus, TransformResult
from modules.image.color_engine import (
    COLOR_GRADING_PRESETS,
    apply_auto_white_balance,
    apply_color_grading,
)
from modules.image.gpu_manager import (
    clear_vision_cache,
    get_cached_hardware_context,
    get_vision_execution_context,
    run_pytorch_vision_forward,
)
from modules.video import (
    MAX_MEMORY_BUDGET_BYTES,
    TemporalColorStabilizer,
    VideoTransformResult,
    classify_temporal_blur,
    detect_scenes,
    extract_keyframes,
    process_video_pipeline,
)


def _write_synthetic_video(path: Path, frames: list[np.ndarray], fps: float = 24.0) -> Path:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "OpenCV VideoWriter failed to open synthetic test file."
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_cached_hardware_context_and_fp16_autocast(monkeypatch):
    clear_vision_cache()
    autocast_dtypes = []
    thread_calls = []

    @contextmanager
    def fake_autocast(*, dtype):
        autocast_dtypes.append(dtype)
        yield

    fake_torch = SimpleNamespace(
        float16="fp16_sentinel",
        set_num_threads=thread_calls.append,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _idx: "Mock RTX GPU",
            amp=SimpleNamespace(autocast=fake_autocast),
        ),
    )
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_dlls=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    ctx = get_vision_execution_context()
    assert ctx["device"] == "cuda"
    assert ctx["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert ctx["autocast_enabled"] is True
    assert 1 <= ctx["cpu_threads"] <= 4
    assert thread_calls == [ctx["cpu_threads"]]

    output = run_pytorch_vision_forward(lambda x: x * 2, 21, execution_context=ctx)
    assert output == 42
    assert autocast_dtypes == ["fp16_sentinel"]
    assert callable(get_cached_hardware_context)


def test_auto_white_balance_gray_world_and_max_rgb_neutralize_cast():
    # Warm-cast gray card (blue channel suppressed, red elevated)
    warm = np.full((32, 32, 3), (60, 120, 180), dtype=np.uint8)
    gw = apply_auto_white_balance(warm, method="gray_world")
    assert gw.shape == warm.shape
    assert gw.dtype == np.uint8
    assert abs(float(gw[:, :, 0].mean()) - float(gw[:, :, 2].mean())) < 3.0

    # Max-RGB stretches brightest per-channel peaks toward 255
    tinted = np.full((32, 32, 3), (100, 150, 200), dtype=np.uint8)
    mr = apply_auto_white_balance(tinted, method="max_rgb")
    assert np.all(mr >= 250)

    # Pure black stays black and BGRA preserves alpha
    bgra = np.zeros((16, 16, 4), dtype=np.uint8)
    bgra[:, :, 3] = 123
    balanced_bgra = apply_auto_white_balance(bgra, method="max_rgb")
    assert balanced_bgra.shape == (16, 16, 4)
    assert np.all(balanced_bgra[:, :, :3] == 0)
    assert np.all(balanced_bgra[:, :, 3] == 123)


def test_color_grading_lift_gamma_gain_and_presets():
    gradient = np.tile(
        np.linspace(20, 220, 64, dtype=np.uint8)[None, :, None],
        (32, 1, 3),
    )
    neutral = apply_color_grading(
        gradient, lift=0.0, gamma=1.0, gain=1.0, saturation=1.0, preset="Custom"
    )
    assert np.array_equal(neutral, gradient)

    brightened = apply_color_grading(
        gradient, lift=0.05, gamma=1.2, gain=1.1, saturation=1.1, preset="Custom"
    )
    assert brightened.mean() > gradient.mean()

    for preset_name in COLOR_GRADING_PRESETS:
        styled = apply_color_grading(gradient, preset=preset_name)
        assert styled.shape == gradient.shape
        assert styled.dtype == np.uint8


def test_video_keyframe_extractor_and_scene_detector(tmp_path: Path):
    scene_a = np.full((48, 64, 3), (20, 40, 220), dtype=np.uint8)  # strong red
    scene_b = np.full((48, 64, 3), (220, 180, 20), dtype=np.uint8)  # strong cyan-blue
    frames = [scene_a.copy() for _ in range(4)] + [scene_b.copy() for _ in range(4)]
    video_path = _write_synthetic_video(tmp_path / "cuts.avi", frames)

    extracted = extract_keyframes(video_path, sample_every_n_frames=1, max_keyframes=20)
    assert extracted["keyframe_count"] == 8
    assert extracted["memory_bytes"] <= MAX_MEMORY_BUDGET_BYTES
    assert extracted["within_memory_budget"] is True

    scenes = detect_scenes(extracted["keyframes"], fps=extracted["fps"], correlation_threshold=0.65)
    assert scenes["scene_cut_indices"] == [4]
    assert scenes["scene_count"] == 2
    assert scenes["min_correlation"] < 0.65


def test_temporal_blur_distinguishes_motion_blur_from_defocus_blur():
    # Sharp textured frame
    sharp = np.zeros((96, 96, 3), dtype=np.uint8)
    for step in range(8, 90, 12):
        cv2.rectangle(sharp, (step, 16), (step + 6, 80), (255, 255, 255), -1)

    sharp_metrics = classify_temporal_blur(sharp, sharp, sharpness_threshold=100.0)
    assert sharp_metrics["blur_type"] == "sharp"
    assert not sharp_metrics["is_blurry"]

    # Static defocus blur: heavily blurred, identical position -> low Laplacian + low flow
    defocused = cv2.GaussianBlur(sharp, (21, 21), 6.0)
    defocus_metrics = classify_temporal_blur(
        defocused, defocused, sharpness_threshold=100.0, motion_threshold=2.0
    )
    assert defocus_metrics["blur_type"] == "defocus_blur"
    assert defocus_metrics["is_defocus_blur"] is True

    # Dynamic motion blur: blurred AND shifted significantly -> low Laplacian + high flow
    shifted = np.roll(defocused, shift=10, axis=1)
    motion_metrics = classify_temporal_blur(
        defocused, shifted, sharpness_threshold=100.0, motion_threshold=2.0
    )
    assert motion_metrics["blur_type"] == "motion_blur"
    assert motion_metrics["is_motion_blur"] is True
    assert motion_metrics["mean_flow_magnitude"] >= 2.0


def test_temporal_ema_color_stabilizer_smooths_gain_jumps():
    stabilizer = TemporalColorStabilizer(alpha=0.15, method="gray_world")
    frame_a = np.full((24, 24, 3), (100, 100, 100), dtype=np.uint8)
    frame_b = np.full((24, 24, 3), (50, 100, 200), dtype=np.uint8)

    stabilizer.step(frame_a)
    stabilizer.step(frame_b)

    g0 = np.array(stabilizer.gain_history[0])
    g1 = np.array(stabilizer.gain_history[1])
    # EMA prevents full instantaneous jump to frame_b's raw blue gain (> 2.0)
    assert np.allclose(g0, [1.0, 1.0, 1.0], atol=1e-4)
    assert g1[0] < 1.35


def test_video_pipeline_returns_standardized_transform_result(tmp_path: Path):
    f1 = np.full((48, 64, 3), (40, 80, 180), dtype=np.uint8)
    cv2.rectangle(f1, (10, 10), (50, 38), (240, 240, 240), 2)
    f2 = np.roll(f1, shift=4, axis=1)
    f3 = np.full((48, 64, 3), (190, 90, 30), dtype=np.uint8)
    video_path = _write_synthetic_video(tmp_path / "sample.avi", [f1, f2, f3])

    result = process_video_pipeline(
        video_path,
        sample_every_n_frames=1,
        ema_alpha=0.15,
        apply_grading=True,
        preset="🎬 Teal & Orange",
    )

    assert isinstance(result, VideoTransformResult)
    assert isinstance(result, TransformResult)
    assert result.status == ExecutionStatus.SUCCESS
    assert len(result.transformed_frames) == 3
    assert result.module_result is not None
    assert result.module_result.metrics["ema_alpha"] == pytest.approx(0.15)
    assert result.module_result.metrics["within_memory_budget"] is True
    assert result.module_result.metrics["scene_cut_count"] >= 1


def test_app_routes_all_four_modalities_including_video(tmp_path: Path):
    for name in ("clip.mp4", "clip.avi", "clip.mov", "clip.mkv", "clip.webm"):
        assert detect_file_modality(name) == "video"

    f1 = np.full((48, 64, 3), (30, 70, 200), dtype=np.uint8)
    f2 = np.full((48, 64, 3), (200, 70, 30), dtype=np.uint8)
    video_path = _write_synthetic_video(tmp_path / "ui_test.avi", [f1, f2])

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=60)

    modality_radio = next(r for r in app.radio if r.label == "Data Modality")
    assert modality_radio.options == [
        "📊 Tabular Datasets",
        "🖼️ Image Datasets",
        "🎙️ Voice / Audio",
        "📹 Video Media",
    ]

    modality_radio.set_value("video").run(timeout=60)
    assert app.session_state["active_modality"] == "video"

    uploader = next(
        u for u in app.get("file_uploader") if u.key == "file_uploader_video"
    )
    uploader.upload("ui_test.avi", video_path.read_bytes(), "video/x-msvideo").run(timeout=60)
    assert not app.exception
    assert app.session_state["video_dataset"]["filename"] == "ui_test.avi"

    next(
        b for b in app.button if b.label == "Run Video Analysis & Color Pipeline"
    ).click().run(timeout=60)

    assert not app.exception
    assert {
        "🎬 Keyframes & Scene Cuts",
        "🔍 Temporal Blur",
        "🎨 EMA Color Grading",
        "📤 Export Studio",
    } <= {tab.label for tab in app.tabs}
    assert any("Workflow & Methodology Guide" in exp.label for exp in app.expander)
