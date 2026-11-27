"""Regression tests for non-mutating image versions and the mounted UI."""

import io
from pathlib import Path
from unittest.mock import Mock
import zipfile

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from modules.image.pipeline_image import GradeSettings, ImagePipelineBuffer
from modules.image.resizer import letterbox_resize
from ui.tabs_image import main_layout, tab_restoration


@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 3), (5, 7, 4)])
def test_buffer_owns_pixels_and_replays_grade_without_compounding(shape):
    source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    raw = source[:, ::-1]
    raw.setflags(write=False)  # Still aliases an externally writable source.
    expected_raw = raw.copy()
    v0 = ImagePipelineBuffer(raw)
    source[:] = 0
    v0.get_base_for_color_grading()[:] = 0
    np.testing.assert_array_equal(v0.get_final_output(), expected_raw)

    recipe = GradeSettings(wb_method="gray_world", gain=1.4)
    raw_grade = v0.with_grading(recipe.apply(v0.get_base_for_color_grading()), recipe)
    restored = np.repeat(np.repeat(expected_raw + 8, 2, axis=0), 2, axis=1)
    replayed = raw_grade.with_restoration(restored)
    np.testing.assert_array_equal(replayed.get_final_output(), recipe.apply(restored))
    np.testing.assert_array_equal(replayed.without_grading().get_final_output(), restored)
    restored[:] = 0
    assert np.any(replayed.restored_bgr)
    assert replayed.raw_bgr is v0.raw_bgr
    assert replayed.without_grading().restored_bgr is replayed.restored_bgr
    with pytest.raises(ValueError):
        replayed.get_final_output().setflags(write=True)
    np.testing.assert_array_equal(replayed.reset_to_raw().get_final_output(), expected_raw)


@pytest.mark.parametrize("invalid", [None, np.empty((0, 0)), np.zeros((4, 4, 2)), np.array([[object()]])])
def test_invalid_image_cannot_start_a_pipeline(invalid):
    with pytest.raises(ValueError, match="raw_bgr"):
        ImagePipelineBuffer(invalid)


def test_zip_restore_grade_rerun_reset_and_export(monkeypatch):
    raw = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    second = np.full((8, 12, 3), (20, 70, 90), dtype=np.uint8)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        for name, image in (("first.png", raw), ("nested/second.png", second)):
            archive.writestr(name, cv2.imencode(".png", image)[1].tobytes())

    calls = []

    def restore(image, *, scale_factor, **kwargs):
        # Deterministic model output: exercise the real Tier-2 UI without requiring weights.
        result = cv2.resize(image // 3 + 10 * (len(calls) + 1), None,
                            fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        calls.append(result.copy())
        return {"image_bgr": result, "runtime": "test", "status": "success", "message": "Restored"}

    monkeypatch.setattr(tab_restoration, "run_neural_restoration", restore)
    extract = Mock(wraps=main_layout.extract_image_items)
    monkeypatch.setattr(main_layout, "extract_image_items", extract)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    app.radio(key="modality_selector").set_value("image").run()
    app.get("file_uploader")[0].upload("images.zip", data.getvalue(), "application/zip").run()
    assert not app.exception
    state = app.session_state
    app.radio(key="image_processing_tier").set_value(tab_restoration.TIER_NEURAL).run()
    app.selectbox(key="restoration_scale_factor_0").set_value(2).run()
    app.button(key="image_run_restoration").click().run()
    assert calls[0].shape == (16, 24, 3)
    recipe = GradeSettings(gain=1.4)
    app.slider(key="image_color_gain").set_value(recipe.gain).run()
    app.button(key="image_apply_color").click().run()
    np.testing.assert_array_equal(state["image_buffer"].get_final_output(), recipe.apply(calls[0]))
    app.button(key="image_run_restoration").click().run()
    assert len(calls) == 2
    buffer = state["image_buffer"]
    np.testing.assert_array_equal(buffer.raw_bgr, raw)
    np.testing.assert_array_equal(buffer.restored_bgr, calls[1])
    np.testing.assert_array_equal(buffer.graded_bgr, recipe.apply(calls[1]))

    app.button(key="image_prepare_export").click().run()
    app.button(key="image_build_zip").click().run()
    expected = letterbox_resize(recipe.apply(calls[1]), target_size=(512, 512))
    with zipfile.ZipFile(io.BytesIO(state["image_export_zip_bytes"])) as archive:
        assert archive.read("first.png") == state["image_export_result"]["bytes"]
        for name, target in (("first.png", expected), ("nested/second.png", letterbox_resize(second, (512, 512)))):
            decoded = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, target)

    app.selectbox(key="image_member_selector").set_value("nested/second.png").run()
    assert state["image_buffer"].grade_settings is None and state["image_buffer"].restored_bgr is None
    assert app.slider(key="image_color_gain").value == 1.0
    app.slider(key="image_color_sat").set_value(0.5).run()
    assert "image_export_zip_bytes" not in state
    app.selectbox(key="image_member_selector").set_value("first.png").run()
    assert state["image_buffer"].grade_settings == recipe
    assert app.slider(key="image_color_gain").value == recipe.gain
    app.button(key="image_reset_color").click().run()
    assert state["image_buffer"].graded_bgr is None
    np.testing.assert_array_equal(state["image_buffer"].get_final_output(), calls[1])
    app.button(key="image_reset_restoration").click().run()
    np.testing.assert_array_equal(state["image_buffer"].get_final_output(), raw)
    assert "image_restoration_metadata" not in state
    assert extract.call_count == 1
    assert not app.exception


def test_same_name_replacement_and_grayscale_blend(monkeypatch):
    # Force the documented CPU fallback, which returns BGR for a grayscale upload.
    monkeypatch.setattr(tab_restoration, "run_neural_restoration", lambda image, **kwargs: {
        "image_bgr": cv2.cvtColor(255 - image, cv2.COLOR_GRAY2BGR),
        "status": "fallback", "message": "CPU fallback", "runtime": "opencv_fallback",
    })
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    app.radio(key="modality_selector").set_value("image").run()

    def upload(payload):
        app.get("file_uploader")[0].upload("same.png", payload, "image/png").run()
        assert not app.exception

    raw = np.arange(36, dtype=np.uint8).reshape(6, 6)
    inputs = [cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tobytes()
              for img in (raw, 255 - raw)]
    assert len(inputs[0]) == len(inputs[1])
    upload(inputs[0])
    app.radio(key="image_processing_tier").set_value(tab_restoration.TIER_NEURAL).run()
    app.slider(key="restoration_blend_alpha").set_value(0.5).run()
    app.button(key="image_run_restoration").click().run()
    assert not app.exception
    np.testing.assert_array_equal(app.session_state["image_buffer"].restored_bgr, np.full((6, 6, 3), 128, np.uint8))
    upload(inputs[1])
    assert app.session_state["image_buffer"].restored_bgr is None
    np.testing.assert_array_equal(app.session_state["image_buffer"].raw_bgr, 255 - raw)
    upload(b"not a PNG")
    assert "image_buffer" not in app.session_state
    assert any("validation failed" in e.value for e in app.error)
