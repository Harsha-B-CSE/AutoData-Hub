"""Video staging, working-set admission, partial results, and deferred export."""

import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core.schema_inference import ExecutionStatus
from modules.video import extract_keyframes, iter_video_frames, open_video_source, process_video_pipeline
from modules.video import keyframe_extractor as extractor
from ui.tabs_video import tab_video_export as export


class ChunkedUpload(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.requests = []

    def getvalue(self):
        pytest.fail("An upload must not be materialized with getvalue().")

    def read(self, n=-1):
        assert 0 < n <= 256 * 1024
        self.requests.append(n)
        return super().read(n)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, (160, 96))
    assert writer.isOpened()
    try:
        for i in range(24):
            frame = np.full((96, 160, 3), (40, 90, 160) if i < 12 else (180, 80, 20), np.uint8)
            cv2.rectangle(frame, (i * 2, 12), (i * 2 + 15, 35), (230, 230, 230), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_staging_uses_bounded_reads_restores_cursor_and_cleans_up():
    data = b"video" * 150_000
    stream = ChunkedUpload(data)
    stream.seek(31)
    with open_video_source({"source": stream, "filename": "sample.avi"}) as (path, name):
        assert name == "sample.avi" and path.read_bytes() == data
        assert stream.tell() == 31
    assert len(stream.requests) > 2
    assert not path.exists()


def test_staging_rejects_oversize_without_materializing_or_leaking(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(extractor.tempfile, "tempdir", str(tmp_path))
    seekable = ChunkedUpload(b"x" * 1025)
    seekable.seek(44)
    with pytest.raises(ValueError, match="staging limit"), open_video_source(seekable):
        pytest.fail("Oversized seekable upload admitted.")
    assert not seekable.requests and seekable.tell() == 44

    class Pipe(ChunkedUpload):
        def tell(self):
            raise io.UnsupportedOperation("not seekable")

    pipe = Pipe(b"y" * 2048)
    with pytest.raises(ValueError, match="staging limit"), open_video_source(pipe):
        pytest.fail("Oversized pipe admitted.")
    assert pipe.requests == [1025]  # only one excess byte needed to reject
    with pytest.raises(ValueError, match="staging limit"), open_video_source(memoryview(b"z" * 1025)):
        pytest.fail("Oversized buffer admitted.")
    with pytest.raises(TypeError, match="binary readable"), open_video_source({"bytes": 10**12}):
        pytest.fail("Integer was implicitly converted into a huge byte string.")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("budget", [0, -1, 1, float("nan"), float("inf")])
def test_unusable_budget_fails_before_reading_upload(budget):
    source = ChunkedUpload(b"not even opened")
    result = process_video_pipeline(source, memory_budget_bytes=budget)
    assert result.status == result.module_result.status == ExecutionStatus.FAILED
    assert "budget" in result.error_message and not result.transformed_frames
    assert not source.requests


@pytest.mark.parametrize("source", [b"", {"filename": "bad.mp4", "bytes": b"not a movie"}])
def test_empty_and_corrupt_sources_fail_without_outputs(source):
    result = process_video_pipeline(source)
    assert result.status == result.module_result.status == ExecutionStatus.FAILED
    assert result.error_message and not result.transformed_frames


class Capture:
    """Metadata can lie; this decoder never allocates above 16x16 in a test."""

    def __init__(self, width, height, reported=5, readable=2):
        self.width, self.height = width, height
        self.reported, self.readable = reported, readable
        self.grabs = self.retrieves = 0
        self.released = False

    def isOpened(self):
        return True

    def release(self):
        self.released = True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FRAME_COUNT: self.reported, cv2.CAP_PROP_FPS: 24.0,
        }[prop]

    def grab(self):
        self.grabs += 1
        return self.grabs <= self.readable

    def retrieve(self):
        self.retrieves += 1
        return True, np.zeros((16, 16, 3), np.uint8)


@pytest.mark.parametrize("width,height,error", [
    (0, 64, "dimensions"), (float("nan"), 48, "dimensions"), (8192, 4320, "dimensions"),
    (4096, 4096, "dimensions"), (3840, 2160, "budget"),
])
def test_source_preflight_precedes_first_decode_even_for_tiny_previews(monkeypatch, width, height, error):
    cap = Capture(width, height)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    result = process_video_pipeline(b"fake", max_dimension=16, memory_budget_bytes=4 * 2**20)
    assert result.status == ExecutionStatus.FAILED and error in result.error_message
    assert cap.grabs == cap.retrieves == 0 and cap.released


def test_one_frame_processing_scratch_is_checked_before_decode(monkeypatch):
    # Both 1080p BGR images and the decode reserve fit 64 MiB; the full-size
    # floating-point working arrays required for grading do not.
    cap = Capture(1920, 1080)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    result = process_video_pipeline(b"fake", max_dimension=None, memory_budget_bytes=64 * 2**20)
    assert result.status == ExecutionStatus.FAILED and "budget" in result.error_message
    assert cap.grabs == cap.retrieves == 0 and cap.released


def test_changed_frame_layout_is_not_analyzed(monkeypatch):
    cap = Capture(8, 8)  # retrieve returns 16x16
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    result = process_video_pipeline(b"fake")
    assert result.status == ExecutionStatus.FAILED and "frame layout" in result.error_message
    assert cap.retrieves == 1 and cap.released and not result.transformed_frames


def test_dimension_change_on_skipped_frame_stops_capture(monkeypatch):
    class VariableSize(Capture):
        def grab(self):
            ok = super().grab()
            if self.grabs == 2:
                self.width = 8192  # only metadata, never allocate this frame
            return ok

    cap = VariableSize(16, 16)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    result = process_video_pipeline(b"fake", sample_every_n_frames=8)
    assert result.status == ExecutionStatus.FAILED and "dimensions changed" in result.error_message
    assert cap.grabs == 2 and cap.retrieves == 1 and cap.released


def test_stream_iterator_limit_does_not_scan_the_next_stride(monkeypatch):
    cap = Capture(16, 16, readable=1000)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    assert len(list(iter_video_frames("fake.avi", stride=1000, max_frames=1))) == 1
    assert cap.grabs == cap.retrieves == 1 and cap.released


@pytest.mark.parametrize("reported,limit,status,reason", [
    (5, 5, ExecutionStatus.PARTIAL, "early_end_of_stream"),
    (0, 1, ExecutionStatus.PARTIAL, "max_keyframes"),
    (0, 2, ExecutionStatus.SUCCESS, "end_of_stream"),
    (1, 5, ExecutionStatus.SUCCESS, "end_of_stream"),
])
def test_sampling_does_not_trust_reported_counts(monkeypatch, reported, limit, status, reason):
    cap = Capture(16, 16, reported=reported)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    result = process_video_pipeline(b"fake", max_keyframes=limit)
    assert result.status == result.module_result.status == status
    assert result.module_result.metrics["sampling_stop_reason"] == reason
    assert cap.retrieves == len(result.transformed_frames) == min(limit, 2)
    assert cap.released


def test_pipeline_budgets_both_frame_sets_and_work_arrays(clip):
    budget = 4 * 2**20
    assert extract_keyframes(clip, memory_budget_bytes=budget)["keyframe_count"] == 24
    upload = ChunkedUpload(clip.read_bytes())
    result = process_video_pipeline(upload, memory_budget_bytes=budget, apply_grading=True)
    assert result.status == result.module_result.status == ExecutionStatus.PARTIAL
    m = result.module_result.metrics
    n = len(result.transformed_frames)
    assert 0 < n < 24 and m["keyframe_count"] == n
    assert m["sampling_stop_reason"] == "memory_budget" and not m["sampling_complete"]
    assert "working memory budget" in result.module_result.issues[0]
    assert m["retained_original_bytes"] == m["retained_output_bytes"] == n * 160 * 96 * 3
    assert m["source_buffer_bytes"] == clip.stat().st_size
    assert m["source_buffer_bytes"] + 2 * m["retained_original_bytes"] < m["estimated_peak_bytes"]
    assert m["estimated_peak_bytes"] <= m["memory_budget_bytes"] == budget and m["within_memory_budget"]
    assert len(result.module_result.artifacts["gain_history"]) == n
    assert m["sharp_frames"] + m["motion_blur_frames"] + m["defocus_blur_frames"] == n
    assert sum(m["stage_times_ms"].values()) == pytest.approx(result.execution_time_ms)


def test_exact_budget_boundary_is_inclusive(clip):
    budget = process_video_pipeline(clip, max_keyframes=2).module_result.metrics["estimated_peak_bytes"]
    exact = process_video_pipeline(clip, memory_budget_bytes=budget)
    below = process_video_pipeline(clip, memory_budget_bytes=budget - 1)
    assert len(exact.transformed_frames) == 2 and len(below.transformed_frames) == 1
    assert exact.module_result.metrics["estimated_peak_bytes"] == budget
    assert below.module_result.metrics["estimated_peak_bytes"] <= budget - 1
    assert exact.status == below.status == ExecutionStatus.PARTIAL


def test_download_is_deferred_and_contains_exactly_the_selected_outputs(monkeypatch, clip):
    result = process_video_pipeline(clip, max_keyframes=3, sample_every_n_frames=2)
    buttons = {}

    class Column:
        def download_button(self, label, *, data, **kwargs):
            buttons[kwargs["key"]] = data

    monkeypatch.setattr(export, "render_tab_guidance_header", lambda **_: None)
    monkeypatch.setattr(export.st, "columns", lambda n: [Column() for _ in range(n)])
    export.render_tab_video_export(result, clip.name)
    download = buttons["video_download_zip"]
    assert callable(download)
    with zipfile.ZipFile(io.BytesIO(download())) as archive:
        names = [f"frames/frame_{i:04d}.png" for i in (0, 2, 4)]
        assert archive.namelist() == ["video_telemetry_manifest.json", *names]
        assert json.loads(archive.read("video_telemetry_manifest.json"))["status"] == "partial"
        for name, expected in zip(names, result.transformed_frames, strict=True):
            image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(image, expected)


def test_ui_warns_on_partial_sampling_and_drops_stale_result(monkeypatch, clip):
    from functools import partial
    from ui.tabs_video import main_layout

    monkeypatch.setattr(main_layout, "process_video_pipeline", partial(process_video_pipeline, max_keyframes=1))
    app = AppTest.from_string(
        "import streamlit as st\nfrom ui.tabs_video.main_layout import render_video_app\n"
        "render_video_app(st.session_state.clip)"
    )
    stream = ChunkedUpload(clip.read_bytes())
    app.session_state["clip"] = {"source": stream, "filename": clip.name, "upload_id": "original"}
    app.run()
    assert not stream.requests and not app.exception
    app.button(key="video_run_pipeline").click().run()
    assert not app.exception and any("Partial sampling" in w.value for w in app.warning)
    assert len(app.tabs) == 4
    app.session_state["clip"] = {"bytes": b"corrupt", "filename": clip.name}
    app.run()
    assert not app.tabs and "video_transform_result" not in app.session_state
    app.button(key="video_run_pipeline").click().run()
    assert not app.exception and app.error and not app.tabs
