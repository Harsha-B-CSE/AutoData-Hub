"""Streamed memory-bounded video frame and keyframe extraction."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterator, Tuple

import cv2
import numpy as np


# Worker buffer allowance, not an RSS limit. Leave headroom for imports,
# allocators, and native codec state inside a roughly 500 MB worker.
MAX_MEMORY_BUDGET_BYTES = 224 * 1024 * 1024
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_CHUNK_BYTES = 256 * 1024
_FIXED_WORKSPACE = 1024 * 1024
_MAX_DECODE_PIXELS = 4096 * 2160


@dataclass(frozen=True)
class VideoMetadata:
    """Stream metadata probed from a video container."""

    source_name: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


@dataclass
class ExtractedKeyframe:
    """Single decoded video frame with stream index and timestamp."""

    frame_index: int
    timestamp_sec: float
    image_bgr: np.ndarray


def _positive_int(value: int, name: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@contextmanager
def open_video_source(
    video_source: Any, *, memory_budget_bytes: int = MAX_MEMORY_BUDGET_BYTES,
) -> Iterator[Tuple[Path, str]]:
    """Stage without cloning uploads; rewind and restore seekable streams."""
    budget = min(_positive_int(memory_budget_bytes, "memory_budget_bytes"), MAX_MEMORY_BUDGET_BYTES)
    if isinstance(video_source, (str, os.PathLike)):
        path = Path(video_source)
        if not path.is_file():
            raise ValueError(f"Video file does not exist: {path}")
        if not path.stat().st_size:
            raise ValueError("Video payload is empty.")
        yield path, path.name
        return

    if isinstance(video_source, dict):
        payload = video_source.get("source", video_source.get("bytes", b""))
        name = str(video_source.get("filename") or video_source.get("name") or "video.mp4")
    else:
        payload = video_source
        name = str(getattr(payload, "name", "video.mp4"))

    # Charge uploads as resident even when a supplied stream is disk-backed.
    # The chunk allowance is separate from the compressed payload itself.
    chunk_size = min(_CHUNK_BYTES, max(1, budget // 8))
    is_buffer = isinstance(payload, (bytes, bytearray, memoryview))
    limit = min(MAX_UPLOAD_BYTES, budget if is_buffer else budget - chunk_size)
    oversize = f"Video upload exceeds the {limit}-byte staging limit for this budget."
    suffix = Path(name).suffix
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ("upload" + (suffix if 0 < len(suffix) <= 10 else ".mp4"))
        with path.open("wb") as target:
            if is_buffer:
                with memoryview(payload) as view:
                    if view.nbytes > limit:
                        raise ValueError(oversize)
                    if not view.c_contiguous:
                        raise ValueError("Video payload must be a contiguous byte buffer.")
                    target.write(view.cast("B"))
            elif hasattr(payload, "read"):
                pos = None
                try:
                    if hasattr(payload, "seek") and hasattr(payload, "tell"):
                        try:
                            pos = payload.tell()
                            size = payload.seek(0, os.SEEK_END)
                        except OSError:  # pipes and other non-seekable uploads
                            pos = None
                        else:
                            if size is not None and size > limit:
                                raise ValueError(oversize)
                            payload.seek(0)
                    total = 0
                    while True:
                        request = min(chunk_size, limit - total + 1)
                        chunk = payload.read(request)
                        if not isinstance(chunk, (bytes, bytearray, memoryview)):
                            raise TypeError("Video stream must return binary data.")
                        count = memoryview(chunk).nbytes
                        if count > request or total + count > limit:
                            raise ValueError(oversize)
                        if not count:
                            break
                        target.write(chunk)
                        total += count
                        del chunk
                finally:
                    if pos is not None:
                        payload.seek(pos)
            else:
                raise TypeError("Expected a video path, byte buffer, or binary readable stream.")
            if not target.tell():
                raise ValueError("Video payload is empty.")
        yield path, name


@contextmanager
def _open_capture(path: Path, name: str) -> Iterator[Tuple[cv2.VideoCapture, VideoMetadata]]:
    # One decoder thread per capture, not a process-wide OpenCV thread setting.
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream with FFmpeg: {name}")
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        if not (
            0 < width <= 4096 and 0 < height <= 4096
            and width == int(width) and height == int(height)
            and width * height <= _MAX_DECODE_PIXELS
        ):
            raise ValueError(
                f"Invalid or oversized source dimensions: {width} x {height} "
                f"(maximum {_MAX_DECODE_PIXELS} pixels; longest edge 4096)."
            )
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        frame_count = int(count) if np.isfinite(count) and count > 0 else 0
        yield cap, VideoMetadata(name, fps, frame_count, int(width), int(height), frame_count / fps)
    finally:
        cap.release()


def probe_video_metadata(video_source: Any) -> VideoMetadata:
    """Inspect video dimensions, FPS, and frame count without buffering frames."""
    with open_video_source(video_source) as (path, name):
        with _open_capture(path, name) as (_, metadata):
            return metadata


def _sampling_plan(
    metadata: VideoMetadata, max_dimension: int | None, budget: int,
    source_bytes: int, reserve_processing: bool,
) -> tuple[tuple[int, int], int, list[tuple[int, int]]]:
    width, height = metadata.width, metadata.height
    if max_dimension is not None:
        edge = _positive_int(max_dimension, "max_dimension")
        if edge < max(width, height):
            scale = edge / max(width, height)
            width, height = max(1, round(width * scale)), max(1, round(height * scale))
    pixels = width * height
    base = source_bytes + _FIXED_WORKSPACE
    # Each phase is (fixed workspace, bytes per retained sample). Decode and
    # analysis do not overlap. Allow for native reference/conversion buffers
    # at SOURCE resolution, even when the result is resized to a tiny preview.
    phases = [(base + 24 * metadata.width * metadata.height, 3 * pixels + 512)]
    if reserve_processing:
        # Grading has several live RGB float32 planes; sharpness/flow have
        # Sobel/flow pyramids. Reserve scratch plus original AND output and
        # Python diagnostics per sample. No change to the analysis algorithms.
        phases.append((base + 128 * pixels, 6 * pixels + 4096))
        # An on-demand Streamlit ZIP is spooled to disk, then returned as one
        # buffer alongside the frame pairs. Allow for incompressible PNGs.
        phases.append((base, 10 * pixels + 8192))
    capacity = min((budget - fixed) // per_sample for fixed, per_sample in phases)
    if capacity < 1:
        raise ValueError(
            f"Working memory budget ({budget} bytes) cannot fit one source "
            f"{metadata.width}x{metadata.height} / analysis {width}x{height} frame. "
            "Reduce the source or analysis resolution, or use a path instead of a resident upload."
        )
    return (width, height), capacity, phases


def _sample_frames(cap, metadata, size, stride, limit, progress) -> Iterator[ExtractedKeyframe]:
    sampled = 0
    while cap.grab():
        idx = progress["frames_scanned"]
        progress["frames_scanned"] += 1
        if (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != (
            metadata.width, metadata.height,
        ):
            raise ValueError("Video dimensions changed during decoding; refusing an unbudgeted frame.")
        if idx % stride:
            continue
        if sampled == limit:  # Look ahead without allocating another BGR frame.
            progress["sampling_stop_reason"] = "sample_limit"
            return
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            progress["sampling_stop_reason"] = "decode_error"
            return
        if frame.shape != (metadata.height, metadata.width, 3) or frame.dtype != np.uint8:
            raise ValueError("Decoded frame layout differs from the budgeted 8-bit BGR dimensions.")
        if size != (metadata.width, metadata.height):
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        yield ExtractedKeyframe(idx, idx / metadata.fps, frame)
        del frame
        sampled += 1
    if progress["frames_scanned"] < metadata.frame_count:
        progress["sampling_stop_reason"] = "early_end_of_stream"


def iter_video_frames(
    video_path: str | os.PathLike[str],
    *,
    stride: int = 1,
    max_frames: int | None = None,
    max_dimension: int | None = None,
) -> Iterator[ExtractedKeyframe]:
    """Yield one sampled frame at a time; frames retained by the caller are external."""
    stride = _positive_int(stride, "stride")
    limit = None if max_frames is None else _positive_int(max_frames, "max_frames")
    path = Path(video_path)
    with _open_capture(path, path.name) as (cap, metadata):
        size, _, _ = _sampling_plan(metadata, max_dimension, MAX_MEMORY_BUDGET_BYTES, 0, False)
        progress = {"frames_scanned": 0, "sampling_stop_reason": "end_of_stream"}
        for count, item in enumerate(_sample_frames(cap, metadata, size, stride, None, progress), 1):
            yield item
            if count == limit:
                return
        if progress["sampling_stop_reason"] in ("decode_error", "early_end_of_stream"):
            raise ValueError(f"Video stream ended unexpectedly after {progress['frames_scanned']} frames.")


def extract_keyframes(
    video_source: Any,
    *,
    sample_every_n_frames: int = 1,
    max_keyframes: int = 120,
    max_dimension: int | None = 1080,
    memory_budget_bytes: int = MAX_MEMORY_BUDGET_BYTES,
    reserve_processing: bool = False,
) -> Dict[str, Any]:
    """Extract sampled BGR frames, optionally reserving the downstream pipeline's workspace."""
    budget = min(_positive_int(memory_budget_bytes, "memory_budget_bytes"), MAX_MEMORY_BUDGET_BYTES)
    stride = _positive_int(sample_every_n_frames, "sample_every_n_frames")
    requested = _positive_int(max_keyframes, "max_keyframes")
    if budget <= _FIXED_WORKSPACE:
        raise ValueError("Working memory budget is too small for the decoder workspace.")
    with open_video_source(video_source, memory_budget_bytes=budget) as (path, name):
        source_bytes = 0 if isinstance(video_source, (str, os.PathLike)) else path.stat().st_size
        if source_bytes + _FIXED_WORKSPACE >= budget:
            raise ValueError("Video upload leaves too little working memory budget for the decoder.")
        with _open_capture(path, name) as (cap, metadata):
            size, capacity, phases = _sampling_plan(metadata, max_dimension, budget, source_bytes, reserve_processing)
            progress = {"frames_scanned": 0, "sampling_stop_reason": "end_of_stream"}
            keyframes = list(_sample_frames(cap, metadata, size, stride, min(capacity, requested), progress))

    if not keyframes:
        raise ValueError("No readable frames found in the video stream.")
    if progress["sampling_stop_reason"] == "sample_limit":
        progress["sampling_stop_reason"] = "memory_budget" if capacity < requested else "max_keyframes"
    retained = sum(item.image_bgr.nbytes for item in keyframes)
    estimated_peak = max(fixed + len(keyframes) * per_sample for fixed, per_sample in phases)
    return {
        **progress,
        "source_name": name,
        "fps": metadata.fps,
        "reported_frame_count": metadata.frame_count,
        "decoded_span_frames": keyframes[-1].frame_index + 1,
        "width": metadata.width,
        "height": metadata.height,
        "analysis_width": size[0],
        "analysis_height": size[1],
        "keyframes": keyframes,
        "keyframe_count": len(keyframes),
        "memory_bytes": retained,
        "memory_mb": round(retained / (1024 * 1024), 3),
        "source_buffer_bytes": source_bytes,
        "estimated_peak_bytes": estimated_peak,
        "memory_budget_bytes": budget,
        "within_memory_budget": estimated_peak <= budget,
    }
