"""Decoupled Video Processing Core for AutoData-Hub."""

from .color_engine_video import (
    DEFAULT_EMA_ALPHA,
    TemporalColorStabilizer,
    apply_temporal_color_balance,
)
from .keyframe_extractor import (
    MAX_MEMORY_BUDGET_BYTES,
    ExtractedKeyframe,
    VideoMetadata,
    extract_keyframes,
    iter_video_frames,
    open_video_source,
    probe_video_metadata,
)
from .pipeline_video import VideoTransformResult, process_video_pipeline
from .scene_detector import (
    DEFAULT_SCENE_CUT_THRESHOLD,
    SceneCut,
    compare_histograms_pearson,
    compute_hsv_histogram,
    detect_scenes,
)
from .temporal_blur import (
    analyze_video_blur,
    classify_temporal_blur,
    compute_farneback_flow_magnitude,
)

__all__ = [
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_SCENE_CUT_THRESHOLD",
    "MAX_MEMORY_BUDGET_BYTES",
    "ExtractedKeyframe",
    "SceneCut",
    "TemporalColorStabilizer",
    "VideoMetadata",
    "VideoTransformResult",
    "analyze_video_blur",
    "apply_temporal_color_balance",
    "classify_temporal_blur",
    "compare_histograms_pearson",
    "compute_farneback_flow_magnitude",
    "compute_hsv_histogram",
    "detect_scenes",
    "extract_keyframes",
    "iter_video_frames",
    "open_video_source",
    "probe_video_metadata",
    "process_video_pipeline",
]
