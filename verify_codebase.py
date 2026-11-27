"""Automated peer-review codebase verification script for AutoData-Hub."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_directory_integrity() -> None:
    required_dirs = [
        "core",
        "modules/tabular",
        "modules/health_audit",
        "modules/privacy",
        "modules/imputation",
        "modules/anomaly",
        "modules/script_gen",
        "modules/image",
        "modules/voice",
        "modules/video",
        "ui/tabs",
        "ui/tabs_image",
        "ui/tabs_voice",
        "ui/tabs_video",
        "backend",
        "tests",
    ]
    for rel in required_dirs:
        target = ROOT / rel
        assert target.is_dir(), f"Missing required directory: {rel}"

    for rel_file in [
        "app.py",
        "ui/utils_ux.py",
        "modules/image/gpu_manager.py",
        "modules/image/color_engine.py",
        "modules/video/keyframe_extractor.py",
        "modules/video/scene_detector.py",
        "modules/video/temporal_blur.py",
        "modules/video/color_engine_video.py",
        "modules/video/pipeline_video.py",
    ]:
        assert (ROOT / rel_file).is_file(), f"Missing required file: {rel_file}"


def _check_math_and_baselines() -> None:
    from modules.anomaly.calibrator import calibrate_ensemble_scores
    from modules.health_audit.auditor import EPSILON, audit_dataset_health
    from modules.image.color_engine import (
        COLOR_GRADING_PRESETS,
        apply_auto_white_balance,
        apply_color_grading,
    )
    from modules.image.neural_restoration import MODEL_REGISTRY
    from modules.video.scene_detector import (
        DEFAULT_SCENE_CUT_THRESHOLD,
        detect_scenes,
    )
    from modules.voice.audio_metrics import compute_sisdr

    # 1. CleanML Q-score epsilon & empty dataframe safety
    assert EPSILON == 1e-9
    empty_res = audit_dataset_health(pd.DataFrame())
    zero_col_res = audit_dataset_health(pd.DataFrame(index=pd.RangeIndex(3)))
    assert np.isfinite(empty_res.metrics["quality_score"])
    assert np.isfinite(zero_col_res.metrics["quality_score"])

    # 2. Quantile / consensus anomaly calibration (V >= 2/3)
    scores = np.column_stack([np.linspace(0, 1, 20), np.linspace(0, 1, 20), np.linspace(0, 1, 20)])
    flags, probs, _ = calibrate_ensemble_scores(
        scores, threshold_mode="fixed_quantile", contamination=0.1, min_vote_ratio=0.66
    )
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert flags[-1] == 1 and flags[0] == 0

    # 3. SI-SDR zero-mean & epsilon floor
    indices = np.arange(1600, dtype=np.float32)
    tone = np.sin(2.0 * np.pi * 440.0 * indices / 16000.0).astype(np.float32)
    assert compute_sisdr(tone, tone + 0.25) > 100.0
    assert compute_sisdr(np.zeros(1600, dtype=np.float32), np.zeros(1600, dtype=np.float32)) == 0.0

    # 4. HSV histogram correlation threshold < 0.65
    assert DEFAULT_SCENE_CUT_THRESHOLD == 0.65
    f_red = np.full((32, 32, 3), (10, 20, 220), dtype=np.uint8)
    f_blue = np.full((32, 32, 3), (220, 20, 10), dtype=np.uint8)
    scene_res = detect_scenes([f_red, f_red, f_blue])
    assert scene_res["scene_cut_indices"] == [2]

    # 5. Vision registry & Color engine presets
    for expected in ("NAFNet", "Restormer", "Real-ESRGAN", "DarkIR"):
        assert any(expected in key for key in MODEL_REGISTRY), f"Missing {expected} in MODEL_REGISTRY"
    assert set(COLOR_GRADING_PRESETS) == {
        "Custom",
        "🎬 Teal & Orange",
        "📻 Vintage Film",
        "❄️ Cold Cyberpunk",
    }
    balanced = apply_auto_white_balance(f_red, method="gray_world")
    graded = apply_color_grading(balanced, preset="🎬 Teal & Orange")
    assert graded.shape == f_red.shape


def _check_hardware_and_ux_headers() -> None:
    import modules.image.gpu_manager as gpu_manager
    import ui.utils_ux as utils_ux

    gpu_src = inspect.getsource(gpu_manager)
    assert "@st.cache_resource" in gpu_src
    assert "get_cached_hardware_context" in gpu_src
    assert "torch.cuda.is_available()" in gpu_src
    assert "get_available_providers()" in gpu_src
    assert "torch.set_num_threads" in gpu_src
    assert "torch.cuda.amp.autocast" in gpu_src

    assert callable(utils_ux.render_tab_guidance_header)

    tab_files = [
        "ui/tabs/tab_audit.py",
        "ui/tabs/tab_privacy.py",
        "ui/tabs/tab_impute.py",
        "ui/tabs/tab_anomaly.py",
        "ui/tabs/tab_export.py",
        "ui/tabs_image/tab_sharpness.py",
        "ui/tabs_image/tab_restoration.py",
        "ui/tabs_image/tab_color_grading.py",
        "ui/tabs_image/tab_export.py",
        "ui/tabs_voice/tab_telemetry.py",
        "ui/tabs_voice/tab_neural_enhancer.py",
        "ui/tabs_voice/tab_custom_tuner.py",
        "ui/tabs_voice/tab_voice_export.py",
        "ui/tabs_voice/tab_live_webrtc.py",
        "ui/tabs_video/tab_video_overview.py",
        "ui/tabs_video/tab_video_quality.py",
        "ui/tabs_video/tab_video_color.py",
        "ui/tabs_video/tab_video_export.py",
    ]
    for rel in tab_files:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "render_tab_guidance_header(" in text, f"UX header missing in {rel}"


def main() -> None:
    _check_directory_integrity()
    _check_math_and_baselines()
    _check_hardware_and_ux_headers()
    print("ALL VERIFICATION CHECKS PASSED (architecture, math baselines, GPU caching, UX headers).")


if __name__ == "__main__":
    main()
