"""Cross-modal degeneracy and boundary regression coverage."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from core.ingestion_engine import detect_encoding, read_delimited
from core.schema_inference import ExecutionStatus
from modules.health_audit.auditor import audit_dataset_health
from modules.image.sharpness_evaluator import (
    evaluate_sharpness_laplacian,
    evaluate_tenengrad,
)
from modules.voice.audio_metrics import compute_sisdr, evaluate_audio_telemetry
from modules.voice.booster import (
    apply_quality_booster,
    calculate_true_peak_dbfs,
)
from ui.tabs_image.common import decode_image_bytes


def test_non_utf8_cp1252_delimited_ingestion_preserves_text():
    payload = (
        "name;note\n"
        "André;“café”\n"
        "Zoë;€uro pricing\n"
    ).encode("cp1252")
    stream = io.BytesIO(payload)

    encoding = detect_encoding(stream)
    frame = read_delimited(stream)

    assert encoding.lower().replace("_", "-") in {
        "cp1252",
        "windows-1252",
    }
    assert frame.shape == (2, 2)
    assert frame.loc[0, "name"] == "André"
    assert frame.loc[0, "note"] == "“café”"
    assert frame.loc[1, "note"] == "€uro pricing"


def test_zero_row_and_zero_column_dataframes_return_finite_audits():
    zero_rows = pd.DataFrame(
        {
            "value": pd.Series(dtype="float64"),
            "label": pd.Series(dtype="string"),
        }
    )
    zero_columns = pd.DataFrame(index=pd.RangeIndex(3))

    rows_result = audit_dataset_health(zero_rows, dataset_version="empty_rows")
    columns_result = audit_dataset_health(
        zero_columns, dataset_version="empty_columns"
    )

    assert rows_result.status == ExecutionStatus.SUCCESS
    assert rows_result.metrics["row_count"] == 0
    assert rows_result.metrics["column_count"] == 2
    assert columns_result.status == ExecutionStatus.SUCCESS
    assert columns_result.metrics["row_count"] == 3
    assert columns_result.metrics["column_count"] == 0
    for result in (rows_result, columns_result):
        assert result.metrics["total_cells"] == 0
        assert result.metrics["quality_score"] == 0.0
        for metric in (
            "missing_ratio",
            "duplicate_ratio",
            "outlier_ratio",
            "type_error_ratio",
        ):
            assert np.isfinite(result.metrics[metric])
            assert result.metrics[metric] == 0.0


def test_constant_columns_have_zero_variance_without_false_outliers():
    frame = pd.DataFrame(
        {
            "constant_numeric": np.full(8, 42.0),
            "constant_text": ["stable"] * 8,
            "varying_numeric": np.arange(8, dtype=np.float64),
        }
    )

    result = audit_dataset_health(frame, dataset_version="constant_columns")

    assert result.status == ExecutionStatus.SUCCESS
    assert "constant_numeric" in result.metrics["zero_variance_columns"]
    numeric = result.metrics["columns"]["constant_numeric"]
    text = result.metrics["columns"]["constant_text"]
    assert numeric["is_constant"] is True
    assert numeric["outlier_count"] == 0
    assert text["is_constant"] is True
    assert np.isfinite(result.metrics["quality_score"])


def test_empty_and_corrupted_image_payloads_fail_deterministically():
    with pytest.raises(ValueError, match="empty"):
        decode_image_bytes(b"")
    with pytest.raises(ValueError, match="could not decode"):
        decode_image_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_sharpness_laplacian(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_tenengrad(np.empty((0, 0, 3), dtype=np.uint8))


def test_uniform_flat_canvas_has_finite_zero_sharpness():
    canvas = np.full((64, 64, 3), 127, dtype=np.uint8)

    metrics = evaluate_sharpness_laplacian(canvas, threshold=100.0)

    assert metrics["laplacian_variance"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["tenengrad_score"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["mean_luminance"] == pytest.approx(127.0)
    assert metrics["is_blurry"] is True
    assert metrics["status"] == "BLURRY"


def test_silent_audio_metrics_and_booster_are_finite():
    silent = np.zeros(16_000, dtype=np.float32)

    telemetry = evaluate_audio_telemetry(silent, sample_rate=16_000)
    boosted = apply_quality_booster(silent, sr=16_000, boost_db=12.0)

    assert compute_sisdr(silent, silent) == pytest.approx(0.0, abs=1e-12)
    for metric in (
        "snr_db",
        "si_sdr_db",
        "pesq_score",
        "stoi_index",
        "clipping_ratio_pct",
        "true_peak_dbfs",
        "rms_dbfs",
        "lufs_est",
        "crest_factor_db",
    ):
        assert np.isfinite(telemetry[metric]), metric
    assert telemetry["clipping_ratio"] == 0.0
    assert telemetry["true_peak_dbfs"] == -120.0
    assert np.array_equal(boosted, silent)

    # SI-SDR must be invariant to a constant DC offset after zero-mean centering.
    indices = np.arange(16_000, dtype=np.float32)
    tone = np.sin(2.0 * np.pi * 440.0 * indices / 16_000).astype(np.float32)
    assert compute_sisdr(tone, tone + 0.35) > 100.0


def test_clipped_square_wave_is_soft_saturated_below_true_peak_ceiling():
    samples = np.arange(16_000)
    clipped_square = np.where((samples // 20) % 2 == 0, 1.0, -1.0).astype(
        np.float32
    )

    telemetry = evaluate_audio_telemetry(clipped_square, sample_rate=16_000)
    boosted = apply_quality_booster(
        clipped_square,
        sr=16_000,
        boost_db=12.0,
        target_peak_dbfs=-0.5,
    )

    assert telemetry["clipping_ratio"] == pytest.approx(1.0)
    assert np.all(np.isfinite(boosted))
    assert not np.array_equal(boosted, clipped_square)
    assert float(np.max(np.abs(boosted))) < 1.0
    assert calculate_true_peak_dbfs(boosted, 16_000) <= -0.49
