"""Regression tests for adaptive ensemble anomaly detection."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from modules.anomaly.calibrator import calibrate_ensemble_scores
from modules.anomaly.detector import AnomalyConfig, detect_anomalies
from modules.script_gen.exporter import PipelineScriptGenerator
from utils.ast_sandbox import validate_code


def _outlier_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    normal = rng.normal(0.0, 1.0, size=(200, 4))
    outliers = rng.normal(12.0, 0.5, size=(8, 4))
    return pd.DataFrame(
        np.vstack([normal, outliers]),
        columns=["feat_a", "feat_b", "feat_c", "feat_d"],
    )


def test_calibrator_empty_constant_and_parameter_guards():
    flags, scores, cutoff = calibrate_ensemble_scores(np.empty((0, 3)))
    assert flags.size == scores.size == 0
    assert cutoff == 0.0

    flags, scores, cutoff = calibrate_ensemble_scores(np.ones((12, 3)))
    assert flags.sum() == 0
    assert cutoff > scores.max()

    with pytest.raises(ValueError, match="2-D"):
        calibrate_ensemble_scores(np.ones(5))
    with pytest.raises(ValueError, match="threshold_mode"):
        calibrate_ensemble_scores(np.ones((3, 1)), threshold_mode="unknown")


def test_calibrator_quantile_and_vote_ratio():
    matrix = np.column_stack([np.arange(10), np.arange(10)])
    flags, scores, cutoff = calibrate_ensemble_scores(
        matrix,
        threshold_mode="fixed_quantile",
        contamination=0.1,
        min_vote_ratio=0.66,
    )
    assert flags[-1] == 1
    assert scores[-1] >= cutoff
    assert flags[:-1].sum() == 0


def test_anomaly_config_is_frozen():
    config = AnomalyConfig()
    with pytest.raises(FrozenInstanceError):
        config.contamination = 0.2


@pytest.mark.parametrize(
    "frame, expected_issue",
    [
        (pd.DataFrame({"x": pd.Series(dtype=float)}), "no rows"),
        (pd.DataFrame({"x": [1.0]}), "Single-row"),
        (pd.DataFrame({"label": ["a", "b"]}), "No continuous numeric"),
    ],
)
def test_clean_edge_cases_return_zero_anomalies(frame, expected_issue):
    transform, result = detect_anomalies(frame)
    assert transform.status.value == "success"
    assert result.status.value == "success"
    assert result.metrics["n_anomalies"] == 0
    assert transform.transformed_df["anomaly_flag"].sum() == 0
    assert any(expected_issue.lower() in issue.lower() for issue in result.issues)


def test_constant_data_and_metadata_columns_are_safe():
    frame = pd.DataFrame({
        "value": [5.0] * 30,
        "value_is_missing": [0] * 30,
        "anomaly_old_score": np.linspace(0.0, 1.0, 30),
        "pii_text": ["person@example.com"] * 30,
    })
    transform, result = detect_anomalies(frame)
    assert result.metrics["n_numeric_features"] == 1
    assert result.metrics["n_anomalies"] == 0
    assert transform.transformed_df["anomaly_flag"].sum() == 0


def test_adaptive_dual_criterion_detects_injected_outliers_without_mutation():
    frame = _outlier_frame()
    original = frame.copy(deep=True)
    transform, result = detect_anomalies(frame)
    output = transform.transformed_df

    assert result.status.value == "success"
    assert result.metrics["threshold_mode"] == "adaptive_3sigma"
    assert result.metrics["required_votes"] == 2
    assert result.metrics["n_anomalies"] > 0
    flagged = output["anomaly_flag"].eq(1)
    assert output.loc[flagged, "anomaly_score_ensemble"].ge(
        result.metrics["dynamic_cutoff"] - 1e-6
    ).all()
    assert output.loc[flagged, "anomaly_vote_count"].ge(
        result.metrics["required_votes"]
    ).all()
    consensus = result.metrics["vote_consensus_counts"]
    assert set(consensus) == {
        "zero_models",
        "one_model",
        "two_models",
        "three_models",
    }
    disagreement = result.artifacts["detector_disagreement_matrix"]
    assert disagreement.equals(disagreement.T)
    assert np.diag(disagreement.to_numpy()).sum() == 0
    assert {
        "anomaly_isolation_forest",
        "anomaly_lof",
        "anomaly_one_class_svm",
    } <= set(transform.applied_actions[0].target_columns)
    pd.testing.assert_frame_equal(frame, original)


def test_single_detector_adapts_consensus_and_replay_is_standalone():
    frame = _outlier_frame()
    transform, result = detect_anomalies(
        frame,
        algorithms=["isolation_forest"],
    )
    action = transform.applied_actions[0]

    assert result.metrics["active_model_count"] == 1
    assert result.metrics["required_votes"] == 1
    assert "anomaly_isolation_forest" in transform.transformed_df
    assert "anomaly_lof" not in transform.transformed_df
    assert "OneClassSVM" not in action.code_snippet
    assert "modules.anomaly" not in action.code_snippet
    validate_code(action.code_snippet)

    script = PipelineScriptGenerator().generate([action], validate_snippets=True)
    assert "anomaly_score_ensemble" in script

    replay = frame.copy(deep=True)
    exec(action.code_snippet, {"df": replay, "pd": pd, "np": np})
    columns = [
        "anomaly_flag",
        "anomaly_score_ensemble",
        "anomaly_vote_count",
        "anomaly_isolation_forest",
    ]
    pd.testing.assert_frame_equal(
        replay[columns],
        transform.transformed_df[columns],
    )
