"""Tests for empirical imputation and downstream performance reports."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer

from modules.evaluation.benchmarker import (
    benchmark_downstream_ml,
    run_imputation_accuracy_experiment,
)


def test_imputation_accuracy_report_contains_all_methods_and_metrics():
    rng = np.random.default_rng(4)
    complete = pd.DataFrame({
        "x": rng.normal(5.0, 1.0, 120),
        "y": rng.normal(10.0, 2.0, 120),
    })
    complete["segment"] = np.where(complete["x"] > 5.0, "high", "low")

    report = run_imputation_accuracy_experiment(
        complete,
        missing_rates=(0.10, 0.20),
        n_splits=3,
        random_state=5,
    )

    assert len(report) == 8
    assert set(report["Method"]) == {
        "SimpleImputer",
        "KNNImputer",
        "MICE",
        "CategoricalClassifierImputer",
    }
    assert report["CV-RMSE"].notna().all()
    assert report["MAE"].notna().all()
    assert report["Categorical Macro-F1"].between(0.0, 1.0).all()


def test_downstream_ml_report_has_valid_deltas():
    features, target = make_classification(
        n_samples=180,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=8,
    )
    raw = pd.DataFrame(features, columns=[f"x{i}" for i in range(5)])
    raw["target"] = target
    rng = np.random.default_rng(8)
    raw.loc[rng.random(len(raw)) < 0.15, "x0"] = np.nan

    cleaned = raw.copy(deep=True)
    feature_columns = [f"x{i}" for i in range(5)]
    cleaned[feature_columns] = SimpleImputer(strategy="median").fit_transform(
        cleaned[feature_columns]
    )

    report = benchmark_downstream_ml(
        raw,
        cleaned,
        "target",
        n_splits=3,
        random_state=8,
    )

    for version in ("v0_raw", "v3_anomaly"):
        assert 0.0 <= report[version]["roc_auc"] <= 1.0
        assert 0.0 <= report[version]["f1"] <= 1.0
    assert set(report["delta"]) == {"roc_auc", "f1"}
