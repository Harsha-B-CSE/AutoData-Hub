import pytest
import pandas as pd
import numpy as np
from backend.engines.quality.scorer import compute_tabular_quality_score, compute_image_quality_score
from backend.engines.quality.comparator import compare_tabular_datasets
from backend.engines.quality.recommender import generate_tabular_recommendations

def test_tabular_quality_score():
    # Clean complete dataset
    df_clean = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    score_clean = compute_tabular_quality_score(df_clean)
    assert score_clean["composite_score"] >= 95.0
    assert score_clean["completeness"] == 100.0

    # Dirty dataset with nulls and duplicates
    df_dirty = pd.DataFrame({
        "a": [1, 1, np.nan, np.nan, 5],
        "b": [10.0, 10.0, np.nan, 40.0, 50000.0],
    })
    score_dirty = compute_tabular_quality_score(df_dirty)
    assert score_dirty["composite_score"] < score_clean["composite_score"]
    assert score_dirty["missing_cells"] == 3
    assert score_dirty["duplicate_rows"] == 1

def test_tabular_comparator():
    df_before = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [10, 20, 30, 40]
    })
    df_after = pd.DataFrame({
        "a": [1, 2, 2.5, 4],
        "b": [10, 20, 30, 40]
    })
    comp = compare_tabular_datasets(df_before, df_after)
    assert comp["missingness"]["resolved_count"] == 1
    assert comp["missingness"]["resolution_percent"] == 100.0

def test_tabular_recommendations():
    profile = {
        "total_rows": 100,
        "duplicate_rows": 5,
        "columns": {
            "income": {
                "null_percent": 18.4,
                "null_count": 18,
                "is_numeric": True,
                "skewness": 3.2,
                "iqr_outliers": 8,
            }
        }
    }
    recs = generate_tabular_recommendations(profile)
    rec_actions = [r["action"] for r in recs]
    assert "remove_duplicates" in rec_actions
    assert "impute" in rec_actions
    # Verify diagnosis message contains skewness information
    income_rec = next(r for r in recs if r["target"] == "income" and r["action"] == "impute")
    assert "skewness" in income_rec["diagnosis"].lower()
    assert income_rec["params"]["strategy"] == "median"

def test_quality_score_contributions_and_formula():
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0],
        "cat": ["A", "B", "C", "D"],
    })
    score = compute_tabular_quality_score(df)
    assert "contributions" in score
    c = score["contributions"]
    assert "completeness" in c
    assert "validity" in c
    assert "uniqueness" in c
    assert "snr" in c

    # Individual contributions sum to composite score
    total_contrib = round(c["completeness"] + c["validity"] + c["uniqueness"] + c["snr"], 2)
    assert total_contrib == score["composite_score"]
    assert "formula" in score
    assert "formula_display" in score
    assert "35%" in score["formula_display"]
    assert "25%" in score["formula_display"]

def test_quality_score_actual_dataset_metrics():
    # Dataset with actual invalid elements: 1 infinity, 1 control char
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.inf, 4.0],
        "text": ["valid", "clean", "bad\x00data", "ok"]
    })
    score = compute_tabular_quality_score(df)
    # Total cells = 8, invalid cells = 2 (1 inf, 1 control char \x00)
    assert score["invalid_cells"] == 2
    # Validity is 1.0 - (2 / 8) = 0.75 -> 75%
    assert score["validity"] == 75.0
    # Validity contribution is 25.0 * 0.75 = 18.75
    assert score["contributions"]["validity"] == 18.75

def test_recommendations_match_selected_configuration():
    profile = {
        "total_rows": 100,
        "duplicate_rows": 10,
        "numeric_columns": ["salary"],
        "categorical_columns": ["dept"],
        "columns": {
            "salary": {
                "null_percent": 15.0,
                "null_count": 15,
                "is_numeric": True,
                "median": 50000.0,
                "mean": 55000.0,
                "iqr_outliers": 4,
            },
            "dept": {
                "null_percent": 0.0,
                "null_count": 0,
                "is_numeric": False,
            }
        }
    }
    # Configure: deduplication disabled, mean imputation, outlier clipping enabled
    config = {
        "cleaning": {
            "remove_duplicates": False,
            "handle_outliers": True,
            "outlier_method": "iqr",
            "outlier_action": "clip",
            "outlier_threshold": 1.5
        },
        "imputation": {
            "numerical_strategy": "mean",
            "categorical_strategy": "mode"
        },
        "scaling": {
            "method": "robust"
        },
        "encoding": {
            "method": "onehot",
            "columns": ["dept"],
            "exclude_columns": []
        }
    }

    recs = generate_tabular_recommendations(profile, config=config)

    # 1. Dedup recommendation reflects that it was disabled
    dedup_rec = next(r for r in recs if r["category"] == "deduplication")
    assert dedup_rec["approved"] is False
    assert "disabled" in dedup_rec["recommendation"].lower()

    # 2. Imputation matches configured strategy (mean, not median)
    sal_rec = next(r for r in recs if r["target"] == "salary" and r["category"] == "imputation")
    assert sal_rec["params"]["strategy"] == "mean"
    assert "mean" in sal_rec["recommendation"].lower()

    # 3. Outlier handling matches configured clipping
    out_rec = next(r for r in recs if r["category"] == "outliers")
    assert out_rec["approved"] is True
    assert "clip" in out_rec["recommendation"].lower()

    # 4. Scaling matches configured RobustScaler
    scale_rec = next(r for r in recs if r["category"] == "scaling")
    assert "robust" in scale_rec["recommendation"].lower()

    # 5. Encoding matches configured One-Hot on dept
    enc_rec = next(r for r in recs if r["category"] == "encoding")
    assert "dept" in enc_rec["recommendation"]
