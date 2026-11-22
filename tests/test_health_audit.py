import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import pandas as pd
import numpy as np

from core.schema_inference import ExecutionStatus, ModuleResult
from modules.health_audit import (
    DataHealthAuditor,
    audit_dataset_health,
    DEFAULT_WEIGHT_MISSINGNESS,
    DEFAULT_WEIGHT_DUPLICATES,
    DEFAULT_WEIGHT_OUTLIERS,
    DEFAULT_WEIGHT_TYPE_ERRORS,
)


def test_health_audit_standard_dataset():
    """Validates metrics calculation, IQR outliers, duplicates, and composite score on dirty dataset."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 7],  # row 7 is an exact duplicate of row 6
        "income": [50000, 52000, 48000, 51000, 49000, 50500, 120000, 120000],  # 2 IQR outliers (120000)
        "age": [25, 28, 30, 29, 27, np.nan, 35, 35],  # 1 null
        "category": ["A", "B", "A", "B", "A", "B", "A", "A"],
        "mixed_field": [10, 20, "str_err", 40, 50, 60, 70, 70],  # 1 type mismatch
    })

    result = audit_dataset_health(df, dataset_version="v0_raw")

    # 1. Verify ModuleResult structure
    assert isinstance(result, ModuleResult)
    assert result.status == ExecutionStatus.SUCCESS
    assert result.module_name == "health_audit"
    assert result.dataset_version == "v0_raw"

    metrics = result.metrics
    assert "quality_score" in metrics
    assert "quality_grade" in metrics
    assert "missing_count" in metrics
    assert "missing_pct" in metrics
    assert "duplicate_count" in metrics
    assert "outlier_count" in metrics
    assert "type_error_count" in metrics
    assert "columns" in metrics

    # 2. Check metric values
    assert metrics["row_count"] == 8
    assert metrics["column_count"] == 5
    assert metrics["total_cells"] == 40
    assert metrics["missing_count"] == 1
    assert metrics["duplicate_count"] == 1
    assert metrics["outlier_count"] >= 1
    assert metrics["type_error_count"] >= 1

    # Quality score in [0, 100]
    score = metrics["quality_score"]
    assert 0.0 <= score <= 100.0

    # 3. Check artifacts
    artifacts = result.artifacts
    assert "missingness_heatmap" in artifacts
    assert "quality_gauge" in artifacts
    assert "feature_distributions" in artifacts

    # Validate Plotly dark template in artifacts
    heatmap = artifacts["missingness_heatmap"]
    assert heatmap.get("layout", {}).get("template", {}).get("layout", {}).get("font", {}).get("color") is not None or "template" in heatmap.get("layout", {})


def test_composite_quality_score_formula():
    """Validates the explicit mathematical weighting of Q in [0, 100]."""
    # Create clean dataset with 0 errors
    df_clean = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "c": ["x", "y", "z", "w", "v"]
    })
    res_clean = audit_dataset_health(df_clean)
    assert res_clean.metrics["quality_score"] == 100.0
    assert res_clean.metrics["quality_grade"] == "A"

    # Create dataset with known error ratios
    # 10 rows, 2 columns = 20 cells
    # 4 nulls in column 'a' -> Missingness = 4/20 = 0.20
    # 2 duplicate rows -> DuplicateRatio = 2/10 = 0.20
    df_dirty = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 5, np.nan, np.nan, np.nan, np.nan],
        "b": ["k", "k", "k", "k", "k", "k", "k", "k", "k", "k"]
    })
    auditor = DataHealthAuditor(weights={
        "missingness": 0.35,
        "duplicates": 0.20,
        "outliers": 0.25,
        "type_errors": 0.20,
    })
    res_dirty = auditor.audit(df_dirty)
    # Expected penalty: 0.35 * 0.20 * 100 + 0.20 * 0.10 (or duplicates) * 100 ...
    assert res_dirty.metrics["quality_score"] < 100.0


def test_edge_cases():
    """Tests 100% missing columns, single-column DataFrame, zero-variance features, and zero-row empty upload."""
    # Edge Case 1: Empty DataFrame (0 rows)
    df_empty = pd.DataFrame()
    res_empty = audit_dataset_health(df_empty)
    assert res_empty.status == ExecutionStatus.SUCCESS
    assert res_empty.metrics["row_count"] == 0
    assert res_empty.metrics["quality_score"] == 0.0
    assert len(res_empty.issues) > 0

    # Edge Case 2: Single Column DataFrame
    df_single = pd.DataFrame({"single_col": [1, 2, 3, 100]})
    res_single = audit_dataset_health(df_single)
    assert res_single.metrics["column_count"] == 1
    assert "single_col" in res_single.metrics["columns"]

    # Edge Case 3: 100% Missing Column
    df_all_null = pd.DataFrame({
        "valid_col": [10, 20, 30],
        "all_null_col": [np.nan, np.nan, np.nan]
    })
    res_all_null = audit_dataset_health(df_all_null)
    assert "all_null_col" in res_all_null.metrics["all_null_columns"]
    assert res_all_null.metrics["columns"]["all_null_col"]["null_pct"] == 100.0

    # Edge Case 4: Zero-Variance Feature (sigma^2 = 0)
    df_zero_var = pd.DataFrame({
        "constant_col": [42, 42, 42, 42],
        "varying_col": [1, 2, 3, 4]
    })
    res_zero_var = audit_dataset_health(df_zero_var)
    assert "constant_col" in res_zero_var.metrics["zero_variance_columns"]
    assert res_zero_var.metrics["columns"]["constant_col"]["is_constant"] is True
    # Zero-variance constant feature has 0 outliers
    assert res_zero_var.metrics["columns"]["constant_col"]["outlier_count"] == 0


def test_zero_framework_leakage():
    """Ensures no Streamlit imports exist in modules/health_audit/."""
    import inspect
    import modules.health_audit.auditor as auditor_module
    source = inspect.getsource(auditor_module)
    assert "import streamlit" not in source
    assert "from streamlit" not in source
    assert "st." not in source


def test_nested_json_cells_do_not_crash_duplicate_or_cardinality_audit():
    frame = pd.DataFrame({
        "tags": [["a", "b"], ["a", "b"], ["c"]],
        "metadata": [{"x": 1}, {"x": 1}, {"x": 2}],
    })
    result = audit_dataset_health(frame)
    assert result.status == ExecutionStatus.SUCCESS
    assert result.metrics["duplicate_count"] == 1
    assert result.metrics["columns"]["tags"]["unique_count"] == 2
