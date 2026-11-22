import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import pandas as pd
import numpy as np

from core.schema_inference import ExecutionStatus, TransformResult, TransformationAction
from modules.imputation import (
    CategoricalClassifierImputer,
    ImputationEngine,
    ImputationStrategy,
    MissingValueImputer,
    impute_missing_values,
)
from modules.imputation.engine import MissingValueImputer as EngineMissingValueImputer
from utils.ast_sandbox import validate_code


def _numeric_missing_frame():
    return pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "x2": [2.0, 4.0, 6.0, 8.0, np.nan, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
    })


def test_mixed_type_imputation_knn():
    """Validates mixed-type imputation using KNNImputer with categorical encoding and decoding."""
    df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "num2": [10.0, 20.0, 30.0, np.nan, 50.0, 60.0],
        "cat1": ["Low", "High", "Low", "Medium", np.nan, "High"],
        "label": [1, 1, 0, 0, 1, 0],
    })

    result = impute_missing_values(
        df=df,
        strategy=ImputationStrategy.KNN,
        add_mnar_indicators=False,
    )

    assert isinstance(result, TransformResult)
    assert result.status == ExecutionStatus.SUCCESS
    assert result.error_message is None

    transformed = result.transformed_df
    assert transformed is not None
    # No NaNs remaining in imputed columns
    assert transformed["num1"].isna().sum() == 0
    assert transformed["num2"].isna().sum() == 0
    assert transformed["cat1"].isna().sum() == 0

    # Categorical preserved as string domain
    assert set(transformed["cat1"].unique()).issubset({"Low", "Medium", "High"})

    # Immutability check
    assert df["num1"].isna().sum() == 1


def test_mice_iterative_imputer():
    """Validates IterativeImputer (MICE) using BayesianRidge."""
    df = pd.DataFrame({
        "feature_a": [10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0, 80.0],
        "feature_b": [100.0, 200.0, np.nan, 400.0, 500.0, 600.0, 700.0, 800.0],
        "feature_c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })

    result = impute_missing_values(
        df=df,
        strategy=ImputationStrategy.MICE_RIDGE,
        add_mnar_indicators=False,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.transformed_df["feature_a"].isna().sum() == 0
    assert result.transformed_df["feature_b"].isna().sum() == 0
    assert len(result.applied_actions) >= 1
    assert "IterativeImputer" in result.applied_actions[0].code_snippet


def test_cv_benchmark_and_model_selection():
    """Tests dynamic Cross-Validation RMSE benchmark on features with artificial missingness."""
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan, 12.0],
        "x2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })

    imputer = MissingValueImputer(default_strategy=ImputationStrategy.AUTO_CV)
    benchmark_res = imputer.benchmark_dataset(df)

    assert "scores" in benchmark_res
    assert "recommendations" in benchmark_res
    assert "x1" in benchmark_res["scores"]
    assert len(benchmark_res["scores"]["x1"]) >= 2
    # Verify recommended imputer chosen
    rec = benchmark_res["recommendations"]["x1"]
    assert rec in ("mean", "median", "knn", "mice_ridge")

    # Run full imputation using AUTO_CV
    res_auto = imputer.impute(df, strategy=ImputationStrategy.AUTO_CV)
    assert res_auto.status == ExecutionStatus.SUCCESS
    assert res_auto.transformed_df["x1"].isna().sum() == 0


def test_mnar_flagging_and_high_missingness_safeguard():
    """Validates MNAR binary indicator column creation and >95% high missingness safeguard."""
    # 20 rows, col_high has 19 nulls (95%), col_mnar has 5 nulls (25% > 15% threshold)
    df = pd.DataFrame({
        "id": list(range(20)),
        "col_high": [np.nan] * 19 + [42.0],  # 95% missing
        "col_mnar": [10.0] * 15 + [np.nan] * 5,  # 25% missing
        "col_clean": list(range(20)),
    })

    imputer = MissingValueImputer(
        high_missing_threshold=0.90,
        mnar_threshold=0.15,
        drop_high_missing=True,
    )
    result = imputer.impute(df)

    assert result.status == ExecutionStatus.SUCCESS
    transformed = result.transformed_df

    # 1. High missing column (>90%) should be dropped when drop_high_missing=True
    assert "col_high" not in transformed.columns

    # 2. MNAR indicator column created for col_mnar
    assert "col_mnar_is_missing" in transformed.columns
    # Indicator is 1 where col_mnar was null, and 0 where it was observed
    assert (transformed["col_mnar_is_missing"] == df["col_mnar"].isna().astype(int)).all()

    # 3. No row dropping (all 20 rows preserved)
    assert len(transformed) == 20

    # 4. Immutability
    assert len(df.columns) == 4


def test_zero_framework_leakage():
    """Verifies that modules/imputation/ contains zero Streamlit or UI library imports."""
    import inspect
    import modules.imputation.imputer as imputer_module
    source = inspect.getsource(imputer_module)
    assert "import streamlit" not in source
    assert "from streamlit" not in source
    assert "st." not in source


def test_single_row_auto_cv_skips_cross_validation():
    frame = pd.DataFrame({"value": [np.nan], "label": ["only-row"]})
    result = impute_missing_values(
        frame,
        strategy=ImputationStrategy.AUTO_CV,
        add_mnar_indicators=False,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.transformed_df["value"].iloc[0] == 0.0
    assert any("skipped cross-validation" in entry for entry in result.audit_log)


def test_categorical_classifier_uses_random_forest_deterministically():
    frame = pd.DataFrame({
        "signal": list(range(20)),
        "segment": ["low" if value < 10 else "high" for value in range(20)],
        "region": ["north", "south"] * 10,
    })
    frame.loc[[2, 7, 12, 17], "segment"] = np.nan
    original = frame.copy(deep=True)

    first = CategoricalClassifierImputer("segment", random_state=7, n_jobs=1)
    second = CategoricalClassifierImputer("segment", random_state=7, n_jobs=1)
    first_output = first.fit_transform(frame)
    second_output = second.fit_transform(frame)

    assert first.strategy_ == "random_forest"
    assert first.n_train_ == 16
    assert first_output["segment"].isna().sum() == 0
    assert set(first_output["segment"]) <= {"low", "high"}
    pd.testing.assert_series_equal(first_output["segment"], second_output["segment"])
    pd.testing.assert_frame_equal(frame, original)


def test_categorical_classifier_fallback_and_unknown_predictors():
    training = pd.DataFrame({
        "value": range(9),
        "region": ["north"] * 6 + ["south"] * 3,
        "target": ["A"] * 7 + ["B", np.nan],
    })
    imputer = CategoricalClassifierImputer("target")
    output = imputer.fit_transform(training)
    assert imputer.strategy_ == "most_frequent"
    assert imputer.fallback_reason_ == "n_train<10"
    assert output["target"].iloc[-1] == "A"

    transform_frame = training.copy(deep=True)
    transform_frame.loc[8, "region"] = "unknown-region"
    assert imputer.transform(transform_frame)["target"].isna().sum() == 0


def test_integrated_categorical_action_and_duplicate_metadata_replay():
    frame = pd.DataFrame({"x": [1, 1], "category": ["A", np.nan]})
    result = impute_missing_values(
        frame,
        strategy=ImputationStrategy.AUTO_CV,
        add_mnar_indicators=False,
    )

    assert {
        key: result.metadata[key]
        for key in (
            "pre_imputation_duplicate_count",
            "post_imputation_duplicate_count",
            "new_duplicates",
        )
    } == {
        "pre_imputation_duplicate_count": 0,
        "post_imputation_duplicate_count": 1,
        "new_duplicates": 1,
    }
    assert result.applied_actions[0].action_type == "impute_categorical_most_frequent"
    validate_code(result.applied_actions[0].code_snippet)

    replay = frame.copy(deep=True)
    exec(result.applied_actions[0].code_snippet, {"df": replay, "pd": pd, "np": np})
    pd.testing.assert_frame_equal(replay, result.transformed_df)


def test_engine_facade_preserves_public_class_identity():
    assert ImputationEngine is MissingValueImputer
    assert EngineMissingValueImputer is MissingValueImputer


def test_integrated_random_forest_action_replays_exactly():
    frame = pd.DataFrame({
        "signal": list(range(20)),
        "target": ["low" if value < 10 else "high" for value in range(20)],
    })
    frame.loc[[1, 6, 13, 18], "target"] = np.nan
    result = impute_missing_values(
        frame,
        strategy=ImputationStrategy.AUTO_CV,
        add_mnar_indicators=False,
    )
    action = next(
        item
        for item in result.applied_actions
        if item.action_type == "impute_categorical_rf"
    )
    validate_code(action.code_snippet)

    replay = frame.copy(deep=True)
    exec(action.code_snippet, {"df": replay, "pd": pd, "np": np})
    pd.testing.assert_series_equal(
        replay["target"],
        result.transformed_df["target"],
    )


def test_all_null_columns_are_filled_and_recorded():
    frame = pd.DataFrame({
        "numeric": [np.nan] * 12,
        "category": pd.Series([None] * 12, dtype="object"),
    })
    result = impute_missing_values(
        frame,
        strategy=ImputationStrategy.KNN,
        add_mnar_indicators=False,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert not result.transformed_df.isna().any().any()
    assert result.transformed_df["numeric"].eq(0.0).all()
    assert result.transformed_df["category"].eq("Missing").all()
    action_types = {action.action_type for action in result.applied_actions}
    assert "fill_all_null_numeric" in action_types
    assert "fill_all_null_categorical" in action_types


def test_auto_cv_scores_survive_in_transform_metadata():
    frame = _numeric_missing_frame()
    result = impute_missing_values(
        frame,
        strategy=ImputationStrategy.AUTO_CV,
        add_mnar_indicators=False,
    )
    benchmark = result.metadata["benchmark_scores"]
    assert benchmark["scores"]
    assert benchmark["recommendations"]


def test_per_column_knn_replay_does_not_overwrite_other_targets():
    frame = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, 4.0, 5.0],
        "y": [10.0, 20.0, np.nan, 40.0, 50.0],
    })
    result = impute_missing_values(
        frame,
        strategy=ImputationStrategy.MEDIAN,
        column_strategies={"x": "knn", "y": "median"},
        add_mnar_indicators=False,
    )
    replay = frame.copy(deep=True)
    for action in result.applied_actions:
        validate_code(action.code_snippet)
        exec(action.code_snippet, {"df": replay, "pd": pd, "np": np})
    pd.testing.assert_frame_equal(replay, result.transformed_df)


def test_mice_rf_replay_serializes_runtime_estimator_configuration():
    result = impute_missing_values(
        _numeric_missing_frame(),
        strategy=ImputationStrategy.MICE_RF,
        add_mnar_indicators=False,
    )
    code = result.applied_actions[-1].code_snippet
    assert "n_estimators=15" in code
    assert "max_depth=6" in code
    assert "random_state=42" in code
    assert "n_jobs=-1" in code
