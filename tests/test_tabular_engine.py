import pytest
import pandas as pd
import numpy as np
from backend.engines.tabular import (
    TabularProfiler,
    TabularCleaner,
    remove_duplicates,
    clean_text_columns,
    infer_and_correct_types,
    TabularImputer,
    TabularScaler,
    TabularEncoder,
)
from backend.engines.tabular.cleaner import handle_outliers

def test_tabular_profiling():
    data = {
        "age": [25, 30, 35, np.nan, 45, 50, 100],
        "salary": [50000, 60000, 75000, 80000, 95000, 110000, 500000],
        "category": ["A", "B", "A", "C", "B", "A", None],
    }
    df = pd.DataFrame(data)
    profile = TabularProfiler.profile(df)

    assert profile["total_rows"] == 7
    assert profile["total_cols"] == 3
    assert "age" in profile["columns"]
    assert profile["columns"]["age"]["null_count"] == 1
    assert profile["columns"]["age"]["null_percent"] == round(1/7 * 100, 2)
    assert profile["columns"]["salary"]["min"] == 50000.0
    assert profile["columns"]["salary"]["max"] == 500000.0

def test_tabular_cleaning_and_types():
    data = {
        "id": [1, 1, 2, 3],
        "name": ["  Alice\x00 ", "  Alice\x00 ", " Bob ", "CHARLIE"],
        "active": ["True", "true", "FALSE", "yes"],
        "price": ["$1,200.50", "$1,200.50", "€450.00", "$99.99"],
    }
    df = pd.DataFrame(data)

    # 1. Deduplication
    df_dedup, dropped = remove_duplicates(df, subset=["id"])
    assert dropped == 1
    assert len(df_dedup) == 3

    # 2. Text cleaning
    df_clean = clean_text_columns(df_dedup, case="lower")
    assert df_clean["name"].iloc[0] == "alice"
    assert df_clean["name"].iloc[2] == "charlie"

    # 3. Type inference & correction
    df_types = infer_and_correct_types(df_clean)
    assert df_types["active"].dtype == bool or set(df_types["active"].dropna().unique()).issubset({True, False})
    assert pd.api.types.is_numeric_dtype(df_types["price"])
    assert df_types["price"].iloc[0] == 1200.50

def test_tabular_imputation():
    data = {
        "num": [10.0, 20.0, np.nan, 40.0],
        "cat": ["Apple", np.nan, "Banana", "Apple"],
    }
    df = pd.DataFrame(data)

    df_imp = TabularImputer.impute(df, numerical_strategy="median", categorical_strategy="mode")
    assert df_imp["num"].isna().sum() == 0
    assert df_imp["num"].iloc[2] == 20.0 # median of [10, 20, 40] is 20.0
    assert df_imp["cat"].isna().sum() == 0
    assert df_imp["cat"].iloc[1] == "Apple" # mode is Apple

def test_tabular_scaling():
    data = {
        "val": [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    df = pd.DataFrame(data)

    df_minmax = TabularScaler.scale(df, method="minmax", feature_range=(0, 1))
    assert df_minmax["val"].min() == 0.0
    assert df_minmax["val"].max() == 1.0

    df_std = TabularScaler.scale(df, method="standard")
    assert pytest.approx(df_std["val"].mean(), 1e-5) == 0.0
    assert pytest.approx(df_std["val"].std(ddof=0), 1e-5) == 1.0

def test_tabular_encoding():
    data = {
        "dept": ["HR", "Engineering", "Marketing", "HR"]
    }
    df = pd.DataFrame(data)

    df_onehot = TabularEncoder.encode(df, method="onehot")
    assert "dept_HR" in df_onehot.columns
    assert "dept_Engineering" in df_onehot.columns

    df_label = TabularEncoder.encode(df, method="label")
    assert pd.api.types.is_integer_dtype(df_label["dept"])

def test_onehot_encoding_default_excludes_name_id():
    data = {
        "user_id": ["U101", "U102", "U103", "U104"],
        "full_name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince"],
        "department": ["Sales", "Engineering", "Sales", "HR"],
    }
    df = pd.DataFrame(data)

    # By default, Name/ID columns MUST be excluded from One-Hot Encoding
    df_encoded = TabularEncoder.encode(df, method="onehot")

    # department should be one-hot encoded into dummies
    assert "department_Sales" in df_encoded.columns
    assert "department_HR" in df_encoded.columns
    # user_id and full_name must NOT be one-hot encoded; they must remain as original columns
    assert "user_id" in df_encoded.columns
    assert "full_name" in df_encoded.columns
    assert not any(c.startswith("user_id_") for c in df_encoded.columns)
    assert not any(c.startswith("full_name_") for c in df_encoded.columns)

def test_onehot_encoding_manual_selection():
    data = {
        "user_id": ["U1", "U2", "U3"],
        "dept": ["HR", "IT", "HR"],
        "role": ["Lead", "Staff", "Intern"],
    }
    df = pd.DataFrame(data)

    # Manually select only 'role' to be encoded
    df_role_only = TabularEncoder.encode(df, method="onehot", columns=["role"])
    assert "role_Lead" in df_role_only.columns
    assert "dept" in df_role_only.columns
    assert "user_id" in df_role_only.columns

    # Explicit exclusion via exclude_columns
    df_excluded = TabularEncoder.encode(df, method="onehot", exclude_columns=["role"])
    assert "dept_HR" in df_excluded.columns
    assert "role" in df_excluded.columns

def test_exact_imputation_strategy_per_column():
    data = {
        "age": [20.0, 30.0, np.nan, 40.0],
        "category": ["A", np.nan, "B", "A"],
        "constant_col": [1.0, np.nan, 3.0, 4.0],
    }
    df = pd.DataFrame(data)

    # Test impute_with_meta and exact strategy reporting
    df_imp, meta = TabularImputer.impute_with_meta(
        df,
        numerical_strategy="median",
        categorical_strategy="mode",
        columns_config={"constant_col": {"strategy": "constant", "constant_value": 99.0}},
    )

    cols = meta["columns"]
    # age: exact median strategy
    assert cols["age"]["strategy"] == "median"
    assert cols["age"]["fill_value"] == 30.0
    assert "Median" in cols["age"]["description"]
    assert df_imp["age"].iloc[2] == 30.0

    # category: exact mode strategy
    assert cols["category"]["strategy"] == "mode"
    assert cols["category"]["fill_value"] == "A"
    assert df_imp["category"].iloc[1] == "A"

    # constant_col: exact constant override strategy
    assert cols["constant_col"]["strategy"] == "constant"
    assert cols["constant_col"]["fill_value"] == 99.0
    assert df_imp["constant_col"].iloc[1] == 99.0

def test_duplicate_removal_options():
    data = {
        "id": [1, 1, 2, 3, 3],
        "val": ["A", "A", "B", "C", "C"],
    }
    df = pd.DataFrame(data)

    # keep="first"
    df_first, d_first = remove_duplicates(df, keep="first")
    assert d_first == 2
    assert len(df_first) == 3
    assert list(df_first.index) == [0, 1, 2]

    # keep="last"
    df_last, d_last = remove_duplicates(df, keep="last")
    assert d_last == 2
    assert len(df_last) == 3

    # keep="false" string drops all duplicates
    df_none, d_none = remove_duplicates(df, keep="false")
    assert d_none == 4
    assert len(df_none) == 1
    assert df_none["id"].iloc[0] == 2

def test_median_imputation_edge_cases():
    # Numeric column with infinity and NaNs
    data = {
        "val": [10.0, 20.0, np.inf, np.nan, 30.0],
    }
    df = pd.DataFrame(data)
    df_imp = TabularImputer.impute(df, numerical_strategy="median")
    # Median of finite values [10, 20, 30] is 20.0
    assert df_imp["val"].isna().sum() == 0
    assert df_imp["val"].iloc[3] == 20.0

    # Column that is all NaN
    df_all_nan = pd.DataFrame({"all_na": [np.nan, np.nan]})
    df_imp_nan = TabularImputer.impute(df_all_nan, numerical_strategy="median", num_constant_value=0.0)
    assert df_imp_nan["all_na"].isna().sum() == 0
    assert df_imp_nan["all_na"].iloc[0] == 0.0

def test_standard_scaler_preserves_nans():
    data = {
        "a": [10.0, 20.0, np.nan, 40.0, 50.0],
        "b": ["x", "y", "z", "w", "v"] # non-numeric column in df
    }
    df = pd.DataFrame(data)

    # Standard scaling should scale column 'a', preserve its NaN at index 2, and leave 'b' untouched
    df_scaled = TabularScaler.scale(df, method="standard")
    assert pd.isna(df_scaled["a"].iloc[2])
    assert df_scaled["b"].tolist() == ["x", "y", "z", "w", "v"]
    # Check that scaled valid values have mean approx 0
    valid_scaled = df_scaled["a"].dropna()
    assert pytest.approx(valid_scaled.mean(), 1e-4) == 0.0

def test_outlier_handling():
    data = {
        "val": [10, 12, 11, 13, 12, 10, 11, 1000] # 1000 is obvious outlier
    }
    df = pd.DataFrame(data)

    df_clipped, counts = handle_outliers(df, method="iqr", action="clip", threshold=1.5)
    assert counts["val"] >= 1
    assert df_clipped["val"].max() < 100
