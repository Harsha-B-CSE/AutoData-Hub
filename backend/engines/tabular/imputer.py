from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def get_column_imputation_details(
    df: pd.DataFrame,
    numerical_strategy: str = "median",
    categorical_strategy: str = "mode",
    num_constant_value: float = 0.0,
    cat_constant_value: str = "MISSING",
    knn_neighbors: int = 5,
    columns_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Determines the exact imputation strategy, fill parameter, and null count for each column in the dataset.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    columns_config = columns_config or {}

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round(float((null_count / max(1, len(df))) * 100.0), 2)
        is_num = bool(pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]))

        col_override = columns_config.get(col, {})
        strategy = col_override.get("strategy")
        if not strategy:
            strategy = numerical_strategy if is_num else categorical_strategy

        fill_desc = ""
        fill_val = None

        if null_count > 0:
            if is_num:
                if strategy == "median":
                    clean_s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                    med = clean_s.median() if len(clean_s) > 0 else np.nan
                    fill_val = float(med) if pd.notna(med) else float(col_override.get("constant_value", num_constant_value))
                    fill_desc = f"Median Imputation (fill_value={round(fill_val, 4) if isinstance(fill_val, float) else fill_val})"
                elif strategy == "mean":
                    clean_s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                    mn = clean_s.mean() if len(clean_s) > 0 else np.nan
                    fill_val = float(mn) if pd.notna(mn) else float(col_override.get("constant_value", num_constant_value))
                    fill_desc = f"Mean Imputation (fill_value={round(fill_val, 4) if isinstance(fill_val, float) else fill_val})"
                elif strategy == "knn":
                    fill_desc = f"KNN Imputation (k={knn_neighbors} neighbors)"
                elif strategy == "constant":
                    fill_val = col_override.get("constant_value", num_constant_value)
                    fill_desc = f"Constant Imputation (fill_value={fill_val})"
                elif strategy == "none":
                    fill_desc = "No Imputation (missing values preserved)"
            else:
                if strategy == "mode":
                    mode_s = df[col].dropna().mode()
                    fill_val = str(mode_s.iloc[0]) if not mode_s.empty else "UNKNOWN"
                    fill_desc = f"Mode Imputation (fill_value='{fill_val}')"
                elif strategy == "constant":
                    fill_val = col_override.get("constant_value", cat_constant_value)
                    fill_desc = f"Constant Imputation (fill_value='{fill_val}')"
                elif strategy == "indicator":
                    fill_val = col_override.get("constant_value", cat_constant_value)
                    fill_desc = f"Indicator Column + Constant ('{fill_val}')"
                elif strategy == "none":
                    fill_desc = "No Imputation (missing values preserved)"
        else:
            fill_desc = "No missing values detected"

        summary[col] = {
            "column": col,
            "is_numeric": is_num,
            "data_type": str(df[col].dtype),
            "null_count": null_count,
            "null_percent": null_pct,
            "strategy": strategy,
            "fill_value": fill_val,
            "description": fill_desc,
        }

    return summary

def impute_missing_values(
    df: pd.DataFrame,
    numerical_strategy: str = "median",  # "mean", "median", "constant", "knn", "none"
    categorical_strategy: str = "mode",  # "mode", "constant", "indicator", "none"
    num_constant_value: float = 0.0,
    cat_constant_value: str = "MISSING",
    knn_neighbors: int = 5,
    columns_config: Optional[Dict[str, Dict[str, Any]]] = None,
    return_metadata: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Imputes missing values across numerical and categorical features.
    Supports global strategies or granular per-column overrides via columns_config.
    """
    res_df = df.copy()

    details = get_column_imputation_details(
        df=df,
        numerical_strategy=numerical_strategy,
        categorical_strategy=categorical_strategy,
        num_constant_value=num_constant_value,
        cat_constant_value=cat_constant_value,
        knn_neighbors=knn_neighbors,
        columns_config=columns_config,
    )

    # Apply per-column or global strategy
    for col in res_df.columns:
        if not res_df[col].isna().any():
            continue

        col_override = columns_config.get(col, {}) if columns_config else {}
        is_num = pd.api.types.is_numeric_dtype(res_df[col]) and not pd.api.types.is_bool_dtype(res_df[col])

        strategy = col_override.get("strategy")
        if not strategy:
            strategy = numerical_strategy if is_num else categorical_strategy

        if strategy == "none":
            continue

        # Numerical Imputation
        if is_num:
            clean_series = res_df[col].replace([np.inf, -np.inf], np.nan)
            if strategy == "mean":
                valid_vals = clean_series.dropna()
                mean_val = float(valid_vals.mean()) if len(valid_vals) > 0 else float(col_override.get("constant_value", num_constant_value))
                res_df[col] = clean_series.fillna(mean_val)
            elif strategy == "median":
                valid_vals = clean_series.dropna()
                median_val = float(valid_vals.median()) if len(valid_vals) > 0 else float(col_override.get("constant_value", num_constant_value))
                if pd.api.types.is_integer_dtype(res_df[col]) and not median_val.is_integer():
                    res_df[col] = clean_series.astype(float).fillna(median_val)
                else:
                    res_df[col] = clean_series.fillna(median_val)
            elif strategy == "constant":
                c_val = col_override.get("constant_value", num_constant_value)
                res_df[col] = clean_series.fillna(c_val)

        # Categorical Imputation
        else:
            if strategy == "mode":
                mode_series = res_df[col].mode(dropna=True)
                fill_val = mode_series.iloc[0] if not mode_series.empty else "UNKNOWN"
                res_df[col] = res_df[col].fillna(fill_val)
            elif strategy == "constant":
                c_val = col_override.get("constant_value", cat_constant_value)
                res_df[col] = res_df[col].fillna(c_val)
            elif strategy == "indicator":
                # Create indicator column and fill sentinel
                indicator_col = f"{col}_is_missing"
                res_df[indicator_col] = res_df[col].isna().astype(int)
                c_val = col_override.get("constant_value", cat_constant_value)
                res_df[col] = res_df[col].fillna(c_val)

    # Global KNN Imputation for any remaining numerical missing fields if selected
    if numerical_strategy == "knn":
        num_cols = res_df.select_dtypes(include=[np.number]).columns.tolist()
        num_with_na = [c for c in num_cols if res_df[c].isna().any()]
        if num_with_na and len(num_cols) >= 2:
            try:
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                res_df[num_cols] = imputer.fit_transform(res_df[num_cols])
            except Exception:
                # Fallback to median on failure
                for c in num_with_na:
                    res_df[c] = res_df[c].fillna(res_df[c].median())

    if return_metadata:
        return res_df, {"columns": details}

    return res_df

class TabularImputer:
    @staticmethod
    def impute(
        df: pd.DataFrame,
        numerical_strategy: str = "median",
        categorical_strategy: str = "mode",
        num_constant_value: float = 0.0,
        cat_constant_value: str = "MISSING",
        knn_neighbors: int = 5,
        columns_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        return impute_missing_values(
            df=df,
            numerical_strategy=numerical_strategy,
            categorical_strategy=categorical_strategy,
            num_constant_value=num_constant_value,
            cat_constant_value=cat_constant_value,
            knn_neighbors=knn_neighbors,
            columns_config=columns_config,
            return_metadata=False,
        )

    @staticmethod
    def impute_with_meta(
        df: pd.DataFrame,
        numerical_strategy: str = "median",
        categorical_strategy: str = "mode",
        num_constant_value: float = 0.0,
        cat_constant_value: str = "MISSING",
        knn_neighbors: int = 5,
        columns_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return impute_missing_values(
            df=df,
            numerical_strategy=numerical_strategy,
            categorical_strategy=categorical_strategy,
            num_constant_value=num_constant_value,
            cat_constant_value=cat_constant_value,
            knn_neighbors=knn_neighbors,
            columns_config=columns_config,
            return_metadata=True,
        )

    @staticmethod
    def get_summary(df: pd.DataFrame, **kwargs) -> Dict[str, Dict[str, Any]]:
        return get_column_imputation_details(df, **kwargs)
