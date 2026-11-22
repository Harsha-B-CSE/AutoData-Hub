from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def scale_features(
    df: pd.DataFrame,
    method: str = "standard",  # "minmax", "standard", "robust", "none"
    columns: Optional[List[str]] = None,
    feature_range: tuple = (0, 1),
) -> pd.DataFrame:
    """
    Scales numerical features using MinMaxScaler, StandardScaler, or RobustScaler.
    """
    if method == "none" or not method:
        return df

    res_df = df.copy()

    # Determine candidate numeric columns
    if columns:
        candidate_cols = [c for c in columns if c in res_df.columns]
    else:
        candidate_cols = res_df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter strictly to numeric, non-boolean columns
    target_cols = [
        c for c in candidate_cols
        if pd.api.types.is_numeric_dtype(res_df[c]) and not pd.api.types.is_bool_dtype(res_df[c])
    ]

    if not target_cols:
        return res_df

    if method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    # Track original NaN / non-finite masks to preserve missingness structure accurately
    nan_masks = {}
    temp_subset = pd.DataFrame(index=res_df.index)

    for col in target_cols:
        s = res_df[col].replace([np.inf, -np.inf], np.nan)
        nan_masks[col] = s.isna()
        valid = s.dropna()
        if len(valid) > 0:
            fill_v = float(valid.median())
        else:
            fill_v = 0.0
        temp_subset[col] = s.fillna(fill_v)

    scaled_vals = scaler.fit_transform(temp_subset)

    for idx, col in enumerate(target_cols):
        col_scaled = pd.Series(scaled_vals[:, idx], index=res_df.index)
        # Restore original missing cells
        if nan_masks[col].any():
            col_scaled[nan_masks[col]] = np.nan
        res_df[col] = col_scaled

    return res_df

class TabularScaler:
    @staticmethod
    def scale(
        df: pd.DataFrame,
        method: str = "standard",
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0, 1),
    ) -> pd.DataFrame:
        return scale_features(
            df=df,
            method=method,
            columns=columns,
            feature_range=feature_range,
        )
