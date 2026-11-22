from typing import List, Optional, Dict
import re
import pandas as pd
import numpy as np

def is_identifier_or_name_column(col_name: str, series: Optional[pd.Series] = None) -> bool:
    """
    Identifies if a column is an identifier, key, or personal name:
    - Match common naming conventions (id, uuid, guid, key, name, email, phone, ssn, username, etc.)
    - High uniqueness ratio in text/object columns (>= 90% unique in >= 10 rows)
    """
    name_clean = str(col_name).strip().lower()
    patterns = [
        r"(^|_)(id|uuid|guid|pk|fk|key)($|_)",
        r"(^|_)(name|full_name|first_name|last_name|fname|lname|surname)($|_)",
        r"(^|_)(email|phone|mobile|ssn|tax_id|passport|account_num|account_number)($|_)",
        r"(^|_)(username|user_name|login|handle)($|_)",
    ]
    for pat in patterns:
        if re.search(pat, name_clean):
            return True

    # High uniqueness heuristic for string/object series
    if series is not None and len(series.dropna()) >= 10:
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            unique_ratio = series.nunique(dropna=True) / len(series.dropna())
            if unique_ratio >= 0.90:
                return True

    return False

def encode_features(
    df: pd.DataFrame,
    method: str = "onehot",  # "onehot", "label", "frequency", "none"
    columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    drop_first: bool = False,
    max_categories: int = 25,  # Prevents combinatorial column explosion
) -> pd.DataFrame:
    """
    Encodes categorical features via One-Hot, Label, or Frequency encoding.
    Excludes Name and ID columns from One-Hot Encoding by default, while allowing
    explicit manual selection via columns.
    """
    if method == "none" or not method:
        return df

    res_df = df.copy()

    # Determine candidate columns
    if columns is not None:
        # Manual selection by user: only encode requested columns
        target_cols = [c for c in columns if c in res_df.columns]
    else:
        # Default: auto-detect categorical features and exclude Name/ID columns by default
        target_cols = [
            c for c in res_df.select_dtypes(include=["object", "string", "category"]).columns
            if not is_identifier_or_name_column(c, res_df[c])
        ]

    # Additionally exclude user-specified excluded columns
    if exclude_columns:
        exclude_set = set(exclude_columns)
        target_cols = [c for c in target_cols if c not in exclude_set]

    if not target_cols:
        return res_df

    for col in target_cols:
        if col not in res_df.columns:
            continue

        unique_count = res_df[col].nunique(dropna=True)
        if unique_count == 0:
            continue

        if method == "onehot":
            # If cardinality exceeds max_categories, group low frequency values as 'OTHER'
            if unique_count > max_categories:
                top_cats = res_df[col].value_counts().nlargest(max_categories - 1).index
                col_series = res_df[col].apply(lambda x: x if x in top_cats or pd.isna(x) else "OTHER")
            else:
                col_series = res_df[col]

            dummies = pd.get_dummies(
                col_series,
                prefix=col,
                drop_first=drop_first,
                dtype=int
            )
            res_df = pd.concat([res_df.drop(columns=[col]), dummies], axis=1)

        elif method == "label":
            # Deterministic sorting
            categories = sorted([str(x) for x in res_df[col].dropna().unique()])
            label_map = {cat: idx for idx, cat in enumerate(categories)}
            res_df[col] = res_df[col].astype(str).map(label_map)

        elif method == "frequency":
            # Relative frequency distribution
            freq_map = res_df[col].value_counts(normalize=True).to_dict()
            res_df[col] = res_df[col].map(freq_map).fillna(0.0)

    return res_df

class TabularEncoder:
    @staticmethod
    def encode(
        df: pd.DataFrame,
        method: str = "onehot",
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        drop_first: bool = False,
        max_categories: int = 25,
    ) -> pd.DataFrame:
        return encode_features(
            df=df,
            method=method,
            columns=columns,
            exclude_columns=exclude_columns,
            drop_first=drop_first,
            max_categories=max_categories,
        )

    @staticmethod
    def detect_identifiers(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if is_identifier_or_name_column(c, df[c])]
