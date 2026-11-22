import re
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: Any = "first"
) -> Tuple[pd.DataFrame, int]:
    """
    Identifies and eliminates identical rows or subset-based duplicates.
    Accurately handles keep strategies ("first", "last", False/"false") and resets index.
    """
    initial_count = len(df)
    if initial_count == 0:
        return df.copy(), 0

    # Sanitize keep parameter for pandas (first, last, or False)
    keep_param = keep
    if isinstance(keep, str):
        k_lower = keep.strip().lower()
        if k_lower in ("false", "none", "drop_all"):
            keep_param = False
        elif k_lower in ("first", "last"):
            keep_param = k_lower
        else:
            keep_param = "first"
    elif not keep:
        keep_param = False

    # Validate subset columns against df
    valid_subset = None
    if subset:
        valid_subset = [c for c in subset if c in df.columns]
        if not valid_subset:
            valid_subset = None

    cleaned_df = df.drop_duplicates(subset=valid_subset, keep=keep_param).reset_index(drop=True)
    dropped_count = initial_count - len(cleaned_df)
    return cleaned_df, dropped_count

def clean_text_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strip_whitespace: bool = True,
    remove_control_chars: bool = True,
    case: str = "preserve",  # "preserve", "lower", "upper", "title"
    regex_replace: Optional[List[Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Cleans string columns: regex replacement, trimming, control character culling,
    and casing standardization.
    """
    res_df = df.copy()
    target_cols = columns if columns else res_df.select_dtypes(include=["object", "string"]).columns.tolist()

    control_char_regex = re.compile(r"[\x00-\x1f\x7f-\x9f]")

    for col in target_cols:
        if col not in res_df.columns:
            continue

        s = res_df[col].astype(str)
        # Preserve actual NaNs
        mask_nan = res_df[col].isna()

        if strip_whitespace:
            s = s.str.strip()

        if remove_control_chars:
            s = s.apply(lambda v: control_char_regex.sub("", v) if isinstance(v, str) else v)

        if regex_replace:
            for rule in regex_replace:
                pattern = rule.get("pattern", "")
                repl = rule.get("replacement", "")
                if pattern:
                    s = s.str.replace(pattern, repl, regex=True)

        if case == "lower":
            s = s.str.lower()
        elif case == "upper":
            s = s.str.upper()
        elif case == "title":
            s = s.str.title()

        res_df[col] = s
        res_df.loc[mask_nan, col] = np.nan

    return res_df

def infer_and_correct_types(
    df: pd.DataFrame,
    convert_booleans: bool = True,
    convert_currency: bool = True,
    convert_datetime: bool = True,
) -> pd.DataFrame:
    """
    Heuristically corrects misclassified types:
    - stringified booleans ('true'/'false', 'yes'/'no', '1'/'0', 't'/'f')
    - numeric fields with currency symbols ($ € £ ¥), commas, percent signs
    - ISO-8601 or standard datetimes
    """
    res_df = df.copy()
    bool_true_set = {"true", "yes", "t", "1", "y"}
    bool_false_set = {"false", "no", "f", "0", "n"}

    for col in res_df.select_dtypes(include=["object", "string"]).columns:
        series_clean = res_df[col].dropna().astype(str).str.strip().str.lower()
        if len(series_clean) == 0:
            continue

        # 1. Check Booleans
        if convert_booleans:
            unique_vals = set(series_clean.unique())
            if unique_vals.issubset(bool_true_set | bool_false_set) and len(unique_vals) > 0:
                bool_map = {v: True for v in bool_true_set}
                bool_map.update({v: False for v in bool_false_set})
                res_df[col] = res_df[col].astype(str).str.strip().str.lower().map(bool_map)
                continue

        # 2. Check Currency / Numbers with symbols
        if convert_currency:
            # Check if strings look like $1,234.56 or 45% or €500
            sample_clean = series_clean.str.replace(r"[$€£¥,%\s]", "", regex=True)
            # Try numeric conversion
            numeric_conv = pd.to_numeric(sample_clean, errors="coerce")
            if numeric_conv.notna().mean() >= 0.85: # 85%+ matched
                res_df[col] = pd.to_numeric(
                    res_df[col].astype(str).str.replace(r"[$€£¥,%\s]", "", regex=True),
                    errors="coerce"
                )
                continue

        # 3. Check Datetime
        if convert_datetime:
            # Only test if string has typical date separators - or /
            has_date_format = series_clean.str.contains(r"[-/:]", regex=True).mean() >= 0.8
            if has_date_format:
                try:
                    dt_series = pd.to_datetime(res_df[col], errors="coerce", format="ISO8601")
                    if dt_series.notna().mean() >= 0.85:
                        res_df[col] = dt_series
                        continue
                except Exception:
                    pass

    return res_df

def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr", # "iqr" or "zscore"
    action: str = "clip", # "clip" (winsorize) or "exclude"
    threshold: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Detects and handles numerical outliers via IQR or Z-score with either clipping or exclusion.
    """
    res_df = df.copy()
    target_cols = columns if columns else res_df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts: Dict[str, int] = {}

    rows_to_drop = set()

    for col in target_cols:
        if col not in res_df.columns:
            continue
        series = res_df[col].dropna()
        if len(series) < 4:
            continue

        if method == "iqr":
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr

            outliers_mask = (res_df[col] < lower_bound) | (res_df[col] > upper_bound)
            count = int(outliers_mask.sum())
            outlier_counts[col] = count

            if action == "clip":
                res_df[col] = res_df[col].clip(lower=lower_bound, upper=upper_bound)
            elif action == "exclude":
                rows_to_drop.update(res_df[outliers_mask].index.tolist())

        elif method == "zscore":
            mean_val = series.mean()
            std_val = series.std()
            if std_val > 0:
                z_scores = np.abs((res_df[col] - mean_val) / std_val)
                outliers_mask = z_scores > threshold
                count = int(outliers_mask.sum())
                outlier_counts[col] = count

                if action == "clip":
                    lower_bound = mean_val - threshold * std_val
                    upper_bound = mean_val + threshold * std_val
                    res_df[col] = res_df[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == "exclude":
                    rows_to_drop.update(res_df[outliers_mask].index.tolist())

    if action == "exclude" and rows_to_drop:
        res_df = res_df.drop(index=list(rows_to_drop)).copy()

    return res_df, outlier_counts

class TabularCleaner:
    @staticmethod
    def clean(
        df: pd.DataFrame,
        remove_dup: bool = True,
        dup_subset: Optional[List[str]] = None,
        dup_keep: str = "first",
        text_clean: bool = True,
        text_case: str = "preserve",
        type_correct: bool = True,
        outlier_handling: bool = False,
        outlier_method: str = "iqr",
        outlier_action: str = "clip",
        outlier_threshold: float = 1.5,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        processed = df.copy()

        if remove_dup:
            processed, dropped = remove_duplicates(processed, subset=dup_subset, keep=dup_keep)
            meta["duplicates_dropped"] = dropped

        if type_correct:
            processed = infer_and_correct_types(processed)
            meta["types_corrected"] = True

        if text_clean:
            processed = clean_text_columns(processed, case=text_case)
            meta["text_cleaned"] = True

        if outlier_handling:
            processed, outliers = handle_outliers(
                processed,
                method=outlier_method,
                action=outlier_action,
                threshold=outlier_threshold,
            )
            meta["outliers_handled"] = outliers

        return processed, meta
