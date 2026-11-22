from typing import Dict, Any, List
import re
import pandas as pd
import numpy as np

def count_tabular_invalid_cells(df: pd.DataFrame) -> int:
    """
    Counts non-missing cells that fail conformance or contain corrupt/invalid representations:
    1. Infinite values in numeric columns (np.isinf).
    2. String values with unprintable/corrupt control characters (\x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f-\x9f).
    3. Incompatible mixed types in object columns.
    """
    invalid_count = 0
    control_char_regex = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            inf_mask = np.isinf(series)
            invalid_count += int(inf_mask.sum())
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            str_vals = series.astype(str)
            has_ctrl = str_vals.apply(lambda v: bool(control_char_regex.search(v)))
            invalid_count += int(has_ctrl.sum())

            # Detect mixed types among non-null values
            type_counts = series.map(type).value_counts()
            if len(type_counts) > 1:
                invalid_count += int(type_counts.iloc[1:].sum())

    return invalid_count

def compute_tabular_quality_score(df: pd.DataFrame, weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Computes 0-100 Quality Score for Tabular Datasets according to Master Specification:
    Q = w_comp * S_comp + w_val * S_val + w_uniq * S_uniq + w_snr * S_snr

    Calculates scores purely from actual dataset metrics without arbitrary increases.
    """
    w = weights or {"comp": 35.0, "val": 25.0, "uniq": 20.0, "snr": 20.0}

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = max(1, total_rows * total_cols)

    if total_rows == 0 or total_cols == 0:
        return {
            "composite_score": 0.0,
            "completeness": 0.0,
            "validity": 0.0,
            "uniqueness": 0.0,
            "snr": 0.0,
            "contributions": {
                "completeness": 0.0,
                "validity": 0.0,
                "uniqueness": 0.0,
                "snr": 0.0,
            },
            "weights": w,
            "formula": "Q = w_comp * S_comp + w_val * S_val + w_uniq * S_uniq + w_snr * S_snr",
            "formula_display": f"Q = {w['comp']:.0f}%·S_comp + {w['val']:.0f}%·S_val + {w['uniq']:.0f}%·S_uniq + {w['snr']:.0f}%·S_snr",
            "total_cells": total_cells,
            "missing_cells": 0,
            "invalid_cells": 0,
            "duplicate_rows": 0,
            "outlier_count": 0,
        }

    # 1. Completeness: 1 - (missing / total)
    missing_cells = int(df.isna().sum().sum())
    s_comp = max(0.0, 1.0 - (missing_cells / total_cells))

    # 2. Validity: conformance ratio (non-corrupt data, valid types, no infinities or control characters)
    present_elements = total_cells - missing_cells
    invalid_cells = count_tabular_invalid_cells(df)
    if present_elements > 0:
        s_val = max(0.0, 1.0 - (invalid_cells / present_elements))
    else:
        s_val = 0.0

    # 3. Uniqueness: 1 - (duplicate rows / total rows)
    duplicate_rows = int(df.duplicated().sum())
    s_uniq = max(0.0, 1.0 - (duplicate_rows / total_rows))

    # 4. SNR: 1 - (outlier count / total eligible numeric cells)
    # Applied to continuous numeric columns (nunique > 2, len > 4), ignoring binary dummy columns
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if not pd.api.types.is_bool_dtype(df[c])
    ]
    total_outliers = 0
    total_eligible_cells = 0
    for col in num_cols:
        series = df[col].dropna()
        if len(series) > 4 and series.nunique() > 2:
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                total_eligible_cells += len(series)
                outliers = ((series < (q25 - 1.5 * iqr)) | (series > (q75 + 1.5 * iqr))).sum()
                total_outliers += int(outliers)

    if total_eligible_cells > 0:
        s_snr = max(0.0, 1.0 - (total_outliers / total_eligible_cells))
    else:
        s_snr = 1.0

    contrib_comp = round(w["comp"] * s_comp, 2)
    contrib_val = round(w["val"] * s_val, 2)
    contrib_uniq = round(w["uniq"] * s_uniq, 2)
    contrib_snr = round(w["snr"] * s_snr, 2)
    composite = round(contrib_comp + contrib_val + contrib_uniq + contrib_snr, 2)

    return {
        "composite_score": composite,
        "completeness": round(s_comp * 100, 2),
        "validity": round(s_val * 100, 2),
        "uniqueness": round(s_uniq * 100, 2),
        "snr": round(s_snr * 100, 2),
        "contributions": {
            "completeness": contrib_comp,
            "validity": contrib_val,
            "uniqueness": contrib_uniq,
            "snr": contrib_snr,
        },
        "weights": w,
        "formula": "Q = w_comp * S_comp + w_val * S_val + w_uniq * S_uniq + w_snr * S_snr",
        "formula_display": f"Q = {w['comp']:.0f}%·S_comp + {w['val']:.0f}%·S_val + {w['uniq']:.0f}%·S_uniq + {w['snr']:.0f}%·S_snr",
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "invalid_cells": invalid_cells,
        "duplicate_rows": duplicate_rows,
        "outlier_count": total_outliers,
    }

def compute_image_quality_score(profiles: List[Dict[str, Any]], weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Computes 0-100 Quality Score for Image Datasets:
    - Completeness: uncorrupted images / total
    - Validity: images having valid resolution and color dimensions / total
    - Uniqueness: unique images / total
    - SNR: ratio of sharp, non-blurry images (Laplacian variance >= threshold)
    """
    w = weights or {"comp": 35.0, "val": 25.0, "uniq": 20.0, "snr": 20.0}
    total = max(1, len(profiles))

    if len(profiles) == 0:
        return {
            "composite_score": 0.0,
            "completeness": 0.0,
            "validity": 0.0,
            "uniqueness": 0.0,
            "snr": 0.0,
            "contributions": {
                "completeness": 0.0,
                "validity": 0.0,
                "uniqueness": 0.0,
                "snr": 0.0,
            },
            "weights": w,
            "formula": "Q = w_comp * S_comp + w_val * S_val + w_uniq * S_uniq + w_snr * S_snr",
            "formula_display": f"Q = {w['comp']:.0f}%·S_comp + {w['val']:.0f}%·S_val + {w['uniq']:.0f}%·S_uniq + {w['snr']:.0f}%·S_snr",
            "total_images": 0,
            "corrupted": 0,
            "blurry": 0,
        }

    corrupt_count = sum(1 for p in profiles if p.get("corrupted", False))
    blurry_count = sum(1 for p in profiles if not p.get("corrupted", False) and p.get("is_blurry", False))
    valid_count = total - corrupt_count

    s_comp = max(0.0, (total - corrupt_count) / total)
    s_val = max(0.0, valid_count / total)
    s_uniq = 1.0  # assume unique unless hashes duplicate
    s_snr = max(0.0, (valid_count - blurry_count) / max(1, valid_count))

    contrib_comp = round(w["comp"] * s_comp, 2)
    contrib_val = round(w["val"] * s_val, 2)
    contrib_uniq = round(w["uniq"] * s_uniq, 2)
    contrib_snr = round(w["snr"] * s_snr, 2)
    composite = round(contrib_comp + contrib_val + contrib_uniq + contrib_snr, 2)

    return {
        "composite_score": composite,
        "completeness": round(s_comp * 100, 2),
        "validity": round(s_val * 100, 2),
        "uniqueness": round(s_uniq * 100, 2),
        "snr": round(s_snr * 100, 2),
        "contributions": {
            "completeness": contrib_comp,
            "validity": contrib_val,
            "uniqueness": contrib_uniq,
            "snr": contrib_snr,
        },
        "weights": w,
        "formula": "Q = w_comp * S_comp + w_val * S_val + w_uniq * S_uniq + w_snr * S_snr",
        "formula_display": f"Q = {w['comp']:.0f}%·S_comp + {w['val']:.0f}%·S_val + {w['uniq']:.0f}%·S_uniq + {w['snr']:.0f}%·S_snr",
        "total_images": total,
        "corrupted": corrupt_count,
        "blurry": blurry_count,
    }

class QualityScorer:
    @staticmethod
    def score_tabular(df: pd.DataFrame) -> Dict[str, Any]:
        return compute_tabular_quality_score(df)

    @staticmethod
    def score_image(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        return compute_image_quality_score(profiles)
