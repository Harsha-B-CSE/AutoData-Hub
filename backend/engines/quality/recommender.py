from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

def generate_tabular_recommendations(
    profile: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    imputation_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Analyzes tabular dataset profile and generates explainable, actionable recommendations.
    When config is provided, recommendations and applied interventions precisely reflect
    the actual selected preprocessing configuration and show the exact imputation strategy
    applied to each column.
    """
    recs: List[Dict[str, Any]] = []
    cols = profile.get("columns", {})
    total_rows = max(1, profile.get("total_rows", 1))
    dup_rows = profile.get("duplicate_rows", 0)

    # 1. Duplicate check
    if dup_rows > 0:
        if config is not None:
            clean_opts = config.get("cleaning", {})
            remove_dup = clean_opts.get("remove_duplicates", True)
            dup_keep = clean_opts.get("duplicate_keep", "first")
            if remove_dup:
                recs.append({
                    "id": "rec_dedup",
                    "category": "deduplication",
                    "priority": "high",
                    "target": "dataset",
                    "diagnosis": f"Detected {dup_rows} exact duplicate rows ({round((dup_rows / total_rows) * 100, 2)}% of dataset).",
                    "recommendation": f"Remove duplicate rows while keeping '{dup_keep}' occurrence as configured.",
                    "action": "remove_duplicates",
                    "params": {"keep": dup_keep},
                    "approved": True
                })
            else:
                recs.append({
                    "id": "rec_dedup",
                    "category": "deduplication",
                    "priority": "low",
                    "target": "dataset",
                    "diagnosis": f"Detected {dup_rows} exact duplicate rows ({round((dup_rows / total_rows) * 100, 2)}% of dataset).",
                    "recommendation": "Deduplication is disabled in selected configuration; duplicate rows retained.",
                    "action": "skip_duplicates",
                    "params": {"keep": "all"},
                    "approved": False
                })
        else:
            recs.append({
                "id": "rec_dedup",
                "category": "deduplication",
                "priority": "high",
                "target": "dataset",
                "diagnosis": f"Detected {dup_rows} exact duplicate rows ({round((dup_rows / total_rows) * 100, 2)}% of dataset).",
                "recommendation": "Remove duplicate rows while keeping first occurrence to avoid bias.",
                "action": "remove_duplicates",
                "params": {"keep": "first"},
                "approved": True
            })

    # 2. Imputation checks (showing exact strategy per column when config is provided)
    if config is not None:
        imp_opts = config.get("imputation", {})
        cols_cfg = imp_opts.get("columns_config", {}) or {}
        num_strat = imp_opts.get("numerical_strategy", "median")
        cat_strat = imp_opts.get("categorical_strategy", "mode")
        knn_k = imp_opts.get("knn_neighbors", 5)

        for col_name, meta in cols.items():
            null_pct = meta.get("null_percent", 0.0)
            null_count = meta.get("null_count", 0)
            is_num = meta.get("is_numeric", False)

            if null_count > 0 or null_pct > 0:
                # Use provided imputation summary or determine from config
                if imputation_summary and col_name in imputation_summary:
                    c_info = imputation_summary[col_name]
                    strategy = c_info.get("strategy", num_strat if is_num else cat_strat)
                    desc = c_info.get("description", "")
                    fill_val = c_info.get("fill_value")
                else:
                    col_override = cols_cfg.get(col_name, {})
                    strategy = col_override.get("strategy") or (num_strat if is_num else cat_strat)
                    if strategy == "median":
                        fill_val = meta.get("median")
                        desc = f"Median Imputation (fill_value={fill_val})"
                    elif strategy == "mean":
                        fill_val = meta.get("mean")
                        desc = f"Mean Imputation (fill_value={fill_val})"
                    elif strategy == "knn":
                        fill_val = None
                        desc = f"Iterative KNN Imputation (k={knn_k} neighbors)"
                    elif strategy == "mode":
                        fill_val = None
                        desc = "Mode Imputation (most frequent category)"
                    elif strategy == "constant":
                        fill_val = col_override.get("constant_value", imp_opts.get("num_constant", 0.0) if is_num else imp_opts.get("cat_constant", "MISSING"))
                        desc = f"Constant Imputation (fill_value='{fill_val}')"
                    elif strategy == "indicator":
                        fill_val = col_override.get("constant_value", imp_opts.get("cat_constant", "MISSING"))
                        desc = f"Indicator Feature + Constant ('{fill_val}')"
                    elif strategy == "none":
                        fill_val = None
                        desc = "No Imputation (missing values retained)"
                    else:
                        fill_val = None
                        desc = f"{strategy.capitalize()} Imputation"

                recs.append({
                    "id": f"rec_impute_{col_name}",
                    "category": "imputation",
                    "priority": "high" if null_pct > 50 else ("medium" if null_pct > 5 else "low"),
                    "target": col_name,
                    "diagnosis": f"Column '{col_name}' ({'numeric' if is_num else 'categorical'}) exhibits {null_count} missing values ({null_pct}%).",
                    "recommendation": f"Applied exact imputation strategy: {desc}.",
                    "action": "impute" if strategy != "none" else "skip_impute",
                    "params": {
                        "column": col_name,
                        "strategy": strategy,
                        "is_numeric": is_num,
                        "null_count": null_count,
                        "null_percent": null_pct,
                        "fill_value": fill_val,
                        "details": desc
                    },
                    "approved": strategy != "none"
                })
    else:
        # Default heuristic recommendations when config is not provided
        for col_name, meta in cols.items():
            null_pct = meta.get("null_percent", 0.0)
            is_num = meta.get("is_numeric", False)
            skew = meta.get("skewness", 0.0)

            if null_pct > 0.0:
                if null_pct > 60.0:
                    recs.append({
                        "id": f"rec_drop_col_{col_name}",
                        "category": "imputation",
                        "priority": "high",
                        "target": col_name,
                        "diagnosis": f"Column '{col_name}' exhibits critical missingness ({null_pct}% nulls).",
                        "recommendation": "Consider dropping or imputing with constant sentinel if required.",
                        "action": "impute",
                        "params": {"column": col_name, "strategy": "constant", "value": "MISSING"},
                        "approved": False
                    })
                elif is_num:
                    if abs(skew) > 1.5:
                        recs.append({
                            "id": f"rec_impute_{col_name}",
                            "category": "imputation",
                            "priority": "medium",
                            "target": col_name,
                            "diagnosis": f"Column '{col_name}' has {null_pct}% missingness and high skewness ({skew}).",
                            "recommendation": "Apply Median Imputation rather than Mean to resist skew bias.",
                            "action": "impute",
                            "params": {"column": col_name, "strategy": "median"},
                            "approved": True
                        })
                    else:
                        recs.append({
                            "id": f"rec_impute_{col_name}",
                            "category": "imputation",
                            "priority": "medium",
                            "target": col_name,
                            "diagnosis": f"Column '{col_name}' exhibits {null_pct}% missingness with balanced distribution.",
                            "recommendation": "Apply Mean or KNN Imputation.",
                            "action": "impute",
                            "params": {"column": col_name, "strategy": "mean"},
                            "approved": True
                        })
                else:
                    recs.append({
                        "id": f"rec_impute_cat_{col_name}",
                        "category": "imputation",
                        "priority": "medium",
                        "target": col_name,
                        "diagnosis": f"Categorical column '{col_name}' has {null_pct}% missing values.",
                        "recommendation": "Apply Mode (most frequent) imputation or set constant sentinel 'MISSING'.",
                        "action": "impute",
                        "params": {"column": col_name, "strategy": "mode"},
                        "approved": True
                    })

    # 3. Outlier check
    for col_name, meta in cols.items():
        is_num = meta.get("is_numeric", False)
        iqr_outliers = meta.get("iqr_outliers", 0)

        if is_num and iqr_outliers > 0:
            if config is not None:
                clean_opts = config.get("cleaning", {})
                handle_outliers = clean_opts.get("handle_outliers", False)
                out_method = clean_opts.get("outlier_method", "iqr")
                out_action = clean_opts.get("outlier_action", "clip")
                out_thresh = clean_opts.get("outlier_threshold", 1.5)

                if handle_outliers:
                    recs.append({
                        "id": f"rec_outliers_{col_name}",
                        "category": "outliers",
                        "priority": "medium" if iqr_outliers < 10 else "high",
                        "target": col_name,
                        "diagnosis": f"Column '{col_name}' contains {iqr_outliers} IQR outliers outside boundary bounds.",
                        "recommendation": f"Applied {out_action.capitalize()} Outlier Handling via {out_method.upper()} (threshold={out_thresh}) as configured.",
                        "action": "clip_outliers" if out_action == "clip" else "exclude_outliers",
                        "params": {"column": col_name, "method": out_method, "action": out_action, "threshold": out_thresh},
                        "approved": True
                    })
                else:
                    recs.append({
                        "id": f"rec_outliers_{col_name}",
                        "category": "outliers",
                        "priority": "low",
                        "target": col_name,
                        "diagnosis": f"Column '{col_name}' contains {iqr_outliers} IQR outliers outside boundary bounds.",
                        "recommendation": "Outlier handling is disabled in selected configuration; outliers retained.",
                        "action": "retain_outliers",
                        "params": {"column": col_name, "method": out_method, "threshold": out_thresh},
                        "approved": False
                    })
            else:
                recs.append({
                    "id": f"rec_outliers_{col_name}",
                    "category": "outliers",
                    "priority": "medium" if iqr_outliers < 10 else "high",
                    "target": col_name,
                    "diagnosis": f"Column '{col_name}' contains {iqr_outliers} IQR outliers outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].",
                    "recommendation": "Winsorize/clip values to boundary bounds or scale with RobustScaler.",
                    "action": "clip_outliers",
                    "params": {"column": col_name, "method": "iqr", "action": "clip", "threshold": 1.5},
                    "approved": True
                })

    # 4. Scaling & Encoding recommendations if config is provided
    if config is not None:
        scale_opts = config.get("scaling", {})
        scale_method = scale_opts.get("method", "standard")
        num_cols = profile.get("numeric_columns", [])
        if num_cols:
            if scale_method != "none":
                recs.append({
                    "id": "rec_scaling",
                    "category": "scaling",
                    "priority": "medium",
                    "target": "numeric_features",
                    "diagnosis": f"Dataset contains {len(num_cols)} numeric features.",
                    "recommendation": f"Applied {scale_method.capitalize()} Scaler across numeric features as configured.",
                    "action": "scale",
                    "params": {"method": scale_method, "columns": scale_opts.get("columns")},
                    "approved": True
                })

        enc_opts = config.get("encoding", {})
        enc_method = enc_opts.get("method", "onehot")
        selected_cols = enc_opts.get("columns")
        excluded_cols = enc_opts.get("exclude_columns") or []
        cat_cols = profile.get("categorical_columns") or [c for c, m in cols.items() if not m.get("is_numeric", False)]

        if enc_method != "none" and cat_cols:
            target_cols = selected_cols if selected_cols is not None else [c for c in cat_cols if c not in excluded_cols]
            recs.append({
                "id": "rec_encoding",
                "category": "encoding",
                "priority": "medium",
                "target": "categorical_features",
                "diagnosis": f"Dataset contains {len(cat_cols)} categorical features. Selected for {enc_method.upper()}: {target_cols}. Excluded from encoding: {excluded_cols}.",
                "recommendation": f"Applied {enc_method.upper()} encoding on {len(target_cols)} features ({', '.join(target_cols) if target_cols else 'none'}). Excluded {len(excluded_cols)} identifier/name columns ({', '.join(excluded_cols) if excluded_cols else 'none'}).",
                "action": "encode",
                "params": {
                    "method": enc_method,
                    "columns": target_cols,
                    "exclude_columns": excluded_cols
                },
                "approved": True
            })

    return recs

def generate_image_recommendations(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyzes visual dataset profile and generates explainable recommendations.
    """
    recs: List[Dict[str, Any]] = []
    total = profile.get("total_images", 0)
    corrupted = profile.get("corrupted_images", 0)
    blurry = profile.get("blurry_images", 0)
    res_dist = profile.get("resolution_distribution", {})

    if corrupted > 0:
        recs.append({
            "id": "rec_corrupt_images",
            "category": "integrity",
            "priority": "critical",
            "target": "dataset",
            "diagnosis": f"Found {corrupted} corrupted/truncated files out of {total} total assets.",
            "recommendation": "Safely quarantine corrupted files to protect training stability.",
            "action": "exclude_corrupt",
            "approved": True
        })

    if blurry > 0:
        pct = round((blurry / max(1, total)) * 100, 1)
        recs.append({
            "id": "rec_blurry_images",
            "category": "sharpness",
            "priority": "high",
            "target": "dataset",
            "diagnosis": f"{blurry} images ({pct}%) exhibit Laplacian variance below 100.0 (noticeable blur).",
            "recommendation": "Apply Unsharp Masking or Bilateral Filtering to restore edge contrast without amplifying noise.",
            "action": "filter",
            "params": {"filter_type": "unsharp"},
            "approved": True
        })

    if len(res_dist) > 1:
        most_common_res = max(res_dist, key=res_dist.get) if res_dist else "512x512"
        w_s, h_s = most_common_res.split("x")
        recs.append({
            "id": "rec_standardize_resolution",
            "category": "geometry",
            "priority": "medium",
            "target": "dataset",
            "diagnosis": f"Dataset has {len(res_dist)} diverse resolutions. Most frequent is {most_common_res}.",
            "recommendation": f"Standardize all images to {w_s}x{h_s} using aspect-ratio-preserving letterboxing.",
            "action": "resize",
            "params": {"width": int(w_s), "height": int(h_s), "preserve_aspect_ratio": True},
            "approved": True
        })

    return recs

class QualityRecommender:
    @staticmethod
    def recommend_tabular(
        profile: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        imputation_summary: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return generate_tabular_recommendations(profile, config=config, imputation_summary=imputation_summary)

    @staticmethod
    def recommend_image(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        return generate_image_recommendations(profile)
