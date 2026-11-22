from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import cv2

def compare_tabular_datasets(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comparative metrics between raw and processed tabular datasets:
    - Missingness resolution (reduction in null count and percent)
    - Row count change (deduplication / outlier drop)
    - Distributional stability via 2-sample Kolmogorov-Smirnov test
    """
    before_nulls = int(df_before.isna().sum().sum())
    after_nulls = int(df_after.isna().sum().sum())
    resolved_nulls = max(0, before_nulls - after_nulls)
    resolution_pct = round((resolved_nulls / max(1, before_nulls)) * 100.0, 2) if before_nulls > 0 else 100.0

    row_delta = len(df_after) - len(df_before)

    # Kolmogorov-Smirnov test for shared numeric columns
    ks_results: Dict[str, Dict[str, Any]] = {}
    common_num_cols = [
        c for c in df_before.select_dtypes(include=[np.number]).columns
        if c in df_after.columns and pd.api.types.is_numeric_dtype(df_after[c])
    ]

    for col in common_num_cols:
        s_bef = df_before[col].dropna()
        s_aft = df_after[col].dropna()
        if len(s_bef) > 5 and len(s_aft) > 5:
            stat, p_val = stats.ks_2samp(s_bef, s_aft)
            # p_val < 0.05 indicates distribution changed significantly (e.g. via scaling or imputation)
            ks_results[col] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
                "distribution_shifted": bool(p_val < 0.05),
            }

    return {
        "missingness": {
            "before_count": before_nulls,
            "after_count": after_nulls,
            "resolved_count": resolved_nulls,
            "resolution_percent": resolution_pct,
        },
        "row_delta": row_delta,
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "cols_before": len(df_before.columns),
        "cols_after": len(df_after.columns),
        "kolmogorov_smirnov": ks_results,
    }

def compute_psnr_and_ssim(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) and approximate SSIM between original and processed image.
    """
    # Resize img2 to img1 if sizes differ for comparison
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # PSNR
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)

    # Simplified SSIM for efficiency
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim = float(np.mean(ssim_map))

    return {
        "psnr_db": round(float(psnr), 2),
        "ssim": round(max(-1.0, min(1.0, ssim)), 4),
        "mse": round(float(mse), 2),
    }

def compare_image_pairs(before_profiles: List[Dict[str, Any]], after_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares image datasets pre and post processing.
    """
    total_before = len(before_profiles)
    total_after = len(after_profiles)

    corrupt_before = sum(1 for p in before_profiles if p.get("corrupted", False))
    corrupt_after = sum(1 for p in after_profiles if p.get("corrupted", False))

    blurry_before = sum(1 for p in before_profiles if not p.get("corrupted", False) and p.get("is_blurry", False))
    blurry_after = sum(1 for p in after_profiles if not p.get("corrupted", False) and p.get("is_blurry", False))

    sharpness_before = np.mean([p.get("laplacian_variance", 0.0) for p in before_profiles if not p.get("corrupted", False)] or [0])
    sharpness_after = np.mean([p.get("laplacian_variance", 0.0) for p in after_profiles if not p.get("corrupted", False)] or [0])

    return {
        "images_before": total_before,
        "images_after": total_after,
        "corrupt_resolved": max(0, corrupt_before - corrupt_after),
        "blurry_resolved": max(0, blurry_before - blurry_after),
        "avg_sharpness_before": round(float(sharpness_before), 2),
        "avg_sharpness_after": round(float(sharpness_after), 2),
        "sharpness_delta": round(float(sharpness_after - sharpness_before), 2),
    }

class QualityComparator:
    @staticmethod
    def compare_tabular(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        return compare_tabular_datasets(df_before, df_after)

    @staticmethod
    def compare_images(before_profiles: List[Dict[str, Any]], after_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        return compare_image_pairs(before_profiles, after_profiles)
