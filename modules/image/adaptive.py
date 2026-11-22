import math
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np

def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimates additive Gaussian noise standard deviation (sigma) using
    Immerkær's fast noise estimation method.
    Fast O(N) convolution invariant to smooth gradients and flat textures.
    """
    h, w = gray.shape[:2]
    if h < 5 or w < 5:
        return 0.0

    kernel = np.array([
        [ 1, -2,  1],
        [-2,  4, -2],
        [ 1, -2,  1]
    ], dtype=np.float32)

    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sigma = np.sum(np.abs(filtered)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
    return float(round(sigma, 2))

def analyze_image_characteristics(img: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes brightness, contrast, noise, sharpness, and resolution
    prior to transformation.
    """
    channels = img.shape[2] if len(img.shape) == 3 else 1
    h, w = img.shape[:2]

    # Extract luminance L* channel in LAB color space and grayscale
    if channels == 1:
        gray = img
        l_chan = img
    elif channels == 4:
        bgr = img[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
    else:  # 3 channels (BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]

    mean_l = float(round(np.mean(l_chan), 2))
    median_l = float(round(np.median(l_chan), 2))
    std_l = float(round(np.std(l_chan), 2))

    p5, p95 = np.percentile(l_chan, [5, 95])
    dynamic_range = float(round(p95 - p5, 2))

    shadow_ratio = float(round(np.mean(l_chan < 40), 3))
    highlight_ratio = float(round(np.mean(l_chan > 220), 3))

    noise_sigma = estimate_noise_sigma(gray)
    sharpness = float(round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2))

    # Diagnostic flags
    is_low_light = (mean_l < 75.0) or (mean_l < 90.0 and shadow_ratio > 0.30)
    is_high_key = (mean_l > 185.0 and highlight_ratio > 0.30)
    is_low_contrast = (std_l < 46.0) or (dynamic_range < 120.0)
    is_noisy = noise_sigma > 5.0
    is_blurry = sharpness < 150.0

    # Image is considered already good when exposure is balanced, contrast is healthy, noise is low, and focus is sharp
    is_already_good = (
        not is_low_light and
        not is_high_key and
        not is_low_contrast and
        not is_noisy and
        sharpness >= 220.0
    )

    return {
        "width": w,
        "height": h,
        "channels": channels,
        "mean_luminance": mean_l,
        "median_luminance": median_l,
        "std_luminance": std_l,
        "dynamic_range": dynamic_range,
        "shadow_ratio": shadow_ratio,
        "highlight_ratio": highlight_ratio,
        "noise_sigma": noise_sigma,
        "sharpness": sharpness,
        "is_low_light": is_low_light,
        "is_high_key": is_high_key,
        "is_low_contrast": is_low_contrast,
        "is_noisy": is_noisy,
        "is_blurry": is_blurry,
        "is_already_good": is_already_good,
    }

def apply_gamma_correction_luminance(bgr_or_gray: np.ndarray, gamma: float) -> np.ndarray:
    """
    Applies gamma correction exclusively to the luminance channel (L* in LAB)
    for color images, brightening shadows without washing highlights or shifting colors.
    """
    if abs(gamma - 1.0) < 0.01:
        return bgr_or_gray

    channels = bgr_or_gray.shape[2] if len(bgr_or_gray.shape) == 3 else 1

    # Precompute lookup table for performance
    lut = np.array([
        np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)

    if channels == 1:
        return cv2.LUT(bgr_or_gray, lut)
    elif channels == 3:
        lab = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.LUT(l, lut)
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    elif channels == 4:
        bgr = bgr_or_gray[:, :, :3]
        alpha = bgr_or_gray[:, :, 3:]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.LUT(l, lut)
        merged = cv2.merge([l, a, b])
        bgr_out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return np.concatenate([bgr_out, alpha], axis=2)

    return bgr_or_gray

def apply_nlm_denoising(img: np.ndarray, h: int = 5) -> np.ndarray:
    """
    Non-Local Means (NLM) denoising.
    Preserves edges, textures, and lines while effectively attenuating noise.
    """
    channels = img.shape[2] if len(img.shape) == 3 else 1
    # Use templateWindowSize=7, searchWindowSize=15 for fast, effective noise suppression
    if channels == 1:
        return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=7, searchWindowSize=15)
    elif channels == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=15)
    elif channels == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:]
        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=15)
        return np.concatenate([denoised, alpha], axis=2)

    return img

def apply_conservative_unsharp_mask(
    img: np.ndarray,
    amount: float = 0.45,
    radius: float = 1.0,
    threshold: float = 2.0
) -> np.ndarray:
    """
    Conservative unsharp masking on luminance channel only.
    Uses coring threshold to avoid sharpening sensor noise or generating edge halos.
    """
    if amount <= 0.01:
        return img

    channels = img.shape[2] if len(img.shape) == 3 else 1

    def sharpen_channel(chan: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(chan, (0, 0), radius)
        diff = chan.astype(np.float32) - blurred.astype(np.float32)
        # Apply coring threshold: ignore micro differences below threshold
        mask = np.abs(diff) >= threshold
        sharpened = chan.astype(np.float32) + (diff * amount * mask)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    if channels == 1:
        return sharpen_channel(img)
    elif channels == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_sharp = sharpen_channel(l)
        merged = cv2.merge([l_sharp, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    elif channels == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_sharp = sharpen_channel(l)
        merged = cv2.merge([l_sharp, a, b])
        bgr_sharp = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return np.concatenate([bgr_sharp, alpha], axis=2)

    return img

def apply_lab_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE exclusively to the L* luminance channel in LAB space.
    Preserves original chrominance (A* and B*) without color cast.
    """
    channels = img.shape[2] if len(img.shape) == 3 else 1
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if channels == 1:
        return clahe.apply(img)
    elif channels == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        merged = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    elif channels == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        merged = cv2.merge([l_clahe, a, b])
        bgr_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return np.concatenate([bgr_clahe, alpha], axis=2)

    return img

PRESET_CONFIGS = {
    "natural": {
        "name": "Natural",
        "description": "Subtle adjustments prioritizing maximum fidelity, zero halos, and texture preservation.",
        "low_light_mean": 65.0,
        "gamma": 0.85,
        "clahe_std_threshold": 38.0,
        "clahe_clip_limit": 1.4,
        "noise_threshold": 6.0,
        "nlm_h": 3,
        "sharpness_threshold": 100.0,
        "unsharp_amount": 0.25,
        "unsharp_radius": 0.8,
    },
    "balanced": {
        "name": "Balanced",
        "description": "Recommended default: harmonious shadow recovery, balanced local contrast, and edge-preserving cleanup.",
        "low_light_mean": 75.0,
        "gamma": 0.75,
        "clahe_std_threshold": 48.0,
        "clahe_clip_limit": 2.0,
        "noise_threshold": 5.0,
        "nlm_h": 5,
        "sharpness_threshold": 180.0,
        "unsharp_amount": 0.45,
        "unsharp_radius": 1.0,
    },
    "strong": {
        "name": "Strong",
        "description": "Deeper shadow recovery, higher contrast lift, and stronger noise suppression for challenging or heavily underexposed captures.",
        "low_light_mean": 85.0,
        "gamma": 0.65,
        "clahe_std_threshold": 55.0,
        "clahe_clip_limit": 2.8,
        "noise_threshold": 4.5,
        "nlm_h": 7,
        "sharpness_threshold": 240.0,
        "unsharp_amount": 0.70,
        "unsharp_radius": 1.2,
    },
}

def adaptive_enhance_image(
    img: np.ndarray,
    preset: str = "balanced",
    custom_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Adaptive image enhancement engine:
    1. Analyzes brightness, contrast, noise, sharpness, and resolution.
    2. Automatically determines necessary operations based on image characteristics and preset.
    3. Executes selected operations in optimal order (Denoise -> Gamma -> CLAHE -> Sharpen).
    4. Skips operations when the image is already good.
    5. Returns processed image and an audit log detailing what was applied and why.
    """
    preset_key = preset.lower() if preset else "balanced"
    if preset_key not in PRESET_CONFIGS:
        preset_key = "balanced"

    cfg = PRESET_CONFIGS[preset_key].copy()
    if custom_overrides:
        cfg.update(custom_overrides)

    # 1. Pre-Analysis
    metrics_before = analyze_image_characteristics(img)
    operations_applied: List[Dict[str, Any]] = []
    skipped_operations: List[Dict[str, Any]] = []

    processed = img.copy()

    # If image is already good, skip heavy processing and preserve authentic details
    if metrics_before["is_already_good"]:
        skipped_operations.append({
            "operation": "all_enhancements",
            "reason": (
                f"Image already has balanced illumination (mean: {metrics_before['mean_luminance']}), "
                f"healthy dynamic range (std: {metrics_before['std_luminance']}), clean signal "
                f"(noise sigma: {metrics_before['noise_sigma']}), and crisp focus (sharpness: {metrics_before['sharpness']})."
            )
        })
        assessment = "Image is already high quality. Preserving original details without unnecessary processing."
        metrics_after = metrics_before.copy()
        audit = {
            "preset": preset_key,
            "preset_name": cfg["name"],
            "assessment": assessment,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "operations_applied": operations_applied,
            "skipped_operations": skipped_operations,
        }
        return processed, audit

    # 2. Non-Local Means Denoising (First so subsequent contrast/sharpness don't amplify noise)
    if metrics_before["noise_sigma"] > cfg["noise_threshold"]:
        # Dynamically scale filter strength h proportional to detected noise level
        h_val = max(int(cfg["nlm_h"]), min(15, int(metrics_before["noise_sigma"] * 0.8)))
        processed = apply_nlm_denoising(processed, h=h_val)
        operations_applied.append({
            "operation": "nlm_denoising",
            "name": "Non-Local Means Denoising",
            "reason": f"Elevated sensor/grain noise detected (noise sigma: {metrics_before['noise_sigma']} > {cfg['noise_threshold']})",
            "parameters": {"h": h_val, "window_size": 15}
        })
    else:
        skipped_operations.append({
            "operation": "nlm_denoising",
            "reason": f"Image has clean signal with low noise (noise sigma: {metrics_before['noise_sigma']} <= {cfg['noise_threshold']})"
        })

    # 3. Gamma Correction (Luminance Shadow Lift for Low-Light / Night Images)
    if metrics_before["is_low_light"] or metrics_before["mean_luminance"] < cfg["low_light_mean"]:
        processed = apply_gamma_correction_luminance(processed, gamma=cfg["gamma"])
        operations_applied.append({
            "operation": "gamma_correction",
            "name": "Luminance Gamma Correction",
            "reason": f"Low-light / underexposed capture detected (mean luminance: {metrics_before['mean_luminance']} < {cfg['low_light_mean']})",
            "parameters": {"gamma": cfg["gamma"], "channel": "LAB_Lightness"}
        })
    else:
        skipped_operations.append({
            "operation": "gamma_correction",
            "reason": f"Image illumination is adequately bright (mean luminance: {metrics_before['mean_luminance']} >= {cfg['low_light_mean']})"
        })

    # 4. CLAHE Contrast Enhancement on LAB Luminance
    # Avoid applying aggressive CLAHE on flat noise-dominated textures
    needs_clahe = (
        (metrics_before["is_low_contrast"] or metrics_before["std_luminance"] < cfg["clahe_std_threshold"]) and
        (metrics_before["dynamic_range"] >= 35.0 or not metrics_before["is_noisy"])
    )
    if needs_clahe:
        clip_lim = float(cfg["clahe_clip_limit"])
        if metrics_before["is_noisy"]:
            clip_lim = min(clip_lim, 1.5)  # Restrain CLAHE on noisy images to prevent amplifying residual grain
        processed = apply_lab_clahe(processed, clip_limit=clip_lim)
        operations_applied.append({
            "operation": "clahe",
            "name": "Smart Local Contrast (CLAHE)",
            "reason": f"Dynamic contrast range requires local equalization (std: {metrics_before['std_luminance']} < {cfg['clahe_std_threshold']})",
            "parameters": {"clip_limit": clip_lim, "channel": "LAB_Lightness"}
        })
    else:
        reason = (
            "Skipping CLAHE on flat noisy image to avoid amplifying residual grain"
            if (metrics_before["is_noisy"] and metrics_before["dynamic_range"] < 35.0)
            else f"Global contrast and dynamic range are already healthy (std: {metrics_before['std_luminance']} >= {cfg['clahe_std_threshold']})"
        )
        skipped_operations.append({
            "operation": "clahe",
            "reason": reason
        })

    # 5. Conservative Unsharp Masking
    # Only sharpen if image is soft and noise is not high
    if metrics_before["sharpness"] < cfg["sharpness_threshold"] and metrics_before["noise_sigma"] <= (cfg["noise_threshold"] + 2.0):
        processed = apply_conservative_unsharp_mask(
            processed,
            amount=cfg["unsharp_amount"],
            radius=cfg["unsharp_radius"],
            threshold=2.0
        )
        operations_applied.append({
            "operation": "unsharp_mask",
            "name": "Conservative Unsharp Masking",
            "reason": f"Soft edge transitions detected (sharpness: {metrics_before['sharpness']} < {cfg['sharpness_threshold']})",
            "parameters": {"amount": cfg["unsharp_amount"], "radius": cfg["unsharp_radius"], "coring_threshold": 2.0}
        })
    else:
        reason = (
            f"Image focus is already sharp (variance: {metrics_before['sharpness']} >= {cfg['sharpness_threshold']})"
            if metrics_before["sharpness"] >= cfg["sharpness_threshold"]
            else f"Skipping sharpening to avoid accentuating noise (noise sigma: {metrics_before['noise_sigma']})"
        )
        skipped_operations.append({
            "operation": "unsharp_mask",
            "reason": reason
        })

    # 6. Post-Analysis Metrics
    metrics_after = analyze_image_characteristics(processed)

    # Human-readable summary
    ops_names = [op["name"] for op in operations_applied]
    if ops_names:
        assessment = f"Applied adaptive enhancements: {', '.join(ops_names)} while preserving natural chrominance."
    else:
        assessment = "No transformations required. Image characteristics already optimal."

    audit = {
        "preset": preset_key,
        "preset_name": cfg["name"],
        "assessment": assessment,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "operations_applied": operations_applied,
        "skipped_operations": skipped_operations,
    }

    return processed, audit
