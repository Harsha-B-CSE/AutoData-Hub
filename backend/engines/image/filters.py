from typing import Optional, Tuple
import cv2
import numpy as np

def convert_color_space(img: np.ndarray, target_space: str = "RGB") -> np.ndarray:
    """
    Inter-converts between color spaces while maintaining OpenCV's standard BGR memory
    representation for color images so that cv2.imwrite, filters, and transforms preserve
    natural RGB color fidelity without inverted channels.
    """
    target = target_space.upper()
    channels = img.shape[2] if len(img.shape) == 3 else 1

    if channels == 1:
        if target in ["RGB", "BGR"]:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif target == "HSV":
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return img

    elif channels == 3:
        if target in ["RGB", "BGR"]:
            # Standard 3-channel color image; keep in standard BGR layout for OpenCV operations & saving
            return img
        elif target in ["GRAY", "GRAYSCALE"]:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif target == "HSV":
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    elif channels == 4:
        if target in ["RGB", "BGR"]:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif target in ["GRAY", "GRAYSCALE"]:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif target == "HSV":
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    return img

def apply_filter(
    img: np.ndarray,
    filter_type: str = "none",  # "gaussian", "median", "bilateral", "unsharp", "none"
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Denoising and sharpening filters.
    Preserves 4-channel alpha transparency and detail-preserving parameters for bilateral filtering.
    """
    if filter_type == "none" or not filter_type:
        return img

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if k < 1:
        k = 1

    channels = img.shape[2] if len(img.shape) == 3 else 1

    if filter_type == "gaussian":
        return cv2.GaussianBlur(img, (k, k), sigma)

    elif filter_type == "median":
        return cv2.medianBlur(img, k)

    elif filter_type == "bilateral":
        # Bilateral filter for edge-preserving smoothing.
        # cv2.bilateralFilter only supports 1 or 3 channels.
        # Balanced sigmaColor=40, sigmaSpace=40 preserves fine architectural edges without color smearing.
        if channels == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3:]
            filtered_bgr = cv2.bilateralFilter(bgr, d=k, sigmaColor=40, sigmaSpace=40)
            return np.concatenate([filtered_bgr, alpha], axis=2)
        return cv2.bilateralFilter(img, d=k, sigmaColor=40, sigmaSpace=40)

    elif filter_type == "unsharp":
        # Unsharp masking: original + amount * (original - blurred)
        if channels == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3:]
            gaussian_bgr = cv2.GaussianBlur(bgr, (k, k), sigma)
            sharpened_bgr = cv2.addWeighted(bgr, 1.5, gaussian_bgr, -0.5, 0)
            return np.concatenate([sharpened_bgr, alpha], axis=2)
        gaussian = cv2.GaussianBlur(img, (k, k), sigma)
        return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    return img

def enhance_contrast(
    img: np.ndarray,
    method: str = "clahe",  # "clahe", "equalize", "none"
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast enhancement via Standard Histogram Equalization or CLAHE.
    Applied strictly to the luminance (L* channel in LAB color space) for color images,
    preserving original chrominance (A* and B* channels), lighting tones, and architecture.
    """
    if method == "none" or not method:
        return img

    channels = img.shape[2] if len(img.shape) == 3 else 1

    if channels == 1:
        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(img)
        elif method == "equalize":
            return cv2.equalizeHist(img)

    elif channels == 3:
        # Convert BGR to LAB color space
        # L* represents perceptual luminance/lightness; A* and B* represent chrominance.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
        elif method == "equalize":
            l = cv2.equalizeHist(l)

        # Merge back with the original untouched chroma channels
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    elif channels == 4:
        # Separate BGR and Alpha, apply LAB CLAHE on BGR, re-merge Alpha
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
        elif method == "equalize":
            l = cv2.equalizeHist(l)

        merged = cv2.merge([l, a, b])
        bgr_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return np.concatenate([bgr_enhanced, alpha], axis=2)

    return img

def normalize_pixels(
    img: np.ndarray,
    mode: str = "scale_0_1"  # "scale_0_1", "imagenet", "none"
) -> np.ndarray:
    """
    Floating-point rescaling.
    """
    if mode == "scale_0_1":
        return img.astype(np.float32) / 255.0
    elif mode == "imagenet":
        # Scale to [0,1], then subtract ImageNet mean and divide by std
        norm = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        if len(norm.shape) == 3 and norm.shape[2] == 3:
            norm = (norm - mean) / std
        return norm
    return img
