import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image

def compute_laplacian_variance(img_bgr: np.ndarray) -> float:
    """
    Computes Laplacian variance as an objective measure of image focus/sharpness.
    Values < 100.0 generally indicate noticeable blur.
    """
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def profile_image(image_path: Path, blur_threshold: float = 100.0) -> Dict[str, Any]:
    """
    Inspects a single image for resolution, color channels, bit depth, sharpness,
    and corruption.
    """
    file_size = image_path.stat().st_size
    if file_size == 0:
        return {
            "path": str(image_path),
            "filename": image_path.name,
            "corrupted": True,
            "error": "Zero byte file",
            "file_size": 0,
        }

    # PIL verification test for truncated/corrupt byte sequence
    try:
        with Image.open(image_path) as pil_img:
            pil_img.verify()
    except Exception as e:
        return {
            "path": str(image_path),
            "filename": image_path.name,
            "corrupted": True,
            "error": f"Corrupt image structure: {str(e)}",
            "file_size": file_size,
        }

    # OpenCV decode
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {
            "path": str(image_path),
            "filename": image_path.name,
            "corrupted": True,
            "error": "Failed to decode raster array with OpenCV",
            "file_size": file_size,
        }

    shape = img.shape
    height = shape[0]
    width = shape[1]
    channels = shape[2] if len(shape) > 2 else 1
    aspect_ratio = round(width / height, 3) if height > 0 else 0.0

    # Color space & bit depth
    dtype = str(img.dtype)
    bit_depth = 16 if "16" in dtype else 8

    color_space = "GRAY"
    if channels == 3:
        color_space = "RGB"
    elif channels == 4:
        color_space = "RGBA"

    sharpness = compute_laplacian_variance(img)
    is_blurry = sharpness < blur_threshold

    return {
        "path": str(image_path),
        "filename": image_path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "aspect_ratio": aspect_ratio,
        "color_space": color_space,
        "bit_depth": bit_depth,
        "laplacian_variance": round(sharpness, 2),
        "is_blurry": is_blurry,
        "corrupted": False,
        "file_size": file_size,
    }

def profile_image_dataset(image_paths: List[Path], blur_threshold: float = 100.0) -> Dict[str, Any]:
    """
    Profiles an entire visual dataset, aggregating statistics:
    - total images, corrupt count, blurry count
    - resolution distributions
    - aspect ratio breakdown
    - average sharpness
    """
    profiles: List[Dict[str, Any]] = []
    total = len(image_paths)
    corrupted_count = 0
    blurry_count = 0
    resolutions = {}
    total_sharpness = 0.0
    valid_count = 0

    for path in image_paths:
        prof = profile_image(path, blur_threshold=blur_threshold)
        profiles.append(prof)
        if prof.get("corrupted", False):
            corrupted_count += 1
        else:
            valid_count += 1
            if prof.get("is_blurry", False):
                blurry_count += 1
            total_sharpness += prof.get("laplacian_variance", 0.0)
            res_key = f"{prof.get('width')}x{prof.get('height')}"
            resolutions[res_key] = resolutions.get(res_key, 0) + 1

    avg_sharpness = round(total_sharpness / valid_count, 2) if valid_count > 0 else 0.0

    return {
        "total_images": total,
        "valid_images": valid_count,
        "corrupted_images": corrupted_count,
        "blurry_images": blurry_count,
        "average_sharpness": avg_sharpness,
        "resolution_distribution": resolutions,
        "sample_profiles": profiles[:50],  # preview top 50
    }

class ImageProfiler:
    @staticmethod
    def profile(image_paths: List[Path], blur_threshold: float = 100.0) -> Dict[str, Any]:
        return profile_image_dataset(image_paths, blur_threshold=blur_threshold)
