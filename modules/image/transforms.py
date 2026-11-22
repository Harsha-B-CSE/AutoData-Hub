import math
from typing import Tuple, Optional, List
import cv2
import numpy as np

INTERPOLATION_MAP = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "nearest": cv2.INTER_NEAREST,
}

def resize_image(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: str = "bilinear",
    preserve_aspect_ratio: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resizes image with optional aspect-ratio-preserving letterboxing.
    """
    interp_flag = INTERPOLATION_MAP.get(interpolation.lower(), cv2.INTER_LINEAR)
    h, w = img.shape[:2]

    if not preserve_aspect_ratio:
        return cv2.resize(img, (target_width, target_height), interpolation=interp_flag)

    # Compute scaling factor preserving aspect ratio
    scale = min(target_width / w, target_height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp_flag)

    # Create padded canvas
    channels = img.shape[2] if len(img.shape) == 3 else 1
    if channels == 1:
        canvas = np.full((target_height, target_width), pad_color[0], dtype=img.dtype)
    elif channels == 3:
        canvas = np.full((target_height, target_width, 3), pad_color, dtype=img.dtype)
    else: # 4 channels
        canvas = np.zeros((target_height, target_width, channels), dtype=img.dtype)

    # Center placement
    pad_top = (target_height - new_h) // 2
    pad_left = (target_width - new_w) // 2

    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas

def crop_image(
    img: np.ndarray,
    crop_type: str = "center",  # "center", "percentage", "bbox"
    target_size: Optional[Tuple[int, int]] = None,
    percentage: float = 0.1,  # margin to crop from all sides
    bbox: Optional[List[int]] = None,  # [ymin, xmin, ymax, xmax]
) -> np.ndarray:
    """
    Applies center crop, percentage crop, or bounding box cropping.
    """
    h, w = img.shape[:2]

    if crop_type == "bbox" and bbox and len(bbox) == 4:
        ymin, xmin, ymax, xmax = bbox
        ymin = max(0, min(ymin, h - 1))
        xmin = max(0, min(xmin, w - 1))
        ymax = max(ymin + 1, min(ymax, h))
        xmax = max(xmin + 1, min(xmax, w))
        return img[ymin:ymax, xmin:xmax]

    elif crop_type == "percentage":
        dy = int(h * percentage)
        dx = int(w * percentage)
        return img[dy:max(dy + 1, h - dy), dx:max(dx + 1, w - dx)]

    elif crop_type == "center" and target_size:
        crop_w, crop_h = target_size
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2
        return img[start_y:start_y + crop_h, start_x:start_x + crop_w]

    return img

def rotate_image(
    img: np.ndarray,
    angle_degrees: float,
    expand_bounds: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Rotates image by arbitrary angle, with auto-expanded bounds so corners aren't clipped.
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    rot_mat = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    if expand_bounds:
        cos = abs(rot_mat[0, 0])
        sin = abs(rot_mat[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]

        return cv2.warpAffine(img, rot_mat, (new_w, new_h), borderValue=pad_color)
    else:
        return cv2.warpAffine(img, rot_mat, (w, h), borderValue=pad_color)

def flip_image(img: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    """
    Flips image horizontally (mode='horizontal'), vertically (mode='vertical'),
    or both (mode='both').
    """
    if mode == "horizontal":
        return cv2.flip(img, 1)
    elif mode == "vertical":
        return cv2.flip(img, 0)
    elif mode == "both":
        return cv2.flip(img, -1)
    return img
