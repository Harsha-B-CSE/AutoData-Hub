import random
from typing import Dict, Any, List
import cv2
import numpy as np

def apply_color_jitter(
    img: np.ndarray,
    brightness: float = 0.2, # delta factor +/-
    contrast: float = 0.2,
    saturation: float = 0.2
) -> np.ndarray:
    """
    Adjusts brightness, contrast, and saturation.
    """
    # Contrast & Brightness: output = alpha * input + beta
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta = random.uniform(-brightness, brightness) * 255.0
    jittered = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Saturation (if 3-channel BGR)
    if len(jittered.shape) == 3 and jittered.shape[2] == 3 and saturation > 0:
        hsv = cv2.cvtColor(jittered, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_mult = 1.0 + random.uniform(-saturation, saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
        jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return jittered

def add_gaussian_noise(
    img: np.ndarray,
    mean: float = 0.0,
    sigma: float = 15.0
) -> np.ndarray:
    """
    Injects additive Gaussian noise: I_noisy = clip(I + N(mean, sigma^2), 0, 255).
    """
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def augment_image(
    img: np.ndarray,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    color_jitter: bool = False,
    gaussian_noise: bool = False,
    jitter_params: Dict[str, float] = None,
    noise_sigma: float = 12.0,
    random_seed: int = None,
) -> np.ndarray:
    """
    Parametric synthetic augmentation for computer vision training pipelines.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    augmented = img.copy()

    if horizontal_flip and random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    if vertical_flip and random.random() > 0.5:
        augmented = cv2.flip(augmented, 0)

    if color_jitter:
        params = jitter_params or {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15}
        augmented = apply_color_jitter(augmented, **params)

    if gaussian_noise:
        augmented = add_gaussian_noise(augmented, sigma=noise_sigma)

    return augmented
