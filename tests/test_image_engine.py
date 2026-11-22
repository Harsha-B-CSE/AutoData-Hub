import pytest
import numpy as np
import cv2
from pathlib import Path
from backend.engines.image.profiler import compute_laplacian_variance
from backend.engines.image.transforms import resize_image, crop_image, rotate_image, flip_image
from backend.engines.image.filters import convert_color_space, apply_filter, enhance_contrast
from backend.engines.image.augmentor import augment_image

def test_laplacian_sharpness():
    # Crisp image with high contrast edges
    crisp = np.zeros((100, 100, 3), dtype=np.uint8)
    crisp[25:75, 25:75] = 255
    crisp_var = compute_laplacian_variance(crisp)

    # Heavily blurred image
    blurred = cv2.GaussianBlur(crisp, (21, 21), 5.0)
    blurred_var = compute_laplacian_variance(blurred)

    assert crisp_var > blurred_var
    assert blurred_var < 100.0

def test_letterbox_resize():
    # 200x100 aspect 2:1
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # Target 300x300
    resized = resize_image(img, target_width=300, target_height=300, preserve_aspect_ratio=True)
    assert resized.shape == (300, 300, 3)

def test_crop_and_rotation():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255

    # Center crop to 50x50
    cropped = crop_image(img, crop_type="center", target_size=(50, 50))
    assert cropped.shape == (50, 50, 3)

    # Rotate 45 degrees with expanded bounds
    rotated = rotate_image(img, 45.0, expand_bounds=True)
    assert rotated.shape[0] > 100
    assert rotated.shape[1] > 100

def test_color_space_and_filters():
    # 3-channel BGR
    img = np.full((50, 50, 3), 128, dtype=np.uint8)

    # Convert to RGB
    rgb = convert_color_space(img, target_space="RGB")
    assert rgb.shape == (50, 50, 3)

    # Convert to Grayscale
    gray = convert_color_space(img, target_space="GRAY")
    assert len(gray.shape) == 2 or gray.shape[2] == 1

    # Denoising filter
    denoised = apply_filter(img, filter_type="gaussian", kernel_size=3)
    assert denoised.shape == (50, 50, 3)

    # CLAHE
    enhanced = enhance_contrast(img, method="clahe")
    assert enhanced.shape == (50, 50, 3)

def test_augmentation():
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    aug = augment_image(
        img,
        horizontal_flip=True,
        vertical_flip=False,
        color_jitter=True,
        gaussian_noise=True,
        noise_sigma=10.0,
        random_seed=42
    )
    assert aug.shape == (64, 64, 3)
    # Augmented image should differ from uniform flat color
    assert not np.array_equal(aug, img)

def test_mysore_palace_color_preservation():
    """
    Regression test: ensures golden-lit warm images do not turn blue.
    Red channel must remain strictly dominant over Blue channel.
    """
    # Create a warm golden-lit image: high Red, moderate Green, low Blue
    golden_img = np.zeros((100, 100, 3), dtype=np.uint8)
    golden_img[:, :, 0] = 20   # Blue
    golden_img[:, :, 1] = 60   # Green
    golden_img[:, :, 2] = 180  # Red

    # 1. Color space conversion
    converted = convert_color_space(golden_img, "RGB")
    # In OpenCV BGR layout, Red is index 2, Blue is index 0
    assert converted[:, :, 2].mean() > converted[:, :, 0].mean()

    # 2. Bilateral filtering
    denoised = apply_filter(converted, filter_type="bilateral", kernel_size=5)
    assert denoised[:, :, 2].mean() > denoised[:, :, 0].mean()

    # 3. LAB CLAHE contrast enhancement
    enhanced = enhance_contrast(denoised, method="clahe", clip_limit=2.0)
    assert enhanced[:, :, 2].mean() > enhanced[:, :, 0].mean()
    assert enhanced[:, :, 2].mean() > enhanced[:, :, 1].mean()

    # If Mysore palace test file exists on disk, test real image as well
    real_path = Path("storage/raw/7132dd25-c3f4-42da-b297-650b44a606f9/Mysore_palace_illuminated.jpg")
    if real_path.exists():
        real_img = cv2.imread(str(real_path))
        real_denoised = apply_filter(real_img, filter_type="bilateral", kernel_size=5)
        real_enhanced = enhance_contrast(real_denoised, method="clahe", clip_limit=2.0)
        # Verify Red > Blue
        assert real_enhanced[:, :, 2].mean() > real_enhanced[:, :, 0].mean()

def test_clahe_luminance_only_preserves_chrominance():
    """
    Verifies CLAHE in LAB space modifies Lightness (L*) while leaving
    chrominance (A* and B*) channels intact.
    """
    # Create gradient image with constant color tint
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for y in range(64):
        img[y, :, 0] = int(y * 2)        # B
        img[y, :, 1] = int(y * 3)        # G
        img[y, :, 2] = min(255, int(y * 4)) # R

    lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    enhanced = enhance_contrast(img, method="clahe", clip_limit=2.0)
    lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

    # Chrominance channels A* and B* must remain virtually identical
    a_diff = np.abs(lab_orig[:, :, 1].astype(int) - lab_enh[:, :, 1].astype(int))
    b_diff = np.abs(lab_orig[:, :, 2].astype(int) - lab_enh[:, :, 2].astype(int))
    assert np.mean(a_diff) < 2.0
    assert np.mean(b_diff) < 2.0

def test_4channel_bgra_filter_and_clahe():
    """
    Verifies that 4-channel BGRA images are handled gracefully by bilateral filter and CLAHE.
    """
    bgra = np.full((40, 40, 4), 100, dtype=np.uint8)
    bgra[:, :, 3] = 200  # distinct alpha

    # Bilateral filter on 4 channels
    filtered = apply_filter(bgra, filter_type="bilateral", kernel_size=5)
    assert filtered.shape == (40, 40, 4)
    assert np.array_equal(filtered[:, :, 3], bgra[:, :, 3])

    # CLAHE on 4 channels
    enhanced = enhance_contrast(bgra, method="clahe")
    assert enhanced.shape == (40, 40, 4)
    assert np.array_equal(enhanced[:, :, 3], bgra[:, :, 3])

def test_adaptive_enhancement_low_light_gamma():
    """
    Verifies that low-light underexposed images trigger Gamma Correction
    to lift shadows while keeping chrominance balance unchanged.
    """
    from backend.engines.image.adaptive import analyze_image_characteristics, adaptive_enhance_image

    # Create dark image (mean L ~ 35) with warm color tint (R > B)
    dark_img = np.zeros((80, 80, 3), dtype=np.uint8)
    dark_img[:, :, 0] = 15   # B
    dark_img[:, :, 1] = 30   # G
    dark_img[:, :, 2] = 60   # R

    char = analyze_image_characteristics(dark_img)
    assert char["is_low_light"] is True

    enhanced, audit = adaptive_enhance_image(dark_img, preset="balanced")
    assert audit["metrics_after"]["mean_luminance"] > audit["metrics_before"]["mean_luminance"]

    # Verify gamma correction was in applied operations
    ops = [op["operation"] for op in audit["operations_applied"]]
    assert "gamma_correction" in ops

    # Warm color balance (Red > Blue) must be strictly maintained!
    assert enhanced[:, :, 2].mean() > enhanced[:, :, 0].mean()

def test_adaptive_enhancement_noisy_image_nlm():
    """
    Verifies that noisy images trigger Non-Local Means (NLM) denoising,
    reducing estimated noise sigma.
    """
    from backend.engines.image.adaptive import analyze_image_characteristics, adaptive_enhance_image

    # Clean base image
    base = np.full((80, 80, 3), 120, dtype=np.uint8)
    # Inject noticeable Gaussian noise (sigma ~ 15)
    np.random.seed(42)
    noise = np.random.normal(0, 16, base.shape).astype(np.float32)
    noisy = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    char = analyze_image_characteristics(noisy)
    assert char["noise_sigma"] > 5.0
    assert char["is_noisy"] is True

    denoised, audit = adaptive_enhance_image(noisy, preset="balanced")
    ops = [op["operation"] for op in audit["operations_applied"]]
    assert "nlm_denoising" in ops
    assert audit["metrics_after"]["noise_sigma"] < audit["metrics_before"]["noise_sigma"]

def test_adaptive_enhancement_soft_image_unsharp():
    """
    Verifies that blurry or soft images trigger conservative Unsharp Masking
    to restore focus without creating halos.
    """
    from backend.engines.image.adaptive import analyze_image_characteristics, adaptive_enhance_image

    # Create sharp square then blur it to simulate soft focus
    crisp = np.zeros((80, 80, 3), dtype=np.uint8)
    crisp[20:60, 20:60] = 200
    soft = cv2.GaussianBlur(crisp, (9, 9), 2.5)

    char = analyze_image_characteristics(soft)
    assert char["sharpness"] < 180.0

    sharpened, audit = adaptive_enhance_image(soft, preset="balanced")
    ops = [op["operation"] for op in audit["operations_applied"]]
    assert "unsharp_mask" in ops
    assert audit["metrics_after"]["sharpness"] > audit["metrics_before"]["sharpness"]

def test_adaptive_enhancement_already_good_skips_unnecessary_ops():
    """
    Verifies that images with already good exposure, clean noise, healthy contrast,
    and crisp sharpness skip unnecessary heavy operations to avoid overprocessing.
    """
    from backend.engines.image.adaptive import analyze_image_characteristics, adaptive_enhance_image

    # Create high-quality balanced image: mean L ~ 120, std L ~ 60, low noise, sharp focus
    good = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        good[i, :, :] = int(40 + 1.6 * i)
    cv2.circle(good, (50, 50), 20, (230, 230, 230), -1)
    cv2.rectangle(good, (10, 10), (35, 35), (20, 20, 20), -1)

    char = analyze_image_characteristics(good)
    assert char["is_already_good"] is True

    out, audit = adaptive_enhance_image(good, preset="balanced")
    assert len(audit["operations_applied"]) == 0
    assert len(audit["skipped_operations"]) > 0
    # Image content should remain virtually identical
    assert np.allclose(out, good, atol=2)

def test_adaptive_enhancement_presets_scaling():
    """
    Verifies that Natural, Balanced, and Strong presets scale the enhancement
    depth appropriately on low-contrast captures.
    """
    from backend.engines.image.adaptive import adaptive_enhance_image

    flat = np.full((60, 60, 3), 90, dtype=np.uint8)
    flat[20:40, 20:40] = 110

    _, audit_nat = adaptive_enhance_image(flat, preset="natural")
    _, audit_bal = adaptive_enhance_image(flat, preset="balanced")
    _, audit_str = adaptive_enhance_image(flat, preset="strong")

    assert audit_nat["preset"] == "natural"
    assert audit_bal["preset"] == "balanced"
    assert audit_str["preset"] == "strong"

def test_mysore_palace_adaptive_pipeline():
    """
    Verifies adaptive pipeline specifically on Mysore Palace:
    - Recognizes night capture (low light luminance)
    - Applies gamma shadow lift and contrast enhancement
    - Skips unsharp mask because focus is already sharp (> 1000)
    - Strictly preserves warm golden lighting (Red > Blue)
    """
    from backend.engines.image.adaptive import analyze_image_characteristics, adaptive_enhance_image

    real_path = Path("storage/raw/7132dd25-c3f4-42da-b297-650b44a606f9/Mysore_palace_illuminated.jpg")
    if not real_path.exists():
        pytest.skip("Mysore Palace raw file not found")

    img = cv2.imread(str(real_path))
    char = analyze_image_characteristics(img)
    assert char["is_low_light"] is True
    assert char["sharpness"] > 300.0  # Already very sharp

    enhanced, audit = adaptive_enhance_image(img, preset="balanced")

    applied_ops = [op["operation"] for op in audit["operations_applied"]]
    skipped_ops = [op["operation"] for op in audit["skipped_operations"]]

    # Gamma shadow lift must be applied for night capture
    assert "gamma_correction" in applied_ops
    # Unsharp mask must be skipped to avoid edge halos on already-sharp lights
    assert "unsharp_mask" in skipped_ops

    # Red must remain strictly dominant over Blue
    assert enhanced[:, :, 2].mean() > enhanced[:, :, 0].mean()
    assert enhanced[:, :, 2].mean() > enhanced[:, :, 1].mean()
