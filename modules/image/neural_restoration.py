"""Optional ONNX image restoration with an audited OpenCV fallback."""

from __future__ import annotations

import os
import importlib.util
import hashlib
import sys
from functools import lru_cache
from pathlib import Path
import time
from typing import Any, Dict

import cv2
import numpy as np

from modules.image.adaptive import apply_conservative_unsharp_mask
from modules.image.clahe_contrast import apply_clahe_contrast
from modules.image.denoiser import apply_bilateral_denoise


MODEL_REGISTRY = {
    "NAFNet (Fast Denoise/Deblur)": {
        "filename": "nafnet.onnx",
        "task": "denoise_deblur",
        "native_scale": 1,
        "sha256": "06b6558d8890db9010c1fc524e26552d01d43fd3ba0c935507e5e9fc33876d76",
        "spatial_multiple": 16,
        "minimum_size": 64,
        "source": "https://github.com/rock1110/nafnet-onnx/releases/tag/v1.0.0",
        "license": "unspecified-third-party-export",
        "recommendation_label": "Fast 1x denoise",
        "recommended_for": "Fast 1× denoising and mild blur cleanup",
        "use_cases": ["sensor noise", "soft photos", "pre-upscale cleanup"],
        "avoid_for": "Resolution enlargement or commercial redistribution without license review",
    },
    "Restormer (High-Res Deblur)": {
        "filename": "restormer.onnx",
        "task": "deblur",
        "native_scale": 1,
        "sha256": "63759916b606b0dace088baf983b6379bacc441e79411cb1af915131eb763653",
        "spatial_multiple": 8,
        "minimum_size": 32,
        "source": "https://github.com/TechFlipsi/FlipsiColor-Models/releases/tag/v0.2",
        "license": "unspecified-third-party-export",
        "recommendation_label": "High-quality 1x deblur",
        "recommended_for": "High-quality 1× deblurring and restoration",
        "use_cases": ["camera shake", "motion blur", "high-resolution photos"],
        "avoid_for": "Fast batch upscaling or commercial redistribution without license review",
    },
    "Real-ESRGAN (4x Super-Res)": {
        "filename": "real_esrgan_x4.onnx",
        "task": "super_resolution",
        "native_scale": 4,
        "sha256": "2d1810f8b75c6a899487decf933697241eafbe6327d7b723674ba6df73a08fda",
        "external_data": "model.data",
        "external_data_sha256": "512d0ec9940c2e9d85d27f2952f12a0b77b7841dc22df4ce9f3ea458bc98f37f",
        "source": "https://huggingface.co/TheGuy444/Real-ESRGAN-ONNX",
        "license": "BSD-3-Clause",
        "recommendation_label": "General-purpose 4x",
        "recommended_for": "General-purpose 4× super-resolution",
        "use_cases": ["mixed photos", "illustrations", "small web images"],
        "avoid_for": "Forensic workflows where generated texture is unacceptable",
    },
    "SwinIR (Transformer)": {
        "filename": "swinir.onnx",
        "task": "restoration",
        "native_scale": 4,
        "sha256": "50a7b6fcaad6c4f342c5e1d059916a3e5d48e89d1d311270a476a7ce6a2ea09f",
        "spatial_multiple": 64,
        "minimum_size": 64,
        "source": "https://huggingface.co/rocca/swin-ir-onnx",
        "license": "Apache-2.0",
        "recommendation_label": "Detailed transformer 4x",
        "recommended_for": "Detailed transformer-based 4× restoration",
        "use_cases": ["natural detail", "architecture", "structured textures"],
        "avoid_for": "Low-latency CPU batch processing",
    },
    "4x UltraSharp V2 (Experimental, Non-Commercial)": {
        "filename": "ultrasharp_v2_x4.onnx",
        "task": "super_resolution",
        "native_scale": 4,
        "sha256": "6c0201e3403745f39a9aa5273c50ed084cf6c4af5a71a4654c30252cf19bf0a5",
        "spatial_multiple": 8,
        "minimum_size": 64,
        "source": "https://huggingface.co/Kim2091/UltraSharpV2",
        "license": "CC-BY-NC-SA-4.0",
        "commercial_use": False,
        "experimental": True,
        "recommendation_label": "Aggressive sharp 4x (NC)",
        "recommended_for": "Aggressive, crisp 4× enhancement for non-commercial work",
        "use_cases": ["game assets", "digital art", "already-clean images"],
        "avoid_for": "Commercial use, subtle portraits, or slow CPU workflows",
    },
    "DAT Nomos8k (Community x4)": {
        "filename": "dat_nomos8k_x4.onnx",
        "task": "super_resolution",
        "native_scale": 4,
        "sha256": "e5c10de92a14544764ca4e4dc0269f7de3cd2e4975b1d149ad687fd195eb9de1",
        "spatial_multiple": 32,
        "minimum_size": 64,
        "source": "https://huggingface.co/huggingworld/onnx-image-models",
        "license": "MIT hub card; community checkpoint provenance",
        "experimental": True,
        "recommendation_label": "Compressed web/JPEG 4x",
        "recommended_for": "Experimental 4× restoration of compressed web imagery",
        "use_cases": ["JPEG artifacts", "social media images", "compressed screenshots"],
        "avoid_for": "Production use without output review or latency-sensitive CPU jobs",
    },
    "RealPLKSR (4x, MIT)": {
        "filename": "realplksr_x4.onnx",
        "task": "super_resolution",
        "native_scale": 4,
        "sha256": "8ce692652494e2e5c31e1f7c7ab61eb084e9a450cde727a374f5d6746cfe054b",
        "tile_overlap": 16,
        "source": "https://huggingface.co/darktable-org/upscale-realplksr-onnx",
        "license": "MIT",
        "recommendation_label": "Photos & portraits 4x",
        "recommended_for": "Natural-looking 4× photographic enhancement",
        "use_cases": ["portraits", "photographs", "skin and natural textures"],
        "avoid_for": "Real-time CPU processing or highly stylized artwork",
    },
}


def _model_root() -> Path:
    configured = os.getenv("AUTODATA_IMAGE_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "models" / "image_restoration"


def get_neural_model_status(model_name: str) -> Dict[str, Any]:
    """Return local weight/runtime readiness without importing ONNX Runtime."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported neural model: {model_name}")
    config = MODEL_REGISTRY[model_name]
    model_path = _model_root() / config["filename"]
    external_path = (
        _model_root() / config["external_data"]
        if config.get("external_data")
        else None
    )
    weights_available = model_path.is_file() and (
        external_path is None or external_path.is_file()
    )
    integrity_valid = weights_available
    if integrity_valid and config.get("sha256"):
        integrity_valid = _sha256(model_path) == config["sha256"]
    if (
        integrity_valid
        and external_path is not None
        and config.get("external_data_sha256")
    ):
        integrity_valid = (
            _sha256(external_path) == config["external_data_sha256"]
        )
    try:
        runtime_available = importlib.util.find_spec("onnxruntime") is not None
    except (ImportError, ValueError):
        runtime_available = "onnxruntime" in sys.modules
    available_providers = []
    if runtime_available:
        try:
            import onnxruntime as ort

            available_providers = list(ort.get_available_providers())
        except (ImportError, OSError):
            runtime_available = False
    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "weights_available": weights_available,
        "integrity_valid": integrity_valid,
        "runtime_available": runtime_available,
        "available_providers": available_providers,
        "cuda_available": "CUDAExecutionProvider" in available_providers,
        "task": config.get("task"),
        "native_scale": int(config.get("native_scale", 1)),
        "license": config.get("license", "unspecified"),
        "commercial_use": bool(config.get("commercial_use", True)),
        "experimental": bool(config.get("experimental", False)),
        "source": config.get("source"),
        "recommendation_label": config.get("recommendation_label", "Restoration"),
        "recommended_for": config.get("recommended_for", "Image restoration"),
        "use_cases": list(config.get("use_cases", [])),
        "avoid_for": config.get("avoid_for"),
        "fallback_only": bool(config.get("fallback_only", False)),
        "availability_note": config.get("availability_note"),
        "ready": weights_available and integrity_valid and runtime_available,
    }


@lru_cache(maxsize=32)
def _sha256_cached(path_string: str, modified_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path), stat.st_mtime_ns, stat.st_size)


def _validate_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("A non-empty image is required.")
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.ndim != 3 or image_bgr.shape[2] not in (3, 4):
        raise ValueError("Neural restoration expects BGR or BGRA image data.")
    if image_bgr.dtype != np.uint8:
        image_bgr = cv2.normalize(
            image_bgr, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    return image_bgr


def _opencv_fallback(
    image_bgr: np.ndarray,
    model_name: str,
    scale_factor: int,
) -> np.ndarray:
    restored = image_bgr.copy()
    restored = apply_bilateral_denoise(
        restored,
        d=7,
        sigma_color=45.0,
        sigma_space=45.0,
    )
    restored = apply_clahe_contrast(
        restored,
        clip_limit=1.8,
        tile_grid_size=(8, 8),
    )
    restored = apply_conservative_unsharp_mask(
        restored,
        amount=0.35,
        radius=1.0,
        threshold=2.0,
    )
    if scale_factor > 1:
        restored = cv2.resize(
            restored,
            (
                restored.shape[1] * scale_factor,
                restored.shape[0] * scale_factor,
            ),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return restored


def _fallback_result(
    image_bgr: np.ndarray,
    model_name: str,
    scale_factor: int,
    device: str,
    reason: str,
    started: float,
) -> Dict[str, Any]:
    fallback = _opencv_fallback(image_bgr, model_name, scale_factor)
    return {
        "image_bgr": fallback,
        "status": "fallback",
        "neural_applied": False,
        "fallback_applied": True,
        "fallback_reason": reason,
        "message": (
            f"{model_name} neural inference was unavailable ({reason}). "
            "Applied CLAHE, bilateral denoising, conservative detail enhancement, "
            "and Lanczos scaling instead."
        ),
        "model_name": model_name,
        "scale_factor": scale_factor,
        "device": device,
        "runtime": "opencv_fallback",
        "providers": [],
        "model_path": str(_model_root() / MODEL_REGISTRY[model_name]["filename"]),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def run_neural_restoration(
    image_bgr: np.ndarray,
    model_name: str,
    scale_factor: int = 2,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run a trusted ONNX restoration model or return an explicit OpenCV fallback."""
    started = time.perf_counter()
    image_bgr = _validate_image(image_bgr)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported neural model: {model_name}")
    if scale_factor not in (1, 2, 4):
        raise ValueError("scale_factor must be 1, 2, or 4.")
    device = str(device).lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")

    model_path = _model_root() / MODEL_REGISTRY[model_name]["filename"]
    model_status = get_neural_model_status(model_name)
    if model_status["fallback_only"]:
        return _fallback_result(
            image_bgr,
            model_name,
            scale_factor,
            device,
            "onnx_not_published",
            started,
        )
    if not model_status["weights_available"]:
        return _fallback_result(
            image_bgr,
            model_name,
            scale_factor,
            device,
            "weights_missing",
            started,
        )
    if not model_status["integrity_valid"]:
        return _fallback_result(
            image_bgr,
            model_name,
            scale_factor,
            device,
            "weights_checksum_mismatch",
            started,
        )

    try:
        import onnxruntime as ort
    except (ImportError, OSError):
        return _fallback_result(
            image_bgr,
            model_name,
            scale_factor,
            device,
            "onnxruntime_unavailable",
            started,
        )

    try:
        if device == "cuda" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        available = ort.get_available_providers()
        requested = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        providers = [provider for provider in requested if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
        model_input = session.get_inputs()[0]
        if len(model_input.shape) != 4:
            raise ValueError("model_input_rank")

        alpha = image_bgr[:, :, 3] if image_bgr.shape[2] == 4 else None
        bgr = image_bgr[:, :, :3]
        original_height, original_width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        input_height = model_input.shape[2]
        input_width = model_input.shape[3]
        output_name = session.get_outputs()[0].name
        input_dtype = (
            np.float16 if "float16" in str(model_input.type) else np.float32
        )

        def run_tile(tile_rgb: np.ndarray) -> np.ndarray:
            tensor = np.ascontiguousarray(
                tile_rgb.astype(input_dtype).transpose(2, 0, 1)[None]
                / input_dtype(255.0)
            )
            raw_output = np.asarray(
                session.run([output_name], {model_input.name: tensor})[0]
            )
            if (
                raw_output.ndim != 4
                or raw_output.shape[0] != 1
                or raw_output.shape[1] != 3
            ):
                raise ValueError("model_output_shape")
            if not np.isfinite(raw_output).all():
                raise ValueError("model_output_nonfinite")
            tile_output = raw_output[0].transpose(1, 2, 0).astype(np.float32)
            if float(np.max(tile_output, initial=0.0)) > 2.0:
                tile_output = tile_output / 255.0
            return np.clip(tile_output, 0.0, 1.0)

        tiled = isinstance(input_height, int) and isinstance(input_width, int)
        if tiled:
            overlap = int(MODEL_REGISTRY[model_name].get("tile_overlap", 0))
            if overlap < 0 or overlap * 2 >= min(input_height, input_width):
                raise ValueError("invalid_tile_overlap")
            padded_height = max(original_height, input_height)
            padded_width = max(original_width, input_width)
            pad_height = padded_height - original_height
            pad_width = padded_width - original_width
            border_mode = (
                cv2.BORDER_REFLECT_101
                if original_height > 1 and original_width > 1
                else cv2.BORDER_REPLICATE
            )
            padded_rgb = cv2.copyMakeBorder(
                rgb,
                0,
                pad_height,
                0,
                pad_width,
                border_mode,
            )
            stride_height = input_height - (2 * overlap)
            stride_width = input_width - (2 * overlap)

            def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
                if length <= tile_size:
                    return [0]
                positions = list(range(0, length - tile_size + 1, stride))
                final_position = length - tile_size
                if positions[-1] != final_position:
                    positions.append(final_position)
                return positions

            y_positions = tile_positions(
                padded_rgb.shape[0], input_height, stride_height
            )
            x_positions = tile_positions(
                padded_rgb.shape[1], input_width, stride_width
            )
            if len(y_positions) * len(x_positions) > 1_024:
                raise ValueError("model_tile_limit")
            first_output = run_tile(padded_rgb[:input_height, :input_width])
            graph_scale_y = first_output.shape[0] / input_height
            graph_scale_x = first_output.shape[1] / input_width
            if (
                not graph_scale_x.is_integer()
                or not graph_scale_y.is_integer()
                or int(graph_scale_x) != int(graph_scale_y)
            ):
                raise ValueError("model_output_scale")
            graph_scale = int(graph_scale_x)
            output_accumulator = np.zeros(
                (
                    padded_rgb.shape[0] * graph_scale,
                    padded_rgb.shape[1] * graph_scale,
                    3,
                ),
                dtype=np.float32,
            )
            output_weights = np.zeros(
                output_accumulator.shape[:2] + (1,), dtype=np.float32
            )
            for row_index, y1 in enumerate(y_positions):
                for column_index, x1 in enumerate(x_positions):
                    tile = padded_rgb[
                        y1:y1 + input_height,
                        x1:x1 + input_width,
                    ]
                    tile_output = (
                        first_output
                        if row_index == 0 and column_index == 0
                        else run_tile(tile)
                    )
                    oy1 = y1 * graph_scale
                    ox1 = x1 * graph_scale
                    output_accumulator[
                        oy1:oy1 + tile_output.shape[0],
                        ox1:ox1 + tile_output.shape[1],
                    ] += tile_output
                    output_weights[
                        oy1:oy1 + tile_output.shape[0],
                        ox1:ox1 + tile_output.shape[1],
                    ] += 1.0
            output_float = output_accumulator / np.maximum(output_weights, 1.0)
            output_float = output_float[
                :original_height * graph_scale,
                :original_width * graph_scale,
            ]
        else:
            spatial_multiple = int(
                MODEL_REGISTRY[model_name].get("spatial_multiple", 1)
            )
            minimum_size = int(
                MODEL_REGISTRY[model_name].get("minimum_size", spatial_multiple)
            )
            padded_height = max(
                minimum_size,
                ((original_height + spatial_multiple - 1) // spatial_multiple)
                * spatial_multiple,
            )
            padded_width = max(
                minimum_size,
                ((original_width + spatial_multiple - 1) // spatial_multiple)
                * spatial_multiple,
            )
            pad_height = padded_height - original_height
            pad_width = padded_width - original_width
            if pad_height or pad_width:
                border_mode = (
                    cv2.BORDER_REFLECT_101
                    if original_height > 1 and original_width > 1
                    else cv2.BORDER_REPLICATE
                )
                inference_rgb = cv2.copyMakeBorder(
                    rgb,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    border_mode,
                )
            else:
                inference_rgb = rgb
            output_float = run_tile(inference_rgb)
            padded_scale_y = output_float.shape[0] / inference_rgb.shape[0]
            padded_scale_x = output_float.shape[1] / inference_rgb.shape[1]
            if abs(padded_scale_x - padded_scale_y) > 1e-6:
                raise ValueError("model_output_scale")
            graph_scale = max(1, int(round(padded_scale_x)))
            output_float = output_float[
                :original_height * graph_scale,
                :original_width * graph_scale,
            ]

        expected_native_scale = int(
            MODEL_REGISTRY[model_name].get("native_scale", graph_scale)
        )
        if graph_scale != expected_native_scale:
            raise ValueError("native_scale_mismatch")

        output = np.round(output_float * 255.0).astype(np.uint8)
        restored_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        target_size = (
            original_width * scale_factor,
            original_height * scale_factor,
        )
        if restored_bgr.shape[1::-1] != target_size:
            restored_bgr = cv2.resize(
                restored_bgr,
                target_size,
                interpolation=cv2.INTER_LANCZOS4,
            )
        if alpha is not None:
            resized_alpha = cv2.resize(
                alpha,
                target_size,
                interpolation=cv2.INTER_LINEAR,
            )
            restored_bgr = np.dstack([restored_bgr, resized_alpha])

        return {
            "image_bgr": restored_bgr,
            "status": "success",
            "neural_applied": True,
            "fallback_applied": False,
            "fallback_reason": None,
            "message": f"Applied {model_name} using ONNX Runtime.",
            "model_name": model_name,
            "scale_factor": scale_factor,
            "requested_device": device,
            "device": (
                "cuda"
                if "CUDAExecutionProvider" in session.get_providers()
                else "cpu"
            ),
            "runtime": "onnxruntime",
            "providers": session.get_providers(),
            "graph_scale": graph_scale,
            "native_scale": expected_native_scale,
            "post_resized": scale_factor != expected_native_scale,
            "tiled": tiled,
            "tile_overlap": (
                int(MODEL_REGISTRY[model_name].get("tile_overlap", 0))
                if tiled
                else 0
            ),
            "model_path": str(model_path),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    except Exception as exc:
        return _fallback_result(
            image_bgr,
            model_name,
            scale_factor,
            device,
            f"inference_error:{type(exc).__name__}",
            started,
        )
