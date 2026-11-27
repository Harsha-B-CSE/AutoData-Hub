"""Hardware discovery for the decoupled voice enhancement engine."""

from __future__ import annotations

from contextlib import nullcontext
import os
from typing import Any, Callable, Dict


def _disabled_autocast() -> nullcontext:
    return nullcontext()


def resolve_execution_device() -> Dict[str, Any]:
    """Resolve safe PyTorch and ONNX Runtime execution settings.

    Import and driver failures are intentionally non-fatal.  Audio DSP remains
    available on CPU even on hosts where an installed accelerator runtime is
    blocked by policy or has mismatched native libraries.
    """

    cpu_count = os.cpu_count() or 1
    cpu_threads = max(1, min(cpu_count - 1, 4))
    torch_module = None
    torch_error = None
    torch_cuda_available = False

    try:
        import torch  # type: ignore

        torch_module = torch
        torch_cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # native DLL/driver failures are expected here
        torch_error = f"{type(exc).__name__}: {exc}"

    ort_error = None
    available_providers: list[str] = []
    try:
        import onnxruntime as ort  # type: ignore

        available_providers = list(ort.get_available_providers())
    except Exception as exc:
        ort_error = f"{type(exc).__name__}: {exc}"

    cpu_provider_available = "CPUExecutionProvider" in available_providers
    onnx_cuda_available = "CUDAExecutionProvider" in available_providers

    device_name = "CPU Host"
    vram_allocated_mb = 0.0
    autocast_factory: Callable[[], Any] = _disabled_autocast

    if torch_cuda_available and torch_module is not None:
        try:
            device_name = str(torch_module.cuda.get_device_name(0))
        except Exception:
            device_name = "CUDA GPU"
        try:
            vram_allocated_mb = float(
                torch_module.cuda.memory_allocated(0) / (1024.0 * 1024.0)
            )
        except Exception:
            vram_allocated_mb = 0.0

        def _cuda_autocast() -> Any:
            return torch_module.cuda.amp.autocast(dtype=torch_module.float16)

        autocast_factory = _cuda_autocast
    else:
        if torch_module is not None:
            try:
                torch_module.set_num_threads(cpu_threads)
            except Exception as exc:
                torch_error = torch_error or f"{type(exc).__name__}: {exc}"

    providers = []
    if onnx_cuda_available:
        providers.append("CUDAExecutionProvider")
    if cpu_provider_available:
        providers.append("CPUExecutionProvider")
    cuda_available = torch_cuda_available or onnx_cuda_available
    device = "cuda" if cuda_available else "cpu"
    if onnx_cuda_available and not torch_cuda_available:
        device_name = "ONNX Runtime CUDA GPU"

    execution_provider = providers[0] if providers else "unavailable"
    return {
        "device": device,
        "pytorch_device": "cuda" if torch_cuda_available else "cpu",
        "device_name": device_name,
        "providers": providers,
        "available_providers": available_providers,
        "execution_provider": execution_provider,
        "vram_allocated_mb": round(vram_allocated_mb, 3),
        "cpu_threads": cpu_threads,
        "cuda_available": cuda_available,
        "torch_cuda_available": torch_cuda_available,
        "onnx_cuda_available": onnx_cuda_available,
        "autocast_enabled": torch_cuda_available,
        "autocast_dtype": "float16" if torch_cuda_available else None,
        "autocast_context": autocast_factory,
        "torch_available": torch_module is not None,
        "torch_error": torch_error,
        "onnxruntime_available": bool(available_providers),
        "onnxruntime_error": ort_error,
    }


def hardware_telemetry(device_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Return only serializable hardware fields suitable for ModuleResult."""

    keys = (
        "device",
        "device_name",
        "vram_allocated_mb",
        "execution_provider",
        "providers",
        "cpu_threads",
        "cuda_available",
        "onnx_cuda_available",
        "autocast_enabled",
    )
    return {key: device_ctx.get(key) for key in keys}
