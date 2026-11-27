"""Hardware and execution-provider resolution for vision workloads."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import types
from typing import Any, Callable, Dict

import streamlit as st


class _HardwareSnapshot(Mapping[str, Any]):
    """Shared read-only discovery metadata; provider lists belong to the reader."""

    def __init__(self, fields: Dict[str, Any]) -> None:
        self._fields = types.MappingProxyType({
            key: tuple(value) if isinstance(value, list) else value
            for key, value in fields.items()
        })

    def __getitem__(self, key: str) -> Any:
        value = self._fields[key]
        return list(value) if isinstance(value, tuple) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)


def _is_real_or_unloaded_module(module_name: str) -> bool:
    """Return True unless sys.modules holds a non-module test double."""
    loaded = sys.modules.get(module_name)
    return loaded is None or isinstance(loaded, types.ModuleType)


def _cpu_thread_budget() -> int:
    """Bound PyTorch and ONNX intra-op CPU threads to avoid worker starvation."""
    return max(1, min((os.cpu_count() or 1) - 1, 4))


def _disabled_autocast() -> Any:
    return nullcontext()


def _cuda_autocast() -> Any:
    import torch  # type: ignore

    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def _probe_vision_execution_context() -> Dict[str, Any]:
    """Discover candidate devices. Only a model session can confirm its providers."""
    cpu_threads = _cpu_thread_budget()
    torch_module = None
    torch_error = None
    torch_cuda_available = False
    device_name = "CPU Host"
    vram_total_mb = None
    try:
        # Import torch before onnxruntime so Windows CUDA/cuDNN DLL paths match.
        import torch  # type: ignore

        torch_module = torch
        if hasattr(torch, "set_num_threads"):
            try:
                torch.set_num_threads(cpu_threads)
            except Exception as exc:
                torch_error = f"{type(exc).__name__}: {exc}"
        torch_cuda_available = bool(torch.cuda.is_available())
        if torch_cuda_available:
            try:
                properties = torch.cuda.get_device_properties(0)
                device_name = str(properties.name)
                vram_total_mb = round(properties.total_memory / 1_000_000)
            except Exception:
                try:
                    device_name = str(torch.cuda.get_device_name(0))
                except Exception:
                    device_name = "CUDA GPU"
    except Exception as exc:
        torch_error = f"{type(exc).__name__}: {exc}"

    available_providers: list[str] = []
    onnxruntime_error = None
    cuda_providers_loaded = False
    onnx_cuda_available = False
    try:
        import onnxruntime as ort  # type: ignore

        available_providers = list(ort.get_available_providers())
        if "CUDAExecutionProvider" in available_providers:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
                # ORT can still reject CUDA when a particular session is constructed.
                cuda_providers_loaded = True
            onnx_cuda_available = True
    except Exception as exc:
        onnxruntime_error = f"{type(exc).__name__}: {exc}"

    providers = []
    if onnx_cuda_available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")

    cuda_available = torch_cuda_available or onnx_cuda_available
    if onnx_cuda_available and not torch_cuda_available:
        device_name = "ONNX Runtime CUDA GPU"
    if cuda_available:
        source = (
            "ONNX CUDA provider advertised" if onnx_cuda_available else "PyTorch CUDA only"
        )
        memory = (
            f"{vram_total_mb:,} MB total VRAM" if vram_total_mb is not None
            else "VRAM not reported"
        )
        status_badge = f"{source}: {device_name} ({memory})"
    else:
        engine = "ONNX CPU" if providers else "OpenCV fallback"
        status_badge = f"CPU Host Mode | {engine} | Neural CPU thread cap: {cpu_threads}"
    return {
        "device": "cuda" if cuda_available else "cpu",
        "is_gpu": cuda_available,
        "pytorch_device": "cuda" if torch_cuda_available else "cpu",
        "device_name": device_name,
        "status_badge": status_badge,
        "vram_total_mb": vram_total_mb,
        "num_threads": cpu_threads,
        "cpu_threads": cpu_threads,
        "cuda_available": cuda_available,
        "torch_cuda_available": torch_cuda_available,
        "onnx_cuda_available": onnx_cuda_available,
        "cuda_providers_loaded": cuda_providers_loaded,
        "autocast_enabled": torch_cuda_available,
        "autocast_dtype": "float16" if torch_cuda_available else None,
        "autocast_context": _cuda_autocast if torch_cuda_available else _disabled_autocast,
        "providers": providers,
        "available_providers": available_providers,
        "execution_provider": providers[0] if providers else "unavailable",
        "torch_available": torch_module is not None,
        "torch_error": torch_error,
        "onnxruntime_available": bool(providers),
        "onnxruntime_error": onnxruntime_error,
    }


@st.cache_resource(show_spinner=False)
def get_cached_hardware_context() -> Mapping[str, Any]:
    """Probe once across sessions without sharing mutable discovery results."""
    return _HardwareSnapshot(_probe_vision_execution_context())


def get_vision_execution_context() -> Dict[str, Any]:
    """Resolve PyTorch and ONNX Runtime acceleration without hard failures."""
    if (
        _is_real_or_unloaded_module("torch")
        and _is_real_or_unloaded_module("onnxruntime")
    ):
        return dict(get_cached_hardware_context())
    return _probe_vision_execution_context()


def _create_onnx_session(
    model_path: str,
    providers: tuple[str, ...],
) -> Any:
    import onnxruntime as ort  # type: ignore

    options = ort.SessionOptions()
    options.intra_op_num_threads = _cpu_thread_budget()
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    try:
        return ort.InferenceSession(
            model_path, sess_options=options, providers=list(providers),
        )
    except Exception:
        if not {"CUDAExecutionProvider", "CPUExecutionProvider"}.issubset(providers):
            raise
        # Some ORT builds raise at CUDA initialization rather than falling back.
        return ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"],
        )


@st.cache_resource(show_spinner=False, max_entries=2)
def _cached_onnx_session(
    model_path: str,
    file_versions: tuple[tuple[str, int, int], ...],
    providers: tuple[str, ...],
) -> Any:
    """Cache ONNX InferenceSession instances to prevent WDDM context thrashing."""
    del file_versions
    return _create_onnx_session(model_path, providers)


def get_vision_inference_session(
    model_path: str | os.PathLike[str],
    providers: list[str] | tuple[str, ...],
    *,
    external_data: str | os.PathLike[str] | None = None,
) -> Any:
    """Reuse a session until its provider order, graph, or external weights change."""
    resolved = Path(model_path).resolve()
    provider_tuple = tuple(providers)
    files = [resolved]
    if external_data is not None:
        files.append((resolved.parent / external_data).resolve())
    versions = []
    for path in files:
        stat = path.stat()
        versions.append((str(path), stat.st_mtime_ns, stat.st_size))
    return _cached_onnx_session(str(resolved), tuple(versions), provider_tuple)


def vision_autocast_context(
    execution_context: Mapping[str, Any] | None = None,
) -> Any:
    """Return FP16 CUDA autocast context when active, otherwise nullcontext."""
    context = (
        get_vision_execution_context() if execution_context is None else execution_context
    )
    if context.get("pytorch_device") == "cuda":
        factory = context.get("autocast_context")
        if callable(factory):
            return factory()
        try:
            return _cuda_autocast()
        except Exception:
            return nullcontext()
    return nullcontext()


def run_pytorch_vision_forward(
    model_or_fn: Callable[..., Any],
    *args: Any,
    execution_context: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a PyTorch vision forward pass inside FP16 autocast when CUDA is active."""
    with vision_autocast_context(execution_context):
        return model_or_fn(*args, **kwargs)


def clear_vision_cache() -> None:
    """Clear cached hardware context and ONNX sessions."""
    get_cached_hardware_context.clear()
    _cached_onnx_session.clear()


def move_torch_tensor_to_vision_device(
    tensor: Any,
    execution_context: Mapping[str, Any] | None = None,
) -> Any:
    """Move a tensor with the non-blocking CUDA contract used by vision models."""
    context = (
        get_vision_execution_context() if execution_context is None else execution_context
    )
    if context.get("pytorch_device") == "cuda":
        return tensor.to(device="cuda", non_blocking=True)
    return tensor.to(device="cpu")
