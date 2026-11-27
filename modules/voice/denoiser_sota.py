"""Classical spectral gating and optional neural speech enhancement adapters."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
import os
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

from modules.voice.audio_loader import downmix_to_mono, resample_audio
from modules.voice.device_manager import resolve_execution_device


DEEPFILTERNET_SAMPLE_RATE = 16_000
_MODEL_FILENAMES = (
    "deepfilternet3.onnx",
    "dfnet3.onnx",
    "deepfilternet3.pt",
    "deepfilternet3.torchscript",
)


def _voice_model_root() -> Path:
    configured = os.getenv("AUTODATA_VOICE_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "models" / "voice"


def _resolve_model_path(device_ctx: dict[str, Any]) -> Path | None:
    require_onnx = bool(device_ctx.get("require_onnx"))
    explicit = device_ctx.get("model_path") or os.getenv("AUTODATA_DFNET3_MODEL")
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if require_onnx and path.suffix.lower() != ".onnx":
            return None
        return path if path.is_file() else None
    root = _voice_model_root()
    filenames = (
        tuple(name for name in _MODEL_FILENAMES if name.endswith(".onnx"))
        if require_onnx
        else _MODEL_FILENAMES
    )
    return next((root / name for name in filenames if (root / name).is_file()), None)


def get_deepfilternet3_status(device_ctx: dict | None = None) -> Dict[str, Any]:
    """Report local neural model readiness without requiring the model."""

    context = device_ctx or resolve_execution_device()
    model_path = _resolve_model_path(context)
    require_onnx = bool(context.get("require_onnx"))
    runtime_available = (
        bool(context.get("onnxruntime_available"))
        if require_onnx
        else bool(context.get("onnxruntime_available") or context.get("torch_available"))
    )
    return {
        "model_name": "DeepFilterNet3",
        "model_path": str(model_path) if model_path else str(_voice_model_root()),
        "weights_available": model_path is not None,
        "runtime_available": runtime_available,
        "execution_provider": context.get("execution_provider", "unavailable"),
        "device": context.get("device", "cpu"),
        "model_contract": "end_to_end_waveform_16khz_v1",
        "onnx_required": require_onnx,
        "ready": model_path is not None and runtime_available,
    }


def apply_spectral_gating(
    audio_signal: np.ndarray,
    sr: int = 16_000,
    attenuation_db: float = 20.0,
) -> np.ndarray:
    """Suppress stationary noise with a phase-preserving STFT soft mask."""

    del sr  # the mask is expressed in STFT bins and is sample-rate agnostic
    audio = downmix_to_mono(audio_signal)
    if audio.size == 0 or attenuation_db <= 0.0:
        return audio.copy()
    if audio.size < 32:
        return audio.copy()

    attenuation_db = float(np.clip(attenuation_db, 0.0, 60.0))
    nperseg = min(1_024, max(32, 2 ** int(np.floor(np.log2(audio.size)))))
    noverlap = min((3 * nperseg) // 4, nperseg - 1)
    _, _, spectrum = stft(
        audio,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum)
    power = np.square(magnitude, dtype=np.float64)
    noise_floor = np.quantile(power, 0.20, axis=1, keepdims=True)
    over_subtraction = 1.0 + attenuation_db / 30.0
    clean_power = np.maximum(power - over_subtraction * noise_floor, 0.0)
    gain_floor = 10.0 ** (-attenuation_db / 20.0)
    gain = np.sqrt(clean_power / (power + np.finfo(np.float64).eps))
    gain = gain_floor + (1.0 - gain_floor) * gain
    if gain.shape[1] > 1:
        gain = uniform_filter(gain, size=(3, 3), mode="nearest")
    gain = np.clip(gain, gain_floor, 1.0)

    # A real-valued gain changes magnitude only; the original phase is retained.
    _, enhanced = istft(
        spectrum * gain,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    enhanced = np.asarray(enhanced[: audio.size], dtype=np.float32)
    if enhanced.size < audio.size:
        enhanced = np.pad(enhanced, (0, audio.size - enhanced.size))
    return np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)


class StreamingSpectralGate:
    """Stateful, chunk-local spectral gate for low-latency PCM streams."""

    def __init__(self, sample_rate: int = 16_000, attenuation_db: float = 20.0):
        self.sample_rate = int(sample_rate)
        self.attenuation_db = float(attenuation_db)
        self._noise_magnitude: np.ndarray | None = None

    def reset(self) -> None:
        self._noise_magnitude = None

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        chunk = downmix_to_mono(audio_chunk)
        if chunk.size < 16 or self.attenuation_db <= 0:
            return chunk.copy()
        fft_size = max(32, 1 << int(np.ceil(np.log2(chunk.size))))
        spectrum = np.fft.rfft(chunk, n=fft_size)
        magnitude = np.abs(spectrum)
        if self._noise_magnitude is None or self._noise_magnitude.size != magnitude.size:
            baseline = float(np.quantile(magnitude, 0.20))
            self._noise_magnitude = np.minimum(magnitude, baseline)
        else:
            # Minimum-statistics tracking follows rising noise slowly and drops fast.
            falling = np.minimum(self._noise_magnitude, magnitude)
            rising = 0.98 * self._noise_magnitude + 0.02 * magnitude
            self._noise_magnitude = np.where(
                magnitude < self._noise_magnitude, falling, rising
            )
        power = np.square(magnitude)
        noise_power = np.square(self._noise_magnitude)
        clean_power = np.maximum(power - noise_power, 0.0)
        floor = 10.0 ** (-float(np.clip(self.attenuation_db, 0, 60)) / 20.0)
        gain = floor + (1.0 - floor) * np.sqrt(clean_power / (power + 1e-12))
        enhanced = np.fft.irfft(spectrum * gain, n=fft_size)[: chunk.size]
        return np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)


@lru_cache(maxsize=4)
def _load_onnx_session(model_path: str, providers: tuple[str, ...]):
    import onnxruntime as ort  # type: ignore

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=options, providers=list(providers))


def _shape_onnx_waveform(audio: np.ndarray, input_meta: Any) -> np.ndarray:
    rank = len(getattr(input_meta, "shape", []) or [])
    input_type = str(getattr(input_meta, "type", "tensor(float)"))
    if "float" not in input_type:
        raise ValueError("The ONNX waveform input must use a floating-point tensor.")
    dtype = np.float16 if "float16" in input_type else np.float32
    if rank in (0, 1):
        shaped = audio
    elif rank == 2:
        shaped = audio[np.newaxis, :]
    elif rank == 3:
        shaped = audio[np.newaxis, np.newaxis, :]
    else:
        raise ValueError("The ONNX wrapper must expose a rank 1-3 waveform input.")
    return np.ascontiguousarray(shaped, dtype=dtype)


def _extract_model_waveform(output: Any, expected_samples: int) -> np.ndarray:
    if isinstance(output, dict):
        output = next(iter(output.values()))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach().float().cpu().numpy()
    result = np.asarray(output)
    if np.iscomplexobj(result):
        raise ValueError("Model returned a spectrum instead of a decoded waveform.")
    result = np.squeeze(result).astype(np.float32, copy=False)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("Model output is not a mono waveform.")
    if result.size != expected_samples:
        raise ValueError(
            f"Model output length {result.size} does not match input length "
            f"{expected_samples}; an end-to-end waveform wrapper is required."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("Model output contains non-finite audio samples.")
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > 1.05:
        result = result / peak
    return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)


def _run_onnx(
    model_path: Path,
    audio: np.ndarray,
    attenuation_db: float,
    device_ctx: dict[str, Any],
) -> np.ndarray:
    providers = tuple(device_ctx.get("providers") or ["CPUExecutionProvider"])
    session = _load_onnx_session(str(model_path), providers)
    inputs = list(session.get_inputs())
    if not inputs:
        raise ValueError("ONNX model declares no inputs.")
    feed: dict[str, np.ndarray] = {
        inputs[0].name: _shape_onnx_waveform(audio, inputs[0])
    }
    for input_meta in inputs[1:]:
        name = input_meta.name.lower()
        if "atten" in name:
            feed[input_meta.name] = np.asarray([attenuation_db], dtype=np.float32)
        elif "sample" in name and "rate" in name:
            feed[input_meta.name] = np.asarray([DEEPFILTERNET_SAMPLE_RATE], dtype=np.int64)
        else:
            raise ValueError(
                f"Unsupported ONNX input {input_meta.name}; use an end-to-end "
                "DeepFilterNet3 waveform export."
            )
    outputs = session.run(None, feed)
    if not outputs:
        raise ValueError("ONNX model produced no outputs.")
    return _extract_model_waveform(outputs[0], audio.size)


@lru_cache(maxsize=4)
def _load_torchscript(model_path: str, device: str, modified_ns: int):
    del modified_ns
    import torch  # type: ignore

    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def _run_torchscript(
    model_path: Path,
    audio: np.ndarray,
    device_ctx: dict[str, Any],
) -> np.ndarray:
    import torch  # type: ignore

    device = str(device_ctx.get("pytorch_device", device_ctx.get("device", "cpu")))
    model = _load_torchscript(str(model_path), device, model_path.stat().st_mtime_ns)
    tensor = torch.from_numpy(audio).to(device).reshape(1, 1, -1)
    autocast_factory = device_ctx.get("autocast_context")
    autocast = autocast_factory() if callable(autocast_factory) else nullcontext()
    with torch.no_grad(), autocast:
        output = model(tensor)
    return _extract_model_waveform(output, audio.size)


def apply_deepfilternet3_sota(
    audio_signal: np.ndarray,
    sr: int = 16_000,
    attenuation_db: float = 20.0,
    device_ctx: dict | None = None,
) -> Dict[str, Any]:
    """Run a local DeepFilterNet3 wrapper and audit any Tier-1 fallback.

    Supported model artifacts are an end-to-end waveform ONNX export or a
    TorchScript module.  Raw training checkpoints are not silently guessed.
    """

    started = time.perf_counter()
    context: dict[str, Any] = dict(device_ctx or resolve_execution_device())
    audio = downmix_to_mono(audio_signal)
    model_audio = (
        resample_audio(audio, int(sr), DEEPFILTERNET_SAMPLE_RATE)
        if int(sr) != DEEPFILTERNET_SAMPLE_RATE
        else audio
    )
    model_path = _resolve_model_path(context)
    fallback_reason = None
    runtime = None
    enhanced_model_audio = None

    if model_path is None:
        fallback_reason = (
            "onnx_weights_missing" if context.get("require_onnx") else "weights_missing"
        )
    elif model_path.suffix.lower() == ".onnx":
        if not context.get("onnxruntime_available"):
            fallback_reason = "onnxruntime_unavailable"
        else:
            try:
                enhanced_model_audio = _run_onnx(
                    model_path, model_audio, attenuation_db, context
                )
                runtime = "onnxruntime"
            except ValueError:
                fallback_reason = "incompatible_model_contract"
            except Exception as exc:
                fallback_reason = f"inference_error:{type(exc).__name__}"
    elif model_path.suffix.lower() in {".pt", ".torchscript"}:
        if not context.get("torch_available"):
            fallback_reason = "pytorch_unavailable"
        else:
            try:
                enhanced_model_audio = _run_torchscript(
                    model_path, model_audio, context
                )
                runtime = "pytorch"
            except ValueError:
                fallback_reason = "incompatible_model_contract"
            except Exception as exc:
                fallback_reason = f"inference_error:{type(exc).__name__}"
    else:
        fallback_reason = "unsupported_weights"

    if enhanced_model_audio is None:
        enhanced_model_audio = apply_spectral_gating(
            model_audio,
            DEEPFILTERNET_SAMPLE_RATE,
            attenuation_db=attenuation_db,
        )
        runtime = "spectral_gate"
        status = "fallback"
        tier_used = "tier1"
    else:
        status = "success"
        tier_used = "tier2"

    enhanced = (
        resample_audio(enhanced_model_audio, DEEPFILTERNET_SAMPLE_RATE, int(sr))
        if int(sr) != DEEPFILTERNET_SAMPLE_RATE
        else enhanced_model_audio
    )
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    result = {
        "audio": enhanced,
        "enhanced_audio": enhanced,
        "transformed_audio": enhanced,
        "sample_rate": int(sr),
        "status": status,
        "tier_requested": "tier2",
        "tier_used": tier_used,
        "model": "DeepFilterNet3",
        "model_contract": "end_to_end_waveform_16khz_v1",
        "model_path": str(model_path) if model_path else None,
        "runtime": runtime,
        "execution_provider": context.get("execution_provider", "unavailable"),
        "device": context.get("device", "cpu"),
        "elapsed_ms": elapsed_ms,
    }
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    return result
