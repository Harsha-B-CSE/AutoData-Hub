"""Vision acceleration and live microphone integration tests."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import av
import numpy as np
from streamlit.testing.v1 import AppTest

from modules.image import gpu_manager, neural_restoration
from modules.image.gpu_manager import (
    clear_vision_cache,
    get_vision_execution_context,
    move_torch_tensor_to_vision_device,
)
from modules.voice.booster import calculate_true_peak_dbfs
from modules.voice.device_manager import resolve_execution_device
from ui.tabs_voice.tab_live_webrtc import LiveAudioDenoiseProcessor
from ui.tabs_voice.tab_live_webrtc import _audio_frame_to_mono_float32


def test_vision_execution_context_handles_cpu_only_runtime(monkeypatch):
    thread_calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        set_num_threads=thread_calls.append,
    )
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"]
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    context = get_vision_execution_context()

    assert isinstance(context, dict)
    assert context["device"] == "cpu"
    assert context["pytorch_device"] == "cpu"
    assert context["providers"] == ["CPUExecutionProvider"]
    assert context["execution_provider"] == "CPUExecutionProvider"
    assert context["cuda_available"] is False


def test_vision_execution_context_prefers_cuda_and_nonblocking_tensor(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Test Vision GPU",
        )
    )
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    context = get_vision_execution_context()

    assert context["device"] == "cuda"
    assert context["device_name"] == "Test Vision GPU"
    assert context["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert context["execution_provider"] == "CUDAExecutionProvider"

    class FakeTensor:
        def __init__(self):
            self.call = None

        def to(self, **kwargs):
            self.call = kwargs
            return self

    tensor = FakeTensor()
    assert move_torch_tensor_to_vision_device(tensor, context) is tensor
    assert tensor.call == {"device": "cuda", "non_blocking": True}


def test_vision_context_uses_onnx_cuda_without_pytorch_cuda(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        set_num_threads=lambda _value: None,
    )
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    context = get_vision_execution_context()
    voice_context = resolve_execution_device()

    assert context["device"] == "cuda"
    assert context["pytorch_device"] == "cpu"
    assert context["onnx_cuda_available"] is True
    assert context["providers"][0] == "CUDAExecutionProvider"
    assert voice_context["device"] == "cuda"
    assert voice_context["pytorch_device"] == "cpu"
    assert voice_context["providers"][0] == "CUDAExecutionProvider"


def test_neural_restoration_passes_cuda_provider_order(monkeypatch, tmp_path: Path):
    provider_calls = []

    class FakeSessionOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None
        log_severity_level = None

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            assert sess_options is not None
            self.providers = list(providers)
            provider_calls.append(self.providers)

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="input", shape=[1, 3, None, None], type="tensor(float)"
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name="output")]

        def run(self, _output_names, feeds):
            return [feeds["input"]]

        def get_providers(self):
            return self.providers

    fake_runtime = SimpleNamespace(
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        preload_dlls=lambda: None,
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Test Vision GPU",
        )
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(neural_restoration, "_model_root", lambda: tmp_path)
    monkeypatch.setitem(
        neural_restoration.MODEL_REGISTRY["NAFNet (Fast Denoise/Deblur)"],
        "sha256",
        None,
    )
    (tmp_path / "nafnet.onnx").write_bytes(b"fake-model")
    context = get_vision_execution_context()
    source = np.full((8, 12, 3), 127, dtype=np.uint8)

    result = neural_restoration.run_neural_restoration(
        source,
        model_name="NAFNet (Fast Denoise/Deblur)",
        scale_factor=1,
        device="cuda",
        execution_context=context,
    )

    assert result["status"] == "success"
    assert result["device"] == "cuda"
    assert provider_calls == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ]

    cpu_result = neural_restoration.run_neural_restoration(
        source,
        model_name="NAFNet (Fast Denoise/Deblur)",
        scale_factor=1,
        device="cpu",
        execution_context=context,
    )
    assert cpu_result["device"] == "cpu"
    assert cpu_result["device_name"] == "CPU Host"
    assert provider_calls[-1] == ["CPUExecutionProvider"]


def test_live_audio_processor_accepts_web_rtc_pcm_frame():
    sample_rate = 16_000
    indices = np.arange(320, dtype=np.float32)
    audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * indices / sample_rate)
    pcm = np.round(audio * 32767.0).astype(np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    frame.sample_rate = sample_rate
    processor = LiveAudioDenoiseProcessor(attenuation_db=6.0, boost_db=0.0)

    output = processor.recv(frame)

    assert isinstance(output, av.AudioFrame)
    assert output.sample_rate == sample_rate
    assert output.layout.name == "mono"
    assert output.samples == frame.samples
    assert np.all(np.isfinite(output.to_ndarray()))
    telemetry = processor.telemetry
    assert telemetry["frames_processed"] == 1
    assert telemetry["frames_failed"] == 0
    assert telemetry["last_latency_ms"] >= 0.0
    processor.update_settings(attenuation_db=18.0, boost_db=2.0)
    assert processor.attenuation_db == 18.0
    assert processor.boost_db == 2.0


def test_unsigned_pcm_silence_is_centered_before_live_dsp():
    pcm = np.full((1, 160), 128, dtype=np.uint8)
    frame = av.AudioFrame.from_ndarray(pcm, format="u8", layout="mono")
    frame.sample_rate = 16_000

    mono, sample_rate = _audio_frame_to_mono_float32(frame)

    assert sample_rate == 16_000
    assert np.max(np.abs(mono)) == 0.0


def test_48khz_ten_millisecond_frames_meet_dsp_compute_budget():
    sample_rate = 48_000
    indices = np.arange(480, dtype=np.float32)
    audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * indices / sample_rate)
    pcm = np.round(audio * 32767.0).astype(np.int16).reshape(1, -1)
    processor = LiveAudioDenoiseProcessor(attenuation_db=6.0, boost_db=0.0)

    started = time.perf_counter()
    for _ in range(20):
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = sample_rate
        output = processor.recv(frame)
        assert output.samples == 480
    average_ms = (time.perf_counter() - started) * 1_000.0 / 20

    assert average_ms < 10.0
    assert processor.telemetry["latency_budget_met"] is True


def test_live_microphone_selector_is_reachable_without_file_upload():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=60)
    next(
        radio for radio in app.radio if radio.label == "Data Modality"
    ).set_value("voice").run(timeout=60)

    source_selector = next(
        radio for radio in app.radio if radio.label == "Voice Input Source"
    )
    assert source_selector.options == [
        "📁 Audio File Upload",
        "🎤 Live Microphone Stream (WebRTC)",
    ]
    source_selector.set_value("🎤 Live Microphone Stream (WebRTC)").run(timeout=60)

    assert not app.exception
    assert app.session_state["voice_execution_mode"] == "live_stream"
    assert any(
        "Real-Time Microphone Enhancement" in subheader.value
        for subheader in app.subheader
    )


def test_gpu_manager_caches_cuda_provider_probe_across_session_calls(monkeypatch):
    clear_vision_cache()
    torch_calls = 0
    ort_calls = 0
    preload_calls = 0

    def fake_probe() -> dict:
        nonlocal torch_calls, ort_calls, preload_calls
        torch_calls += 1
        ort_calls += 1
        preload_calls += 1
        return {
            "device": "cuda",
            "pytorch_device": "cuda",
            "device_name": "Cached Test GPU",
            "cuda_available": True,
            "torch_cuda_available": True,
            "onnx_cuda_available": True,
            "cuda_providers_loaded": True,
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "execution_provider": "CUDAExecutionProvider",
            "torch_available": True,
            "torch_error": None,
            "onnxruntime_available": True,
            "onnxruntime_error": None,
        }

    monkeypatch.setattr(gpu_manager, "_probe_vision_execution_context", fake_probe)

    first = get_vision_execution_context()
    second = get_vision_execution_context()

    assert first == second
    assert first is not second
    assert first["cuda_providers_loaded"] is True
    assert torch_calls == 1
    assert ort_calls == 1
    assert preload_calls == 1
    clear_vision_cache()


def test_restoration_tab_displays_hardware_status_badge():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=60)
    next(
        radio for radio in app.radio if radio.label == "Data Modality"
    ).set_value("image").run(timeout=60)

    import cv2

    image = np.zeros((24, 24, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    next(
        uploader
        for uploader in app.get("file_uploader")
        if uploader.key == "file_uploader_image"
    ).upload("badge.png", buffer.tobytes(), "image/png").run(timeout=60)

    assert not app.exception
    assert any("adh-badge" in str(markdown.value) for markdown in app.markdown)


def test_live_audio_processor_limits_hot_stereo_20ms_pcm_frames():
    sample_rate = 48_000
    samples_per_channel = 960  # 20 ms at 48 kHz
    indices = np.arange(samples_per_channel, dtype=np.float32)
    tone = 0.95 * np.sin(2.0 * np.pi * 440.0 * indices / sample_rate)
    stereo_interleaved = np.column_stack((tone, tone)).reshape(1, -1)
    pcm = np.round(stereo_interleaved * 32767.0).astype(np.int16)

    processor = LiveAudioDenoiseProcessor(attenuation_db=0.0, boost_db=6.0)
    outputs = []
    for _ in range(5):
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="stereo")
        frame.sample_rate = sample_rate
        out_frame = processor.recv(frame)
        assert out_frame.samples == samples_per_channel
        outputs.append(out_frame.to_ndarray().reshape(-1).astype(np.float32) / 32767.0)

    combined = np.concatenate(outputs)
    assert calculate_true_peak_dbfs(combined, sample_rate) <= -0.45
    assert processor.telemetry["latency_budget_met"] is True
