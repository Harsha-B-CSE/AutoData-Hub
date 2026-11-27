"""Vision hardware discovery, session reuse, and actual forward-pass contracts."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from modules.image import gpu_manager as gpu, neural_restoration as restoration


CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
MODEL = "NAFNet (Fast Denoise/Deblur)"


@pytest.fixture(autouse=True)
def isolated_cache():
    gpu.clear_vision_cache()
    yield
    gpu.clear_vision_cache()


@pytest.fixture
def ort(monkeypatch):
    runtime = ModuleType("onnxruntime")
    runtime.get_available_providers = Mock(return_value=[CPU, CUDA])
    runtime.preload_dlls = Mock()
    runtime.SessionOptions = SimpleNamespace
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setitem(sys.modules, "torch", None)
    return runtime


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    monkeypatch.setattr(restoration, "_model_root", lambda: tmp_path)
    monkeypatch.setitem(restoration.MODEL_REGISTRY[MODEL], "sha256", None)
    path = tmp_path / "nafnet.onnx"
    path.write_bytes(b"test graph")
    return path


@pytest.fixture
def identity_session(ort):
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="x", shape=[1, 3, None, None], type="tensor(float)")
    ]
    session.get_outputs.return_value = [SimpleNamespace(name="y")]
    session.run.side_effect = lambda _outputs, inputs: [inputs["x"]]
    session.get_providers.return_value = [CPU]
    ort.InferenceSession = Mock(return_value=session)
    return session


def test_one_probe_with_onnx_cuda_only_and_isolated_provider_lists(ort):
    snapshot = gpu.get_cached_hardware_context()
    assert snapshot is gpu.get_cached_hardware_context()
    assert snapshot["device"] == "cuda" and snapshot["is_gpu"]
    assert snapshot["providers"] == [CUDA, CPU]
    assert snapshot["available_providers"] == [CPU, CUDA]
    assert snapshot["pytorch_device"] == "cpu" and not snapshot["autocast_enabled"]
    assert snapshot["torch_error"] and not snapshot["torch_available"]
    assert "CUDA" in snapshot["status_badge"] and "active" not in snapshot["status_badge"].lower()
    with pytest.raises(TypeError):
        snapshot["device"] = "cpu"
    snapshot["providers"].clear()
    local = gpu.get_vision_execution_context()
    local["providers"].reverse()
    local["available_providers"].clear()
    assert gpu.get_vision_execution_context()["providers"] == [CUDA, CPU]
    assert snapshot["available_providers"] == [CPU, CUDA]
    ort.get_available_providers.assert_called_once_with()
    ort.preload_dlls.assert_called_once_with()


@pytest.mark.parametrize(("cores", "budget"), [(None, 1), (1, 1), (3, 2), (12, 4)])
def test_cpu_thread_budget_reaches_both_runtimes(ort, model_path, monkeypatch, cores, budget):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=Mock(return_value=False))
    torch.set_num_threads = Mock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(gpu.os, "cpu_count", lambda: cores)
    ort.get_available_providers.return_value = [CPU]
    ort.InferenceSession = Mock()

    ctx = gpu.get_vision_execution_context()
    assert ctx["device"] == "cpu" and not ctx["is_gpu"]
    assert ctx["num_threads"] == ctx["cpu_threads"] == budget
    assert f"thread cap: {budget}" in ctx["status_badge"]
    assert gpu.get_vision_inference_session(model_path, ctx["providers"]) is ort.InferenceSession.return_value
    assert gpu.get_vision_execution_context() == ctx
    options = ort.InferenceSession.call_args.kwargs["sess_options"]
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (budget, 1)
    torch.set_num_threads.assert_called_once_with(budget)
    torch.cuda.is_available.assert_called_once_with()
    ort.preload_dlls.assert_not_called()


def test_preload_failure_preserves_torch_but_excludes_onnx_cuda(ort, model_path, monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _: SimpleNamespace(name="Test GPU", total_memory=4_000_000_000),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    ort.preload_dlls.side_effect = OSError("missing CUDA DLL")

    ctx = gpu.get_vision_execution_context()
    assert ctx["is_gpu"] and ctx["pytorch_device"] == "cuda"
    assert ctx["available_providers"] == [CPU, CUDA] and ctx["providers"] == [CPU]
    assert not ctx["cuda_providers_loaded"] and not ctx["onnx_cuda_available"]
    assert "missing CUDA DLL" in ctx["onnxruntime_error"]
    assert "PyTorch CUDA only" in ctx["status_badge"]
    assert "Test GPU" in ctx["status_badge"] and "4,000 MB" in ctx["status_badge"]
    status = restoration.get_neural_model_status(MODEL, execution_context=ctx)
    assert status["device"] == "cpu" and status["device_name"] == "CPU Host"
    assert status["ready"] and status["execution_provider"] == CPU
    ort.preload_dlls.assert_called_once_with()


def test_missing_optional_runtimes_use_opencv(ort, model_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    ctx = gpu.get_vision_execution_context()
    assert ctx["providers"] == [] and not ctx["onnxruntime_available"]
    assert not ctx["is_gpu"] and not ctx["torch_available"]
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    result = restoration.run_neural_restoration(image, MODEL, 1, "cuda", ctx)
    assert result["status"] == "fallback" and result["runtime"] == "opencv_fallback"
    assert result["device"] == "cpu" and result["providers"] == []
    assert result["fallback_reason"] == "onnxruntime_unavailable"
    assert result["image_bgr"].shape == image.shape


@pytest.mark.parametrize("failure", ["preload", "construction", "run"])
def test_cpu_provider_fallback_is_reported_after_run_and_reused(ort, identity_session, model_path, failure):
    if failure == "preload":
        ort.preload_dlls.side_effect = OSError("missing DLL")
    elif failure == "construction":
        ort.InferenceSession.side_effect = [RuntimeError("CUDA init failed"), identity_session]
    else:
        identity_session.get_providers.return_value = [CUDA, CPU]

        def run(_outputs, inputs):
            identity_session.get_providers.return_value = [CPU]
            return [inputs["x"]]

        identity_session.run.side_effect = run

    context = gpu.get_vision_execution_context()
    autocast = Mock(side_effect=AssertionError("ONNX forward entered PyTorch autocast"))
    context.update(pytorch_device="cuda", autocast_context=autocast)
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    for _ in range(2):
        result = restoration.run_neural_restoration(image, MODEL, 1, "cuda", context)
        assert result["status"] == "success" and result["neural_applied"]
        assert result["provider_fallback"] and not result["fallback_applied"]
        assert result["device"] == "cpu" and result["requested_device"] == "cuda"
        assert result["providers"] == [CPU] and result["execution_provider"] == CPU
        assert result["device_name"] == "CPU Host" and CPU in result["message"]
        np.testing.assert_array_equal(result["image_bgr"], image)
    expected = {"preload": [[CPU]], "construction": [[CUDA, CPU], [CPU]], "run": [[CUDA, CPU]]}
    assert [call.kwargs["providers"] for call in ort.InferenceSession.call_args_list] == expected[failure]
    ort.preload_dlls.assert_called_once_with()
    ort.get_available_providers.assert_called_once_with()
    autocast.assert_not_called()


def test_restoration_honors_partial_execution_context(ort, model_path, identity_session):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    result = restoration.run_neural_restoration(image, MODEL, 1, "cuda", {"onnx_cuda_available": False})
    assert result["status"] == "success" and result["provider_fallback"]
    assert ort.InferenceSession.call_args.kwargs["providers"] == [CPU]
    ort.get_available_providers.assert_called_once_with()


def test_session_cache_tracks_graph_sidecar_and_provider_order(ort, model_path):
    ort.InferenceSession = Mock(side_effect=lambda *args, **kwargs: object())
    weights = model_path.with_name("model.data")
    weights.write_bytes(b"v1")

    def get(providers):
        return gpu.get_vision_inference_session(model_path, providers, external_data=weights.name)

    first = get([CUDA, CPU])
    assert get([CUDA, CPU]) is first
    assert get([CPU, CUDA]) is not first
    weights.write_bytes(b"new weights")
    updated = get([CUDA, CPU])
    assert updated is not first
    model_path.write_bytes(b"new graph version")
    assert get([CUDA, CPU]) is not updated
    assert ort.InferenceSession.call_count == 4


def test_restoration_forwards_registry_sidecar_to_session_cache(ort, model_path, identity_session, monkeypatch):
    weights = model_path.with_name("model.data")
    weights.write_bytes(b"weights")
    monkeypatch.setitem(restoration.MODEL_REGISTRY[MODEL], "external_data", weights.name)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    ctx = {"providers": [CPU]}
    for _ in range(2):
        assert restoration.run_neural_restoration(image, MODEL, 1, execution_context=ctx)["neural_applied"]
    assert ort.InferenceSession.call_count == 1
    weights.write_bytes(b"updated weights")
    assert restoration.run_neural_restoration(image, MODEL, 1, execution_context=ctx)["neural_applied"]
    assert ort.InferenceSession.call_count == 2
    ort.get_available_providers.assert_not_called()


def test_real_pytorch_forward_dtype_and_device():
    ctx = gpu.get_vision_execution_context()
    if not ctx["torch_available"]:
        pytest.skip("PyTorch is optional")
    import torch

    device = ctx["pytorch_device"]
    sample = gpu.move_torch_tensor_to_vision_device(torch.ones(2, 16), ctx)
    layer = torch.nn.Linear(16, 8, bias=False).to(device)
    with torch.no_grad():
        layer.weight.fill_(1)
        output = gpu.run_pytorch_vision_forward(layer, sample, execution_context=ctx)
    assert output.device.type == device
    assert output.dtype == (torch.float16 if device == "cuda" else torch.float32)
    assert output.shape == (2, 8) and torch.all(output == 16).item()


def test_classical_badge_never_starts_neural_discovery(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from modules.image.pipeline_image import ImagePipelineBuffer
    from ui.tabs_image import tab_restoration

    probe = Mock(side_effect=AssertionError("Unexpected neural hardware probe"))
    monkeypatch.setattr(tab_restoration, "get_vision_execution_context", probe)
    app = AppTest.from_string(
        "from ui.tabs_image.tab_restoration import render_tab_restoration\n"
        "render_tab_restoration(None)"
    )
    app.session_state["image_buffer"] = ImagePipelineBuffer(np.zeros((8, 12, 3), dtype=np.uint8))
    app.run(timeout=30)
    assert not app.exception
    assert any("CPU Host Mode" in md.value and "OpenCV CPU" in md.value for md in app.markdown)
    probe.assert_not_called()
