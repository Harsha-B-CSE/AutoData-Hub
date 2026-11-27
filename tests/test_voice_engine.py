"""Dual-tier audio engine and streaming regression tests."""

from contextlib import nullcontext
import io
import os
from types import SimpleNamespace
import wave

import numpy as np

from core.schema_inference import ExecutionStatus, TransformResult
from modules.voice.audio_loader import (
    _decode_pyav,
    encode_flac_bytes,
    encode_mp3_bytes,
    encode_ogg_bytes,
    encode_wav_bytes,
    load_audio,
    pcm_bytes_to_float32,
    resample_audio,
)
from modules.voice.audio_metrics import (
    compute_sisdr,
    compute_snr_db,
    evaluate_audio_telemetry,
)
from modules.voice.booster import (
    apply_quality_booster,
    apply_true_peak_limiter,
    calculate_true_peak_dbfs,
    estimate_integrated_lufs,
)
from modules.voice.bandwidth_extension import (
    apply_high_freq_reconstruction,
    apply_voicefixer_sota,
)
from modules.voice.denoiser_sota import (
    apply_deepfilternet3_sota,
    apply_spectral_gating,
)
from modules.voice.device_manager import resolve_execution_device
from modules.voice.pipeline_voice import PCMStreamEnhancer, enhance_voice
from modules.voice.tuner import analyze_audio_spectrum, apply_parametric_eq


def _tone(sample_rate: int = 16_000, seconds: float = 0.1) -> np.ndarray:
    samples = np.arange(int(sample_rate * seconds), dtype=np.float32)
    return (0.2 * np.sin(2.0 * np.pi * 440.0 * samples / sample_rate)).astype(
        np.float32
    )


def test_cpu_device_resolution_bounds_threads(monkeypatch):
    thread_calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        set_num_threads=thread_calls.append,
    )
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"]
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    context = resolve_execution_device()

    expected_threads = max(1, min((os.cpu_count() or 1) - 1, 4))
    assert context["device"] == "cpu"
    assert context["providers"] == ["CPUExecutionProvider"]
    assert context["execution_provider"] == "CPUExecutionProvider"
    assert thread_calls == [expected_threads]


def test_cuda_device_resolution_prefers_cuda_provider(monkeypatch):
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _index: "Test GPU",
        memory_allocated=lambda _index: 64 * 1024 * 1024,
        amp=SimpleNamespace(autocast=lambda **_kwargs: nullcontext()),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, float16="float16")
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    context = resolve_execution_device()

    assert context["device"] == "cuda"
    assert context["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert context["device_name"] == "Test GPU"
    assert context["vram_allocated_mb"] == 64.0
    assert context["autocast_enabled"] is True


def test_wav_loader_downmixes_and_resamples_to_16khz():
    left = _tone(8_000)
    right = np.zeros_like(left)
    stereo = np.column_stack((left, right))
    interleaved = np.round(stereo * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(interleaved)

    audio, sample_rate = load_audio(
        {"filename": "stereo.wav", "bytes": buffer.getvalue()}, target_sr=16_000
    )

    assert sample_rate == 16_000
    assert audio.dtype == np.float32
    assert audio.shape == (1_600,)
    assert 0.08 < float(np.max(np.abs(audio))) < 0.12


def test_pyav_decoder_uses_audio_resampler(monkeypatch):
    frame = SimpleNamespace(
        sample_rate=48_000,
        to_ndarray=lambda: np.stack(
            (np.full(16, 0.25, dtype=np.float32), np.full(16, -0.25, dtype=np.float32))
        ),
    )
    stream = SimpleNamespace(type="audio", rate=48_000)

    class FakeContainer:
        streams = [stream]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, _stream):
            return [frame]

    class FakeResampler:
        def __init__(self, *, format):
            assert format == "fltp"

        def resample(self, input_frame):
            return [] if input_frame is None else [input_frame]

    fake_av = SimpleNamespace(
        AudioResampler=FakeResampler,
        open=lambda *_args, **_kwargs: FakeContainer(),
    )
    monkeypatch.setitem(__import__("sys").modules, "av", fake_av)

    decoded, sample_rate, channels = _decode_pyav(b"fake-container")

    assert decoded.shape == (16, 2)
    assert sample_rate == 48_000
    assert channels == 2


def test_audio_metrics_are_finite_and_identify_surrogate():
    reference = _tone()
    noise = np.full_like(reference, 0.01)

    assert compute_snr_db(reference, noise) > 20.0
    assert compute_sisdr(reference, reference) > 100.0
    metrics = evaluate_audio_telemetry(reference, 16_000)

    assert np.isfinite(metrics["snr_db"])
    assert np.isfinite(metrics["si_sdr_db"])
    assert metrics["pesq_method"] in {"pesq", "pesq_heuristic_surrogate"}
    assert 0.0 <= metrics["stoi_index"] <= 1.0
    assert metrics["clipping_ratio"] == 0.0
    assert np.isfinite(metrics["crest_factor_db"])
    assert np.isfinite(metrics["lufs_est"])


def test_quality_booster_increases_loudness_without_true_peak_clipping():
    quiet = _tone(seconds=1.0) * 0.1
    boosted = apply_quality_booster(quiet, 16_000, boost_db=3.0)

    input_rms = float(np.sqrt(np.mean(np.square(quiet, dtype=np.float64))))
    output_rms = float(np.sqrt(np.mean(np.square(boosted, dtype=np.float64))))
    assert output_rms > input_rms * 3.0
    assert calculate_true_peak_dbfs(boosted, 16_000) <= -0.49
    assert -13.5 <= estimate_integrated_lufs(boosted, 16_000) <= -9.0


def test_smart_tuner_detects_mid_heavy_audio_and_applies_eq():
    sample_rate = 16_000
    indices = np.arange(sample_rate, dtype=np.float32)
    mids = (0.2 * np.sin(2.0 * np.pi * 500.0 * indices / sample_rate)).astype(
        np.float32
    )

    analysis = analyze_audio_spectrum(mids, sample_rate)
    equalized = apply_parametric_eq(mids, sample_rate, mid_gain_db=-6.0)

    assert analysis["band_energies"]["mids"] > 0.9
    assert "Mids heavy: dip 350Hz" in analysis["suggestions"]
    assert analysis["crest_factor_db"] > 0
    assert np.sqrt(np.mean(equalized**2)) < np.sqrt(np.mean(mids**2))


def test_positive_eq_preserves_float_headroom_until_limiter():
    source = _tone(seconds=1.0) * 4.0
    equalized = apply_parametric_eq(source, 16_000, low_gain_db=12.0, mid_gain_db=12.0)
    limited = apply_true_peak_limiter(equalized, 16_000, target_peak_dbfs=-0.5)

    assert not np.any(np.isclose(np.abs(equalized), 1.0, atol=1e-7))
    assert np.max(np.abs(equalized)) > 1.0
    assert calculate_true_peak_dbfs(limited, 16_000) <= -0.49


def test_spectral_gate_and_neural_missing_weight_fallback(tmp_path):
    audio = _tone()
    unchanged = apply_spectral_gating(audio, attenuation_db=0.0)
    assert np.array_equal(unchanged, audio)

    context = {
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
        "execution_provider": "CPUExecutionProvider",
        "onnxruntime_available": True,
        "torch_available": False,
        "model_path": str(tmp_path / "missing.onnx"),
    }
    result = apply_deepfilternet3_sota(audio, device_ctx=context)

    assert result["status"] == "fallback"
    assert result["fallback_reason"] == "weights_missing"
    assert result["runtime"] == "spectral_gate"
    assert result["audio"].shape == audio.shape


def test_neural_wrapper_rejects_wrong_output_length(monkeypatch, tmp_path):
    class FakeSessionOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            assert sess_options is not None
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="waveform", shape=[1, 1, None], type="tensor(float)"
                )
            ]

        def run(self, _names, feeds):
            waveform = feeds["waveform"]
            return [waveform[..., :-1]]

    fake_runtime = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_runtime)
    model_path = tmp_path / "deepfilternet3.onnx"
    model_path.write_bytes(b"fake")
    context = {
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
        "execution_provider": "CPUExecutionProvider",
        "onnxruntime_available": True,
        "torch_available": False,
        "model_path": str(model_path),
    }

    result = apply_deepfilternet3_sota(_tone(), device_ctx=context)

    assert result["status"] == "fallback"
    assert result["fallback_reason"] == "incompatible_model_contract"


def test_neural_waveform_onnx_wrapper_executes(monkeypatch, tmp_path):
    class FakeSessionOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            assert sess_options is not None
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="waveform", shape=[1, 1, None], type="tensor(float)"
                )
            ]

        def run(self, _names, feeds):
            return [feeds["waveform"] * 0.5]

    monkeypatch.setitem(
        __import__("sys").modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=FakeSessionOptions,
            InferenceSession=FakeSession,
        ),
    )
    model_path = tmp_path / "deepfilternet3.onnx"
    model_path.write_bytes(b"fake")
    context = {
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
        "execution_provider": "CPUExecutionProvider",
        "onnxruntime_available": True,
        "torch_available": False,
        "model_path": str(model_path),
    }

    result = apply_deepfilternet3_sota(_tone(), device_ctx=context)

    assert result["status"] == "success"
    assert result["runtime"] == "onnxruntime"
    assert result["model_contract"] == "end_to_end_waveform_16khz_v1"


def test_neural_wrapper_rejects_non_finite_audio(monkeypatch, tmp_path):
    class FakeSessionOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            assert sess_options is not None
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="waveform", shape=[1, 1, None], type="tensor(float)"
                )
            ]

        def run(self, _names, feeds):
            return [np.full_like(feeds["waveform"], np.nan)]

    monkeypatch.setitem(
        __import__("sys").modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=FakeSessionOptions,
            InferenceSession=FakeSession,
        ),
    )
    model_path = tmp_path / "deepfilternet3_nonfinite.onnx"
    model_path.write_bytes(b"fake")
    context = {
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
        "execution_provider": "CPUExecutionProvider",
        "onnxruntime_available": True,
        "torch_available": False,
        "model_path": str(model_path),
    }

    result = apply_deepfilternet3_sota(_tone(), device_ctx=context)

    assert result["status"] == "fallback"
    assert result["fallback_reason"] == "incompatible_model_contract"
    assert np.all(np.isfinite(result["audio"]))


def test_bandwidth_extension_reconstructs_48khz_waveform():
    source = _tone(seconds=0.2)
    extended = apply_high_freq_reconstruction(source, 16_000, 48_000)

    assert extended.dtype == np.float32
    assert extended.shape == (source.size * 3,)
    assert np.all(np.isfinite(extended))
    assert float(np.max(np.abs(extended))) <= 1.0


def test_bandwidth_waveform_onnx_wrapper_executes(monkeypatch, tmp_path):
    class FakeSessionOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            assert sess_options is not None
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="waveform", shape=[1, 1, None], type="tensor(float)"
                )
            ]

        def run(self, _names, feeds):
            return [np.repeat(feeds["waveform"], 3, axis=-1)]

    monkeypatch.setitem(
        __import__("sys").modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=FakeSessionOptions,
            InferenceSession=FakeSession,
        ),
    )
    model_path = tmp_path / "voicefixer.onnx"
    model_path.write_bytes(b"fake")
    context = {
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
        "execution_provider": "CPUExecutionProvider",
        "onnxruntime_available": True,
        "torch_available": False,
        "bandwidth_model_path": str(model_path),
    }

    result = apply_voicefixer_sota(_tone(), device_ctx=context)

    assert result["status"] == "success"
    assert result["runtime"] == "onnxruntime"
    assert result["audio"].size == _tone().size * 3


def test_pipeline_returns_audio_transform_and_hardware_telemetry(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODATA_VOICE_MODEL_DIR", str(tmp_path))
    context = {
        "device": "cpu",
        "device_name": "CPU Host",
        "providers": ["CPUExecutionProvider"],
        "available_providers": ["CPUExecutionProvider"],
        "execution_provider": "CPUExecutionProvider",
        "vram_allocated_mb": 0.0,
        "cpu_threads": 2,
        "cuda_available": False,
        "onnx_cuda_available": False,
        "autocast_enabled": False,
        "onnxruntime_available": True,
        "torch_available": False,
    }

    result = enhance_voice(
        _tone(),
        16_000,
        processing_mode="tier2",
        enable_bandwidth_extension=True,
        device_ctx=context,
    )

    assert isinstance(result, TransformResult)
    assert result.status == ExecutionStatus.PARTIAL
    assert result.transformed_audio is not None
    assert result.sample_rate == 48_000
    assert result.transformed_audio.size == _tone().size * 3
    assert result.module_result.metrics["device_name"] == "CPU Host"
    assert result.module_result.metrics["execution_provider"] == "CPUExecutionProvider"
    assert result.execution_timing["total"] > 0
    assert [item["step"] for item in result.step_queue] == [
        "hardware_resolution",
        "ingestion",
        "denoising",
        "parametric_eq",
        "bandwidth_extension",
        "quality_booster",
        "telemetry",
    ]
    assert result.module_result.metrics["mode"] == "file_offline"
    assert result.module_result.metrics["multiband_eq"] is True
    assert result.module_result.metrics["bandwidth_extension"] is True


def test_live_mode_forces_onnx_path_and_disables_heavy_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODATA_VOICE_MODEL_DIR", str(tmp_path))
    context = {
        "device": "cpu",
        "device_name": "CPU Host",
        "providers": ["CPUExecutionProvider"],
        "execution_provider": "CPUExecutionProvider",
        "vram_allocated_mb": 0.0,
        "cpu_threads": 2,
        "cuda_available": False,
        "onnx_cuda_available": False,
        "autocast_enabled": False,
        "onnxruntime_available": True,
        "torch_available": False,
    }

    result = enhance_voice(
        _tone(),
        16_000,
        processing_mode="tier1",
        mode="live_stream",
        enable_bandwidth_extension=True,
        boost_db=12.0,
        device_ctx=context,
    )

    steps = [item["step"] for item in result.step_queue]
    assert result.sample_rate == 16_000
    assert result.transformed_audio.shape == _tone().shape
    assert "true_peak_limiter" in steps
    assert "parametric_eq" not in steps
    assert "bandwidth_extension" not in steps
    assert "quality_booster" not in steps
    assert result.module_result.metrics["tier_effective"] == "tier2"
    assert result.module_result.metrics["latency_guard_active"] is True
    assert result.module_result.metrics["multiband_eq"] is False
    assert result.module_result.metrics["bandwidth_extension"] is False


def test_pcm_stream_enhancer_preserves_chunk_framing():
    audio = _tone()
    pcm = np.round(audio * 32767.0).astype("<i2").tobytes()
    stream = PCMStreamEnhancer(
        processing_mode="tier1",
        input_sample_rate=16_000,
        channels=1,
        sample_width=2,
    )

    enhanced_pcm = b"".join(stream.process_stream([pcm]))
    decoded = pcm_bytes_to_float32(enhanced_pcm, sample_width=2, channels=1)

    assert len(enhanced_pcm) == len(pcm)
    assert decoded.shape == audio.shape
    assert stream.module_result.metrics["chunks_processed"] == 1
    assert stream.module_result.metrics["samples_processed"] == audio.size
    assert stream.processing_mode == "tier2"
    assert stream.enable_bandwidth_extension is False
    assert stream.module_result.metrics["limiter_lookahead_ms"] == 1.0
    assert stream.module_result.metrics["pipeline_latency_ms"] < 20.0


def test_streaming_limiter_holds_true_peak_across_chunk_boundaries():
    positive = np.full(320, 0.99, dtype=np.float32)
    negative = np.full(320, -0.99, dtype=np.float32)
    chunks = [
        np.round(chunk * 32767.0).astype("<i2").tobytes()
        for chunk in (positive, negative, positive, negative)
    ]
    stream = PCMStreamEnhancer(
        input_sample_rate=16_000,
        channels=1,
        sample_width=2,
        attenuation_db=0.0,
    )

    enhanced_pcm = b"".join(stream.process_stream(chunks))
    enhanced = pcm_bytes_to_float32(
        enhanced_pcm, sample_width=2, channels=1
    )

    assert enhanced.size == 1_280
    assert calculate_true_peak_dbfs(enhanced, 16_000) <= -0.45


def test_pcm_stream_buffers_frames_and_avoids_resampling_drift():
    source_rate = 44_100
    sample_indices = np.arange(10_000, dtype=np.float32)
    audio = (0.2 * np.sin(2.0 * np.pi * 440.0 * sample_indices / source_rate)).astype(
        np.float32
    )
    pcm = np.round(audio * 32767.0).astype("<i2").tobytes()
    chunks = [pcm[:1], pcm[1:997], pcm[997:8_001], pcm[8_001:]]
    stream = PCMStreamEnhancer(
        processing_mode="tier1",
        input_sample_rate=source_rate,
        channels=1,
        sample_width=2,
    )

    enhanced_pcm = b"".join(stream.process_stream(chunks))

    expected_frames = round(audio.size * 16_000 / source_rate)
    assert len(enhanced_pcm) // 2 == expected_frames
    assert stream.module_result.metrics["samples_processed"] == audio.size


def test_pcm_stream_resampler_rejects_out_of_band_aliases():
    source_rate = 44_100
    sample_indices = np.arange(10_000, dtype=np.float32)
    out_of_band = (0.5 * np.sin(2.0 * np.pi * 12_000.0 * sample_indices / source_rate)).astype(
        np.float32
    )
    pcm = np.round(out_of_band * 32767.0).astype("<i2").tobytes()
    stream = PCMStreamEnhancer(
        processing_mode="tier1",
        attenuation_db=0.0,
        input_sample_rate=source_rate,
        channels=1,
        sample_width=2,
    )

    enhanced_pcm = b"".join(
        stream.process_stream([pcm[:2_001], pcm[2_001:9_777], pcm[9_777:]])
    )
    enhanced = pcm_bytes_to_float32(
        enhanced_pcm, sample_width=2, channels=1
    )

    steady_state = enhanced[200:-200]
    rms = float(np.sqrt(np.mean(np.square(steady_state, dtype=np.float64))))
    assert rms < 0.01


def test_pcm_stream_rejects_incomplete_final_frame():
    stream = PCMStreamEnhancer(sample_width=2, channels=1)
    assert stream.process_pcm_chunk(b"\x00") == b""
    with np.testing.assert_raises_regex(ValueError, "incomplete channel frame"):
        stream.flush()


def test_24bit_wav_export_round_trip():
    source = _tone()
    payload = encode_wav_bytes(source, 16_000, bit_depth=24)
    decoded, sample_rate = load_audio(
        {"filename": "enhanced.wav", "bytes": payload}, target_sr=16_000
    )

    assert sample_rate == 16_000
    assert decoded.shape == source.shape
    assert np.max(np.abs(decoded - source)) < 2e-6


def test_all_export_formats_round_trip_through_static_decoder():
    source = _tone(seconds=0.2)
    encoders = {
        "wav": lambda: encode_wav_bytes(source, 16_000, bit_depth=16),
        "mp3": lambda: encode_mp3_bytes(source, 16_000, bit_rate=320_000),
        "flac": lambda: encode_flac_bytes(source, 16_000),
        "ogg": lambda: encode_ogg_bytes(source, 16_000),
    }

    for extension, encode in encoders.items():
        payload = encode()
        decoded, sample_rate = load_audio(
            {"filename": f"export.{extension}", "bytes": payload},
            target_sr=16_000,
        )
        assert len(payload) > 100
        assert sample_rate == 16_000
        assert abs(decoded.size - source.size) < 2_000
        assert np.all(np.isfinite(decoded))


def test_export_rate_and_lossy_codec_keep_true_peak_headroom():
    source_rate = 48_000
    target_rate = 44_100
    indices = np.arange(source_rate, dtype=np.float32)
    hot = (0.94 * np.sin(2.0 * np.pi * 997.0 * indices / source_rate)).astype(
        np.float32
    )
    resampled = resample_audio(hot, source_rate, target_rate)

    lossless_ready = apply_true_peak_limiter(
        resampled, target_rate, target_peak_dbfs=-0.5
    )
    wav_payload = encode_wav_bytes(lossless_ready, target_rate, bit_depth=24)
    wav_audio, _ = load_audio(
        {"filename": "safe.wav", "bytes": wav_payload}, target_rate
    )

    lossy_ready = apply_true_peak_limiter(
        resampled, target_rate, target_peak_dbfs=-3.0
    )
    ogg_payload = encode_ogg_bytes(lossy_ready, target_rate)
    ogg_audio, _ = load_audio(
        {"filename": "safe.ogg", "bytes": ogg_payload}, target_rate
    )

    assert calculate_true_peak_dbfs(wav_audio, target_rate) <= -0.45
    assert calculate_true_peak_dbfs(ogg_audio, target_rate) <= -0.45
