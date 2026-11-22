"""Tests for portable, checksum-verified neural model provisioning."""

import hashlib
import io

from utils import setup_vision_models


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def test_model_manifest_has_expected_local_filenames_and_hashes():
    filenames = {
        entry["filename"]
        for entries in setup_vision_models.MODEL_FILES.values()
        for entry in entries
    }
    assert filenames == {
        "nafnet.onnx",
        "restormer.onnx",
        "real_esrgan_x4.onnx",
        "model.data",
        "swinir.onnx",
        "ultrasharp_v2_x4.onnx",
        "dat_nomos8k_x4.onnx",
        "realplksr_x4.onnx",
    }
    for entries in setup_vision_models.MODEL_FILES.values():
        for entry in entries:
            assert len(entry["sha256"]) == 64
            int(entry["sha256"], 16)
            assert entry["url"].startswith("https://")
    assert "realplksr" in setup_vision_models.DEFAULT_MODEL_IDS
    assert "ultrasharp-v2" not in setup_vision_models.DEFAULT_MODEL_IDS
    assert "dat-nomos8k" not in setup_vision_models.DEFAULT_MODEL_IDS


def test_verified_download_is_atomic_and_idempotent(monkeypatch, tmp_path):
    payload = b"verified-model-payload"
    expected_hash = hashlib.sha256(payload).hexdigest()
    entry = {
        "filename": "model.onnx",
        "url": "https://example.invalid/model.onnx",
        "sha256": expected_hash,
        "license": "test",
    }
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return _Response(payload)

    monkeypatch.setattr(setup_vision_models, "urlopen", fake_urlopen)
    destination = tmp_path / "model.onnx"
    assert (
        setup_vision_models.download_verified_file(entry, destination)
        == "downloaded"
    )
    assert destination.read_bytes() == payload
    assert (
        setup_vision_models.download_verified_file(entry, destination)
        == "verified-existing"
    )
    assert calls == [entry["url"]]
