"""Tests for portable, checksum-verified neural model provisioning."""

import hashlib
import io
import sys

import pytest

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
        "DarkIR_384.pt",
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
    assert setup_vision_models.MODEL_DOWNLOAD_MAP == {
        key: files[0]["url"] for key, files in setup_vision_models.MODEL_FILES.items()
    }
    darkir, = setup_vision_models.MODEL_FILES["darkir"]
    assert darkir["url"].startswith("https://huggingface.co/Cidaut/DarkIR/resolve/")
    assert darkir["sha256"] == "61eee7d5cfb593d408fba4c716874449b324ed1f49cbff97c2a25ce2fcbe4fde"


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


@pytest.mark.parametrize(("options", "expected"), [
    (["--model", "darkir"], ["darkir"]),
    (["--models", "darkir"], ["darkir"]),
    (["--models", "darkir", "nafnet", "--model", "real-esrgan"], ["darkir", "nafnet", "real-esrgan"]),
])
def test_model_flags_select_manifest_entries(monkeypatch, tmp_path, options, expected):
    calls = []

    def provision(ids, directory, force):
        calls.append((ids, directory, force))
        return []

    monkeypatch.setattr(setup_vision_models, "provision_models", provision)
    monkeypatch.setattr(sys, "argv", ["setup_vision_models", "--model-dir", str(tmp_path), *options])
    setup_vision_models.main()
    assert calls == [(expected, tmp_path, False)]


@pytest.mark.parametrize("failure", ["checksum", "interrupted"])
def test_failed_download_preserves_existing_file(monkeypatch, tmp_path, failure):
    entry = setup_vision_models.MODEL_FILES["darkir"][0]
    destination = tmp_path / entry["filename"]
    destination.write_bytes(b"previous version")

    def download(request, timeout):
        if failure == "interrupted":
            raise OSError("interrupted")
        return _Response(b"incomplete checkpoint")

    monkeypatch.setattr(setup_vision_models, "urlopen", download)
    with pytest.raises((OSError, RuntimeError), match="interrupted|Checksum mismatch"):
        setup_vision_models.download_verified_file(entry, destination, force=True)
    assert destination.read_bytes() == b"previous version"
    assert list(tmp_path.iterdir()) == [destination]
