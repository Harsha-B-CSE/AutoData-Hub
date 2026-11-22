import io
import os
import zipfile
import pytest
from pathlib import Path
from backend.core.security import (
    resolve_path_within,
    safe_extract_zip,
    safe_extract_zip_member,
    verify_file_magic_and_classify,
    make_read_only,
    unlock_read_only,
)
from backend.core.exceptions import (
    BadRequestException,
    EntityTooLargeException,
    UnsupportedMediaTypeException,
)

def test_empty_file_rejection(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_bytes(b"")

    with pytest.raises(BadRequestException):
        verify_file_magic_and_classify(empty_file, "empty.csv")

def test_spoofed_image_extension(tmp_path):
    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"this is plain text pretending to be png")

    with pytest.raises(UnsupportedMediaTypeException):
        verify_file_magic_and_classify(fake_png, "fake.png")

def test_zip_slip_path_traversal_prevention(tmp_path):
    zip_path = tmp_path / "malicious.zip"
    extract_target = tmp_path / "extracted"
    extract_target.mkdir()

    # Craft zip with directory traversal
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../escape.txt", "malicious payload")

    with pytest.raises(BadRequestException) as exc_info:
        safe_extract_zip(zip_path, extract_target)

    assert "Zip Slip" in str(exc_info.value.detail)


def test_zip_slip_rejects_sibling_prefix_bypass(tmp_path):
    zip_path = tmp_path / "prefix-bypass.zip"
    extract_target = tmp_path / "extracted"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../extracted-evil/payload.txt", "malicious payload")

    with pytest.raises(BadRequestException):
        safe_extract_zip(zip_path, extract_target)

    assert not (tmp_path / "extracted-evil" / "payload.txt").exists()


def test_single_member_extraction_enforces_containment(tmp_path):
    zip_path = tmp_path / "images.zip"
    extract_target = tmp_path / "preview"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("safe/image.png", b"image bytes")
        zf.writestr("../outside.png", b"malicious payload")

    extracted = safe_extract_zip_member(zip_path, "safe/image.png", extract_target)
    assert extracted == extract_target / "safe" / "image.png"

    with pytest.raises(BadRequestException):
        safe_extract_zip_member(zip_path, "../outside.png", extract_target)


def test_resolve_path_within_rejects_escape(tmp_path):
    base_dir = tmp_path / "processed"
    assert resolve_path_within(base_dir, "job", "image.png") == base_dir / "job" / "image.png"

    with pytest.raises(BadRequestException):
        resolve_path_within(base_dir, "..", "processed-evil", "image.png")

def test_zip_bomb_prevention(tmp_path):
    zip_path = tmp_path / "bomb.zip"
    extract_target = tmp_path / "extracted_bomb"
    extract_target.mkdir()

    # Create repetitive zeroes that compress to high ratio using chunks
    chunk = b"0" * (1024 * 1024) # 1 MB
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open("huge_zeroes.dat", "w") as dest:
            for _ in range(60):
                dest.write(chunk)

    # Ratio will exceed 10.0:1 on 60MB zeros
    with pytest.raises(BadRequestException) as exc_info:
        safe_extract_zip(zip_path, extract_target)

    assert "ratio" in str(exc_info.value.detail).lower()

def test_read_only_isolation(tmp_path):
    file_path = tmp_path / "locked_asset.dat"
    file_path.write_bytes(b"immutable original raw data")

    make_read_only(file_path)

    # Attempting to write in append or write mode should fail or raise PermissionError
    try:
        with open(file_path, "wb") as f:
            f.write(b"overwritten")
        pytest.fail("Should have failed to write to read-only file")
    except PermissionError:
        pass # Expected

    # Unlock and cleanup
    unlock_read_only(file_path)
    file_path.unlink()
