import os
import stat
import hashlib
import zipfile
from pathlib import Path
from typing import Tuple, List
from backend.config import settings
from backend.core.exceptions import (
    UnsupportedMediaTypeException,
    EntityTooLargeException,
    BadRequestException,
)


def resolve_path_within(
    base_dir: Path,
    *path_parts: str | Path,
    error_detail: str = "Path traversal outside the allowed directory is not permitted.",
) -> Path:
    """Resolve a path and require it to remain inside the given directory."""
    resolved_base = base_dir.resolve()
    destination = resolved_base.joinpath(*path_parts).resolve()
    try:
        destination.relative_to(resolved_base)
    except ValueError as exc:
        raise BadRequestException(error_detail) from exc
    return destination

# Common Magic Byte Signatures
MAGIC_SIGNATURES = {
    "jpeg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "bmp": [b"BM"],
    "webp": [b"RIFF"], # Checked with WEBP at offset 8
    "tiff": [b"II*\x00", b"MM\x00*"],
    "zip": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],
    "xls": [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"],
}

def compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_file_magic_and_classify(file_path: Path, original_filename: str) -> Tuple[str, str]:
    """
    Validates file magic signatures against extension to prevent spoofing,
    and returns (file_type, canonical_extension).
    file_type: 'tabular' | 'vision'
    """
    ext = Path(original_filename).suffix.lower()
    if not ext:
        raise UnsupportedMediaTypeException(f"File '{original_filename}' has no extension.")

    with open(file_path, "rb") as f:
        header = f.read(32)

    if len(header) == 0:
        raise BadRequestException("File is empty (0 bytes).")

    # Image check
    if ext in settings.IMAGE_EXTENSIONS:
        if ext in [".jpg", ".jpeg"] and not header.startswith(b"\xff\xd8\xff"):
            raise UnsupportedMediaTypeException(f"Invalid JPEG header for '{original_filename}'.")
        elif ext == ".png" and not header.startswith(b"\x89PNG\r\n\x1a\n"):
            raise UnsupportedMediaTypeException(f"Invalid PNG header for '{original_filename}'.")
        elif ext == ".bmp" and not header.startswith(b"BM"):
            raise UnsupportedMediaTypeException(f"Invalid BMP header for '{original_filename}'.")
        elif ext == ".webp" and not (header.startswith(b"RIFF") and header[8:12] == b"WEBP"):
            raise UnsupportedMediaTypeException(f"Invalid WEBP header for '{original_filename}'.")
        elif ext in [".tiff", ".tif"] and not (header.startswith(b"II*\x00") or header.startswith(b"MM\x00*")):
            raise UnsupportedMediaTypeException(f"Invalid TIFF header for '{original_filename}'.")
        return "vision", ext

    # Tabular check
    if ext in [".csv", ".tsv"]:
        # Check text readability
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                _ = f.read(1024)
            return "tabular", ext
        except Exception:
            raise UnsupportedMediaTypeException(f"Corrupted or unreadable CSV/TSV file '{original_filename}'.")

    if ext == ".xls":
        if not header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            raise UnsupportedMediaTypeException(f"Invalid Excel XLS binary header for '{original_filename}'.")
        return "tabular", ext

    if ext == ".xlsx":
        if not (header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06")):
            raise UnsupportedMediaTypeException(f"Invalid Excel XLSX archive header for '{original_filename}'.")
        # Validate that it is indeed an OpenXML spreadsheet
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()
                if not any("xl/" in n or "workbook.xml" in n for n in names):
                    raise UnsupportedMediaTypeException(f"File '{original_filename}' is a zip archive, but not a valid XLSX workbook.")
            return "tabular", ext
        except zipfile.BadZipFile:
            raise UnsupportedMediaTypeException(f"Corrupted XLSX file '{original_filename}'.")

    # ZIP archive check
    if ext in settings.ARCHIVE_EXTENSIONS:
        if not (header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06")):
            raise UnsupportedMediaTypeException(f"Invalid ZIP archive header for '{original_filename}'.")
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()
                image_count = sum(1 for n in names if Path(n).suffix.lower() in settings.IMAGE_EXTENSIONS)
                tabular_count = sum(1 for n in names if Path(n).suffix.lower() in settings.TABULAR_EXTENSIONS)

                if image_count > 0:
                    return "vision", ext
                elif tabular_count > 0:
                    return "tabular", ext
                else:
                    return "vision", ext # default zip archive to vision
        except zipfile.BadZipFile:
            raise UnsupportedMediaTypeException(f"Corrupted ZIP archive '{original_filename}'.")

    raise UnsupportedMediaTypeException(f"Unsupported file format: '{ext}'. Supported: CSV, TSV, XLS, XLSX, JPG, PNG, WEBP, BMP, TIFF, ZIP.")

def safe_extract_zip(zip_path: Path, target_dir: Path) -> List[Path]:
    """
    Safely extracts a ZIP archive with protections:
    - Zip Slip path traversal mitigation
    - Decompression bomb mitigation (size limit & compression ratio ceiling)
    """
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[Path] = []
    total_uncompressed_bytes = 0
    compressed_bytes = zip_path.stat().st_size

    with zipfile.ZipFile(zip_path, "r") as zf:
        infolist = zf.infolist()
        for member in infolist:
            # Check uncompressed size
            total_uncompressed_bytes += member.file_size
            if total_uncompressed_bytes > settings.MAX_ZIP_EXTRACT_SIZE_BYTES:
                raise EntityTooLargeException(
                    f"Archive uncompressed size exceeds limit of {settings.MAX_ZIP_EXTRACT_SIZE_BYTES / (1024**3):.1f} GB"
                )

            if compressed_bytes > 0:
                ratio = total_uncompressed_bytes / compressed_bytes
                if ratio > settings.MAX_ZIP_RATIO and total_uncompressed_bytes > (50 * 1024 * 1024):
                    raise BadRequestException(f"Archive compression ratio ({ratio:.1f}) exceeds safety ceiling of {settings.MAX_ZIP_RATIO}:1.")

            destination = resolve_path_within(
                target_dir,
                member.filename,
                error_detail=f"Potential Zip Slip path traversal detected in member: {member.filename}",
            )

            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, open(destination, "wb") as target:
                while chunk := source.read(65536):
                    target.write(chunk)

            extracted_files.append(destination)

    return extracted_files


def safe_extract_zip_member(zip_path: Path, member_name: str, target_dir: Path) -> Path:
    """Extract one regular ZIP member after path, size, and ratio validation."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            member = zf.getinfo(member_name)
        except KeyError as exc:
            raise BadRequestException("Archive member was not found.") from exc

        if member.is_dir():
            raise BadRequestException("Archive member is not a file.")
        if member.file_size > settings.MAX_ZIP_EXTRACT_SIZE_BYTES:
            raise EntityTooLargeException(
                f"Archive member exceeds limit of {settings.MAX_ZIP_EXTRACT_SIZE_BYTES / (1024**3):.1f} GB"
            )

        compressed_size = max(member.compress_size, 1)
        ratio = member.file_size / compressed_size
        if ratio > settings.MAX_ZIP_RATIO and member.file_size > (50 * 1024 * 1024):
            raise BadRequestException(
                f"Archive compression ratio ({ratio:.1f}) exceeds safety ceiling of {settings.MAX_ZIP_RATIO}:1."
            )

        destination = resolve_path_within(
            target_dir,
            member.filename,
            error_detail=f"Potential Zip Slip path traversal detected in member: {member.filename}",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as source, open(destination, "wb") as target:
            extracted_bytes = 0
            while chunk := source.read(65536):
                extracted_bytes += len(chunk)
                if extracted_bytes > settings.MAX_ZIP_EXTRACT_SIZE_BYTES:
                    raise EntityTooLargeException("Archive member exceeded the extraction size limit.")
                target.write(chunk)

    return destination

def make_read_only(file_path: Path):
    """
    Enforces strict read-only isolation on raw files so they cannot be modified in place.
    """
    try:
        current_perms = stat.S_IMODE(os.lstat(file_path).st_mode)
        # Remove write permissions
        read_only_perms = current_perms & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH
        os.chmod(file_path, read_only_perms)
    except Exception:
        pass

def unlock_read_only(file_path: Path):
    """
    Restores write permission for file removal or administrative cleanup.
    """
    try:
        current_perms = stat.S_IMODE(os.lstat(file_path).st_mode)
        write_perms = current_perms | stat.S_IWRITE
        os.chmod(file_path, write_perms)
    except Exception:
        pass
