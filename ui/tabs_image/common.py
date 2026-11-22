"""Shared decoding, display, ZIP, and encoding helpers for image tabs."""

from __future__ import annotations

import io
from pathlib import PurePosixPath
import zipfile

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}
MAX_ARCHIVE_ENTRIES = 2_000
MAX_ARCHIVE_IMAGES = 500
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_DIMENSION = 16_384
MAX_COMPRESSION_RATIO = 100.0


def uploaded_name(uploaded_file) -> str:
    if isinstance(uploaded_file, dict):
        return str(uploaded_file.get("filename", "uploaded_image"))
    return str(getattr(uploaded_file, "name", "uploaded_image"))


def uploaded_bytes(uploaded_file) -> bytes:
    if isinstance(uploaded_file, dict):
        return bytes(uploaded_file.get("bytes", b""))
    if hasattr(uploaded_file, "getvalue"):
        return bytes(uploaded_file.getvalue())
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file)
    raise TypeError("Unsupported uploaded image object.")


def decode_image_bytes(payload: bytes) -> np.ndarray:
    if not payload:
        raise ValueError("Image payload is empty.")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError("OpenCV could not decode the image payload.")
    height, width = image.shape[:2]
    if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        raise ValueError("Image dimensions exceed the 16,384 pixel safety limit.")
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError("Decoded image exceeds the 25 megapixel safety limit.")
    if image.dtype == np.uint16:
        image = np.round(image.astype(np.float32) / 257.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def extract_image_items(uploaded_file) -> list[dict]:
    """Return validated in-memory image members without extracting to disk."""
    name = uploaded_name(uploaded_file)
    payload = uploaded_bytes(uploaded_file)
    if not name.lower().endswith(".zip"):
        decode_image_bytes(payload)
        return [{"name": name, "bytes": payload}]

    items = []
    total_uncompressed = 0
    normalized_names = set()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise ValueError("ZIP archive contains too many entries.")
            for info in entries:
                if info.is_dir():
                    continue
                path = PurePosixPath(info.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("ZIP archive contains an unsafe path.")
                normalized = path.as_posix().casefold()
                if normalized in normalized_names:
                    raise ValueError("ZIP archive contains duplicate normalized filenames.")
                normalized_names.add(normalized)
                if info.flag_bits & 0x1:
                    raise ValueError("Encrypted ZIP members are not supported.")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("ZIP archive exceeds the uncompressed size limit.")
                if info.file_size > MAX_IMAGE_BYTES:
                    continue
                if info.compress_size > 0:
                    ratio = info.file_size / info.compress_size
                    if ratio > MAX_COMPRESSION_RATIO:
                        raise ValueError("ZIP member compression ratio is unsafe.")
                if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                member_bytes = archive.read(info)
                decode_image_bytes(member_bytes)
                items.append({"name": path.as_posix(), "bytes": member_bytes})
                if len(items) > MAX_ARCHIVE_IMAGES:
                    raise ValueError("ZIP archive contains too many image files.")
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded ZIP archive is corrupted or invalid.") from exc

    if not items:
        raise ValueError("The ZIP archive contains no supported, decodable images.")
    return items


def bgr_to_display(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    if image_bgr.shape[2] == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA)
    raise ValueError("Unsupported image channel count.")


def encode_image(
    image_bgr: np.ndarray,
    output_format: str,
    *,
    quality: int = 95,
    png_compression: int = 4,
    jpeg_background_bgr: tuple[int, int, int] = (0, 0, 0),
) -> tuple[bytes, str, str]:
    """Encode a BGR/BGRA image and return bytes, extension, and MIME type."""
    output_format = output_format.lower()
    image_to_encode = image_bgr
    if output_format in {"jpg", "jpeg"} and image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        alpha = image_bgr[:, :, 3:4].astype(np.float32) / 255.0
        background = np.full(
            image_bgr[:, :, :3].shape,
            jpeg_background_bgr,
            dtype=np.uint8,
        ).astype(np.float32)
        image_to_encode = np.round(
            image_bgr[:, :, :3].astype(np.float32) * alpha
            + background * (1.0 - alpha)
        ).astype(np.uint8)

    if output_format in {"jpg", "jpeg"}:
        extension = ".jpg"
        mime = "image/jpeg"
        parameters = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif output_format == "webp":
        extension = ".webp"
        mime = "image/webp"
        parameters = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    else:
        extension = ".png"
        mime = "image/png"
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]

    success, buffer = cv2.imencode(extension, image_to_encode, parameters)
    if not success:
        raise ValueError(f"OpenCV failed to encode {extension} output.")
    return buffer.tobytes(), extension, mime
