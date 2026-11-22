import os
import shutil
import uuid
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile
from backend.config import settings
from backend.core.security import (
    compute_file_sha256,
    verify_file_magic_and_classify,
    make_read_only,
    unlock_read_only,
)
from backend.core.exceptions import EntityTooLargeException

class StorageService:
    @staticmethod
    async def save_raw_upload(upload_file: UploadFile) -> Tuple[str, Path, int, str, str]:
        """
        Saves uploaded file under raw storage partition:
        /storage/raw/{dataset_id}/{filename}
        Enforces size limits, magic bytes verification, sha256 checksum, and read-only isolation.
        Returns: (dataset_id, saved_path, file_size_bytes, sha256_hash, file_type)
        """
        dataset_id = str(uuid.uuid4())
        raw_dir = settings.RAW_STORAGE_DIR / dataset_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        original_filename = upload_file.filename or "dataset.dat"
        # Sanitize filename
        safe_filename = Path(original_filename).name.replace(" ", "_")
        target_path = raw_dir / safe_filename

        total_bytes = 0
        with open(target_path, "wb") as f:
            while chunk := await upload_file.read(1024 * 1024): # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                    # Clean up partial file
                    f.close()
                    if target_path.exists():
                        target_path.unlink()
                    raise EntityTooLargeException(
                        f"Uploaded file exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f} MB."
                    )
                f.write(chunk)

        # Magic byte verification and classification
        file_type, _ = verify_file_magic_and_classify(target_path, safe_filename)
        sha256_hash = compute_file_sha256(target_path)

        # Lock raw file to read-only to ensure non-destructive governance
        make_read_only(target_path)

        return dataset_id, target_path, total_bytes, sha256_hash, file_type

    @staticmethod
    def get_intermediate_dir(job_id: str) -> Path:
        path = settings.INTERMEDIATE_STORAGE_DIR / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_processed_dir(job_id: str) -> Path:
        path = settings.PROCESSED_STORAGE_DIR / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def cleanup_intermediate(job_id: str):
        path = settings.INTERMEDIATE_STORAGE_DIR / job_id
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass

    @staticmethod
    def delete_dataset(dataset_id: str):
        path = settings.RAW_STORAGE_DIR / dataset_id
        if path.exists():
            for root, dirs, files in os.walk(path):
                for f in files:
                    unlock_read_only(Path(root) / f)
            shutil.rmtree(path, ignore_errors=True)
