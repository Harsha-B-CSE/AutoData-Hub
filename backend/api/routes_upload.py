import zipfile
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.db.models import Dataset
from backend.services.storage_service import StorageService
from backend.core.security import resolve_path_within, safe_extract_zip_member
from backend.engines.tabular import load_tabular_file, TabularProfiler
from backend.engines.image import ImageProfiler, profile_image
from backend.config import settings

router = APIRouter(prefix="/api/v1", tags=["Ingestion & Upload"])

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Receives multipart file upload, verifies magic bytes, enforces read-only storage isolation,
    creates Dataset metadata record, and returns classification.
    """
    dataset_id, storage_path, file_size, sha256_hash, file_type = await StorageService.save_raw_upload(file)

    ds = Dataset(
        id=dataset_id,
        original_filename=file.filename or "unknown.dat",
        file_type=file_type,
        file_size_bytes=file_size,
        sha256_hash=sha256_hash,
        storage_path=str(storage_path),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    return {
        "dataset_id": ds.id,
        "filename": ds.original_filename,
        "type": ds.file_type,
        "size": ds.file_size_bytes,
        "sha256": ds.sha256_hash,
        "created_at": ds.created_at.isoformat(),
    }

@router.get("/datasets")
def list_datasets(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).limit(limit).all()
    return [
        {
            "id": d.id,
            "filename": d.original_filename,
            "file_type": d.file_type,
            "file_size_bytes": d.file_size_bytes,
            "sha256": d.sha256_hash,
            "created_at": d.created_at.isoformat(),
        }
        for d in datasets
    ]

@router.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": ds.id,
        "filename": ds.original_filename,
        "file_type": ds.file_type,
        "file_size_bytes": ds.file_size_bytes,
        "sha256": ds.sha256_hash,
        "created_at": ds.created_at.isoformat(),
    }

@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(
    dataset_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Returns virtualized tabular rows or image asset gallery with profiling metrics.
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    raw_path = Path(ds.storage_path)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Dataset physical file missing")

    if ds.file_type == "tabular":
        df = load_tabular_file(raw_path)
        total_rows = len(df)
        total_pages = max(1, (total_rows + limit - 1) // limit)

        start_idx = (page - 1) * limit
        end_idx = min(start_idx + limit, total_rows)
        slice_df = df.iloc[start_idx:end_idx]

        # Convert values to JSON-safe primitives
        clean_rows = []
        for _, row in slice_df.iterrows():
            clean_rows.append({
                col: (None if pd.isna(val) else (val.item() if hasattr(val, "item") else val))
                for col, val in row.items()
            })

        profile = TabularProfiler.profile(df)

        return {
            "type": "tabular",
            "columns": list(df.columns),
            "rows": clean_rows,
            "total_rows": total_rows,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "profile": profile,
        }

    elif ds.file_type == "vision":
        # Vision preview
        images_info = []
        if raw_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(raw_path, "r") as zf:
                valid_members = [
                    m for m in zf.infolist()
                    if not m.is_dir() and Path(m.filename).suffix.lower() in settings.IMAGE_EXTENSIONS
                ]
                total_images = len(valid_members)
                start_idx = (page - 1) * limit
                end_idx = min(start_idx + limit, total_images)

                for m in valid_members[start_idx:end_idx]:
                    images_info.append({
                        "name": Path(m.filename).name,
                        "relative_path": m.filename,
                        "size": m.file_size,
                    })
        else:
            total_images = 1
            prof = profile_image(raw_path)
            images_info.append({
                "name": raw_path.name,
                "relative_path": raw_path.name,
                "size": raw_path.stat().st_size,
                "profile": prof,
            })

        return {
            "type": "vision",
            "total_images": total_images,
            "page": page,
            "limit": limit,
            "images": images_info,
        }

@router.get("/datasets/{dataset_id}/image/{image_name:path}")
def get_raw_dataset_image(
    dataset_id: str,
    image_name: str,
    db: Session = Depends(get_db)
):
    """
    Streams a raw image from single upload or extracted zip for preview.
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    raw_path = Path(ds.storage_path)
    if raw_path.suffix.lower() == ".zip":
        # Extract to temporary preview in raw dir if not extracted
        preview_dir = resolve_path_within(settings.RAW_STORAGE_DIR, dataset_id, "preview_extracted")
        target_file = resolve_path_within(preview_dir, image_name)
        if target_file.suffix.lower() not in settings.IMAGE_EXTENSIONS:
            raise HTTPException(status_code=404, detail="Image not found in archive")
        if not target_file.exists():
            target_file = safe_extract_zip_member(raw_path, image_name, preview_dir)
        if target_file.is_file():
            return FileResponse(target_file)
        raise HTTPException(status_code=404, detail="Image not found")
    else:
        if raw_path.exists():
            return FileResponse(raw_path)
        raise HTTPException(status_code=404, detail="Image file not found")
