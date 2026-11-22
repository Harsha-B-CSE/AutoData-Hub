from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.db.models import ProcessingJob, Dataset
from backend.services.export_service import ExportService
from backend.config import settings

router = APIRouter(prefix="/api/v1", tags=["Export Management"])

@router.get("/jobs/{job_id}/download-image")
def download_single_processed_image(
    job_id: str,
    format: str = Query("png", pattern="^(png|jpg|jpeg)$"),
    image_name: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Downloads an individual processed image in JPEG or PNG format.
    Ensures downloaded raster identically matches the displayed processed preview.
    Restricted strictly to visual datasets.
    """
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Cannot export image from job in state '{job.status}'")

    ds = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    if not ds or ds.file_type != "vision":
        raise HTTPException(
            status_code=400,
            detail="Image downloads are only available for visual datasets (not tabular datasets)"
        )

    proc_img_dir = settings.PROCESSED_STORAGE_DIR / job_id / "processed_images"
    if not proc_img_dir.exists():
        raise HTTPException(status_code=404, detail="No processed images directory found for this job")

    available_images = [p for p in proc_img_dir.glob("*.*") if p.suffix.lower() in settings.IMAGE_EXTENSIONS]
    if not available_images:
        raise HTTPException(status_code=404, detail="No processed images found for this job")

    target_file: Optional[Path] = None
    if image_name:
        # Match exact name, or with 'proc_' prefix, or basename
        clean_name = Path(image_name).name
        for p in available_images:
            if p.name == clean_name or p.name == f"proc_{clean_name}" or p.name.replace("proc_", "") == clean_name:
                target_file = p
                break
        if not target_file:
            raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found in processed artifacts")
    else:
        target_file = available_images[0]

    img = cv2.imread(str(target_file), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read processed image from disk")

    normalized_fmt = format.lower()
    stem = target_file.stem
    if stem.startswith("proc_"):
        stem = stem[5:]

    if normalized_fmt in ("jpg", "jpeg"):
        # JPEG doesn't support 4th alpha channel, convert BGRA to BGR
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to JPEG")
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"Content-Disposition": f'attachment; filename="{stem}_processed.jpg"'}
        )
    else:  # PNG
        success, buffer = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 4])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to PNG")
        return Response(
            content=buffer.tobytes(),
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{stem}_processed.png"'}
        )

@router.get("/jobs/{job_id}/export")
def export_job_artifacts(
    job_id: str,
    format: str = Query("csv", pattern="^(csv|xlsx|zip|pdf|json|png|jpg|jpeg)$"),
    image_name: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Downloads processed artifacts in target format:
    - Tabular datasets: CSV, XLSX, PDF, JSON
    - Vision datasets: PNG, JPG/JPEG, ZIP, PDF, JSON
    Enforces strict dataset-specific validation.
    """
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Cannot export job in state '{job.status}'")

    ds = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    dataset_type = ds.file_type if ds else "tabular"

    # Enforce strict dataset-type export constraints
    if dataset_type == "tabular":
        if format in ("png", "jpg", "jpeg", "zip"):
            raise HTTPException(
                status_code=400,
                detail=f"Image export format '{format.upper()}' is not supported for tabular datasets. Choose CSV, XLSX, PDF, or JSON."
            )
    elif dataset_type == "vision":
        if format in ("csv", "xlsx"):
            raise HTTPException(
                status_code=400,
                detail=f"Tabular export format '{format.upper()}' is not supported for image datasets. Choose PNG, JPEG, ZIP, PDF, or JSON."
            )

    if format in ("png", "jpg", "jpeg"):
        return download_single_processed_image(job_id=job_id, format=format, image_name=image_name, db=db)

    proc_dir = settings.PROCESSED_STORAGE_DIR / job_id

    if format == "pdf":
        pdf_bytes = ExportService.generate_pdf_report(job_id)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="autodata_audit_{job_id[:8]}.pdf"'}
        )

    elif format == "json":
        manifest_path = proc_dir / "manifest.json"
        if manifest_path.exists():
            return FileResponse(
                manifest_path,
                media_type="application/json",
                filename=f"autodata_manifest_{job_id[:8]}.json"
            )
        raise HTTPException(status_code=404, detail="Manifest file not found")

    elif format == "csv":
        csv_path = proc_dir / "cleaned_data.csv"
        if csv_path.exists():
            return FileResponse(
                csv_path,
                media_type="text/csv",
                filename=f"cleaned_data_{job_id[:8]}.csv"
            )
        raise HTTPException(status_code=404, detail="Processed CSV not available for this job")

    elif format == "xlsx":
        xlsx_path = proc_dir / "cleaned_data.xlsx"
        if xlsx_path.exists():
            return FileResponse(
                xlsx_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=f"cleaned_data_{job_id[:8]}.xlsx"
            )
        raise HTTPException(status_code=404, detail="Processed Excel file not available for this job")

    elif format == "zip":
        zip_path = proc_dir / "processed_images.zip"
        if zip_path.exists():
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename=f"processed_images_{job_id[:8]}.zip"
            )
        raise HTTPException(status_code=404, detail="Processed ZIP archive not available for this job")

    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
