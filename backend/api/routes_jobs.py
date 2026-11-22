import json
import zipfile
from pathlib import Path
from typing import Optional, List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.db.models import ProcessingJob, Dataset, QualityReport, BenchmarkLog
from backend.engines.tabular import load_tabular_file
from backend.config import settings
from backend.core.security import resolve_path_within

router = APIRouter(prefix="/api/v1", tags=["Job Management & Telemetry"])

@router.get("/jobs")
def list_jobs(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    jobs = db.query(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(limit).all()
    results = []
    for j in jobs:
        ds = db.query(Dataset).filter(Dataset.id == j.dataset_id).first()
        results.append({
            "job_id": j.id,
            "dataset_id": j.dataset_id,
            "filename": ds.original_filename if ds else "Unknown",
            "file_type": ds.file_type if ds else "unknown",
            "status": j.status,
            "progress_percent": j.progress_percent,
            "current_stage": j.current_stage,
            "execution_mode": j.execution_mode,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            "error_message": j.error_message,
            "created_at": j.created_at.isoformat(),
        })
    return results

@router.get("/jobs/{job_id}")
def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    ds = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()

    elapsed_ms = None
    if job.started_at:
        end_time = job.completed_at or job.started_at.__class__.now(job.started_at.tzinfo)
        elapsed_ms = round((end_time - job.started_at).total_seconds() * 1000.0, 2)

    return {
        "job_id": job.id,
        "dataset_id": job.dataset_id,
        "filename": ds.original_filename if ds else "Unknown",
        "file_type": ds.file_type if ds else "unknown",
        "status": job.status,
        "progress": job.progress_percent,
        "current_stage": job.current_stage,
        "execution_mode": job.execution_mode,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "elapsed_ms": elapsed_ms,
        "error_message": job.error_message,
        "output_path": job.output_path,
    }

@router.get("/jobs/{job_id}/benchmarks")
def get_job_benchmarks(
    job_id: str,
    db: Session = Depends(get_db)
):
    benchmarks = db.query(BenchmarkLog).filter(BenchmarkLog.job_id == job_id).order_by(BenchmarkLog.id.asc()).all()
    return [
        {
            "id": b.id,
            "stage_name": b.stage_name,
            "latency_ms": b.latency_ms,
            "memory_delta_mb": b.memory_delta_mb,
            "cpu_percent": b.cpu_percent,
            "timestamp": b.timestamp.isoformat(),
        }
        for b in benchmarks
    ]

@router.get("/jobs/{job_id}/quality")
def get_job_quality(
    job_id: str,
    db: Session = Depends(get_db)
):
    qr = db.query(QualityReport).filter(QualityReport.job_id == job_id).first()
    if not qr:
        raise HTTPException(status_code=404, detail="Quality report not found for this job")

    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    ds = db.query(Dataset).filter(Dataset.id == job.dataset_id).first() if job else None

    return {
        "job_id": qr.job_id,
        "pre_score": qr.pre_score,
        "post_score": qr.post_score,
        "score_delta": round(qr.post_score - qr.pre_score, 2),
        "metrics": json.loads(qr.metrics_json),
        "recommendations": json.loads(qr.recommendations_json),
        "dataset_type": ds.file_type if ds else "tabular",
        "created_at": qr.created_at.isoformat(),
    }

@router.get("/jobs/{job_id}/diff")
def get_job_diff(
    job_id: str,
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Returns before and after sample data for side-by-side tabular comparison
    or image comparison.
    """
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    ds = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    raw_path = Path(ds.storage_path)

    if ds.file_type == "tabular":
        if not raw_path.exists():
            raise HTTPException(status_code=404, detail="Raw dataset file missing")

        df_raw = load_tabular_file(raw_path)
        processed_file = Path(job.output_path) if job.output_path else settings.PROCESSED_STORAGE_DIR / job_id / "cleaned_data.csv"

        if not processed_file.exists():
            raise HTTPException(status_code=400, detail="Processed output not ready yet")

        df_proc = load_tabular_file(processed_file)

        raw_sample = [
            {col: (None if pd.isna(val) else (val.item() if hasattr(val, "item") else val)) for col, val in row.items()}
            for _, row in df_raw.head(limit).iterrows()
        ]
        proc_sample = [
            {col: (None if pd.isna(val) else (val.item() if hasattr(val, "item") else val)) for col, val in row.items()}
            for _, row in df_proc.head(limit).iterrows()
        ]

        return {
            "type": "tabular",
            "columns_before": list(df_raw.columns),
            "columns_after": list(df_proc.columns),
            "rows_before": raw_sample,
            "rows_after": proc_sample,
            "total_before": len(df_raw),
            "total_after": len(df_proc),
        }

    elif ds.file_type == "vision":
        # Vision diff: list of matched original vs processed image paths
        proc_dir = settings.PROCESSED_STORAGE_DIR / job_id / "processed_images"
        if not proc_dir.exists():
            raise HTTPException(status_code=400, detail="Processed images not ready yet")

        processed_images = [p.name for p in proc_dir.glob("*.*") if p.suffix.lower() in settings.IMAGE_EXTENSIONS]

        enh_audit = {}
        enh_file = settings.PROCESSED_STORAGE_DIR / job_id / "adaptive_enhancement.json"
        if enh_file.exists():
            try:
                with open(enh_file, "r") as f:
                    enh_audit = json.load(f)
            except Exception:
                pass

        comparisons = []
        for p_name in processed_images[:limit]:
            # proc_{orig_name}
            orig_name = p_name.replace("proc_", "")
            audit_info = enh_audit.get(p_name, enh_audit.get(orig_name, {}))
            comparisons.append({
                "original_name": orig_name,
                "processed_name": p_name,
                "before_url": f"/api/v1/datasets/{ds.id}/image/{orig_name}",
                "after_url": f"/api/v1/jobs/{job.id}/image/{p_name}",
                "enhancements": audit_info,
            })

        return {
            "type": "vision",
            "comparisons": comparisons,
            "total_images": len(processed_images)
        }

@router.get("/jobs/{job_id}/image/{image_name:path}")
def get_processed_job_image(
    job_id: str,
    image_name: str
):
    """
    Streams a processed image raster for split diff preview.
    """
    proc_dir = resolve_path_within(settings.PROCESSED_STORAGE_DIR, job_id, "processed_images")
    proc_img = resolve_path_within(proc_dir, image_name)
    if proc_img.is_file() and proc_img.suffix.lower() in settings.IMAGE_EXTENSIONS:
        return FileResponse(proc_img)
    raise HTTPException(status_code=404, detail="Processed image not found")
