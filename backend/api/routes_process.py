import json
import uuid
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.db.models import Dataset, ProcessingJob, PipelineConfig
from backend.core.orchestrator import ProcessingOrchestrator
from backend.api.routes_config import DEFAULT_PRESETS

router = APIRouter(prefix="/api/v1", tags=["Processing & Execution"])

class ProcessRequest(BaseModel):
    dataset_id: str
    config_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    execution_mode: Optional[str] = "batch"  # sequential, batch, parallel

@router.post("/process")
def dispatch_process_job(
    payload: ProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Submits a dataset preprocessing job, creates Job record in database,
    and dispatches execution asynchronously.
    """
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    resolved_config_id = None
    final_parameters = payload.parameters or {}

    # Resolve config if provided
    if payload.config_id:
        # Check if built-in
        builtin = next((p for p in DEFAULT_PRESETS if p["id"] == payload.config_id), None)
        if builtin:
            # Create a persistent config record for this run
            cfg = PipelineConfig(
                name=builtin["name"],
                engine_type=builtin["engine_type"],
                parameters_json=json.dumps(final_parameters if final_parameters else builtin["parameters"]),
            )
            db.add(cfg)
            db.commit()
            db.refresh(cfg)
            resolved_config_id = cfg.id
        else:
            db_cfg = db.query(PipelineConfig).filter(PipelineConfig.id == payload.config_id).first()
            if db_cfg:
                resolved_config_id = db_cfg.id
                if not final_parameters and db_cfg.parameters_json:
                    final_parameters = json.loads(db_cfg.parameters_json)
    elif final_parameters:
        # Save custom one-off config
        cfg = PipelineConfig(
            name="Custom Execution",
            engine_type=dataset.file_type,
            parameters_json=json.dumps(final_parameters),
        )
        db.add(cfg)
        db.commit()
        db.refresh(cfg)
        resolved_config_id = cfg.id

    mode = payload.execution_mode.lower() if payload.execution_mode else "batch"
    if mode not in ["sequential", "batch", "parallel"]:
        mode = "batch"

    job = ProcessingJob(
        id=str(uuid.uuid4()),
        dataset_id=dataset.id,
        config_id=resolved_config_id,
        status="QUEUED",
        progress_percent=0.0,
        current_stage="Queued in Dispatcher",
        execution_mode=mode,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Launch in background
    background_tasks.add_task(ProcessingOrchestrator.run_job, job.id)

    return {
        "job_id": job.id,
        "dataset_id": job.dataset_id,
        "status": job.status,
        "execution_mode": job.execution_mode,
        "created_at": job.created_at.isoformat(),
    }
