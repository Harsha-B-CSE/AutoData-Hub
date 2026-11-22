import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, BigInteger, Float, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.db.session import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # 'tabular' or 'vision'
    file_size_bytes = Column(BigInteger, nullable=False)
    sha256_hash = Column(String(64), nullable=False, index=True)
    storage_path = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    jobs = relationship("ProcessingJob", back_populates="dataset", cascade="all, delete-orphan")

class PipelineConfig(Base):
    __tablename__ = "pipeline_configs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    engine_type = Column(String(20), nullable=False)  # 'tabular' or 'vision'
    parameters_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    jobs = relationship("ProcessingJob", back_populates="config")

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    config_id = Column(String(36), ForeignKey("pipeline_configs.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(20), nullable=False, default="QUEUED", index=True)  # QUEUED, PROCESSING, COMPLETED, FAILED
    progress_percent = Column(Float, default=0.0, nullable=False)
    current_stage = Column(String(100), nullable=True)
    execution_mode = Column(String(20), default="batch", nullable=False)  # sequential, batch, parallel
    output_path = Column(String(512), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    dataset = relationship("Dataset", back_populates="jobs")
    config = relationship("PipelineConfig", back_populates="jobs")
    benchmarks = relationship("BenchmarkLog", back_populates="job", cascade="all, delete-orphan")
    quality_report = relationship("QualityReport", back_populates="job", uselist=False, cascade="all, delete-orphan")

class BenchmarkLog(Base):
    __tablename__ = "benchmark_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    stage_name = Column(String(100), nullable=False)
    latency_ms = Column(Float, nullable=False)
    memory_delta_mb = Column(Float, nullable=False)
    cpu_percent = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utc_now, nullable=False)

    job = relationship("ProcessingJob", back_populates="benchmarks")

class QualityReport(Base):
    __tablename__ = "quality_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    job_id = Column(String(36), ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, unique=True)
    pre_score = Column(Float, nullable=False)
    post_score = Column(Float, nullable=False)
    metrics_json = Column(Text, nullable=False)
    recommendations_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    job = relationship("ProcessingJob", back_populates="quality_report")
