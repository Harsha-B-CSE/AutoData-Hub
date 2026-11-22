import os
import json
import time
import zipfile
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import cv2

from backend.config import settings
from backend.db.session import SessionLocal
from backend.db.models import ProcessingJob, BenchmarkLog, QualityReport, Dataset, PipelineConfig
from backend.engines.performance.benchmark import BenchmarkContext
from backend.engines.performance.cache import PipelineCache
from backend.engines.performance.pool_manager import ExecutionMode, execute_workload
from backend.engines.tabular import (
    load_tabular_file,
    TabularProfiler,
    TabularCleaner,
    TabularImputer,
    TabularScaler,
    TabularEncoder,
)
from backend.engines.image import (
    ImageProfiler,
    profile_image,
    resize_image,
    crop_image,
    rotate_image,
    flip_image,
    convert_color_space,
    apply_filter,
    enhance_contrast,
    augment_image,
    adaptive_enhance_image,
)
from backend.engines.quality import (
    QualityScorer,
    QualityComparator,
    QualityRecommender,
)
from backend.core.security import safe_extract_zip, make_read_only

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ProcessingOrchestrator:
    @staticmethod
    def log_benchmark(db, job_id: str, stage_name: str, latency_ms: float, mem_delta: float, cpu: float):
        b_log = BenchmarkLog(
            job_id=job_id,
            stage_name=stage_name,
            latency_ms=latency_ms,
            memory_delta_mb=mem_delta,
            cpu_percent=cpu,
            timestamp=utc_now(),
        )
        db.add(b_log)
        db.commit()

    @staticmethod
    def update_job_progress(db, job_id: str, progress: float, stage: str, status: str = "PROCESSING"):
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if job:
            job.status = status
            job.progress_percent = progress
            job.current_stage = stage
            if status == "PROCESSING" and not job.started_at:
                job.started_at = utc_now()
            db.commit()

    @classmethod
    def run_job(cls, job_id: str):
        """
        Main worker execution entrypoint called in background.
        Manages DB session, runs tabular or vision pipeline, records benchmarks and reports.
        """
        db = SessionLocal()
        try:
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if not job:
                return

            dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            if not dataset:
                job.status = "FAILED"
                job.error_message = "Dataset record not found"
                db.commit()
                return

            # Retrieve parameters from pipeline_configs
            parameters: Dict[str, Any] = {}
            if job.config_id:
                cfg = db.query(PipelineConfig).filter(PipelineConfig.id == job.config_id).first()
                if cfg and cfg.parameters_json:
                    parameters = json.loads(cfg.parameters_json)

            raw_path = Path(dataset.storage_path)
            exec_mode = ExecutionMode(job.execution_mode) if job.execution_mode in ["sequential", "batch", "parallel"] else ExecutionMode.BATCH

            # Smart Skipping check via content & parameter hash caching
            should_skip, cached_result = PipelineCache.should_skip(
                dataset.sha256_hash,
                dataset.file_type,
                parameters
            )

            if should_skip and cached_result:
                cls.update_job_progress(db, job_id, 100.0, "Skipped (Retrieved from Cache)", status="COMPLETED")
                job.output_path = cached_result["output_path"]
                job.completed_at = utc_now()
                # Store cached quality report
                qr = QualityReport(
                    job_id=job_id,
                    pre_score=cached_result["pre_score"],
                    post_score=cached_result["post_score"],
                    metrics_json=json.dumps(cached_result["metrics"]),
                    recommendations_json=json.dumps(cached_result["recommendations"]),
                )
                db.add(qr)
                db.commit()
                return

            # Mark PROCESSING
            cls.update_job_progress(db, job_id, 5.0, "Initializing Engine", status="PROCESSING")

            if dataset.file_type == "tabular":
                cls._execute_tabular_pipeline(db, job, raw_path, parameters, exec_mode, dataset.sha256_hash)
            elif dataset.file_type == "vision":
                cls._execute_vision_pipeline(db, job, raw_path, parameters, exec_mode, dataset.sha256_hash)
            else:
                raise ValueError(f"Unknown dataset engine file_type: {dataset.file_type}")

        except Exception as e:
            traceback_str = traceback.format_exc()
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                job.status = "FAILED"
                job.error_message = f"{str(e)}\n\nTraceback:\n{traceback_str}"
                job.completed_at = utc_now()
                db.commit()
        finally:
            db.close()

    @classmethod
    def _execute_tabular_pipeline(
        cls,
        db,
        job: ProcessingJob,
        raw_path: Path,
        params: Dict[str, Any],
        mode: ExecutionMode,
        raw_sha256: str
    ):
        processed_dir = settings.PROCESSED_STORAGE_DIR / job.id
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Load file
        with BenchmarkContext("Tabular Ingestion") as b:
            cls.update_job_progress(db, job.id, 15.0, "Ingesting Tabular Data")
            df_raw = load_tabular_file(raw_path)
        cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

        # Stage 2: Profiling & Baseline Quality Score
        with BenchmarkContext("Tabular Profiling") as b:
            cls.update_job_progress(db, job.id, 30.0, "Profiling & Baseline Scoring")
            pre_profile = TabularProfiler.profile(df_raw)
            pre_quality = QualityScorer.score_tabular(df_raw)
            recommendations = QualityRecommender.recommend_tabular(pre_profile)
        cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

        # Stage 3: Cleaning & Deduplication & Outliers
        with BenchmarkContext("Tabular Cleaning") as b:
            cls.update_job_progress(db, job.id, 50.0, "Cleaning & Normalizing")
            clean_opts = params.get("cleaning", {})
            df_cleaned, _ = TabularCleaner.clean(
                df_raw,
                remove_dup=clean_opts.get("remove_duplicates", True),
                dup_subset=clean_opts.get("duplicate_subset"),
                dup_keep=clean_opts.get("duplicate_keep", "first"),
                text_clean=clean_opts.get("clean_text", True),
                text_case=clean_opts.get("text_case", "preserve"),
                type_correct=clean_opts.get("infer_types", True),
                outlier_handling=clean_opts.get("handle_outliers", False),
                outlier_method=clean_opts.get("outlier_method", "iqr"),
                outlier_action=clean_opts.get("outlier_action", "clip"),
                outlier_threshold=clean_opts.get("outlier_threshold", 1.5),
            )
        cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

        # Stage 4: Imputation
        with BenchmarkContext("Tabular Imputation") as b:
            cls.update_job_progress(db, job.id, 65.0, "Imputing Missing Values")
            imp_opts = params.get("imputation", {})
            df_imputed, imp_meta = TabularImputer.impute_with_meta(
                df_cleaned,
                numerical_strategy=imp_opts.get("numerical_strategy", "median"),
                categorical_strategy=imp_opts.get("categorical_strategy", "mode"),
                num_constant_value=imp_opts.get("num_constant", 0.0),
                cat_constant_value=imp_opts.get("cat_constant", "MISSING"),
                knn_neighbors=imp_opts.get("knn_neighbors", 5),
                columns_config=imp_opts.get("columns_config"),
            )
            imputation_summary = imp_meta.get("columns", {})
        cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

        # Stage 5: Scaling & Normalization
        with BenchmarkContext("Tabular Scaling") as b:
            cls.update_job_progress(db, job.id, 75.0, "Scaling & Normalizing Features")
            scale_opts = params.get("scaling", {})
            df_scaled = TabularScaler.scale(
                df_imputed,
                method=scale_opts.get("method", "standard"),
                columns=scale_opts.get("columns"),
                feature_range=tuple(scale_opts.get("feature_range", [0, 1])),
            )
        cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

        # Stage 6: Feature Encoding
        with BenchmarkContext("Tabular Encoding") as b:
            cls.update_job_progress(db, job.id, 85.0, "Encoding Categorical Features")
            enc_opts = params.get("encoding", {})
            df_final = TabularEncoder.encode(
                df_scaled,
                method=enc_opts.get("method", "onehot"),
                columns=enc_opts.get("columns"),
                exclude_columns=enc_opts.get("exclude_columns"),
                drop_first=enc_opts.get("drop_first", False),
                max_categories=enc_opts.get("max_categories", 25),
            )
        cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

        # Stage 7: Quality Evaluation & Comparison
        with BenchmarkContext("Quality Evaluation") as b:
            cls.update_job_progress(db, job.id, 92.0, "Assessing Post-Processing Quality")
            post_quality = QualityScorer.score_tabular(df_final)
            comparison = QualityComparator.compare_tabular(df_raw, df_final)
            recommendations = QualityRecommender.recommend_tabular(
                pre_profile,
                config=params,
                imputation_summary=imputation_summary
            )
        cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

        # Stage 8: Output Serialization
        cls.update_job_progress(db, job.id, 96.0, "Serializing Cleaned Datasets")
        out_csv = processed_dir / "cleaned_data.csv"
        out_xlsx = processed_dir / "cleaned_data.xlsx"
        df_final.to_csv(out_csv, index=False, encoding="utf-8")
        try:
            df_final.to_excel(out_xlsx, index=False, engine="openpyxl")
        except Exception:
            pass

        # Write manifest.json
        manifest = {
            "job_id": job.id,
            "raw_sha256": raw_sha256,
            "engine": "tabular",
            "parameters": params,
            "rows_in": len(df_raw),
            "rows_out": len(df_final),
            "columns_in": len(df_raw.columns),
            "columns_out": len(df_final.columns),
            "pre_score": pre_quality["composite_score"],
            "post_score": post_quality["composite_score"],
            "comparison": comparison,
            "imputation_strategies": imputation_summary,
            "timestamp": utc_now().isoformat(),
        }
        with open(processed_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Record Quality Report
        qr = QualityReport(
            job_id=job.id,
            pre_score=pre_quality["composite_score"],
            post_score=post_quality["composite_score"],
            metrics_json=json.dumps({
                "pre_quality": pre_quality,
                "post_quality": post_quality,
                "comparison": comparison,
                "imputation_strategies": imputation_summary,
            }),
            recommendations_json=json.dumps(recommendations),
        )
        db.add(qr)

        job.output_path = str(out_csv)
        job.status = "COMPLETED"
        job.progress_percent = 100.0
        job.current_stage = "Completed"
        job.completed_at = utc_now()
        db.commit()

        # Cache result
        PipelineCache.set(
            PipelineCache.generate_cache_key(raw_sha256, "tabular", params),
            {
                "output_path": str(out_csv),
                "pre_score": pre_quality["composite_score"],
                "post_score": post_quality["composite_score"],
                "metrics": {
                    "pre_quality": pre_quality,
                    "post_quality": post_quality,
                    "comparison": comparison,
                    "imputation_strategies": imputation_summary,
                },
                "recommendations": recommendations,
            }
        )

    @classmethod
    def _execute_vision_pipeline(
        cls,
        db,
        job: ProcessingJob,
        raw_path: Path,
        params: Dict[str, Any],
        mode: ExecutionMode,
        raw_sha256: str
    ):
        inter_dir = settings.INTERMEDIATE_STORAGE_DIR / job.id
        inter_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = settings.PROCESSED_STORAGE_DIR / job.id
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_images_dir = processed_dir / "processed_images"
        out_images_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage 1: Extraction / Ingestion
            with BenchmarkContext("Vision Ingestion & Extraction") as b:
                cls.update_job_progress(db, job.id, 15.0, "Extracting & Ingesting Visual Assets")
                raw_ext = raw_path.suffix.lower()
                image_files: List[Path] = []

                if raw_ext == ".zip":
                    extracted = safe_extract_zip(raw_path, inter_dir)
                    for p in extracted:
                        if p.suffix.lower() in settings.IMAGE_EXTENSIONS:
                            image_files.append(p)
                else:
                    # Single image file: copy to intermediate
                    stg_copy = inter_dir / raw_path.name
                    shutil.copy2(raw_path, stg_copy)
                    image_files.append(stg_copy)

            cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

            # Stage 2: Profiling & Baseline Quality Scoring
            with BenchmarkContext("Vision Profiling & Screening") as b:
                cls.update_job_progress(db, job.id, 30.0, "Screening Images & Sharpness Profiling")
                pre_profiles = [profile_image(p) for p in image_files]
                pre_quality = QualityScorer.score_image(pre_profiles)
                dataset_profile = ImageProfiler.profile(image_files)
                recommendations = QualityRecommender.recommend_image(dataset_profile)
            cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

            # Filter out corrupted images
            valid_image_files = [p for p, prof in zip(image_files, pre_profiles) if not prof.get("corrupted", False)]

            # Define per-image transformation worker function
            trans_opts = params.get("transform", {})
            filter_opts = params.get("filter", {})
            aug_opts = params.get("augmentation", {})
            adaptive_audits: Dict[str, Any] = {}

            def process_single_image(src_path: Path) -> Optional[Path]:
                try:
                    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        return None

                    # Resizing
                    if trans_opts.get("resize", True):
                        target_w = trans_opts.get("width", 512)
                        target_h = trans_opts.get("height", 512)
                        preserve_aspect = trans_opts.get("preserve_aspect_ratio", True)
                        interp = trans_opts.get("interpolation", "bilinear")
                        img = resize_image(
                            img,
                            target_width=target_w,
                            target_height=target_h,
                            interpolation=interp,
                            preserve_aspect_ratio=preserve_aspect
                        )

                    # Rotation & Flips
                    if trans_opts.get("rotation_angle"):
                        img = rotate_image(img, float(trans_opts["rotation_angle"]), expand_bounds=True)
                    if trans_opts.get("flip_mode"):
                        img = flip_image(img, mode=trans_opts["flip_mode"])

                    # Color Space Conversion
                    if filter_opts.get("color_space"):
                        img = convert_color_space(img, target_space=filter_opts["color_space"])

                    # Adaptive Image Enhancement Pipeline
                    adaptive_enabled = filter_opts.get("adaptive", True)
                    if adaptive_enabled:
                        preset = filter_opts.get("adaptive_preset", "balanced")
                        img, audit = adaptive_enhance_image(img, preset=preset)
                        adaptive_audits[src_path.name] = audit
                        adaptive_audits[f"proc_{src_path.name}"] = audit
                    else:
                        # Manual / Fixed Denoising & Contrast Enhancement
                        if filter_opts.get("filter_type"):
                            img = apply_filter(
                                img,
                                filter_type=filter_opts["filter_type"],
                                kernel_size=filter_opts.get("kernel_size", 5)
                            )
                        if filter_opts.get("contrast_enhancement"):
                            clip_lim = float(filter_opts.get("clahe_clip_limit", 2.0))
                            img = enhance_contrast(
                                img,
                                method=filter_opts["contrast_enhancement"],
                                clip_limit=clip_lim
                            )

                    # Augmentation (if enabled)
                    if aug_opts.get("enabled", False):
                        img = augment_image(
                            img,
                            horizontal_flip=aug_opts.get("horizontal_flip", True),
                            vertical_flip=aug_opts.get("vertical_flip", False),
                            color_jitter=aug_opts.get("color_jitter", False),
                            gaussian_noise=aug_opts.get("gaussian_noise", False)
                        )

                    # Save to processed directory
                    dest_path = out_images_dir / f"proc_{src_path.name}"
                    cv2.imwrite(str(dest_path), img)
                    return dest_path
                except Exception:
                    return None

            # Stage 3-5: Process workload through Performance Engine
            with BenchmarkContext("Vision Processing & Transformation") as b:
                cls.update_job_progress(db, job.id, 65.0, f"Applying Transformations ({mode.value.capitalize()} Mode)")
                processed_paths = execute_workload(
                    func=process_single_image,
                    items=valid_image_files,
                    mode=mode,
                    batch_size=20,
                    use_threads=True # Use ThreadPoolExecutor to prevent memory serialization overhead
                )
                processed_paths = [p for p in processed_paths if p is not None]
            cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

            # Stage 6: Post-Processing Profiling & Quality Evaluation
            with BenchmarkContext("Vision Quality Evaluation") as b:
                cls.update_job_progress(db, job.id, 85.0, "Evaluating Post-Processing Vision Quality")
                post_profiles = [profile_image(p) for p in processed_paths]
                post_quality = QualityScorer.score_image(post_profiles)
                comparison = QualityComparator.compare_images(pre_profiles, post_profiles)
            cls.log_benchmark(db, job.id, b.stage_name, b.result.latency_ms, b.result.memory_delta_mb, b.result.cpu_percent)

            # Stage 7: Packaging ZIP Archive & Manifest
            cls.update_job_progress(db, job.id, 92.0, "Packaging Output Archive & Manifest")
            out_zip_path = processed_dir / "processed_images.zip"
            with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for img_p in processed_paths:
                    zf.write(img_p, arcname=img_p.name)

            # Save adaptive enhancement audit log
            with open(processed_dir / "adaptive_enhancement.json", "w") as f:
                json.dump(adaptive_audits, f, indent=2)

            manifest = {
                "job_id": job.id,
                "raw_sha256": raw_sha256,
                "engine": "vision",
                "execution_mode": mode.value,
                "total_input_images": len(image_files),
                "total_processed_images": len(processed_paths),
                "pre_score": pre_quality["composite_score"],
                "post_score": post_quality["composite_score"],
                "comparison": comparison,
                "adaptive_enhancements": adaptive_audits,
                "timestamp": utc_now().isoformat(),
            }
            with open(processed_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            qr = QualityReport(
                job_id=job.id,
                pre_score=pre_quality["composite_score"],
                post_score=post_quality["composite_score"],
                metrics_json=json.dumps({
                    "engine_type": "vision",
                    "pre_quality": pre_quality,
                    "post_quality": post_quality,
                    "comparison": comparison,
                    "adaptive_enhancements": adaptive_audits,
                }),
                recommendations_json=json.dumps(recommendations),
            )
            db.add(qr)

            job.output_path = str(out_zip_path)
            job.status = "COMPLETED"
            job.progress_percent = 100.0
            job.current_stage = "Completed"
            job.completed_at = utc_now()
            db.commit()

            # Cache result
            PipelineCache.set(
                PipelineCache.generate_cache_key(raw_sha256, "vision", params),
                {
                    "output_path": str(out_zip_path),
                    "pre_score": pre_quality["composite_score"],
                    "post_score": post_quality["composite_score"],
                    "metrics": {
                        "pre_quality": pre_quality,
                        "post_quality": post_quality,
                        "comparison": comparison,
                    },
                    "recommendations": recommendations,
                }
            )

        finally:
            # Clean up intermediate staging files
            shutil.rmtree(inter_dir, ignore_errors=True)
