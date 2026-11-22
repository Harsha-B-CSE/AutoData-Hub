"""
AutoData-Hub Ensemble Anomaly Detection Engine (Module 3).

Combines three complementary anomaly detection algorithms:
  - Isolation Forest (global tree-based density isolation)
  - Local Outlier Factor (LOF, neighborhood density contrast)
  - One-Class SVM (kernel-based boundary estimation)

Ensemble Voting:
  A record is flagged as an anomaly when >= 2 of 3 detectors independently
  classify it as an outlier (majority vote: V >= 2/3).

Probability Calibration:
  Empirical probability scores are computed per detector via quantile
  calibration of raw decision function / score_samples outputs, yielding
  a unified [0, 1] anomaly probability column (anomaly_score_ensemble).

Outputs:
  - ModuleResult with metrics, column-level artifact figures (Plotly dark),
    and anomaly-flagged DataFrame.
  - TransformResult with zero in-place mutations and reproducible code snippets.

Strictly decoupled from UI frameworks (zero Streamlit dependencies).
"""

from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from core.schema_inference import (
    ExecutionStatus,
    ModuleResult,
    TransformationAction,
    TransformResult,
)
from modules.anomaly.calibrator import (
    _normalize_score_matrix,
    calibrate_ensemble_scores,
)


# ---------------------------------------------------------------------------
# Enums & Frozen Config Dataclasses
# ---------------------------------------------------------------------------


class AnomalyAlgorithm(str, Enum):
    """Supported anomaly detection algorithms."""
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    ONE_CLASS_SVM = "one_class_svm"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class AnomalyConfig:
    """
    Immutable configuration for the Ensemble Anomaly Engine.

    Parameters
    ----------
    contamination : float
        Expected fraction of outliers in the dataset [0.01, 0.5].
    n_estimators : int
        Number of trees for Isolation Forest.
    lof_n_neighbors : int
        Number of neighbors for Local Outlier Factor.
    svm_nu : float
        Upper bound on the fraction of margin errors for OneClassSVM.
    svm_kernel : str
        Kernel type for OneClassSVM ('rbf', 'poly', 'sigmoid').
    vote_threshold : int
        Minimum number of detectors required to flag an anomaly (default 2).
    random_state : int
        Reproducibility seed for stochastic algorithms.
    max_features_svm : int
        Maximum numeric features to pass to SVM (avoids high-dim blow-up).
    """
    contamination: float = 0.05
    n_estimators: int = 100
    lof_n_neighbors: int = 20
    svm_nu: float = 0.05
    svm_kernel: str = "rbf"
    vote_threshold: int = 2
    random_state: int = 42
    max_features_svm: int = 30
    threshold_mode: str = "adaptive_3sigma"
    k_sigma: float = 3.0
    min_vote_ratio: float = 0.66
    generate_artifacts: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination < 1.0:
            raise ValueError("contamination must be between 0 and 1.")
        if self.threshold_mode not in {"adaptive_3sigma", "fixed_quantile"}:
            raise ValueError("Unsupported anomaly threshold mode.")
        if self.k_sigma < 0.0:
            raise ValueError("k_sigma must be non-negative.")
        if not 0.0 < self.min_vote_ratio <= 1.0:
            raise ValueError("min_vote_ratio must be in (0, 1].")


@dataclass(frozen=True)
class DetectorScore:
    """Per-detector result for a single algorithm run."""
    algorithm: AnomalyAlgorithm
    raw_scores: np.ndarray          # Decision function or score_samples output
    binary_flags: np.ndarray        # 1 = normal, -1 = outlier (sklearn convention)
    calibrated_probs: np.ndarray    # [0, 1] anomaly probability
    n_detected: int
    detection_rate: float           # fraction flagged as anomalous

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "n_detected": self.n_detected,
            "detection_rate": round(self.detection_rate, 4),
        }


# ---------------------------------------------------------------------------
# Utility: Quantile Probability Calibration
# ---------------------------------------------------------------------------


def _calibrate_scores(raw_scores: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    Calibrates raw detector decision function output to [0, 1] anomaly probability
    via empirical rank-based quantile mapping.

    Parameters
    ----------
    raw_scores : array of shape (n_samples,)
        Raw decision function or score_samples output (lower = more anomalous).
    invert : bool
        If True, invert so that lower raw scores -> higher anomaly probability.

    Returns
    -------
    calibrated : array of shape (n_samples,) in [0, 1]
    """
    scores = raw_scores.copy().astype(float)
    if invert:
        scores = -scores  # Now higher = more anomalous

    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.zeros(len(scores))

    # MinMax scale to [0, 1]
    s_min = scores[finite_mask].min()
    s_max = scores[finite_mask].max()
    if s_max == s_min:
        return np.zeros(len(scores))

    calibrated = np.where(finite_mask, (scores - s_min) / (s_max - s_min), 0.0)
    return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Core Ensemble Detector Class
# ---------------------------------------------------------------------------


class EnsembleAnomalyDetector:
    """
    Multi-algorithm ensemble anomaly detection engine.

    Combines Isolation Forest, LOF, and One-Class SVM predictions via
    configurable majority voting and emits unified anomaly probability scores.
    """

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()

    # -----------------------------------------------------------------------
    # Individual Detectors
    # -----------------------------------------------------------------------

    def _run_isolation_forest(self, X: np.ndarray) -> DetectorScore:
        """Runs Isolation Forest on numeric matrix X."""
        cfg = self.config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iso = IsolationForest(
                n_estimators=cfg.n_estimators,
                contamination=(
                    "auto" if cfg.threshold_mode == "adaptive_3sigma"
                    else cfg.contamination
                ),
                random_state=cfg.random_state,
                n_jobs=-1,
            )
            iso.fit(X)
            raw = iso.score_samples(X)         # Lower = more anomalous
            flags = iso.predict(X)             # -1 = outlier, 1 = normal

        probs = _calibrate_scores(raw, invert=True)
        n_detected = int((flags == -1).sum())
        return DetectorScore(
            algorithm=AnomalyAlgorithm.ISOLATION_FOREST,
            raw_scores=raw,
            binary_flags=flags,
            calibrated_probs=probs,
            n_detected=n_detected,
            detection_rate=n_detected / len(flags),
        )

    def _run_lof(self, X: np.ndarray) -> DetectorScore:
        """Runs Local Outlier Factor on numeric matrix X."""
        cfg = self.config
        n_neighbors = min(cfg.lof_n_neighbors, max(1, len(X) - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=(
                    "auto" if cfg.threshold_mode == "adaptive_3sigma"
                    else cfg.contamination
                ),
                novelty=False,
                n_jobs=-1,
            )
            flags = lof.fit_predict(X)         # -1 = outlier, 1 = normal
            raw = lof.negative_outlier_factor_  # More negative = more anomalous

        probs = _calibrate_scores(raw, invert=True)
        n_detected = int((flags == -1).sum())
        return DetectorScore(
            algorithm=AnomalyAlgorithm.LOF,
            raw_scores=raw,
            binary_flags=flags,
            calibrated_probs=probs,
            n_detected=n_detected,
            detection_rate=n_detected / len(flags),
        )

    def _run_one_class_svm(self, X: np.ndarray) -> DetectorScore:
        """Runs One-Class SVM on numeric matrix X (max_features_svm features)."""
        cfg = self.config
        # Clip to max features to avoid quadratic kernel cost on wide datasets
        X_svm = X[:, : cfg.max_features_svm] if X.shape[1] > cfg.max_features_svm else X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svm = OneClassSVM(
                nu=cfg.svm_nu,
                kernel=cfg.svm_kernel,
            )
            svm.fit(X_svm)
            raw = svm.decision_function(X_svm)  # Negative = outlier side
            flags = svm.predict(X_svm)           # -1 = outlier, 1 = normal

        probs = _calibrate_scores(raw, invert=True)
        n_detected = int((flags == -1).sum())
        return DetectorScore(
            algorithm=AnomalyAlgorithm.ONE_CLASS_SVM,
            raw_scores=raw,
            binary_flags=flags,
            calibrated_probs=probs,
            n_detected=n_detected,
            detection_rate=n_detected / len(flags),
        )

    # -----------------------------------------------------------------------
    # Preprocessing: Numeric Feature Extraction
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_numeric_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[Any]]:
        """
        Extracts numeric (non-boolean, non-indicator) columns, imputes residual
        NaNs with column medians, and scales with RobustScaler.

        Returns (scaled_matrix, column_names)
        """
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
            and not str(c).endswith("_is_missing")
            and not str(c).startswith("anomaly_")
        ]

        if not numeric_cols:
            return np.empty((len(df), 0)), []

        sub = df[numeric_cols].copy().replace([np.inf, -np.inf], np.nan)

        # Residual NaN imputation: fill with column median
        for col in numeric_cols:
            if sub[col].isna().any():
                fill_val = sub[col].median()
                if pd.isna(fill_val):
                    fill_val = 0.0
                sub[col] = sub[col].fillna(fill_val)

        X = sub.to_numpy(dtype=float)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, numeric_cols

    # -----------------------------------------------------------------------
    # Main Detection Pipeline
    # -----------------------------------------------------------------------

    def detect(
        self,
        df: pd.DataFrame,
        dataset_version: str = "v2_imputed",
        start_step_id: int = 1,
        algorithms: Optional[List[str]] = None,
    ) -> Tuple[TransformResult, ModuleResult]:
        """
        Runs the full ensemble anomaly detection pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Imputed dataset (v2_imputed) from Module 2.
        dataset_version : str
            Active dataset version tag.
        start_step_id : int
            Step ID counter offset for action ledger.
        algorithms : Optional[List[str]]
            Subset of algorithms to run; defaults to all three.

        Returns
        -------
        (TransformResult, ModuleResult)
        """
        if df is None:
            return (
                TransformResult.failure("Cannot run anomaly detection on None DataFrame."),
                ModuleResult(
                    status=ExecutionStatus.FAILED,
                    module_name="anomaly",
                    dataset_version=dataset_version,
                    issues=["Input DataFrame is None."],
                ),
            )
        if not df.columns.is_unique:
            message = "Anomaly detection requires unique column labels."
            return (
                TransformResult.failure(message),
                ModuleResult(
                    status=ExecutionStatus.FAILED,
                    module_name="anomaly",
                    dataset_version=dataset_version,
                    issues=[message],
                ),
            )

        provided_algorithms = (
            ["isolation_forest", "lof", "one_class_svm"]
            if algorithms is None
            else list(dict.fromkeys(algorithms))
        )
        canonical_order = ["isolation_forest", "lof", "one_class_svm"]
        requested_algorithms = [
            name for name in canonical_order if name in provided_algorithms
        ] + [
            name for name in provided_algorithms if name not in canonical_order
        ]
        result_df = df.copy(deep=True)
        actions: List[TransformationAction] = []
        audit_log: List[str] = []
        issues: List[str] = []

        def ts() -> str:
            return datetime.datetime.now(datetime.timezone.utc).isoformat()

        audit_log.append(
            f"[{ts()}] [ANOMALY] Starting Ensemble Anomaly Engine on dataset version '{dataset_version}'."
        )

        def zero_anomaly_result(message: str, numeric_feature_count: int = 0):
            result_df["anomaly_flag"] = np.zeros(len(result_df), dtype=int)
            result_df["anomaly_score_ensemble"] = np.zeros(len(result_df), dtype=float)
            result_df["anomaly_vote_count"] = np.zeros(len(result_df), dtype=int)
            audit_log.append(f"[{ts()}] [ANOMALY] {message}")
            zero_action = TransformationAction(
                step_id=start_step_id,
                module="anomaly",
                action_type="annotate_zero_anomalies",
                target_columns=[
                    "anomaly_flag",
                    "anomaly_score_ensemble",
                    "anomaly_vote_count",
                ],
                parameters={"reason": message, "n_anomalies": 0},
                code_snippet=(
                    "df['anomaly_flag'] = 0\n"
                    "df['anomaly_score_ensemble'] = 0.0\n"
                    "df['anomaly_vote_count'] = 0"
                ),
            )
            metrics = {
                "n_rows": len(result_df),
                "n_numeric_features": numeric_feature_count,
                "n_anomalies": 0,
                "anomaly_rate": 0.0,
                "anomaly_rate_pct": 0.0,
                "threshold_mode": self.config.threshold_mode,
                "dynamic_cutoff": 0.0,
                "k_sigma": self.config.k_sigma,
                "min_vote_ratio": self.config.min_vote_ratio,
                "required_votes": 0,
                "active_model_count": 0,
                "active_algorithms": [],
                "detector_summaries": {},
                "vote_consensus_counts": {
                    "zero_models": len(result_df),
                    "one_model": 0,
                    "two_models": 0,
                    "three_models": 0,
                },
                "detector_disagreement_counts": {},
                "contamination_param": self.config.contamination,
            }
            return (
                TransformResult(
                    status=ExecutionStatus.SUCCESS,
                    transformed_df=result_df,
                    applied_actions=[zero_action],
                    audit_log=audit_log,
                ),
                ModuleResult(
                    status=ExecutionStatus.SUCCESS,
                    module_name="anomaly",
                    dataset_version=dataset_version,
                    metrics=metrics,
                    issues=[message],
                    artifacts={},
                ),
            )

        if len(result_df) == 0:
            return zero_anomaly_result("Dataset has no rows; zero anomalies flagged.")

        X, numeric_cols = self._extract_numeric_matrix(result_df)
        if X.shape[1] == 0:
            return zero_anomaly_result(
                "No continuous numeric columns available for anomaly scoring."
            )
        if len(result_df) == 1:
            return zero_anomaly_result(
                "Single-row dataset cannot support ensemble fitting; zero anomalies flagged.",
                numeric_feature_count=len(numeric_cols),
            )

        n_rows = len(result_df)
        audit_log.append(
            f"[{ts()}] [ANOMALY] Extracted {X.shape[1]} numeric features from {n_rows} rows."
        )

        algo_map = {
            "isolation_forest": self._run_isolation_forest,
            "lof": self._run_lof,
            "one_class_svm": self._run_one_class_svm,
        }
        invalid_algorithms = [name for name in requested_algorithms if name not in algo_map]
        if invalid_algorithms:
            issues.append(
                "Ignored unsupported detectors: " + ", ".join(invalid_algorithms)
            )

        detector_scores: Dict[str, DetectorScore] = {}
        raw_score_columns: List[np.ndarray] = []
        successful_names: List[str] = []
        for algo_name in requested_algorithms:
            runner = algo_map.get(algo_name)
            if runner is None:
                continue
            try:
                audit_log.append(f"[{ts()}] [ANOMALY] Running {algo_name.upper()}...")
                score = runner(X)
                detector_scores[algo_name] = score
                successful_names.append(algo_name)
                raw_score_columns.append(-np.asarray(score.raw_scores, dtype=float))
            except Exception as exc:
                error = f"[ANOMALY] {algo_name.upper()} failed: {exc!s}"
                audit_log.append(f"[{ts()}] {error}")
                issues.append(error)

        if not raw_score_columns:
            message = "No anomaly detector completed successfully; zero anomalies flagged."
            if issues:
                message += " " + "; ".join(issues)
            return zero_anomaly_result(message, numeric_feature_count=len(numeric_cols))

        oriented_scores = np.column_stack(raw_score_columns)
        normalized_scores = _normalize_score_matrix(oriented_scores)
        ensemble_flags, ensemble_probs, dynamic_cutoff = calibrate_ensemble_scores(
            oriented_scores,
            threshold_mode=self.config.threshold_mode,
            contamination=self.config.contamination,
            k_sigma=self.config.k_sigma,
            min_vote_ratio=self.config.min_vote_ratio,
        )
        individual_flags = (normalized_scores >= 0.5).astype(int)
        vote_counts = individual_flags.sum(axis=1)
        active_model_count = len(successful_names)
        required_votes = int(np.ceil(self.config.min_vote_ratio * active_model_count))
        vote_consensus_counts = {
            "zero_models": int((vote_counts == 0).sum()),
            "one_model": int((vote_counts == 1).sum()),
            "two_models": int((vote_counts == 2).sum()),
            "three_models": int((vote_counts == 3).sum()),
        }
        disagreement_counts: Dict[str, Dict[str, int]] = {}
        disagreement_matrix = np.zeros(
            (active_model_count, active_model_count), dtype=int
        )
        for row_index, row_name in enumerate(successful_names):
            disagreement_counts[row_name] = {}
            for column_index, column_name in enumerate(successful_names):
                count = int(
                    np.not_equal(
                        individual_flags[:, row_index],
                        individual_flags[:, column_index],
                    ).sum()
                )
                disagreement_matrix[row_index, column_index] = count
                disagreement_counts[row_name][column_name] = count

        for index, algo_name in enumerate(successful_names):
            original = detector_scores[algo_name]
            model_flags = individual_flags[:, index]
            n_detected = int(model_flags.sum())
            detector_scores[algo_name] = DetectorScore(
                algorithm=original.algorithm,
                raw_scores=original.raw_scores,
                binary_flags=np.where(model_flags == 1, -1, 1),
                calibrated_probs=normalized_scores[:, index],
                n_detected=n_detected,
                detection_rate=n_detected / n_rows,
            )
            result_df[f"anomaly_{algo_name}"] = model_flags

        result_df["anomaly_flag"] = ensemble_flags
        result_df["anomaly_score_ensemble"] = np.round(ensemble_probs, 6)
        result_df["anomaly_vote_count"] = vote_counts

        n_anomalies = int(ensemble_flags.sum())
        anomaly_rate = round(n_anomalies / n_rows, 6)

        model_contamination = (
            "auto" if self.config.threshold_mode == "adaptive_3sigma"
            else self.config.contamination
        )
        code_lines = [
            "from sklearn.preprocessing import RobustScaler",
            "import numpy as np",
        ]
        if "isolation_forest" in successful_names:
            code_lines.append("from sklearn.ensemble import IsolationForest")
        if "lof" in successful_names:
            code_lines.append("from sklearn.neighbors import LocalOutlierFactor")
        if "one_class_svm" in successful_names:
            code_lines.append("from sklearn.svm import OneClassSVM")
        code_lines.extend([
            "",
            f"numeric_cols = {numeric_cols!r}",
            "_anomaly_data = df[numeric_cols].copy().replace([np.inf, -np.inf], np.nan)",
            "for _anomaly_column in numeric_cols:",
            "    _anomaly_fill = _anomaly_data[_anomaly_column].median()",
            "    if pd.isna(_anomaly_fill):",
            "        _anomaly_fill = 0.0",
            "    _anomaly_data[_anomaly_column] = _anomaly_data[_anomaly_column].fillna(_anomaly_fill)",
            "_anomaly_X = RobustScaler().fit_transform(_anomaly_data.to_numpy(dtype=float))",
            "_anomaly_raw_scores = []",
        ])
        if "isolation_forest" in successful_names:
            code_lines.extend([
                f"_anomaly_iso = IsolationForest(n_estimators={self.config.n_estimators}, "
                f"contamination={model_contamination!r}, random_state={self.config.random_state}, n_jobs=-1)",
                "_anomaly_iso.fit(_anomaly_X)",
                "_anomaly_raw_scores.append(-_anomaly_iso.score_samples(_anomaly_X))",
            ])
        if "lof" in successful_names:
            code_lines.extend([
                f"_anomaly_neighbors = min({self.config.lof_n_neighbors}, max(1, len(_anomaly_X) - 1))",
                f"_anomaly_lof = LocalOutlierFactor(n_neighbors=_anomaly_neighbors, "
                f"contamination={model_contamination!r}, n_jobs=-1)",
                "_anomaly_lof.fit_predict(_anomaly_X)",
                "_anomaly_raw_scores.append(-_anomaly_lof.negative_outlier_factor_)",
            ])
        if "one_class_svm" in successful_names:
            code_lines.extend([
                f"_anomaly_X_svm = _anomaly_X[:, :{self.config.max_features_svm}]",
                f"_anomaly_svm = OneClassSVM(nu={self.config.svm_nu}, kernel={self.config.svm_kernel!r})",
                "_anomaly_svm.fit(_anomaly_X_svm)",
                "_anomaly_raw_scores.append(-_anomaly_svm.decision_function(_anomaly_X_svm))",
            ])
        code_lines.extend([
            "_anomaly_scores = np.column_stack(_anomaly_raw_scores)",
            "for _anomaly_index in range(_anomaly_scores.shape[1]):",
            "    _anomaly_column = _anomaly_scores[:, _anomaly_index]",
            "    _anomaly_finite = np.isfinite(_anomaly_column)",
            "    if not _anomaly_finite.any():",
            "        _anomaly_scores[:, _anomaly_index] = 0.0",
            "    else:",
            "        _anomaly_scores[~_anomaly_finite, _anomaly_index] = _anomaly_column[_anomaly_finite].min()",
            "_anomaly_mins = np.min(_anomaly_scores, axis=0)",
            "_anomaly_ptp = np.ptp(_anomaly_scores, axis=0)",
            "_anomaly_ptp[_anomaly_ptp == 0] = 1.0",
            "_anomaly_norm = np.clip((_anomaly_scores - _anomaly_mins) / (_anomaly_ptp + 1e-9), 0.0, 1.0)",
            "_anomaly_ensemble = np.mean(_anomaly_norm, axis=1)",
            f"_anomaly_threshold_mode = {self.config.threshold_mode!r}",
            "_anomaly_mu = float(np.mean(_anomaly_ensemble))",
            "_anomaly_sigma = float(np.std(_anomaly_ensemble))",
            "if _anomaly_threshold_mode == 'adaptive_3sigma':",
            f"    _anomaly_cutoff = _anomaly_mu + 1.0 if _anomaly_sigma < 1e-6 else _anomaly_mu + ({self.config.k_sigma} * _anomaly_sigma)",
            "else:",
            f"    _anomaly_cutoff = float(np.quantile(_anomaly_ensemble, {1.0 - self.config.contamination!r}))",
            "_anomaly_individual = (_anomaly_norm >= 0.5).astype(int)",
            "df['anomaly_vote_count'] = np.sum(_anomaly_individual, axis=1)",
            f"_anomaly_required = int(np.ceil({self.config.min_vote_ratio} * _anomaly_norm.shape[1]))",
            "df['anomaly_score_ensemble'] = np.round(_anomaly_ensemble, 6)",
            "df['anomaly_flag'] = ((_anomaly_ensemble >= _anomaly_cutoff) & "
            "(df['anomaly_vote_count'] >= _anomaly_required)).astype(int)",
        ])
        for index, algo_name in enumerate(successful_names):
            code_lines.append(
                f"df['anomaly_{algo_name}'] = _anomaly_individual[:, {index}]"
            )
        code_snippet = "\n".join(code_lines)

        actions.append(TransformationAction(
            step_id=start_step_id,
            module="anomaly",
            action_type="ensemble_anomaly_detection",
            target_columns=(
                ["anomaly_flag", "anomaly_score_ensemble", "anomaly_vote_count"]
                + [f"anomaly_{name}" for name in successful_names]
            ),
            parameters={
                "algorithms": successful_names,
                "threshold_mode": self.config.threshold_mode,
                "dynamic_cutoff": dynamic_cutoff,
                "contamination": self.config.contamination,
                "k_sigma": self.config.k_sigma,
                "min_vote_ratio": self.config.min_vote_ratio,
                "required_votes": required_votes,
                "n_anomalies": n_anomalies,
                "anomaly_rate": anomaly_rate,
            },
            code_snippet=code_snippet,
        ))

        audit_log.append(
            f"[{ts()}] [ANOMALY] Calibration complete: {n_anomalies} anomalies flagged "
            f"({anomaly_rate * 100:.2f}%) at cutoff {dynamic_cutoff:.6f}; "
            f"required votes {required_votes}/{active_model_count}."
        )
        artifacts = {}
        if self.config.generate_artifacts:
            artifacts = self._build_figures(
                result_df=result_df,
                numeric_cols=numeric_cols,
                detector_scores=detector_scores,
                ensemble_probs=ensemble_probs,
                ensemble_flags=ensemble_flags,
            )
            artifacts["detector_disagreement_matrix"] = pd.DataFrame(
                disagreement_matrix,
                index=successful_names,
                columns=successful_names,
            )
            artifacts["pre_mitigation_flags"] = ensemble_flags.tolist()
            artifacts["flagged_records"] = result_df.loc[
                result_df["anomaly_flag"].eq(1)
            ].copy(deep=True)
        metrics = {
            "n_rows": n_rows,
            "n_numeric_features": len(numeric_cols),
            "n_anomalies": n_anomalies,
            "anomaly_rate": anomaly_rate,
            "anomaly_rate_pct": round(anomaly_rate * 100, 2),
            "vote_threshold": required_votes,
            "required_votes": required_votes,
            "active_model_count": active_model_count,
            "threshold_mode": self.config.threshold_mode,
            "dynamic_cutoff": dynamic_cutoff,
            "k_sigma": self.config.k_sigma,
            "min_vote_ratio": self.config.min_vote_ratio,
            "active_algorithms": successful_names,
            "detector_summaries": {
                name: score.to_dict() for name, score in detector_scores.items()
            },
            "vote_consensus_counts": vote_consensus_counts,
            "detector_disagreement_counts": disagreement_counts,
            "contamination_param": self.config.contamination,
        }
        final_status = ExecutionStatus.SUCCESS if not issues else ExecutionStatus.PARTIAL
        return (
            TransformResult(
                status=final_status,
                transformed_df=result_df,
                applied_actions=actions,
                audit_log=audit_log,
                error_message="; ".join(issues) if issues else None,
            ),
            ModuleResult(
                status=final_status,
                module_name="anomaly",
                dataset_version=dataset_version,
                metrics=metrics,
                issues=issues,
                artifacts=artifacts,
            ),
        )

    # -----------------------------------------------------------------------
    # Plotly Figure Builders
    # -----------------------------------------------------------------------

    def _build_figures(
        self,
        result_df: pd.DataFrame,
        numeric_cols: List[Any],
        detector_scores: Dict[str, DetectorScore],
        ensemble_probs: np.ndarray,
        ensemble_flags: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Builds 4 interactive Plotly dark-themed figures for the anomaly tab.
        Falls back gracefully if plotly is unavailable.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return {}

        artifacts: Dict[str, Any] = {}

        # Figure 1: Ensemble Anomaly Score Distribution
        try:
            normal_mask = ensemble_flags == 0
            anomaly_mask = ensemble_flags == 1

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=ensemble_probs[normal_mask],
                name="Normal",
                marker_color="rgba(56, 189, 248, 0.75)",
                nbinsx=40,
                opacity=0.85,
            ))
            fig_dist.add_trace(go.Histogram(
                x=ensemble_probs[anomaly_mask],
                name="Anomaly",
                marker_color="rgba(248, 113, 113, 0.85)",
                nbinsx=40,
                opacity=0.85,
            ))
            fig_dist.update_layout(
                template="plotly_dark",
                title="Ensemble Anomaly Score Distribution",
                xaxis_title="Anomaly Probability Score",
                yaxis_title="Record Count",
                barmode="overlay",
                legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
                paper_bgcolor="rgba(15,17,26,1)",
                plot_bgcolor="rgba(22,27,42,1)",
                margin=dict(l=50, r=30, t=60, b=50),
            )
            artifacts["anomaly_score_distribution"] = fig_dist
        except Exception:
            pass

        # Figure 2: Vote Count Distribution
        try:
            vote_counts_arr = result_df["anomaly_vote_count"].to_numpy()
            max_votes = max(1, len(detector_scores))
            vote_hist = np.bincount(vote_counts_arr, minlength=max_votes + 1)

            fig_vote = go.Figure(go.Bar(
                x=[f"{i} detectors" for i in range(max_votes + 1)],
                y=vote_hist.tolist(),
                marker_color=["#38bdf8", "#818cf8", "#fb923c", "#f87171"][:max_votes + 1],
                text=vote_hist.tolist(),
                textposition="outside",
            ))
            fig_vote.update_layout(
                template="plotly_dark",
                title="Detector Agreement: Vote Count Distribution",
                xaxis_title="Number of Detectors Flagging as Anomaly",
                yaxis_title="Record Count",
                paper_bgcolor="rgba(15,17,26,1)",
                plot_bgcolor="rgba(22,27,42,1)",
                margin=dict(l=50, r=30, t=60, b=50),
            )
            artifacts["detector_vote_distribution"] = fig_vote
        except Exception:
            pass

        # Figure 3: Per-Detector Detection Rate
        try:
            algo_names = list(detector_scores.keys())
            detection_rates = [detector_scores[a].detection_rate * 100 for a in algo_names]
            display_names = [a.replace("_", " ").title() for a in algo_names]

            fig_rates = go.Figure(go.Bar(
                x=display_names,
                y=detection_rates,
                marker_color=["#38bdf8", "#a78bfa", "#fb923c"],
                text=[f"{r:.1f}%" for r in detection_rates],
                textposition="outside",
            ))
            fig_rates.update_layout(
                template="plotly_dark",
                title="Per-Detector Anomaly Detection Rate",
                xaxis_title="Algorithm",
                yaxis_title="Detection Rate (%)",
                paper_bgcolor="rgba(15,17,26,1)",
                plot_bgcolor="rgba(22,27,42,1)",
                margin=dict(l=50, r=30, t=60, b=50),
            )
            artifacts["detector_comparison"] = fig_rates
        except Exception:
            pass

        # Figure 4: PCA 2D Scatter with Anomaly Overlay
        try:
            if len(numeric_cols) >= 2:
                from sklearn.decomposition import PCA

                X_2d, _ = self._extract_numeric_matrix(result_df)
                if X_2d.shape[1] >= 2 and not np.all(
                    np.std(X_2d, axis=0) < 1e-12
                ):
                    pca = PCA(n_components=2, random_state=42)
                    components = pca.fit_transform(X_2d)

                    normal_x = components[ensemble_flags == 0, 0]
                    normal_y = components[ensemble_flags == 0, 1]
                    anom_x = components[ensemble_flags == 1, 0]
                    anom_y = components[ensemble_flags == 1, 1]

                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=normal_x.tolist(),
                        y=normal_y.tolist(),
                        mode="markers",
                        name="Normal",
                        marker=dict(color="rgba(56,189,248,0.5)", size=5),
                    ))
                    fig_scatter.add_trace(go.Scatter(
                        x=anom_x.tolist(),
                        y=anom_y.tolist(),
                        mode="markers",
                        name="Anomaly",
                        marker=dict(
                            color="rgba(248,113,113,0.9)",
                            size=8,
                            symbol="x",
                            line=dict(width=1, color="white"),
                        ),
                    ))
                    fig_scatter.update_layout(
                        template="plotly_dark",
                        title=f"PCA 2D Projection — Anomaly Overlay (n={int(ensemble_flags.sum())} flagged)",
                        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
                        paper_bgcolor="rgba(15,17,26,1)",
                        plot_bgcolor="rgba(22,27,42,1)",
                        margin=dict(l=50, r=30, t=60, b=50),
                        legend=dict(orientation="h", y=1.02),
                    )
                    artifacts["pca_anomaly_scatter"] = fig_scatter
        except Exception:
            pass

        return artifacts


# ---------------------------------------------------------------------------
# Convenience Module API
# ---------------------------------------------------------------------------


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    algorithms: Optional[List[str]] = None,
    vote_threshold: int = 2,
    dataset_version: str = "v2_imputed",
    random_state: int = 42,
    start_step_id: int = 1,
    threshold_mode: str = "adaptive_3sigma",
    k_sigma: float = 3.0,
    min_vote_ratio: float = 0.66,
    generate_artifacts: bool = True,
) -> Tuple[TransformResult, ModuleResult]:
    """
    Convenience orchestrator for Ensemble Anomaly Detection (Module 3).

    Parameters
    ----------
    df : pd.DataFrame
        Imputed dataset (v2_imputed) from Module 2.
    contamination : float
        Expected fraction of outliers [0.01, 0.5].
    algorithms : Optional[List[str]]
        List of algorithms to activate: 'isolation_forest', 'lof', 'one_class_svm'.
    vote_threshold : int
        Number of detector agreements required to flag an anomaly (default 2).
    dataset_version : str
        Active version tag.
    random_state : int
        Reproducibility seed.
    start_step_id : int
        Starting step ID for action ledger.

    Returns
    -------
    (TransformResult, ModuleResult)
    """
    config = AnomalyConfig(
        contamination=contamination,
        vote_threshold=vote_threshold,
        random_state=random_state,
        threshold_mode=threshold_mode,
        k_sigma=k_sigma,
        min_vote_ratio=min_vote_ratio,
        generate_artifacts=generate_artifacts,
    )
    detector = EnsembleAnomalyDetector(config=config)
    return detector.detect(
        df=df,
        dataset_version=dataset_version,
        start_step_id=start_step_id,
        algorithms=algorithms,
    )
