"""
AutoData-Hub Module 3: Ensemble Anomaly Detection Engine.

Exports the primary detection interface:
  - detect_anomalies(): Convenience function for the full ensemble pipeline.
  - EnsembleAnomalyDetector: Configurable class-based interface.
  - AnomalyConfig: Frozen immutable configuration dataclass.
  - AnomalyAlgorithm: Enum of supported algorithm identifiers.
"""

from modules.anomaly.detector import (
    AnomalyAlgorithm,
    AnomalyConfig,
    DetectorScore,
    EnsembleAnomalyDetector,
    detect_anomalies,
)
from modules.anomaly.calibrator import calibrate_ensemble_scores

__all__ = [
    "AnomalyAlgorithm",
    "AnomalyConfig",
    "DetectorScore",
    "EnsembleAnomalyDetector",
    "detect_anomalies",
    "calibrate_ensemble_scores",
]
