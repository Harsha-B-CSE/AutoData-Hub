"""Score calibration utilities for the anomaly ensemble."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _normalize_score_matrix(scores_matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize anomaly-oriented model scores without mutating input."""
    scores = np.asarray(scores_matrix, dtype=float).copy()
    if scores.ndim != 2:
        raise ValueError("scores_matrix must be a 2-D array.")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        return np.zeros(scores.shape, dtype=float)

    for column_index in range(scores.shape[1]):
        column = scores[:, column_index]
        finite_mask = np.isfinite(column)
        if not finite_mask.any():
            scores[:, column_index] = 0.0
            continue
        finite_min = float(column[finite_mask].min())
        scores[~finite_mask, column_index] = finite_min

    minimums = np.min(scores, axis=0)
    ptp_values = np.ptp(scores, axis=0)
    ptp_values[ptp_values == 0] = 1.0
    normalized = (scores - minimums) / (ptp_values + 1e-9)
    return np.clip(normalized, 0.0, 1.0)


def calibrate_ensemble_scores(
    scores_matrix: np.ndarray,
    threshold_mode: str = "adaptive_3sigma",
    contamination: float = 0.05,
    k_sigma: float = 3.0,
    min_vote_ratio: float = 0.66,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calibrate multi-model scores using adaptive 3-sigma or a quantile cutoff."""
    scores = np.asarray(scores_matrix, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores_matrix must be a 2-D array.")
    if threshold_mode not in {"adaptive_3sigma", "fixed_quantile"}:
        raise ValueError("threshold_mode must be 'adaptive_3sigma' or 'fixed_quantile'.")
    if not 0.0 < contamination < 1.0:
        raise ValueError("contamination must be between 0 and 1.")
    if k_sigma < 0.0:
        raise ValueError("k_sigma must be non-negative.")
    if not 0.0 < min_vote_ratio <= 1.0:
        raise ValueError("min_vote_ratio must be in (0, 1].")

    n_samples, n_models = scores.shape
    if n_samples == 0 or n_models == 0:
        return (
            np.zeros(n_samples, dtype=int),
            np.zeros(n_samples, dtype=float),
            0.0,
        )

    normalized_scores = _normalize_score_matrix(scores)
    ensemble_score = np.mean(normalized_scores, axis=1)

    mean_score = float(np.mean(ensemble_score))
    score_sigma = float(np.std(ensemble_score))
    if threshold_mode == "adaptive_3sigma":
        dynamic_cutoff = (
            mean_score + 1.0
            if score_sigma < 1e-6
            else mean_score + (k_sigma * score_sigma)
        )
    else:
        dynamic_cutoff = float(np.quantile(ensemble_score, 1.0 - contamination))

    individual_flags = (normalized_scores >= 0.5).astype(int)
    vote_counts = np.sum(individual_flags, axis=1)
    required_votes = int(np.ceil(min_vote_ratio * n_models))
    anomaly_flags = (
        (ensemble_score >= dynamic_cutoff)
        & (vote_counts >= required_votes)
    ).astype(int)

    return anomaly_flags, ensemble_score, float(dynamic_cutoff)
