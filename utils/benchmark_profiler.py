"""Scalability profiler for the AutoData-Hub v0-to-v3 pipeline."""

from __future__ import annotations

import gc
import time
import tracemalloc
from typing import Sequence

import numpy as np
import pandas as pd

from modules.anomaly.detector import detect_anomalies
from modules.imputation import ImputationStrategy, impute_missing_values
from modules.privacy.detector import PIIDetector
from modules.privacy.masker import PIIMasker


DEFAULT_SIZES = (1_000, 10_000, 50_000, 100_000)


def _synthetic_frame(size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "feature_a": rng.normal(10.0, 2.0, size),
        "feature_b": rng.normal(25.0, 5.0, size),
        "feature_c": rng.gamma(2.0, 2.0, size),
        "feature_d": rng.uniform(-1.0, 1.0, size),
        "email": [f"user{index}@example.com" for index in range(size)],
    })
    for column in ("feature_a", "feature_b", "feature_c"):
        frame.loc[rng.random(size) < 0.10, column] = np.nan
    outlier_count = max(1, size // 100)
    frame.loc[:outlier_count - 1, "feature_a"] += 30.0
    return frame


def _execute_pipeline(v0_raw: pd.DataFrame) -> pd.DataFrame:
    detections = PIIDetector().detect(v0_raw)
    privacy = PIIMasker(session_salt="benchmark-session-salt").mask_dataframe(
        v0_raw,
        detections=detections,
        strategy_overrides={column: "redact" for column in detections},
    )
    if privacy.transformed_df is None:
        raise RuntimeError(privacy.error_message or "Privacy stage failed.")

    imputed = impute_missing_values(
        privacy.transformed_df,
        strategy=ImputationStrategy.MEDIAN,
        add_mnar_indicators=False,
        dataset_version="v1_privacy",
    )
    if imputed.transformed_df is None:
        raise RuntimeError(imputed.error_message or "Imputation stage failed.")

    anomaly_transform, _ = detect_anomalies(
        imputed.transformed_df,
        algorithms=["isolation_forest"],
        threshold_mode="adaptive_3sigma",
        min_vote_ratio=0.66,
        dataset_version="v2_imputed",
        generate_artifacts=False,
    )
    if anomaly_transform.transformed_df is None:
        raise RuntimeError(anomaly_transform.error_message or "Anomaly stage failed.")

    v3_anomaly = anomaly_transform.transformed_df.copy(deep=True)
    numeric_columns = [
        column
        for column in imputed.transformed_df.columns
        if pd.api.types.is_numeric_dtype(imputed.transformed_df[column])
        and not pd.api.types.is_bool_dtype(imputed.transformed_df[column])
        and not str(column).endswith("_is_missing")
        and not str(column).startswith("anomaly_")
    ]
    for column in numeric_columns:
        lower = imputed.transformed_df[column].quantile(0.01)
        upper = imputed.transformed_df[column].quantile(0.99)
        v3_anomaly[column] = v3_anomaly[column].clip(lower=lower, upper=upper)
    return v3_anomaly


def profile_pipeline(
    sizes: Sequence[int] = DEFAULT_SIZES,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Profile the real linear pipeline and return required scalability metrics."""
    records = []
    for offset, size in enumerate(sizes):
        if size <= 0:
            raise ValueError("Dataset sizes must be positive integers.")
        v0_raw = _synthetic_frame(int(size), random_state + offset)
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        try:
            v3_anomaly = _execute_pipeline(v0_raw)
            latency = time.perf_counter() - started
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        if len(v3_anomaly) != size:
            raise RuntimeError("Pipeline changed row count during winsorization profile.")
        records.append({
            "Dataset Size (N)": int(size),
            "Wall-Clock Latency (sec)": latency,
            "Peak RAM Footprint (MB)": peak_bytes / (1024 ** 2),
            "Throughput (Rows/sec)": size / max(latency, 1e-12),
        })
        del v0_raw, v3_anomaly
        gc.collect()

    return pd.DataFrame(records)


def to_markdown(report: pd.DataFrame) -> str:
    headers = list(report.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in report.itertuples(index=False, name=None):
        values = [
            f"{value:,.0f}" if index in (0, 3) else f"{value:.3f}"
            for index, value in enumerate(row)
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    print("# AutoData-Hub Scalability Benchmark")
    print(to_markdown(profile_pipeline()))


if __name__ == "__main__":
    main()
