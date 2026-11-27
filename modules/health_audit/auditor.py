"""
AutoData-Hub Data Health Audit Engine (Module 1).

Calculates composite dataset Quality Score Q in [0, 100], missingness ratios,
row duplicates, cardinality per feature, type mismatches, mixed string/numeric fields,
and univariate outlier boundaries using the Interquartile Range (IQR).

Emits pure ModuleResult dataclasses with Plotly dark-themed visualization artifacts.
Strictly decoupled from UI frameworks (zero Streamlit dependencies) and 100% vectorized.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.schema_inference import ExecutionStatus, ModuleResult


def _hashable_value(value: Any) -> Any:
    """Canonicalize nested JSON-like cells for duplicate/cardinality operations."""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return repr(value)
    return value


# ---------------------------------------------------------------------------
# Default Quality Score Weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHT_MISSINGNESS = 0.35
DEFAULT_WEIGHT_DUPLICATES = 0.20
DEFAULT_WEIGHT_OUTLIERS = 0.25
DEFAULT_WEIGHT_TYPE_ERRORS = 0.20
EPSILON = 1e-9


def _safe_ratio(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """Return a finite ratio while preserving zero-denominator semantics."""

    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator) / max(float(denominator), EPSILON)


class DataHealthAuditor:
    """
    Automated dataset health audit engine.

    Performs comprehensive quality assessment across 4 key dimensions:
    - Missingness ratio (w1 = 0.35)
    - Duplicate row ratio (w2 = 0.20)
    - Univariate IQR outlier ratio (w3 = 0.25)
    - Type mismatch / error ratio (w4 = 0.20)

    Produces composite Quality Score:
      Q = max(0, 100 - (w1*M + w2*D + w3*O + w4*T) * 100)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        w = weights or {}
        self.w_missing = float(w.get("missingness", DEFAULT_WEIGHT_MISSINGNESS))
        self.w_duplicate = float(w.get("duplicates", DEFAULT_WEIGHT_DUPLICATES))
        self.w_outlier = float(w.get("outliers", DEFAULT_WEIGHT_OUTLIERS))
        self.w_type_error = float(w.get("type_errors", DEFAULT_WEIGHT_TYPE_ERRORS))

        # Normalize weights if sum != 1.0
        total_w = self.w_missing + self.w_duplicate + self.w_outlier + self.w_type_error
        if total_w > EPSILON:
            stable_total = max(total_w, EPSILON)
            self.w_missing /= stable_total
            self.w_duplicate /= stable_total
            self.w_outlier /= stable_total
            self.w_type_error /= stable_total

    def audit(
        self,
        df: pd.DataFrame,
        dataset_version: str = "v0_raw",
    ) -> ModuleResult:
        """
        Executes a vectorized health audit on the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to evaluate.
        dataset_version : str
            Active version identifier (e.g. 'v0_raw').

        Returns
        -------
        ModuleResult
            Standardized execution telemetry containing metrics and Plotly artifacts.
        """
        # Edge Case: None or Zero-row empty DataFrame
        if df is None or df.empty or len(df.columns) == 0:
            return self._handle_empty_dataset(df, dataset_version)

        row_count = int(len(df))
        col_count = int(len(df.columns))
        total_cells = row_count * col_count

        issues: List[str] = []

        # 1. Missingness Profiling (Vectorized)
        missing_count, missing_ratio, col_missing = self._calculate_missingness(df, total_cells)
        missing_pct = round(missing_ratio * 100.0, 2)

        all_null_columns = [col for col, meta in col_missing.items() if meta["null_pct"] >= 100.0]
        if all_null_columns:
            issues.append(f"Detected {len(all_null_columns)} column(s) with 100% missing values: {all_null_columns}")

        # 2. Duplicate Detection (Vectorized)
        duplicate_count, duplicate_ratio = self._calculate_duplicates(df, row_count)
        duplicate_pct = round(duplicate_ratio * 100.0, 2)
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate row(s) ({duplicate_pct}% of dataset)")

        # 3. Univariate IQR Outlier Detection (Vectorized)
        outlier_count, outlier_ratio, col_outliers, zero_var_cols = self._calculate_outliers(df)
        outlier_pct = round(outlier_ratio * 100.0, 2)
        if zero_var_cols:
            issues.append(f"Identified {len(zero_var_cols)} zero-variance constant feature(s): {zero_var_cols}")
        if outlier_count > 0:
            issues.append(f"Identified {outlier_count} univariate IQR outlier values ({outlier_pct}%)")

        # 4. Type Error & Mismatch Profiling (Vectorized)
        type_error_count, type_error_ratio, col_type_errors = self._calculate_type_errors(df, total_cells)
        type_error_pct = round(type_error_ratio * 100.0, 2)
        if type_error_count > 0:
            mismatched_cols = [c for c, count in col_type_errors.items() if count > 0]
            issues.append(f"Identified {type_error_count} type mismatch value(s) across columns: {mismatched_cols}")

        # 5. Composite Quality Score Calculation
        quality_score, grade, penalties = self._calculate_quality_score(
            missing_ratio=missing_ratio,
            duplicate_ratio=duplicate_ratio,
            outlier_ratio=outlier_ratio,
            type_error_ratio=type_error_ratio,
        )

        # 6. Column-level Breakdown Synthesis
        column_breakdown = self._synthesize_column_breakdown(
            df=df,
            col_missing=col_missing,
            col_outliers=col_outliers,
            col_type_errors=col_type_errors,
            zero_var_cols=zero_var_cols,
        )

        # 7. Generate Plotly Dark Artifacts
        artifacts = self._generate_artifacts(
            df=df,
            quality_score=quality_score,
            grade=grade,
            column_breakdown=column_breakdown,
        )

        metrics: Dict[str, Any] = {
            "quality_score": quality_score,
            "quality_grade": grade,
            "penalty_breakdown": penalties,
            "row_count": row_count,
            "column_count": col_count,
            "total_cells": total_cells,
            "missing_count": missing_count,
            "missing_ratio": round(missing_ratio, 4),
            "missing_pct": missing_pct,
            "duplicate_count": duplicate_count,
            "duplicate_ratio": round(duplicate_ratio, 4),
            "duplicate_pct": duplicate_pct,
            "outlier_count": outlier_count,
            "outlier_ratio": round(outlier_ratio, 4),
            "outlier_pct": outlier_pct,
            "type_error_count": type_error_count,
            "type_error_ratio": round(type_error_ratio, 4),
            "type_error_pct": type_error_pct,
            "zero_variance_columns": zero_var_cols,
            "all_null_columns": all_null_columns,
            "columns": column_breakdown,
        }

        return ModuleResult(
            status=ExecutionStatus.SUCCESS,
            module_name="health_audit",
            dataset_version=dataset_version,
            metrics=metrics,
            issues=issues,
            artifacts=artifacts,
        )

    # -----------------------------------------------------------------------
    # Internal Calculation Engines (100% Vectorized)
    # -----------------------------------------------------------------------

    def _calculate_missingness(
        self,
        df: pd.DataFrame,
        total_cells: int,
    ) -> Tuple[int, float, Dict[str, Dict[str, Any]]]:
        """Vectorized missingness calculation per column and across dataset."""
        null_series = df.isna().sum()
        total_missing = int(null_series.sum())
        missing_ratio = _safe_ratio(total_missing, total_cells)

        row_count = len(df)
        col_missing: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            n_count = int(null_series[col])
            n_pct = round(_safe_ratio(n_count, row_count) * 100.0, 2)
            col_missing[str(col)] = {
                "null_count": n_count,
                "null_pct": n_pct,
            }

        return total_missing, missing_ratio, col_missing

    def _calculate_duplicates(
        self,
        df: pd.DataFrame,
        row_count: int,
    ) -> Tuple[int, float]:
        """Vectorized duplicate row detection."""
        hashable_df = df.apply(lambda series: series.map(_hashable_value))
        dup_series = hashable_df.duplicated()
        duplicate_count = int(dup_series.sum())
        duplicate_ratio = _safe_ratio(duplicate_count, row_count)
        return duplicate_count, duplicate_ratio

    def _calculate_outliers(
        self,
        df: pd.DataFrame,
    ) -> Tuple[int, float, Dict[str, Dict[str, Any]], List[str]]:
        """
        Calculates baseline univariate outlier bounds using Interquartile Range (IQR):
          Lower = Q1 - 1.5 * IQR
          Upper = Q3 + 1.5 * IQR
        Handles zero-variance features and single-column DataFrames safely.
        """
        total_outliers = 0
        total_numeric_cells = 0
        col_outliers: Dict[str, Dict[str, Any]] = {}
        zero_var_cols: List[str] = []

        for col in df.columns:
            series = df[col]
            # Inspect numeric columns (excluding booleans)
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                non_null = series.dropna()
                n_count = len(non_null)
                total_numeric_cells += n_count

                if n_count == 0:
                    col_outliers[str(col)] = {
                        "outlier_count": 0,
                        "outlier_bounds": None,
                        "variance": 0.0,
                    }
                    continue

                variance = float(non_null.var()) if n_count > 1 else 0.0
                if abs(variance) <= EPSILON or non_null.nunique() <= 1:
                    zero_var_cols.append(str(col))

                # Quantile calculation
                q1 = float(non_null.quantile(0.25))
                q3 = float(non_null.quantile(0.75))
                iqr = float(q3 - q1)

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                if abs(iqr) <= EPSILON and abs(variance) <= EPSILON:
                    # Constant column has 0 outliers
                    outlier_count = 0
                else:
                    outlier_mask = (non_null < lower_bound) | (non_null > upper_bound)
                    outlier_count = int(outlier_mask.sum())

                total_outliers += outlier_count
                col_outliers[str(col)] = {
                    "outlier_count": outlier_count,
                    "outlier_bounds": {
                        "lower": round(lower_bound, 4),
                        "upper": round(upper_bound, 4),
                        "q1": round(q1, 4),
                        "q3": round(q3, 4),
                        "iqr": round(iqr, 4),
                    },
                    "variance": round(variance, 4),
                }

        outlier_ratio = _safe_ratio(total_outliers, total_numeric_cells)
        return total_outliers, outlier_ratio, col_outliers, zero_var_cols

    def _calculate_type_errors(
        self,
        df: pd.DataFrame,
        total_cells: int,
    ) -> Tuple[int, float, Dict[str, int]]:
        """
        Vectorized type error detection for mixed string/numeric fields and schema mismatches.
        """
        total_type_errors = 0
        col_type_errors: Dict[str, int] = {}

        for col in df.columns:
            series = df[col]
            col_str = str(col)
            # Only object and string dtypes can hold mixed types in memory
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna()
                if len(non_null) == 0:
                    col_type_errors[col_str] = 0
                    continue

                errors_in_col = 0

                # 1. Check for heterogeneous native Python types in the column
                type_counts = non_null.map(lambda x: type(x).__name__).value_counts()
                if len(type_counts) > 1:
                    # Count everything outside dominant type as error
                    dominant_count = int(type_counts.iloc[0])
                    errors_in_col += (len(non_null) - dominant_count)

                # 2. Check if predominantly numeric strings contain invalid non-numeric entries
                else:
                    str_non_null = non_null.astype(str)
                    numeric_attempt = pd.to_numeric(str_non_null, errors="coerce")
                    valid_num_count = int(numeric_attempt.notna().sum())
                    num_ratio = _safe_ratio(valid_num_count, len(non_null))

                    # If majority (>=50%) parse as numbers, non-numeric strings are type mismatches
                    if 0.5 <= num_ratio < 1.0:
                        errors_in_col += (len(non_null) - valid_num_count)

                col_type_errors[col_str] = errors_in_col
                total_type_errors += errors_in_col
            else:
                col_type_errors[col_str] = 0

        type_error_ratio = _safe_ratio(total_type_errors, total_cells)
        return total_type_errors, type_error_ratio, col_type_errors

    def _calculate_quality_score(
        self,
        missing_ratio: float,
        duplicate_ratio: float,
        outlier_ratio: float,
        type_error_ratio: float,
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculates Composite Quality Score:
          Q = max(0, 100 - (w1*M + w2*D + w3*O + w4*T) * 100)
        """
        penalty_m = self.w_missing * missing_ratio * 100.0
        penalty_d = self.w_duplicate * duplicate_ratio * 100.0
        penalty_o = self.w_outlier * outlier_ratio * 100.0
        penalty_t = self.w_type_error * type_error_ratio * 100.0

        total_penalty = penalty_m + penalty_d + penalty_o + penalty_t
        raw_score = 100.0 - total_penalty
        quality_score = round(float(np.clip(raw_score, 0.0, 100.0)), 2)

        # Categorical Letter Grade
        if quality_score >= 90.0:
            grade = "A"
        elif quality_score >= 80.0:
            grade = "B"
        elif quality_score >= 70.0:
            grade = "C"
        elif quality_score >= 60.0:
            grade = "D"
        else:
            grade = "F"

        penalties = {
            "missingness_penalty": round(penalty_m, 2),
            "duplicate_penalty": round(penalty_d, 2),
            "outlier_penalty": round(penalty_o, 2),
            "type_error_penalty": round(penalty_t, 2),
            "total_penalty": round(total_penalty, 2),
        }

        return quality_score, grade, penalties

    def _synthesize_column_breakdown(
        self,
        df: pd.DataFrame,
        col_missing: Dict[str, Dict[str, Any]],
        col_outliers: Dict[str, Dict[str, Any]],
        col_type_errors: Dict[str, int],
        zero_var_cols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Synthesizes comprehensive per-column audit metrics."""
        breakdown: Dict[str, Dict[str, Any]] = {}
        row_count = len(df)

        for col in df.columns:
            col_str = str(col)
            series = df[col]
            non_null = series.dropna()
            unique_count = int(non_null.map(_hashable_value).nunique())
            card_ratio = round(_safe_ratio(unique_count, len(non_null)), 4)

            # Safe sample values
            sample_vals = non_null.head(3).tolist()
            clean_samples = [str(v) for v in sample_vals]

            outlier_info = col_outliers.get(col_str, {})
            missing_info = col_missing.get(col_str, {"null_count": 0, "null_pct": 0.0})

            breakdown[col_str] = {
                "name": col_str,
                "dtype": str(series.dtype),
                "null_count": missing_info["null_count"],
                "null_pct": missing_info["null_pct"],
                "unique_count": unique_count,
                "cardinality_ratio": card_ratio,
                "is_constant": col_str in zero_var_cols or unique_count <= 1,
                "is_unique": unique_count == row_count and row_count > 0,
                "type_errors": col_type_errors.get(col_str, 0),
                "outlier_count": outlier_info.get("outlier_count", 0),
                "outlier_bounds": outlier_info.get("outlier_bounds"),
                "sample_values": clean_samples,
            }

        return breakdown

    # -----------------------------------------------------------------------
    # Plotly Dark-Themed Artifact Generation
    # -----------------------------------------------------------------------

    def _generate_artifacts(
        self,
        df: pd.DataFrame,
        quality_score: float,
        grade: str,
        column_breakdown: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates dark-themed Plotly visualization dictionaries (template="plotly_dark").
        """
        artifacts: Dict[str, Any] = {}

        # 1. Spatial Missingness Heatmap
        artifacts["missingness_heatmap"] = self._create_missingness_heatmap(df)

        # 2. Quality Score Gauge Chart
        artifacts["quality_gauge"] = self._create_quality_gauge(quality_score, grade)

        # 3. Feature Distribution Histograms (Top numeric and categorical features)
        artifacts["feature_distributions"] = self._create_feature_distributions(df)

        return artifacts

    def _create_missingness_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Creates a Plotly binary heatmap showing spatial missingness across columns."""
        if df.empty or len(df.columns) == 0:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark", title="Missingness Heatmap (No Data)")
            return fig.to_dict()

        # Downsample rows for fast frontend rendering if dataset > 500 rows
        step = max(1, len(df) // 500)
        sample_matrix = df.isna().astype(int).iloc[::step]

        fig = go.Figure(
            data=go.Heatmap(
                z=sample_matrix.values,
                x=[str(c) for c in df.columns],
                y=list(range(len(sample_matrix))),
                colorscale=[[0, "#0F172A"], [1, "#EF4444"]],
                showscale=False,
                hoverongaps=False,
            )
        )
        fig.update_layout(
            template="plotly_dark",
            title="Missingness Spatial Distribution (Red = Missing, Navy = Present)",
            xaxis_title="Features",
            yaxis_title="Downsampled Record Index",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        return fig.to_dict()

    def _create_quality_gauge(self, score: float, grade: str) -> Dict[str, Any]:
        """Creates a Plotly dark gauge indicator for overall dataset health."""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": f"Composite Quality Score (Grade {grade})"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                    "bar": {"color": "#38BDF8"},
                    "steps": [
                        {"range": [0, 60], "color": "#7F1D1D"},
                        {"range": [60, 80], "color": "#78350F"},
                        {"range": [80, 100], "color": "#064E3B"},
                    ],
                    "threshold": {
                        "line": {"color": "#F8FAFC", "width": 4},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            )
        )
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=30, r=30, t=60, b=30),
            height=280,
        )
        return fig.to_dict()

    def _create_feature_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates distribution histograms for numeric and frequency bars for categorical features."""
        dist_dict: Dict[str, Any] = {}

        # Numeric histograms (up to top 8 columns)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        for col in numeric_cols[:8]:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue

            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=vals,
                        nbinsx=30,
                        marker_color="#38BDF8",
                        opacity=0.85,
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                title=f"Distribution — {col}",
                xaxis_title=str(col),
                yaxis_title="Frequency",
                margin=dict(l=40, r=40, t=45, b=40),
                height=260,
            )
            dist_dict[str(col)] = fig.to_dict()

        # Categorical bar charts (up to top 4 columns with reasonable cardinality)
        cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]
        for col in cat_cols[:4]:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            counts = vals.map(_hashable_value).value_counts().head(12)
            if len(counts) > 1:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=[str(k) for k in counts.index],
                            y=counts.values,
                            marker_color="#818CF8",
                            opacity=0.85,
                        )
                    ]
                )
                fig.update_layout(
                    template="plotly_dark",
                    title=f"Top Categories — {col}",
                    xaxis_title=str(col),
                    yaxis_title="Count",
                    margin=dict(l=40, r=40, t=45, b=40),
                    height=260,
                )
                dist_dict[str(col)] = fig.to_dict()

        return dist_dict

    def _handle_empty_dataset(
        self,
        df: Optional[pd.DataFrame],
        dataset_version: str,
    ) -> ModuleResult:
        """Handles zero-row or None dataset gracefully."""
        fig_empty = go.Figure()
        fig_empty.update_layout(template="plotly_dark", title="Empty Dataset")
        empty_artifact = fig_empty.to_dict()

        return ModuleResult(
            status=ExecutionStatus.SUCCESS,
            module_name="health_audit",
            dataset_version=dataset_version,
            metrics={
                "quality_score": 0.0,
                "quality_grade": "F",
                "penalty_breakdown": {},
                "row_count": 0 if df is None else int(len(df)),
                "column_count": 0 if df is None else len(df.columns),
                "total_cells": 0,
                "missing_count": 0,
                "missing_ratio": 0.0,
                "missing_pct": 0.0,
                "duplicate_count": 0,
                "duplicate_ratio": 0.0,
                "duplicate_pct": 0.0,
                "outlier_count": 0,
                "outlier_ratio": 0.0,
                "outlier_pct": 0.0,
                "type_error_count": 0,
                "type_error_ratio": 0.0,
                "type_error_pct": 0.0,
                "zero_variance_columns": [],
                "all_null_columns": [],
                "columns": {},
            },
            issues=["Dataset is empty (0 rows or 0 columns)."],
            artifacts={
                "missingness_heatmap": empty_artifact,
                "quality_gauge": empty_artifact,
                "feature_distributions": {},
            },
        )


# ---------------------------------------------------------------------------
# Convenience Module API
# ---------------------------------------------------------------------------

def audit_dataset_health(
    df: pd.DataFrame,
    dataset_version: str = "v0_raw",
    weights: Optional[Dict[str, float]] = None,
) -> ModuleResult:
    """
    Convenience functional endpoint for executing Data Health Audit.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    dataset_version : str
        Version identifier tag.
    weights : Optional[Dict[str, float]]
        Custom weighting map for quality scoring.

    Returns
    -------
    ModuleResult
        Standardized telemetry payload with metrics, diagnostic issues, and Plotly artifacts.
    """
    auditor = DataHealthAuditor(weights=weights)
    return auditor.audit(df=df, dataset_version=dataset_version)
