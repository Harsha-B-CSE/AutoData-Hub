"""
AutoData-Hub Core Data Contracts & Schema Inference Engine.

Defines the system-wide dataclass contracts for pipeline transformations,
module execution telemetry, audit logging, and automated column schema inference.
Strictly decoupled from UI frameworks (zero UI framework dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core Execution Enums & Dataclasses
# ---------------------------------------------------------------------------

class ExecutionStatus(str, Enum):
    """Execution status for module and transformation steps."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class TransformationAction:
    """
    Immutable specification of a single dataset transformation step.

    Used by the Script & Report Exporter (Module 5) to compile AST
    pipelines and audit logs into standalone Python scripts.
    """
    step_id: int
    module: str
    action_type: str
    target_columns: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    code_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the action to a serializable dictionary."""
        return {
            "step_id": self.step_id,
            "module": self.module,
            "action_type": self.action_type,
            "target_columns": list(self.target_columns),
            "parameters": dict(self.parameters),
            "code_snippet": self.code_snippet,
        }


@dataclass
class ModuleResult:
    """
    Standardized telemetry and outcome payload returned by all processing modules.

    Captures metrics, detected issues, and diagnostic artifacts without mutating
    the underlying dataset.
    """
    status: ExecutionStatus
    module_name: str
    dataset_version: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        module_name: str,
        dataset_version: str,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        issues: Optional[List[str]] = None,
    ) -> ModuleResult:
        """Convenience constructor for successful execution."""
        return cls(
            status=ExecutionStatus.SUCCESS,
            module_name=module_name,
            dataset_version=dataset_version,
            metrics=metrics if metrics is not None else {},
            issues=issues if issues is not None else [],
            artifacts=artifacts if artifacts is not None else {},
        )

    @classmethod
    def failure(
        cls,
        module_name: str,
        dataset_version: str,
        error_message: str,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        """Convenience constructor for failed execution."""
        return cls(
            status=ExecutionStatus.FAILED,
            module_name=module_name,
            dataset_version=dataset_version,
            metrics=metrics if metrics is not None else {},
            issues=[error_message],
            artifacts=artifacts if artifacts is not None else {},
        )


@dataclass
class TransformResult:
    """
    Result of an operational pipeline transformation.

    Guarantees non-mutating updates by wrapping a newly assigned pd.DataFrame,
    the sequence of TransformationAction records applied, and complete audit logs.
    """
    status: ExecutionStatus
    transformed_df: Optional[pd.DataFrame] = None
    applied_actions: List[TransformationAction] = field(default_factory=list)
    audit_log: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        transformed_df: pd.DataFrame,
        applied_actions: Optional[List[TransformationAction]] = None,
        audit_log: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransformResult:
        """Convenience constructor for successful transformation."""
        return cls(
            status=ExecutionStatus.SUCCESS,
            transformed_df=transformed_df,
            applied_actions=applied_actions if applied_actions is not None else [],
            audit_log=audit_log if audit_log is not None else [],
            error_message=None,
            metadata=dict(metadata) if metadata is not None else {},
        )

    @classmethod
    def failure(
        cls,
        error_message: str,
        audit_log: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransformResult:
        """Convenience constructor for failed transformation."""
        return cls(
            status=ExecutionStatus.FAILED,
            transformed_df=None,
            applied_actions=[],
            audit_log=audit_log if audit_log is not None else [],
            error_message=error_message,
            metadata=dict(metadata) if metadata is not None else {},
        )


# ---------------------------------------------------------------------------
# Schema Inference Contracts & Engine
# ---------------------------------------------------------------------------

class ColumnType(str, Enum):
    """Categorization of column semantics in the inferred dataset schema."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    ID = "id"
    UNKNOWN = "unknown"


@dataclass
class ColumnSchema:
    """Detailed profile and semantic schema inferred for a single column."""
    name: str
    inferred_type: ColumnType
    original_dtype: str
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    is_unique: bool = False
    sample_values: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSchema:
    """Global schema profile inferred for a pd.DataFrame."""
    row_count: int
    column_count: int
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    boolean_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    memory_usage_bytes: int = 0


# ID pattern detection: common ID column naming conventions
_ID_NAME_PATTERNS = re.compile(
    r"^(id|.*_id|id_.*|uuid|guid|pk|.*_pk|sku|ssn|vin)$",
    re.IGNORECASE,
)


def infer_column_type(
    series: pd.Series,
    categorical_threshold: float = 0.05,
    max_categorical_cardinality: int = 50,
) -> ColumnType:
    """
    Infers the semantic type of a pandas Series.

    Parameters
    ----------
    series : pd.Series
        The column data to inspect.
    categorical_threshold : float
        Ratio of unique values to non-null rows below which object columns are
        treated as categorical rather than free-form text.
    max_categorical_cardinality : int
        Absolute max number of distinct categories for an object column to qualify
        as categorical.

    Returns
    -------
    ColumnType
        Inferred semantic classification.
    """
    # 1. Check for boolean dtypes
    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN

    # 2. Check for explicit datetime dtypes
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return ColumnType.DATETIME

    # 3. Check for numeric types (int, float)
    if pd.api.types.is_numeric_dtype(series):
        # Heuristic for IDs: integer series matching ID names or monotonically increasing unique integers
        non_null = series.dropna()
        if len(non_null) > 0 and pd.api.types.is_integer_dtype(series):
            is_id_name = bool(_ID_NAME_PATTERNS.match(str(series.name).strip()))
            is_all_unique = (non_null.nunique() == len(non_null))
            if is_id_name and is_all_unique:
                return ColumnType.ID
        return ColumnType.NUMERIC

    # 4. Handle object / string / categorical columns
    non_null = series.dropna()
    total_non_null = len(non_null)

    if total_non_null == 0:
        return ColumnType.UNKNOWN

    # Check if object column holds booleans
    sample_values = non_null.head(100)
    bool_candidates = {True, False, "true", "false", "True", "False", "1", "0", "yes", "no", "Y", "N"}
    if all(str(v).strip().lower() in {"true", "false", "1", "0", "yes", "no", "y", "n"} for v in sample_values):
        return ColumnType.BOOLEAN

    # Check for ID naming + uniqueness
    is_id_name = bool(_ID_NAME_PATTERNS.match(str(series.name).strip()))
    unique_count = non_null.nunique()
    if is_id_name and (unique_count == total_non_null):
        return ColumnType.ID

    # Check if object column can be parsed as datetimes (only if strings contain common date delimiters)
    if total_non_null > 0:
        sample = non_null.head(30).astype(str)
        has_date_delimiters = sample.str.contains(r"[-/:]", regex=True).all()
        if has_date_delimiters and not all(s.replace(".", "", 1).isdigit() for s in sample):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(sample, errors="raise", format="mixed")
                return ColumnType.DATETIME
            except (ValueError, TypeError, OverflowError):
                pass

    # Check if strings are long enough to indicate free-form text
    avg_str_len = float(non_null.astype(str).str.len().mean()) if total_non_null > 0 else 0.0
    if avg_str_len > 35:
        return ColumnType.TEXT

    # Categorical vs Text evaluation
    cardinality_ratio = unique_count / total_non_null if total_non_null > 0 else 0.0
    if isinstance(series.dtype, pd.CategoricalDtype):
        return ColumnType.CATEGORICAL

    if unique_count <= max_categorical_cardinality or cardinality_ratio <= categorical_threshold:
        return ColumnType.CATEGORICAL

    # Free-form string / text
    return ColumnType.TEXT


def infer_schema(df: pd.DataFrame) -> DatasetSchema:
    """
    Performs full schema inference and statistical profiling across a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.

    Returns
    -------
    DatasetSchema
        Comprehensive schema metadata and column classifications.
    """
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    row_count = int(len(df))
    column_count = int(len(df.columns))
    memory_usage = int(df.memory_usage(deep=True).sum()) if row_count > 0 else 0

    columns: Dict[str, ColumnSchema] = {}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    boolean_cols: List[str] = []
    text_cols: List[str] = []
    id_cols: List[str] = []

    for col_name in df.columns:
        col_str = str(col_name)
        series = df[col_name]
        col_type = infer_column_type(series)

        # Compute missingness stats
        null_count = int(series.isna().sum())
        null_pct = float((null_count / row_count) * 100.0) if row_count > 0 else 0.0

        # Unique stats
        non_null_series = series.dropna()
        unique_count = int(non_null_series.nunique())
        is_unique = (unique_count == (row_count - null_count)) and (row_count - null_count > 0)

        # Sample values (up to 5 unique non-null samples converted to native Python types)
        raw_samples = non_null_series.drop_duplicates().head(5).tolist()
        clean_samples = []
        for val in raw_samples:
            if isinstance(val, (np.integer, np.int64, np.int32)):
                clean_samples.append(int(val))
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                clean_samples.append(float(val))
            elif isinstance(val, (np.bool_, bool)):
                clean_samples.append(bool(val))
            elif isinstance(val, pd.Timestamp):
                clean_samples.append(val.isoformat())
            else:
                clean_samples.append(str(val))

        metadata: Dict[str, Any] = {}
        if col_type == ColumnType.NUMERIC and len(non_null_series) > 0:
            metadata["min"] = float(non_null_series.min()) if pd.notna(non_null_series.min()) else None
            metadata["max"] = float(non_null_series.max()) if pd.notna(non_null_series.max()) else None
            metadata["mean"] = float(non_null_series.mean()) if pd.notna(non_null_series.mean()) else None
            metadata["std"] = float(non_null_series.std()) if len(non_null_series) > 1 and pd.notna(non_null_series.std()) else 0.0

        elif col_type == ColumnType.CATEGORICAL and len(non_null_series) > 0:
            top_counts = non_null_series.value_counts().head(5).to_dict()
            metadata["top_categories"] = {str(k): int(v) for k, v in top_counts.items()}

        col_schema = ColumnSchema(
            name=col_str,
            inferred_type=col_type,
            original_dtype=str(series.dtype),
            null_count=null_count,
            null_percentage=round(null_pct, 2),
            unique_count=unique_count,
            is_unique=is_unique,
            sample_values=clean_samples,
            metadata=metadata,
        )
        columns[col_str] = col_schema

        # Map to categorised lists
        if col_type == ColumnType.NUMERIC:
            numeric_cols.append(col_str)
        elif col_type == ColumnType.CATEGORICAL:
            categorical_cols.append(col_str)
        elif col_type == ColumnType.DATETIME:
            datetime_cols.append(col_str)
        elif col_type == ColumnType.BOOLEAN:
            boolean_cols.append(col_str)
        elif col_type == ColumnType.TEXT:
            text_cols.append(col_str)
        elif col_type == ColumnType.ID:
            id_cols.append(col_str)

    return DatasetSchema(
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        boolean_columns=boolean_cols,
        text_columns=text_cols,
        id_columns=id_cols,
        memory_usage_bytes=memory_usage,
    )
