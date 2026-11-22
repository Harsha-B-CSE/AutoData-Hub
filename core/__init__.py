"""
AutoData-Hub Core Module.
Exposes foundational contracts, ingestion engines, dataset managers, and orchestrators.
"""

from core.schema_inference import (
    ColumnSchema,
    ColumnType,
    DatasetSchema,
    ExecutionStatus,
    ModuleResult,
    TransformationAction,
    TransformResult,
    infer_column_type,
    infer_schema,
)
from core.ingestion_engine import (
    DBConnectorConfig,
    DatabaseConnector,
    detect_delimiter,
    detect_encoding,
    get_excel_sheet_names,
    ingest_dataset,
    parse_mixed_dates,
    read_arrow,
    read_delimited,
    read_excel,
    read_json_flattened,
    sanitize_strings_vectorized,
)
from core.dataset_manager import (
    DatasetManager,
    DatasetVersionInfo,
    DeliverableArtifact,
)

__all__ = [
    # Schema Inference & Dataclass Contracts
    "ExecutionStatus",
    "TransformationAction",
    "ModuleResult",
    "TransformResult",
    "ColumnType",
    "ColumnSchema",
    "DatasetSchema",
    "infer_column_type",
    "infer_schema",
    # Ingestion Engine
    "DBConnectorConfig",
    "DatabaseConnector",
    "detect_encoding",
    "detect_delimiter",
    "sanitize_strings_vectorized",
    "parse_mixed_dates",
    "read_delimited",
    "get_excel_sheet_names",
    "read_excel",
    "read_arrow",
    "read_json_flattened",
    "ingest_dataset",
    # Dataset Manager (4-Tier Storage)
    "DatasetManager",
    "DatasetVersionInfo",
    "DeliverableArtifact",
]
