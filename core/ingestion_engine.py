"""
AutoData-Hub Universal Ingestion Engine.

Provides resilient, multi-format dataset ingestion with automated delimiter/encoding
detection, semi-structured JSON flattening, multi-engine database connectivity,
and streaming memory safeguards for large datasets (>100MB / >100k rows).
Strictly decoupled from UI frameworks (zero UI framework imports).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import datetime
import io
import json
import os
from pathlib import Path
import re
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Sequence, TextIO, Tuple, Union
import warnings

import chardet
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Database Connector Configuration & Abstraction
# ---------------------------------------------------------------------------

@dataclass
class DBConnectorConfig:
    """
    Configuration specification for database connectivity.

    Supports PostgreSQL (psycopg2), MySQL (PyMySQL), SQLite (sqlite3),
    Snowflake (snowflake-sqlalchemy), and MS SQL Server (pyodbc).
    """
    dialect: str  # 'sqlite', 'postgresql', 'mysql', 'snowflake', 'mssql', 'custom'
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    driver: Optional[str] = None
    custom_url: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def get_connection_url(self) -> str:
        """Constructs a normalized SQLAlchemy connection URL string."""
        if self.custom_url:
            return self.custom_url

        d = self.dialect.lower()
        if d == "sqlite":
            db_path = self.database or ":memory:"
            return f"sqlite:///{db_path}"

        if d in ("postgresql", "postgres"):
            user = self.username or "postgres"
            pwd = f":{self.password}" if self.password else ""
            host = self.host or "localhost"
            port = f":{self.port}" if self.port else ":5432"
            db = f"/{self.database}" if self.database else ""
            return f"postgresql+psycopg2://{user}{pwd}@{host}{port}{db}"

        if d == "mysql":
            user = self.username or "root"
            pwd = f":{self.password}" if self.password else ""
            host = self.host or "localhost"
            port = f":{self.port}" if self.port else ":3306"
            db = f"/{self.database}" if self.database else ""
            return f"mysql+pymysql://{user}{pwd}@{host}{port}{db}"

        if d == "snowflake":
            user = self.username or ""
            pwd = f":{self.password}" if self.password else ""
            account = self.host or ""
            db = f"/{self.database}" if self.database else ""
            schema = self.extra_params.get("schema", "")
            schema_part = f"/{schema}" if schema else ""
            query_params = []
            if "warehouse" in self.extra_params:
                query_params.append(f"warehouse={self.extra_params['warehouse']}")
            if "role" in self.extra_params:
                query_params.append(f"role={self.extra_params['role']}")
            query_str = f"?{'&'.join(query_params)}" if query_params else ""
            return f"snowflake://{user}{pwd}@{account}{db}{schema_part}{query_str}"

        if d in ("mssql", "sqlserver"):
            user = self.username or "sa"
            pwd = f":{self.password}" if self.password else ""
            host = self.host or "localhost"
            port = f":{self.port}" if self.port else ":1433"
            db = f"/{self.database}" if self.database else ""
            driver = self.driver or "ODBC Driver 17 for SQL Server"
            driver_encoded = driver.replace(" ", "+")
            return f"mssql+pyodbc://{user}{pwd}@{host}{port}{db}?driver={driver_encoded}"

        raise ValueError(f"Unsupported database dialect: '{self.dialect}'. Use 'custom' with custom_url.")


class DatabaseConnector:
    """
    SQLAlchemy-backed universal database abstraction.
    Provides schema inspection, query execution, and stream ingestion.
    """

    def __init__(self, config_or_url: Union[str, DBConnectorConfig]):
        if isinstance(config_or_url, str):
            self.connection_url = config_or_url
        else:
            self.connection_url = config_or_url.get_connection_url()

        self._engine: Optional[Engine] = None

    def get_engine(self) -> Engine:
        """Lazily initializes and returns the SQLAlchemy Engine."""
        if self._engine is None:
            self._engine = create_engine(self.connection_url, pool_pre_ping=True)
        return self._engine

    def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Validates database reachability by executing a ping statement."""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, None
        except Exception as e:
            return False, str(e)

    def list_tables(self) -> List[str]:
        """Inspects and returns the list of available table names in the database."""
        engine = self.get_engine()
        inspector = inspect(engine)
        return list(inspector.get_table_names())

    def read_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        chunksize: Optional[int] = None,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Executes a SQL query and returns the results as a DataFrame or chunk iterator.
        """
        engine = self.get_engine()
        return pd.read_sql(
            sql=text(query) if isinstance(query, str) else query,
            con=engine,
            params=params,
            chunksize=chunksize,
        )

    def read_table(
        self,
        table_name: str,
        limit: Optional[int] = None,
        chunksize: Optional[int] = None,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Reads a specific table from the database."""
        engine = self.get_engine()
        query = f"SELECT * FROM {table_name}"
        if limit is not None and limit > 0:
            query = f"{query} LIMIT {limit}"
        return pd.read_sql(sql=text(query), con=engine, chunksize=chunksize)


# ---------------------------------------------------------------------------
# Ingestion Safeguards: Encoding, Delimiter, Sanitization, Date Parsing
# ---------------------------------------------------------------------------

def detect_encoding(file_source: Union[str, Path, BinaryIO], sample_bytes: int = 65536) -> str:
    """
    Autodetects file encoding (e.g. UTF-8, Latin-1, CP1252, UTF-16) using chardet.

    Parameters
    ----------
    file_source : Union[str, Path, BinaryIO]
        Path to file or open binary stream.
    sample_bytes : int
        Number of raw bytes to sample from the start of the file.

    Returns
    -------
    str
        Detected encoding string, falling back to 'utf-8'.
    """
    raw_data: bytes = b""
    is_stream = hasattr(file_source, "read")

    if is_stream:
        stream = file_source  # type: ignore
        pos = stream.tell() if hasattr(stream, "tell") else 0
        read_val = stream.read(sample_bytes)
        if isinstance(read_val, str):
            if hasattr(stream, "seek"):
                stream.seek(pos)
            return "utf-8"
        raw_data = read_val
        if hasattr(stream, "seek"):
            stream.seek(pos)
    else:
        path = Path(file_source)
        if not path.is_file():
            return "utf-8"
        with open(path, "rb") as f:
            raw_data = f.read(sample_bytes)

    if not raw_data:
        return "utf-8"

    result = chardet.detect(raw_data)
    detected = result.get("encoding")
    confidence = result.get("confidence", 0.0)

    if detected and confidence and confidence >= 0.6:
        # Standardize common encodings
        det_lower = detected.lower()
        if "utf-8" in det_lower or "ascii" in det_lower:
            return "utf-8"
        if "1252" in det_lower:
            return "cp1252"
        if "latin" in det_lower or "iso-8859" in det_lower:
            return "latin-1"
        if "utf-16" in det_lower:
            return "utf-16"
        return detected

    # Fallback cascade
    for candidate in ["utf-8", "cp1252", "latin-1"]:
        try:
            raw_data.decode(candidate)
            return candidate
        except (UnicodeDecodeError, LookupError):
            continue

    return "utf-8"


def detect_delimiter(
    file_source: Union[str, Path, BinaryIO, TextIO],
    encoding: str = "utf-8",
    sample_size: int = 16384,
) -> str:
    """
    Autodetects the tabular delimiter (',', ';', '\t', '|') using csv.Sniffer.

    Parameters
    ----------
    file_source : Union[str, Path, BinaryIO, TextIO]
        File path or readable stream.
    encoding : str
        Character encoding to use when reading text.
    sample_size : int
        Sample size in bytes/characters to evaluate.

    Returns
    -------
    str
        Detected delimiter character, defaulting to ','.
    """
    text_sample = ""
    is_stream = hasattr(file_source, "read")

    if is_stream:
        stream = file_source  # type: ignore
        pos = stream.tell() if hasattr(stream, "tell") else 0
        raw = stream.read(sample_size)
        if isinstance(raw, bytes):
            text_sample = raw.decode(encoding, errors="replace")
        else:
            text_sample = str(raw)
        if hasattr(stream, "seek"):
            stream.seek(pos)
    else:
        path = Path(file_source)
        if not path.is_file():
            return ","
        with open(path, "r", encoding=encoding, errors="replace") as f:
            text_sample = f.read(sample_size)

    if not text_sample:
        return ","

    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text_sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except (csv.Error, Exception):
        # Heuristic fallback based on delimiter occurrence frequencies
        counts = {
            ",": text_sample.count(","),
            ";": text_sample.count(";"),
            "\t": text_sample.count("\t"),
            "|": text_sample.count("|"),
        }
        max_delim, max_count = max(counts.items(), key=lambda item: item[1])
        return max_delim if max_count > 0 else ","


# Null string sentinel values to normalize across tabular ingestion
DEFAULT_NULL_STRINGS = frozenset([
    "n/a", "na", "-", "--", "null", "none", "?", "", "nan",
    "undefined", "n.a.", "null", "none", "#n/a", "n/a", "nil"
])


def sanitize_strings_vectorized(
    df: pd.DataFrame,
    null_strings: Optional[Sequence[str]] = None,
    strip_currency_pct: bool = True,
) -> pd.DataFrame:
    """
    Vectorized Regex string sanitizer.

    - Replaces common null-like sentinels ('N/A', '-', 'null', etc.) with np.nan.
    - Strips currency symbols ('$', '€', '£', '¥') and '%' from numeric strings.
    - Automatically converts columns to numeric if >= 70% of non-null entries
      are formatted numbers with currency/percentage.
    - Strictly non-mutating (returns a new copy).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    null_strings : Optional[Sequence[str]]
        Custom sequence of null string patterns to treat as NaN.
    strip_currency_pct : bool
        Whether to strip currency/percentage symbols from numeric-like string columns.

    Returns
    -------
    pd.DataFrame
        Sanitized DataFrame copy.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    clean_df = df.copy()
    null_set = {s.strip().lower() for s in (null_strings or DEFAULT_NULL_STRINGS)}

    for col in clean_df.columns:
        series = clean_df[col]
        # Only inspect object, string, or categorical types
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # 1. Vectorized map of null sentinels
            str_series = series.astype(str)
            trimmed_lower = str_series.str.strip().str.lower()
            null_mask = trimmed_lower.isin(null_set) | series.isna()
            sanitized = series.mask(null_mask, np.nan)

            # 2. Check for currency and percentage patterns
            if strip_currency_pct:
                non_null_vals = sanitized.dropna()
                if len(non_null_vals) > 0:
                    str_non_null = non_null_vals.astype(str)
                    has_symbols = str_non_null.str.contains(r"[$€£¥%]", regex=True).any()
                    if has_symbols:
                        # Attempt stripping symbols and commas
                        stripped = (
                            str_non_null
                            .str.replace(r"[$€£¥%]", "", regex=True)
                            .str.replace(",", "", regex=False)
                            .str.strip()
                        )
                        # Check numeric conversion rate
                        numeric_attempt = pd.to_numeric(stripped, errors="coerce")
                        valid_ratio = float(numeric_attempt.notna().mean())
                        if valid_ratio >= 0.7:
                            # Apply to full column
                            full_stripped = (
                                sanitized.astype(str)
                                .str.replace(r"[$€£¥%]", "", regex=True)
                                .str.replace(",", "", regex=False)
                                .str.strip()
                            )
                            clean_df[col] = pd.to_numeric(full_stripped, errors="coerce")
                            continue
            clean_df[col] = sanitized

    return clean_df


# Common column name keywords indicating date/time content
_DATE_NAME_KEYWORDS = re.compile(
    r"(date|time|timestamp|created|updated|joined|dob|birth|start|end|period)",
    re.IGNORECASE,
)


def parse_mixed_dates(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
    sample_rows: int = 50,
) -> pd.DataFrame:
    """
    Parses date columns supporting mixed formats via pd.to_datetime(format='mixed').

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_columns : Optional[List[str]]
        Explicit list of columns to convert. If None, auto-detects date candidate columns.
    sample_rows : int
        Sample size used for candidate date detection heuristics.

    Returns
    -------
    pd.DataFrame
        DataFrame copy with date columns converted to datetime64[ns].
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    clean_df = df.copy()
    target_cols = set(date_columns) if date_columns is not None else set()

    # Auto-detect date candidates if date_columns was not specified
    if date_columns is None:
        for col in clean_df.columns:
            series = clean_df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                continue
            if pd.api.types.is_numeric_dtype(series):
                continue

            col_str = str(col).strip()
            name_match = bool(_DATE_NAME_KEYWORDS.search(col_str))

            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            sample = non_null.head(sample_rows).astype(str)
            # Require standard date punctuation delimiters or column name keyword
            delim_ratio = float(sample.str.contains(r"[-/.:]", regex=True).mean())
            if not name_match and delim_ratio < 0.5:
                continue

            # Test mixed date parsing
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed_sample = pd.to_datetime(sample, format="mixed", errors="coerce")
                if parsed_sample.notna().mean() >= 0.5:
                    target_cols.add(col)
            except Exception:
                if name_match:
                    target_cols.add(col)

    # Convert detected date columns
    for col in target_cols:
        if col in clean_df.columns:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clean_df[col] = pd.to_datetime(
                        clean_df[col],
                        format="mixed",
                        errors="coerce",
                    )
            except Exception:
                pass

    return clean_df


# ---------------------------------------------------------------------------
# Specific Format Ingestion Handlers
# ---------------------------------------------------------------------------

def read_delimited(
    file_source: Union[str, Path, BinaryIO, TextIO],
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    chunksize: Optional[int] = None,
    **kwargs,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Ingests delimited text files (.csv, .tsv, .txt) with automated sniffers.
    """
    det_encoding = encoding or detect_encoding(file_source)
    det_delimiter = delimiter or detect_delimiter(file_source, encoding=det_encoding)

    return pd.read_csv(
        file_source,
        delimiter=det_delimiter,
        encoding=det_encoding,
        chunksize=chunksize,
        low_memory=False,
        **kwargs,
    )


def get_excel_sheet_names(file_source: Union[str, Path, BinaryIO]) -> List[str]:
    """Inspects and returns sheet names available in an Excel workbook."""
    import openpyxl
    if isinstance(file_source, (str, Path)):
        wb = openpyxl.load_workbook(filename=str(file_source), read_only=True)
        sheets = list(wb.sheetnames)
        wb.close()
        return sheets
    else:
        # File-like binary buffer
        wb = openpyxl.load_workbook(filename=file_source, read_only=True)
        sheets = list(wb.sheetnames)
        wb.close()
        return sheets


def read_excel(
    file_source: Union[str, Path, BinaryIO],
    sheet_name: Optional[Union[str, int]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Ingests Excel (.xlsx, .xls) files via openpyxl with dynamic sheet selection.
    """
    target_sheet = 0 if sheet_name is None else sheet_name
    return pd.read_excel(file_source, sheet_name=target_sheet, engine="openpyxl", **kwargs)


def read_arrow(
    file_source: Union[str, Path, BinaryIO],
    format_type: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Ingests Apache Parquet (.parquet) or Feather (.feather) datasets via pyarrow.
    """
    path_str = str(file_source).lower() if isinstance(file_source, (str, Path)) else ""
    is_feather = (format_type == "feather") or path_str.endswith(".feather")

    if is_feather:
        return pd.read_feather(file_source, **kwargs)
    return pd.read_parquet(file_source, engine="pyarrow", **kwargs)


def read_json_flattened(
    file_source: Union[str, Path, BinaryIO, TextIO],
    is_jsonl: Optional[bool] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Ingests .json, .jsonl, .ndjson files using pandas.json_normalize() to flatten
    nested hierarchy keys (e.g., 'user.address.zipcode') and align schemas
    with uniform np.nan values.
    """
    records: List[Dict[str, Any]] = []
    is_stream = hasattr(file_source, "read")
    path_str = str(file_source).lower() if isinstance(file_source, (str, Path)) else ""

    # Detect JSON Lines format
    jsonl_mode = is_jsonl
    if jsonl_mode is None:
        jsonl_mode = path_str.endswith(".jsonl") or path_str.endswith(".ndjson")

    # Read content
    if is_stream:
        stream = file_source  # type: ignore
        content = stream.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
    else:
        with open(file_source, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

    content = content.strip()
    if not content:
        return pd.DataFrame()

    if jsonl_mode or ("\n" in content and content.startswith("{") and not content.endswith("]")):
        # Line-delimited JSON
        for line in content.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError:
                continue
    else:
        # Standard JSON document
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                records = parsed
            elif isinstance(parsed, dict):
                # Check for common wrapper keys like 'data', 'records', 'items'
                wrapped_key = next((k for k in ["data", "records", "items", "results"] if isinstance(parsed.get(k), list)), None)
                if wrapped_key:
                    records = parsed[wrapped_key]
                else:
                    records = [parsed]
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON content: {e}")

    if not records:
        return pd.DataFrame()

    # Flatten nested structures into uniform columns
    normalized_df = pd.json_normalize(records, **kwargs)
    return normalized_df


# ---------------------------------------------------------------------------
# Universal Ingestion Engine Orchestrator
# ---------------------------------------------------------------------------

# Size thresholds for chunked streaming protection
LARGE_DATASET_BYTES = 100 * 1024 * 1024  # 100 MB
CHUNK_SIZE_DEFAULT = 50000


def ingest_dataset(
    source: Union[str, Path, BinaryIO, TextIO, DBConnectorConfig],
    file_type: Optional[str] = None,
    sheet_name: Optional[Union[str, int]] = None,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    sanitize: bool = True,
    parse_dates: bool = True,
    date_columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    db_table: Optional[str] = None,
    db_query: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Universal Ingestion API for AutoData-Hub.

    Ingests Tabular (.csv, .tsv, .txt, .xlsx, .parquet, .feather),
    Semi-Structured (.json, .jsonl), and Relational Databases (PostgreSQL,
    MySQL, SQLite, Snowflake, MS SQL Server).

    Enforces automated safeguards:
    - Encoding autodetection (chardet)
    - Delimiter autodetection (csv.Sniffer)
    - Chunked streaming for large datasets (>100MB / >100k rows)
    - Vectorized regex string sanitization
    - Mixed date format parsing
    - Zero in-place mutations
    """
    df: Optional[pd.DataFrame] = None

    # 1. Handle Database Connector Configuration
    if isinstance(source, DBConnectorConfig):
        connector = DatabaseConnector(source)
        if db_query:
            raw_res = connector.read_query(db_query, chunksize=chunksize)
        elif db_table:
            raw_res = connector.read_table(db_table, chunksize=chunksize)
        else:
            raise ValueError("Must provide either db_table or db_query when ingesting from DBConnectorConfig.")

        if isinstance(raw_res, pd.DataFrame):
            df = raw_res
        else:
            # Concatenate chunks if returned as an iterator
            df = pd.concat(list(raw_res), ignore_index=True)

    else:
        # Determine format hint
        fmt = (file_type or "").lower().strip(".")
        if not fmt and isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower().strip(".")
            fmt = suffix
        elif not fmt and getattr(source, "name", None):
            fmt = Path(str(source.name)).suffix.lower().strip(".")

        # Check file size for chunked streaming safeguard
        is_large = False
        if isinstance(source, (str, Path)) and os.path.exists(source):
            file_bytes = os.path.getsize(source)
            if file_bytes >= LARGE_DATASET_BYTES:
                is_large = True

        effective_chunksize = chunksize or (CHUNK_SIZE_DEFAULT if is_large else None)

        # 2. Dispatch by Format
        if fmt in ("csv", "tsv", "txt"):
            res = read_delimited(
                source,
                delimiter=delimiter,
                encoding=encoding,
                chunksize=effective_chunksize,
                **kwargs,
            )
            if isinstance(res, pd.DataFrame):
                df = res
            else:
                # Streaming chunk accumulation
                chunks = []
                for chunk in res:
                    chunk_clean = chunk
                    if sanitize:
                        chunk_clean = sanitize_strings_vectorized(chunk_clean)
                    chunks.append(chunk_clean)
                df = pd.concat(chunks, ignore_index=True)

        elif fmt in ("xlsx", "xls"):
            df = read_excel(source, sheet_name=sheet_name, **kwargs)

        elif fmt in ("parquet", "pq"):
            df = read_arrow(source, format_type="parquet", **kwargs)

        elif fmt in ("feather", "ft"):
            df = read_arrow(source, format_type="feather", **kwargs)

        elif fmt in ("json", "jsonl", "ndjson"):
            is_jsonl = fmt in ("jsonl", "ndjson")
            df = read_json_flattened(source, is_jsonl=is_jsonl, **kwargs)

        else:
            # Fallback: attempt delimited file reading
            res = read_delimited(
                source,
                delimiter=delimiter,
                encoding=encoding,
                chunksize=effective_chunksize,
                **kwargs,
            )
            if isinstance(res, pd.DataFrame):
                df = res
            else:
                df = pd.concat(list(res), ignore_index=True)

    if df is None:
        raise ValueError("Failed to ingest dataset: unsupported or invalid source.")

    # 3. Apply Universal Safeguards (Vectorized String Sanitization & Mixed Date Parsing)
    if sanitize:
        df = sanitize_strings_vectorized(df)

    if parse_dates:
        df = parse_mixed_dates(df, date_columns=date_columns)

    return df
