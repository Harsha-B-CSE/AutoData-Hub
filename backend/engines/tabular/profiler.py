import csv
import io
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from backend.engines.tabular.encoder import is_identifier_or_name_column

def detect_delimiter_and_encoding(file_path: Path) -> Tuple[str, str]:
    """
    Detects delimiter (comma, semicolon, tab, pipe) and encoding for text-based tables.
    """
    encodings = ["utf-8", "latin-1", "cp1252"]
    sample_bytes = b""
    detected_encoding = "utf-8"

    for enc in encodings:
        try:
            with open(file_path, "rb") as f:
                sample_bytes = f.read(16384)
            sample_bytes.decode(enc)
            detected_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    sample_text = sample_bytes.decode(detected_encoding, errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except Exception:
        # Fallback heuristic
        counts = {d: sample_text.count(d) for d in [",", "\t", ";", "|"]}
        delimiter = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

    return delimiter, detected_encoding

def load_tabular_file(file_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Loads CSV, TSV, XLS, or XLSX file into a pandas DataFrame.
    """
    ext = file_path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        delimiter, encoding = detect_delimiter_and_encoding(file_path)
        return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, low_memory=False)
    elif ext in [".xlsx", ".xls"]:
        engine = "openpyxl" if ext == ".xlsx" else None
        target_sheet = sheet_name if sheet_name else 0
        return pd.read_excel(file_path, sheet_name=target_sheet, engine=engine)
    else:
        raise ValueError(f"Unsupported tabular extension: {ext}")

class TabularProfiler:
    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes missing value ratios, unique values, memory footprint, skewness,
        kurtosis, outlier counts, and correlation matrix.
        """
        total_rows, total_cols = df.shape
        memory_bytes = int(df.memory_usage(deep=True).sum())
        duplicate_rows = int(df.duplicated().sum())

        columns_profile: Dict[str, Dict[str, Any]] = {}
        numeric_cols: List[str] = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_percent = float((null_count / total_rows) * 100.0) if total_rows > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))
            dtype_str = str(series.dtype)

            is_ident = is_identifier_or_name_column(str(col), series)
            col_meta: Dict[str, Any] = {
                "name": str(col),
                "dtype": dtype_str,
                "null_count": null_count,
                "null_percent": round(null_percent, 2),
                "unique_count": unique_count,
                "is_numeric": pd.api.types.is_numeric_dtype(series),
                "is_identifier_or_name": is_ident,
                "sample_values": [None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
                                  for v in series.dropna().head(5).tolist()]
            }

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                numeric_cols.append(col)
                valid_num = series.dropna()
                if len(valid_num) > 0:
                    q25 = float(valid_num.quantile(0.25))
                    q75 = float(valid_num.quantile(0.75))
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    iqr_outliers = int(((valid_num < lower_bound) | (valid_num > upper_bound)).sum())

                    mean_val = float(valid_num.mean())
                    std_val = float(valid_num.std()) if len(valid_num) > 1 else 0.0
                    if std_val > 0:
                        z_scores = np.abs((valid_num - mean_val) / std_val)
                        zscore_outliers = int((z_scores > 3.0).sum())
                    else:
                        zscore_outliers = 0

                    skew_val = float(stats.skew(valid_num, nan_policy="omit")) if len(valid_num) > 2 else 0.0
                    kurt_val = float(stats.kurtosis(valid_num, nan_policy="omit")) if len(valid_num) > 3 else 0.0

                    col_meta.update({
                        "min": float(valid_num.min()),
                        "max": float(valid_num.max()),
                        "mean": round(mean_val, 4),
                        "median": round(float(valid_num.median()), 4),
                        "std": round(std_val, 4),
                        "q25": round(q25, 4),
                        "q75": round(q75, 4),
                        "iqr": round(iqr, 4),
                        "iqr_outliers": iqr_outliers,
                        "zscore_outliers": zscore_outliers,
                        "skewness": round(skew_val if not math.isnan(skew_val) else 0.0, 4),
                        "kurtosis": round(kurt_val if not math.isnan(kurt_val) else 0.0, 4),
                    })

            columns_profile[str(col)] = col_meta

        # Correlation matrix for numerical features
        corr_matrix: Dict[str, Dict[str, float]] = {}
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr(method="pearson").round(4)
            for c1 in numeric_cols:
                corr_matrix[str(c1)] = {}
                for c2 in numeric_cols:
                    val = corr_df.loc[c1, c2]
                    corr_matrix[str(c1)][str(c2)] = None if pd.isna(val) else float(val)

        categorical_cols = [
            str(c) for c in df.select_dtypes(include=["object", "string", "category"]).columns
        ]
        identifier_cols = [
            str(c) for c in df.columns if is_identifier_or_name_column(str(c), df[c])
        ]

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "memory_bytes": memory_bytes,
            "memory_mb": round(memory_bytes / (1024 * 1024), 2),
            "duplicate_rows": duplicate_rows,
            "columns": columns_profile,
            "correlation_matrix": corr_matrix,
            "numeric_columns": [str(c) for c in numeric_cols],
            "categorical_columns": categorical_cols,
            "identifier_columns": identifier_cols,
        }
