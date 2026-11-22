import numpy as np
import pandas as pd
from typing import Generator, List, Any

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 5000) -> Generator[pd.DataFrame, None, None]:
    """
    Generator streaming dataframe in chunks of rows to prevent memory pressure.
    """
    total_rows = len(df)
    for start in range(0, total_rows, chunk_size):
        yield df.iloc[start:start + chunk_size]

def vectorize_series(series: pd.Series) -> np.ndarray:
    """
    Ensures series data is extracted into a contiguous C-order NumPy array for SIMD execution.
    """
    arr = series.to_numpy()
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts numeric types where possible without loss of precision to save RAM.
    """
    optimized = df.copy()
    for col in optimized.select_dtypes(include=["int64", "int32"]).columns:
        c_min = optimized[col].min()
        c_max = optimized[col].max()
        if pd.notna(c_min) and pd.notna(c_max):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                optimized[col] = optimized[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                optimized[col] = optimized[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                optimized[col] = optimized[col].astype(np.int32)

    for col in optimized.select_dtypes(include=["float64"]).columns:
        optimized[col] = pd.to_numeric(optimized[col], downcast="float")

    return optimized
