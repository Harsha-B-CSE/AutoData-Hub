import os
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import pandas as pd
import numpy as np
from core.ingestion_engine import ingest_dataset

def test_scenario_a_non_utf8_encoding(tmp_path):
    """Test cp1252 encoded CSV with semicolon separators and currency symbols."""
    file_path = tmp_path / "dirty_latin1.csv"
    raw_content = "ID;Name;Salary\n1;Müller;$1,200.50\n2;García;€2,300.00\n3;Smith;N/A".encode("cp1252")
    file_path.write_bytes(raw_content)

    # FIXED: Captures the single pd.DataFrame return from the engine payload
    df = ingest_dataset(str(file_path))

    assert df is not None
    assert len(df) == 3
    assert "Salary" in df.columns

    # Verify symbol stripping coerced floats correctly
    assert df["Salary"].iloc[0] == 1200.50
    assert pd.isna(df["Salary"].iloc[2])


def test_scenario_b_nested_json_schema_union(tmp_path):
    """Test heterogeneous nested JSON records with missing keys."""
    file_path = tmp_path / "nested.json"
    json_data = '''[
        {"id": 1, "user": {"name": "Alice", "address": {"zip": "10001"}}},
        {"id": 2, "user": {"name": "Bob"}},
        {"id": 3, "user": {"name": "Charlie", "address": {"zip": "90210"}}, "age": 30}
    ]'''
    file_path.write_text(json_data)

    # FIXED: Captures the single pd.DataFrame return from the engine payload
    df = ingest_dataset(str(file_path))

    assert "user.address.zip" in df.columns
    assert pd.isna(df.loc[1, "user.address.zip"])  # Bob has no zip
    assert df.loc[2, "age"] == 30.0
