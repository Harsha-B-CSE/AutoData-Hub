"""Regression tests for file-like upload format inference."""

import io
import json

from core.ingestion_engine import ingest_dataset


def test_file_like_json_uses_name_suffix_instead_of_csv_fallback():
    uploaded = io.BytesIO(
        json.dumps([
            {"tags": ["a", "b"], "meta": {"x": 1}},
            {"tags": ["a", "b"], "meta": {"x": 1}},
        ]).encode("utf-8")
    )
    uploaded.name = "nested.json"
    frame = ingest_dataset(uploaded, parse_dates=False)
    assert frame.shape == (2, 2)
    assert frame.columns.tolist() == ["tags", "meta.x"]
