"""Privacy regression for audit-report metric serialization."""

from pathlib import Path
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from pypdf import PdfReader

from core.schema_inference import ExecutionStatus, ModuleResult
from modules.script_gen.exporter import export_audit_pdf


def test_pdf_excludes_raw_health_sample_values(tmp_path):
    raw_email = "secret.person@example.com"
    result = ModuleResult(
        status=ExecutionStatus.SUCCESS,
        module_name="health_audit",
        dataset_version="v0_raw",
        metrics={
            "quality_score": 100.0,
            "columns": {
                "email": {
                    "dtype": "object",
                    "sample_values": [raw_email],
                }
            },
        },
        issues=[],
        artifacts={},
    )
    artifact = export_audit_pdf(
        module_results=[result],
        actions=[],
        output_dir=tmp_path,
    )
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(Path(artifact.filepath)).pages
    )
    assert raw_email not in text
    assert "quality_score" in text
