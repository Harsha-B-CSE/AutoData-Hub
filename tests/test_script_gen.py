"""Test suite for Module 5: Script Generator & PDF Exporter."""
import os

import numpy as np
import pandas as pd
from core.schema_inference import (
    TransformationAction, ExecutionStatus, ModuleResult
)
from modules.script_gen.exporter import (
    PipelineScriptGenerator, export_pipeline_script, DeliverableArtifact
)
from utils.ast_sandbox import validate_code, ASTSandboxError, is_safe

# --------------------------------------------------------------------------
# Test 1: AST Sandbox — safe code passes
# --------------------------------------------------------------------------
safe_snippet = """
import pandas as pd
import numpy as np
df['age'] = df['age'].fillna(df['age'].median())
"""
violations = validate_code(safe_snippet, raise_on_first=False)
assert violations == [], f"Unexpected violations: {violations}"
print("Test 1 PASS: AST sandbox accepts safe Pandas code")

# --------------------------------------------------------------------------
# Test 2: AST Sandbox — blocked imports caught
# --------------------------------------------------------------------------
dangerous_snippets = [
    "import os\nos.system('rm -rf /')",
    "import subprocess\nsubprocess.run(['ls'])",
    "eval('__import__(\"os\").system(\"whoami\")')",
    "exec('import sys')",
]
for snip in dangerous_snippets:
    assert not is_safe(snip), f"Should have blocked: {snip[:40]}"
print("Test 2 PASS: AST sandbox blocks os/subprocess/eval/exec correctly")

# --------------------------------------------------------------------------
# Test 3: AST Sandbox — network imports blocked
# --------------------------------------------------------------------------
net_snips = [
    "import requests\nrequests.get('http://evil.com')",
    "from urllib import request",
]
for snip in net_snips:
    assert not is_safe(snip), f"Should have blocked: {snip[:40]}"
print("Test 3 PASS: AST sandbox blocks network imports correctly")

# --------------------------------------------------------------------------
# Test 4: Script Generator — basic compilation
# --------------------------------------------------------------------------
actions = [
    TransformationAction(
        step_id=1,
        module="imputation",
        action_type="impute_median",
        target_columns=["age"],
        parameters={"strategy": "median"},
        code_snippet="df['age'] = df['age'].fillna(df['age'].median())",
    ),
    TransformationAction(
        step_id=2,
        module="imputation",
        action_type="impute_mode",
        target_columns=["dept"],
        parameters={"strategy": "most_frequent"},
        code_snippet="df['dept'] = df['dept'].fillna(df['dept'].mode()[0])",
    ),
    TransformationAction(
        step_id=3,
        module="anomaly",
        action_type="ensemble_anomaly_detection",
        target_columns=["anomaly_flag", "anomaly_score_ensemble"],
        parameters={"contamination": 0.05},
        code_snippet=(
            "from sklearn.ensemble import IsolationForest\n"
            "from sklearn.preprocessing import RobustScaler\n"
            "X = df[['age']].fillna(0).to_numpy()\n"
            "X_scaled = RobustScaler().fit_transform(X)\n"
            "iso = IsolationForest(contamination=0.05, random_state=42)\n"
            "iso.fit(X_scaled)\n"
            "df['anomaly_flag'] = (iso.predict(X_scaled) == -1).astype(int)"
        ),
    ),
]

gen = PipelineScriptGenerator()
script_src = gen.generate(
    actions=actions,
    dataset_name="test_dataset",
    version_chain=["v0_raw", "v1_privacy", "v2_imputed", "v3_anomaly"],
    input_csv_path="data/test.csv",
    validate_snippets=True,
)
assert "import pandas as pd" in script_src
assert "import numpy as np" in script_src
assert "IsolationForest" in script_src
assert "impute median" in script_src.lower() or "Impute Median" in script_src or "impute_median" in script_src.lower()
assert "# ── DATA LOADING" in script_src
assert "# ── MODULE 2" in script_src or "imputation" in script_src.lower()
assert "# ── MODULE 3" in script_src or "anomaly" in script_src.lower()
print("Test 4 PASS: Script generator produces valid Python with correct sections")

# --------------------------------------------------------------------------
# Test 5: Script is valid Python (parseable)
# --------------------------------------------------------------------------
import ast
ast.parse(script_src)  # Will raise SyntaxError if invalid
print("Test 5 PASS: Generated pipeline.py is syntactically valid Python")

# --------------------------------------------------------------------------
# Test 6: Import deduplication
# --------------------------------------------------------------------------
assert script_src.count("import pandas as pd") == 1, "pandas should appear exactly once"
assert script_src.count("import numpy as np") == 1, "numpy should appear exactly once"
# IsolationForest import appears in snippet, should be deduplicated in header
print("Test 6 PASS: Import deduplication works correctly")

# --------------------------------------------------------------------------
# Test 7: Dangerous snippet is excluded (not silently passing)
# --------------------------------------------------------------------------
evil_action = TransformationAction(
    step_id=99,
    module="rogue",
    action_type="os_shell",
    target_columns=[],
    parameters={},
    code_snippet="import os\nos.system('whoami')",
)
try:
    gen.generate(actions=[evil_action], validate_snippets=True)
    raise AssertionError("Dangerous snippets must fail pipeline synthesis.")
except ASTSandboxError:
    pass
print("Test 7 PASS: Dangerous snippet fails pipeline synthesis closed")

# --------------------------------------------------------------------------
# Test 8: write() produces DeliverableArtifact with checksum
# --------------------------------------------------------------------------
import tempfile
with tempfile.TemporaryDirectory() as tmp_dir:
    artifact = export_pipeline_script(
        actions=actions,
        dataset_name="test_ds",
        output_dir=tmp_dir,
        validate_snippets=True,
    )
    assert artifact.filename == "pipeline.py"
    assert os.path.exists(artifact.filepath)
    assert artifact.size_bytes > 0
    assert len(artifact.sha256) == 64  # SHA-256 is 64 hex chars
    assert artifact.created_at != ""
    print(f"  Generated: {artifact.filename} ({artifact.size_bytes} bytes)")
    print(f"  SHA-256: {artifact.sha256[:16]}...")
print("Test 8 PASS: export_pipeline_script() produces valid DeliverableArtifact with SHA-256")

# --------------------------------------------------------------------------
# Test 9: Empty action list produces minimal valid script
# --------------------------------------------------------------------------
empty_script = gen.generate(actions=[], validate_snippets=False)
assert "import pandas as pd" in empty_script
assert "pd.read_csv" in empty_script
ast.parse(empty_script)
print("Test 9 PASS: Empty action list produces valid minimal script")

print()
print("=== ALL 9 TESTS PASSED === Module 5 Script Generator & AST Sandbox validated.")
