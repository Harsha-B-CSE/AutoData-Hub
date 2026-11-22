import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import pandas as pd
import numpy as np

from core.schema_inference import ExecutionStatus, TransformResult, TransformationAction
from modules.privacy import (
    PIIType,
    PIIDetector,
    PIIMasker,
    MaskingStrategy,
    apply_privacy_firewall,
)
from modules.script_gen.exporter import PipelineScriptGenerator
from utils.ast_sandbox import validate_code


def test_pii_detection_categories():
    """Validates regex and semantic header detection for Email, Phone, SSN, Credit Card, IP, DOB, Name, Address."""
    df = pd.DataFrame({
        "contact_email": ["john.doe@example.com", "sarah.connor@sky.net", "info@company.org"],
        "telephone": ["+1 (555) 234-5678", "555-876-5432", "(555) 999-0000"],
        "ssn": ["123-45-6789", "987-65-4321", "555-00-1234"],
        "card_number": ["4532015112830366", "4532015112830366", "4532015112830366"],  # Valid Luhn Visa
        "client_ip": ["192.168.1.1", "10.0.0.254", "172.16.0.5"],
        "dob": ["1990-05-12", "1985-11-23", "2001-08-04"],
        "full_name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "street_address": ["123 Main St, Springfield", "456 Elm Ave, Metropolis", "789 Oak Blvd, Gotham"],
    })

    detector = PIIDetector()
    detections = detector.scan_dataframe(df)

    assert "contact_email" in detections
    assert detections["contact_email"].pii_type == PIIType.EMAIL

    assert "telephone" in detections
    assert detections["telephone"].pii_type == PIIType.PHONE

    assert "ssn" in detections
    assert detections["ssn"].pii_type == PIIType.SSN

    assert "card_number" in detections
    assert detections["card_number"].pii_type == PIIType.CREDIT_CARD

    assert "client_ip" in detections
    assert detections["client_ip"].pii_type == PIIType.IP_ADDRESS

    assert "dob" in detections
    assert detections["dob"].pii_type == PIIType.DOB

    assert "full_name" in detections
    assert detections["full_name"].pii_type == PIIType.NAME

    assert "street_address" in detections
    assert detections["street_address"].pii_type == PIIType.ADDRESS


def test_false_positive_safeguards():
    """Ensures numeric IDs and surrogate keys are NOT incorrectly flagged as SSN or Phone numbers."""
    df_ids = pd.DataFrame({
        "user_id": [100000001, 100000002, 100000003, 100000004],  # 9-digit integers
        "order_id": [9876543210, 9876543211, 9876543212, 9876543213],  # 10-digit integers
        "item_id": [123456789, 234567890, 345678901, 456789012],
        "transaction_code": ["9999-8888-7777-6666", "1111-2222-3333-4444", "0000-0000-0000-0000", "5555-5555-5555-5555"],  # Invalid Luhn
        "product_sku": ["SKU-990-12", "SKU-990-13", "SKU-990-14", "SKU-990-15"],
    })

    detector = PIIDetector()
    detections = detector.scan_dataframe(df_ids)

    # None of these surrogate keys should be detected as SSN, Phone, or Credit Card
    assert "user_id" not in detections
    assert "order_id" not in detections
    assert "item_id" not in detections
    assert "transaction_code" not in detections
    assert "product_sku" not in detections


def test_masking_strategies():
    """Tests Redaction, Salted SHA-256 Hashing, and Anonymization operations."""
    df = pd.DataFrame({
        "ssn": ["123-45-6789", "987-65-4321"],
        "email": ["user1@test.com", "user2@test.com"],
        "name": ["Alice Smith", "Bob Jones"],
    })

    masker = PIIMasker(session_salt="fixed_test_salt_123")

    # 1. Redaction on SSN
    s_redacted, code_redact = masker.mask_series(df["ssn"], PIIType.SSN, MaskingStrategy.REDACT)
    assert s_redacted.iloc[0] == "XXX-XX-6789"
    assert s_redacted.iloc[1] == "XXX-XX-4321"
    assert "XXX-XX-" in code_redact

    # 2. Salted Hashing on Email
    s_hashed, code_hash = masker.mask_series(df["email"], PIIType.EMAIL, MaskingStrategy.HASH)
    assert len(s_hashed.iloc[0]) == 64
    assert s_hashed.iloc[0] != df["email"].iloc[0]
    # Consistent with same salt
    s_hashed_2, _ = masker.mask_series(df["email"], PIIType.EMAIL, MaskingStrategy.HASH)
    assert s_hashed.iloc[0] == s_hashed_2.iloc[0]

    # Different with different salt
    other_masker = PIIMasker(session_salt="different_salt_456")
    s_hashed_diff, _ = other_masker.mask_series(df["email"], PIIType.EMAIL, MaskingStrategy.HASH)
    assert s_hashed.iloc[0] != s_hashed_diff.iloc[0]

    # 3. Anonymization on Name
    s_anon, code_anon = masker.mask_series(df["name"], PIIType.NAME, MaskingStrategy.ANONYMIZE)
    assert s_anon.iloc[0] == "Anon_User_0001"
    assert s_anon.iloc[1] == "Anon_User_0002"
    assert "Anon_User" in code_anon


def test_privacy_firewall_transform_result():
    """Validates the complete Privacy Firewall execution returning a pure TransformResult."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "email": ["alice@company.com", "bob@company.com", "charlie@company.com"],
        "ssn": ["123-45-6789", "234-56-7890", "345-67-8901"],
        "score": [95.5, 88.0, 72.3],
    })

    # Apply firewall
    result = apply_privacy_firewall(
        df,
        strategy_overrides={"email": "hash", "ssn": "redact"},
        session_salt="firewall_test_salt"
    )

    # 1. Validate TransformResult contract
    assert isinstance(result, TransformResult)
    assert result.status == ExecutionStatus.SUCCESS
    assert result.error_message is None
    assert len(result.audit_log) >= 2
    assert len(result.applied_actions) == 2

    # 2. Validate Immutability (original df not mutated)
    assert df["ssn"].iloc[0] == "123-45-6789"
    assert df["email"].iloc[0] == "alice@company.com"

    # Transformed copy is sanitized
    transformed = result.transformed_df
    assert transformed["ssn"].iloc[0] == "XXX-XX-6789"
    assert transformed["email"].iloc[0] != "alice@company.com"
    assert transformed["score"].iloc[0] == 95.5  # Unchanged non-PII column

    # 3. Validate executable Pandas code snippets in TransformationActions
    for action in result.applied_actions:
        assert isinstance(action, TransformationAction)
        assert action.module == "privacy"
        assert len(action.code_snippet) > 0

        # Execute code snippet on fresh copy and verify it executes without error
        test_df_copy = df.copy()
        exec_scope = {"df": test_df_copy, "pd": pd, "np": np}
        exec(action.code_snippet, exec_scope)
        # Verify the target column in test_df_copy now matches transformed
        target_col = action.target_columns[0]
        assert (test_df_copy[target_col] == transformed[target_col]).all()


def test_header_boundaries_and_full_column_scan():
    """Header matching avoids substrings and scanning cannot be bypassed after row 200."""
    notes = ["ordinary note"] * 250
    notes[-1] = "Contact late.user@example.com"
    df = pd.DataFrame({
        "shipping_address": ["123 Main St"] * 250,
        "company_name": ["Acme Corp"] * 250,
        "capacity": list(range(250)),
        "description": notes,
    })

    detections = PIIDetector().detect(df)

    assert detections["shipping_address"].pii_type == PIIType.ADDRESS
    assert detections["description"].pii_type == PIIType.EMAIL
    assert "company_name" not in detections
    assert "capacity" not in detections


def test_detector_mapping_preserves_types_and_redacts_fail_closed():
    """Detector dictionaries retain types and every detected non-null cell changes."""
    df = pd.DataFrame({
        "card_number": ["378282246310005", "371449635398431"],
        "client_ip": ["2001:0db8:85a3:0000:0000:8a2e:0370:7334", "::1"],
        "ein": ["12-3456789", "98-7654321"],
        "ssn": ["123456789", "234567890"],
    })
    findings = PIIDetector().detect(df)

    result = PIIMasker(session_salt="mapping_test_salt").mask(
        df,
        findings=findings,
        mode="redact",
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert findings["card_number"].pii_type == PIIType.CREDIT_CARD
    assert findings["client_ip"].pii_type == PIIType.IP_ADDRESS
    assert findings["ein"].pii_type == PIIType.SSN
    assert all(not result.transformed_df[col].equals(df[col]) for col in findings)

    replay = df.copy(deep=True)
    for action in result.applied_actions:
        validate_code(action.code_snippet)
        exec(action.code_snippet, {"df": replay, "pd": pd, "np": np})
    pd.testing.assert_frame_equal(replay, result.transformed_df)


def test_privacy_snippets_are_secure_and_exportable():
    """Hash/anonymize actions replay exactly without embedding original PII."""
    df = pd.DataFrame({
        "email": ["alice@example.com", "bob@example.com"],
        "full_name": ["Alice Smith", "Bob Jones"],
    })
    findings = PIIDetector().detect(df)
    result = PIIMasker(session_salt="safe_export_salt").mask_dataframe(
        df,
        findings,
        strategy_overrides={"email": "hash", "full_name": "anonymize"},
    )

    for action in result.applied_actions:
        validate_code(action.code_snippet)
        assert "alice@example.com" not in action.code_snippet
        assert "Alice Smith" not in action.code_snippet

    script = PipelineScriptGenerator().generate(
        result.applied_actions,
        validate_snippets=True,
    )
    assert "import hashlib" in script
    assert "alice@example.com" not in script
    assert "Alice Smith" not in script

    replay = df.copy(deep=True)
    for action in result.applied_actions:
        exec(action.code_snippet, {"df": replay, "pd": pd, "np": np})
    pd.testing.assert_frame_equal(replay, result.transformed_df)


def test_duplicate_columns_fail_explicitly():
    """Ambiguous duplicate labels must not be reported as successfully sanitized."""
    df = pd.DataFrame(
        [["a@example.com", "b@example.com"]],
        columns=["email", "email"],
    )

    with pytest.raises(ValueError, match="unique column labels"):
        PIIDetector().detect(df)


def test_non_string_and_quoted_columns_generate_safe_code():
    """Column labels and salts are represented safely in replay snippets."""
    df = pd.DataFrame({
        "email'o": ["alice@example.com"],
        7: ["bob@example.com"],
    })
    findings = PIIDetector().detect(df)
    result = PIIMasker(session_salt="salt'with\nnewline").mask_dataframe(
        df,
        findings,
        strategy_overrides={col: "hash" for col in findings},
    )

    replay = df.copy(deep=True)
    for action in result.applied_actions:
        validate_code(action.code_snippet)
        exec(action.code_snippet, {"df": replay, "pd": pd, "np": np})

    pd.testing.assert_frame_equal(replay, result.transformed_df)


def test_camelcase_headers_ipv6_sparse_phone_and_surrogate_guards():
    frame = pd.DataFrame({
        "fullName": ["Alice Smith", "Bob Jones", "Carol Lee", "Dan Ray"],
        "streetAddress": ["1 Main St", "2 Oak St", "3 Pine St", "4 Elm St"],
        "dateOfBirth": ["1990-01-01", "1988-02-02", "1977-03-03", "1999-04-04"],
        "ssnNumber": ["123456789", "234567890", "345678901", "456789012"],
        "customerId": ["4155552671", "4155552672", "4155552673", "4155552674"],
        "order_id": ["4111111111111111"] * 4,
        "network_value": ["2001:db8::1", "plain", "plain", "plain"],
        "contact_note": ["+44 207 946 0958", "none", "none", "none"],
    })
    findings = PIIDetector().detect(frame)

    assert findings["fullName"].pii_type == PIIType.NAME
    assert findings["streetAddress"].pii_type == PIIType.ADDRESS
    assert findings["dateOfBirth"].pii_type == PIIType.DOB
    assert findings["ssnNumber"].pii_type == PIIType.SSN
    assert findings["network_value"].pii_type == PIIType.IP_ADDRESS
    assert findings["contact_note"].pii_type == PIIType.PHONE
    assert "customerId" not in findings
    assert "order_id" not in findings


def test_dob_hash_replay_canonicalizes_timestamp_and_csv_string():
    timestamp_frame = pd.DataFrame({"dateOfBirth": [pd.Timestamp("2000-01-01")]})
    finding = PIIDetector().detect(timestamp_frame)
    result = PIIMasker(session_salt="canonical-date-salt").mask_dataframe(
        timestamp_frame,
        finding,
        strategy_overrides={"dateOfBirth": "hash"},
    )
    action = result.applied_actions[0]
    validate_code(action.code_snippet)

    replay = pd.DataFrame({"dateOfBirth": ["2000-01-01"]})
    exec(action.code_snippet, {"df": replay, "pd": pd, "np": np})
    assert replay["dateOfBirth"].iloc[0] == result.transformed_df["dateOfBirth"].iloc[0]


@pytest.mark.parametrize("strategy", ["redact", "hash", "anonymize"])
def test_regex_only_findings_mask_matched_rows_without_erasing_notes(strategy):
    frame = pd.DataFrame({
        "notes": [
            "ordinary operational note",
            "person@example.com",
            "another ordinary note",
        ]
    })
    findings = PIIDetector().detect(frame)
    assert findings["notes"].detection_method == "regex"
    result = PIIMasker(session_salt="selective-salt").mask_dataframe(
        frame,
        findings,
        strategy_overrides={"notes": strategy},
    )

    assert result.transformed_df["notes"].iloc[0] == frame["notes"].iloc[0]
    assert result.transformed_df["notes"].iloc[2] == frame["notes"].iloc[2]
    assert result.transformed_df["notes"].iloc[1] != frame["notes"].iloc[1]
    assert result.applied_actions[0].parameters["changed_cells"] == 1

    replay = frame.copy(deep=True)
    exec(
        result.applied_actions[0].code_snippet,
        {"df": replay, "pd": pd, "np": np},
    )
    pd.testing.assert_frame_equal(replay, result.transformed_df)


def test_continuous_numeric_features_are_not_scanned_as_identifier_strings():
    frame = pd.DataFrame({
        "feature_a": [40.06838553450637, 12.719495080619923, 8.979385846424664]
    })
    assert PIIDetector().detect(frame) == {}


def test_zero_framework_leakage():
    """Confirms no Streamlit or UI library dependencies exist in modules/privacy/."""
    import inspect
    import modules.privacy.detector as detector_module
    import modules.privacy.masker as masker_module

    for mod in [detector_module, masker_module]:
        source = inspect.getsource(mod)
        assert "import streamlit" not in source
        assert "from streamlit" not in source
        assert "st." not in source
