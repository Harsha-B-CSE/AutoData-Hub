"""Empirical precision/recall checks for supported synthetic PII patterns."""

import pandas as pd

from modules.privacy.detector import PIIDetector, PIIType, luhn_checksum_valid


def test_synthetic_pii_precision_recall_f1():
    cases = [
        ("123-45-6789", PIIType.SSN),
        ("001-01-0001", PIIType.SSN),
        ("000-12-3456", None),
        ("666-12-3456", None),
        ("900-12-3456", None),
        ("123-00-6789", None),
        ("4111111111111111", PIIType.CREDIT_CARD),
        ("378282246310005", PIIType.CREDIT_CARD),
        ("4111 1111 1111 1111", PIIType.CREDIT_CARD),
        ("4111111111111112", None),
        ("378282246310006", None),
        ("user@example.com", PIIType.EMAIL),
        ("first.last+tag@sub.example.co.uk", PIIType.EMAIL),
        ("user@", None),
        ("not-an-email", None),
        ("(415) 555-2671", PIIType.PHONE),
        ("+44 207 946 0958", PIIType.PHONE),
        ("+91 987 654 3210", PIIType.PHONE),
        ("555-12", None),
        ("1234567890123456", None),
    ]

    detector = PIIDetector()
    true_positive = false_positive = false_negative = 0
    predictions = []

    for index, (value, expected_type) in enumerate(cases):
        result = detector.detect_column(
            pd.Series([value], name=f"candidate_{index}")
        )
        predicted_type = result.pii_type if result is not None else None
        predictions.append((value, expected_type, predicted_type))

        if expected_type is None:
            false_positive += int(predicted_type is not None)
        elif predicted_type == expected_type:
            true_positive += 1
        else:
            false_negative += 1
            false_positive += int(predicted_type is not None)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1_score = 2 * precision * recall / max(1e-12, precision + recall)

    print(
        f"Synthetic PII metrics: precision={precision:.3f}, "
        f"recall={recall:.3f}, F1={f1_score:.3f}"
    )
    assert luhn_checksum_valid("4111111111111111")
    assert not luhn_checksum_valid("4111111111111112")
    assert f1_score >= 0.95, predictions
