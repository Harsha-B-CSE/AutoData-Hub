"""
AutoData-Hub Local PII & Privacy Detector (Module 4).

Performs strict local detection of Personally Identifiable Information (PII)
governed by GDPR, HIPAA, and CCPA standards. Combines high-precision regex detectors
with a Semantic Header Classifier and False Positive safeguards for surrogate IDs.

Zero external API dependencies. Strictly decoupled from UI frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class PIIType(str, Enum):
    """Supported Personally Identifiable Information classifications."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DOB = "dob"
    NAME = "name"
    ADDRESS = "address"
    GENERIC_PII = "generic_pii"


@dataclass(frozen=True)
class PIIDetectionResult:
    """Outcome of PII analysis on an individual dataset column."""
    column: Any
    pii_type: PIIType
    confidence: float
    detection_method: str  # 'regex', 'semantic_header', 'hybrid'
    match_count: int = 0
    match_ratio: float = 0.0
    sample_matches: List[str] = field(default_factory=list)
    recommended_strategy: str = "redact"  # 'redact', 'hash', 'anonymize'
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "column": self.column,
            "pii_type": self.pii_type.value,
            "confidence": round(self.confidence, 4),
            "detection_method": self.detection_method,
            "match_count": self.match_count,
            "match_ratio": round(self.match_ratio, 4),
            "sample_matches": list(self.sample_matches),
            "recommended_strategy": self.recommended_strategy,
            "details": dict(self.details),
        }


# ---------------------------------------------------------------------------
# High-Precision Regex Patterns
# ---------------------------------------------------------------------------

REGEX_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

REGEX_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)

REGEX_SSN_FORMATTED = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

REGEX_SSN_RAW = re.compile(
    r"^\d{9}$"
)

REGEX_CREDIT_CARD = re.compile(
    r"(?<!\d)(?:\d[-\s]?){12,18}\d(?!\d)"
)

REGEX_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)

REGEX_IPV6 = re.compile(
    r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b|\b::(?:[A-Fa-f0-9]{1,4}:){0,5}[A-Fa-f0-9]{1,4}\b"
)

REGEX_DATE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)


# ---------------------------------------------------------------------------
# Semantic Header Keyword Mappings & Exclusion Safeguards
# ---------------------------------------------------------------------------

SEMANTIC_HEADER_KEYWORDS: Dict[PIIType, Set[str]] = {
    PIIType.EMAIL: {
        "email", "e_mail", "mail_address", "electronic_mail", "user_email", "contact_email", "customer_email"
    },
    PIIType.PHONE: {
        "phone", "telephone", "mobile", "cell", "fax", "contact_num", "cellphone", "tel", "phone_number"
    },
    PIIType.SSN: {
        "ssn", "social_security", "social_security_number", "tax_id", "itin", "ein", "sin"
    },
    PIIType.CREDIT_CARD: {
        "credit_card", "card_number", "ccn", "cvv", "pan", "debit_card", "account_number", "iban", "swift"
    },
    PIIType.IP_ADDRESS: {
        "ip", "ip_address", "ipv4", "ipv6", "client_ip", "host_ip", "remote_addr", "ip_addr"
    },
    PIIType.DOB: {
        "dob", "birth_date", "date_of_birth", "birthday", "bday", "birthdate"
    },
    PIIType.NAME: {
        "first_name", "last_name", "full_name", "fname", "lname", "surname", "patient_name",
        "client_name", "customer_name", "user_name", "author_name", "employee_name", "full_names"
    },
    PIIType.ADDRESS: {
        "address", "street", "street_address", "zipcode", "postal_code", "city", "residence",
        "home_address", "billing_address", "shipping_address", "residential_address", "zip_code"
    },
}

# Exclusion keywords for surrogate/database keys to prevent false positive PII tagging
ID_EXCLUSION_KEYWORDS: Set[str] = {
    "id", "uid", "uuid", "guid", "pk", "fk", "item_id", "product_id", "order_id",
    "transaction_id", "row_id", "index", "sku", "category_id", "record_id", "entity_id",
    "code", "transaction_code", "tracking_code", "confirmation_code", "ref_code", "product_sku"
}


def luhn_checksum_valid(num_str: str) -> bool:
    """
    Validates Luhn formula (Mod 10) on potential credit card numbers,
    verifying standard Major Industry Identifier (MII) prefixes and non-triviality.
    """
    digits = [int(c) for c in str(num_str) if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    # Reject dummy all-identical digits (e.g. 0000... or 1111...)
    if len(set(digits)) <= 1:
        return False
    # Check standard payment network prefixes (Visa=4, MC=5 or 2, Amex=34/37, Discover=6)
    d_str = "".join(map(str, digits))
    if not re.match(r"^(4|5[1-5]|2[2-7]|3[47]|6)", d_str):
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled if doubled < 10 else (doubled - 9)
        else:
            checksum += d
    return checksum % 10 == 0


class PIIDetector:
    """
    Local PII Detection Engine combining regex pattern scanners with semantic
    header classification and ID false-positive mitigation safeguards.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        match_threshold: float = 0.25,
        min_sample_size: int = 10,
    ):
        self.config = config or {}
        self.match_threshold = self.config.get("match_threshold", match_threshold)
        self.min_sample_size = self.config.get("min_sample_size", min_sample_size)

    def detect_column(
        self,
        series: pd.Series,
    ) -> Optional[PIIDetectionResult]:
        """
        Scans a single pandas Series for PII characteristics.

        Parameters
        ----------
        series : pd.Series
            Column to evaluate.

        Returns
        -------
        Optional[PIIDetectionResult]
            Detection details if PII is identified, else None.
        """
        col_name = str(series.name).strip()
        non_null = series.dropna()

        if len(non_null) == 0:
            return None

        # Privacy scanning must inspect the full column; head-only sampling lets
        # sensitive values later in a dataset bypass the firewall.
        sample = non_null.astype(str)

        # 1. Semantic Header Analysis
        header_match_type = self._match_semantic_header(col_name)

        # 2. Check False Positive Safeguard: Numeric surrogate IDs
        is_surrogate_id = self._is_surrogate_id_header(col_name)

        # 3. Value-level Regex Scanners
        if pd.api.types.is_numeric_dtype(series) and header_match_type is None:
            regex_results = None
        else:
            regex_results = self._scan_regex_patterns(
                sample, col_name, is_surrogate_id
            )

        # 4. Synthesize Detection & Safeguards
        # Case A: Strong Regex Match Found
        if regex_results:
            best_type, match_ratio, match_count, matched_indices = regex_results

            # If surrogate ID header detected and pattern was raw numeric, suppress false positive
            if (
                is_surrogate_id
                and best_type in (PIIType.SSN, PIIType.PHONE, PIIType.CREDIT_CARD)
                and header_match_type is None
            ):
                return None

            # Calculate confidence
            confidence = 0.85
            detection_method = "regex"
            if header_match_type == best_type:
                confidence = 0.99
                detection_method = "hybrid"

            recommended_strategy = self._get_recommended_strategy(best_type)

            return PIIDetectionResult(
                column=series.name,
                pii_type=best_type,
                confidence=confidence,
                detection_method=detection_method,
                match_count=match_count,
                match_ratio=match_ratio,
                sample_matches=[],
                recommended_strategy=recommended_strategy,
                details={
                    "header_match": header_match_type.value if header_match_type else None,
                    "matched_indices": matched_indices,
                },
            )

        # Case B: Semantic Header Match (even if text regex doesn't match, e.g. names or addresses)
        if header_match_type is not None:
            # Confirm column content is reasonable for the header type (e.g. not all pure numbers for name)
            if header_match_type == PIIType.NAME and pd.api.types.is_numeric_dtype(series):
                return None  # Numeric column named 'first_name' is suspect/invalid

            recommended_strategy = self._get_recommended_strategy(header_match_type)
            return PIIDetectionResult(
                column=series.name,
                pii_type=header_match_type,
                confidence=0.80,
                detection_method="semantic_header",
                match_count=len(non_null),
                match_ratio=1.0,
                sample_matches=[],
                recommended_strategy=recommended_strategy,
                details={"header_keyword_match": True},
            )

        return None

    def scan_dataframe(
        self,
        df: pd.DataFrame,
    ) -> Dict[Any, PIIDetectionResult]:
        """
        Scans all columns in a DataFrame and returns detected PII specifications.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to evaluate.

        Returns
        -------
        Dict[str, PIIDetectionResult]
            Map of column name to detection result.
        """
        results: Dict[Any, PIIDetectionResult] = {}
        if df is None or df.empty:
            return results

        if not df.columns.is_unique:
            raise ValueError("PII scanning requires unique column labels.")

        for col in df.columns:
            detection = self.detect_column(df[col])
            if detection is not None:
                results[col] = detection

        return results

    def detect(self, df: pd.DataFrame) -> Dict[Any, PIIDetectionResult]:
        """UI Integration Alias: maps detect() to scan_dataframe()."""
        return self.scan_dataframe(df)

    # -----------------------------------------------------------------------
    # Helper Heuristic Detectors
    # -----------------------------------------------------------------------

    def _match_semantic_header(self, col_name: str) -> Optional[PIIType]:
        """Matches column header against known PII semantic keywords."""
        cleaned = self._normalize_header(col_name, separator=" ")
        tokens = set(cleaned.split())

        for pii_type, keywords in SEMANTIC_HEADER_KEYWORDS.items():
            for kw in keywords:
                normalized_kw = re.sub(r"[^a-z0-9]+", " ", kw.lower()).strip()
                keyword_tokens = normalized_kw.split()
                if (
                    (len(keyword_tokens) == 1 and normalized_kw in tokens)
                    or (
                        len(keyword_tokens) > 1
                        and re.search(rf"(?:^| ){re.escape(normalized_kw)}(?: |$)", cleaned)
                    )
                ):
                    return pii_type
        return None

    def _is_surrogate_id_header(self, col_name: str) -> bool:
        """Determines if a column is a surrogate database ID (e.g. customer_id, order_id, transaction_code)."""
        cleaned = self._normalize_header(col_name, separator="_")
        if cleaned in ID_EXCLUSION_KEYWORDS:
            return True
        # Matches patterns like user_id, order_id, product_id, transaction_code, but NOT ssn_id or tax_id
        if cleaned.endswith("_id") or cleaned.endswith("_uid") or cleaned.endswith("_code"):
            base = cleaned.rsplit("_", 1)[0]
            if base in {"ssn", "tax", "social_security", "card"}:
                return False
            return True
        return False

    @staticmethod
    def _normalize_header(col_name: str, separator: str = " ") -> str:
        """Normalize snake, kebab, spaced, and camelCase headers consistently."""
        camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(col_name).strip())
        normalized = re.sub(r"[^A-Za-z0-9]+", separator, camel_split).lower()
        return normalized.strip(separator)

    def _scan_regex_patterns(
        self,
        sample: pd.Series,
        col_name: str,
        is_surrogate_id: bool,
    ) -> Optional[Tuple[PIIType, float, int, List[str]]]:
        """Runs vectorized regex matching against sample records."""
        total = len(sample)
        if total == 0:
            return None

        def has_match(matches: pd.Series, high_precision: bool = False) -> bool:
            count = int(matches.sum())
            if count == 0:
                return False
            ratio = count / total
            return high_precision or ratio >= self.match_threshold or (
                total < self.min_sample_size and count == total
            )

        # 1. Email check
        email_matches = sample.str.contains(REGEX_EMAIL, regex=True)
        ratio = float(email_matches.mean())
        if has_match(email_matches, high_precision=True):
            matched_indices = sample[email_matches].index.tolist()
            return PIIType.EMAIL, ratio, int(email_matches.sum()), matched_indices

        # 2. Formatted SSN check (\d{3}-\d{2}-\d{4})
        ssn_matches = sample.map(self._contains_valid_ssn)
        ratio = float(ssn_matches.mean())
        if has_match(ssn_matches, high_precision=True):
            matched_indices = sample[ssn_matches].index.tolist()
            return PIIType.SSN, ratio, int(ssn_matches.sum()), matched_indices

        # 3. Credit Card check (Regex + Luhn formula)
        cc_candidates = sample.str.contains(REGEX_CREDIT_CARD, regex=True)
        if cc_candidates.any():
            valid_cards = sample.map(
                lambda value: any(
                    luhn_checksum_valid(match.group(0))
                    for match in REGEX_CREDIT_CARD.finditer(value)
                )
            )
            valid_luhn_count = int(valid_cards.sum())
            luhn_ratio = valid_luhn_count / total
            if valid_luhn_count > 0:
                matched_indices = sample[valid_cards].index.tolist()
                return PIIType.CREDIT_CARD, luhn_ratio, valid_luhn_count, matched_indices

        # 4. IP Address check (IPv4 / IPv6)
        ipv4_matches = sample.str.contains(REGEX_IPV4, regex=True)
        ipv6_matches = sample.str.contains(REGEX_IPV6, regex=True)
        validated_ip_matches = sample.map(self._contains_valid_ip)
        ip_matches = ipv4_matches | ipv6_matches | validated_ip_matches
        ratio = float(ip_matches.mean())
        if has_match(ip_matches, high_precision=True):
            matched_indices = sample[ip_matches].index.tolist()
            return PIIType.IP_ADDRESS, ratio, int(ip_matches.sum()), matched_indices

        # 5. Unformatted 9-digit SSN (only if column name semantically matches SSN)
        if self._match_semantic_header(col_name) == PIIType.SSN:
            ssn_raw_matches = sample.str.fullmatch(REGEX_SSN_RAW)
            ratio = float(ssn_raw_matches.mean())
            if has_match(ssn_raw_matches):
                matched_indices = sample[ssn_raw_matches].index.tolist()
                return PIIType.SSN, ratio, int(ssn_raw_matches.sum()), matched_indices

        # 6. Phone number check (Formatted)
        phone_matches = sample.str.contains(REGEX_PHONE, regex=True)
        ratio = float(phone_matches.mean())
        if has_match(phone_matches, high_precision=True):
            # Safeguard: if numbers are simple integers and column name is an ID, do not flag as phone
            if not (is_surrogate_id and sample.str.isnumeric().all()):
                matched_indices = sample[phone_matches].index.tolist()
                return PIIType.PHONE, ratio, int(phone_matches.sum()), matched_indices

        # 7. Custom Dates / Date of Birth (when column name suggests birth/dob)
        if self._match_semantic_header(col_name) == PIIType.DOB:
            date_matches = sample.str.contains(REGEX_DATE, regex=True)
            ratio = float(date_matches.mean())
            if has_match(date_matches):
                matched_indices = sample[date_matches].index.tolist()
                return PIIType.DOB, ratio, int(date_matches.sum()), matched_indices

        return None

    @staticmethod
    def _contains_valid_ssn(value: str) -> bool:
        """Validate US SSN ranges instead of accepting format alone."""
        for match in REGEX_SSN_FORMATTED.finditer(value):
            area, group, serial = (int(part) for part in match.group(0).split("-"))
            if area not in (0, 666) and area < 900 and group != 0 and serial != 0:
                return True
        return False

    @staticmethod
    def _contains_valid_ip(value: str) -> bool:
        try:
            ipaddress.ip_address(str(value).strip())
            return True
        except ValueError:
            return False

    @staticmethod
    def _get_recommended_strategy(pii_type: PIIType) -> str:
        """Returns standard default masking strategy for PII classification."""
        if pii_type in (PIIType.SSN, PIIType.CREDIT_CARD):
            return "redact"
        if pii_type in (PIIType.EMAIL, PIIType.PHONE):
            return "hash"
        if pii_type in (PIIType.NAME, PIIType.ADDRESS, PIIType.DOB):
            return "anonymize"
        return "redact"
