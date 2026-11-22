"""
AutoData-Hub Local PII & Privacy Masker (Module 4).

Executes local data sanitization using Redaction, Hashing (SHA-256 with unique session salt),
and Anonymization / Pseudonymization.

Emits pure TransformResult contracts with exact standalone Pandas code snippets
for AST compilation and audit reproducibility.
Zero in-place mutations (strictly non-mutating deep copies).
"""

from __future__ import annotations

import datetime
from enum import Enum
import hashlib
import re
import secrets
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from core.schema_inference import ExecutionStatus, TransformationAction, TransformResult
from modules.privacy.detector import PIIDetectionResult, PIIDetector, PIIType


class MaskingStrategy(str, Enum):
    """Supported sanitization strategies."""
    REDACT = "redact"
    HASH = "hash"
    ANONYMIZE = "anonymize"


_REDACTION_RULES = [
    (r"([^@\s]{1})[^@\s]*(@[^\s]+)", r"\1***\2"),
    (r"(\d{3}-\d{2}-)(\d{4})", r"XXX-XX-\2"),
    (r"(?<!\d)\d{9}(?!\d)", "XXX-XX-XXXX"),
    (r"(?<!\d)(?:\d[-\s]?){12,18}\d(?!\d)", "[CARD-REDACTED]"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP-REDACTED]"),
    (r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b", "[IP-REDACTED]"),
    (r"\b::(?:[A-Fa-f0-9]{1,4}:){0,5}[A-Fa-f0-9]{1,4}\b", "[IP-REDACTED]"),
    (r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE-REDACTED]"),
]

_FULL_VALUE_PATTERNS = {
    PIIType.EMAIL: r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    PIIType.PHONE: r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    PIIType.SSN: r"(?:\d{3}-\d{2}-\d{4}|\d{9})",
    PIIType.CREDIT_CARD: r"(?:\d[-\s]?){12,18}\d",
    PIIType.IP_ADDRESS: (
        r"(?:(?:\d{1,3}\.){3}\d{1,3}|"
        r"(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}|"
        r"::(?:[A-Fa-f0-9]{1,4}:){0,5}[A-Fa-f0-9]{1,4})"
    ),
}


class PIIMasker:
    """
    Applies privacy-preserving transformations (Redaction, Salted SHA-256 Hashing,
    Anonymization) on identified PII columns.
    """

    def __init__(self, session_salt: Optional[str] = None, salt: Optional[str] = None, **kwargs):
        # Support session_salt, salt, or any extraneous init kwargs seamlessly
        self.session_salt = session_salt or salt or secrets.token_hex(16)

    def mask_series(
        self,
        series: pd.Series,
        pii_type: PIIType,
        strategy: MaskingStrategy,
        col_name: Optional[Any] = None,
    ) -> Tuple[pd.Series, str]:
        """
        Masks a single Series and returns the sanitized copy along with the
        exact, reproducible Pandas transformation code string.
        """
        name = series.name if col_name is None else col_name
        s_copy = series.copy()

        if strategy == MaskingStrategy.HASH:
            return self._apply_hashing(s_copy, name, pii_type)
        elif strategy == MaskingStrategy.ANONYMIZE:
            return self._apply_anonymization(s_copy, pii_type, name)
        else:
            return self._apply_redaction(s_copy, pii_type, name)

    def mask_dataframe(
        self,
        df: pd.DataFrame,
        detections: Dict[Any, PIIDetectionResult],
        strategy_overrides: Optional[Dict[Any, str]] = None,
        start_step_id: int = 1,
    ) -> TransformResult:
        """
        Executes privacy masking across all detected PII columns in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to sanitize.
        detections : Dict[str, PIIDetectionResult]
            PII detection outputs per column.
        strategy_overrides : Optional[Dict[str, str]]
            Optional user overrides mapping column name to 'redact', 'hash', or 'anonymize'.
        start_step_id : int
            Starting index for logged TransformationAction steps.

        Returns
        -------
        TransformResult
            Contract containing the sanitized DataFrame copy, audit logs, and
            compiled code snippets.
        """
        if df is None:
            return TransformResult(
                status=ExecutionStatus.FAILED,
                transformed_df=None,
                applied_actions=[],
                audit_log=["[PRIVACY] Error: Input DataFrame is None."],
                error_message="Cannot mask None DataFrame.",
            )

        if not df.columns.is_unique:
            return TransformResult.failure(
                "Privacy masking requires unique column labels.",
                ["[PRIVACY] Error: Duplicate column labels are not supported."],
            )

        if df.empty or not detections:
            return TransformResult(
                status=ExecutionStatus.SUCCESS,
                transformed_df=df.copy(deep=True),
                applied_actions=[],
                audit_log=["[PRIVACY] No PII columns targeted for masking."],
                error_message=None,
            )

        clean_df = df.copy(deep=True)
        actions: List[TransformationAction] = []
        audit_log: List[str] = []
        step_id = start_step_id

        overrides = strategy_overrides or {}

        for col, detection in detections.items():
            if col not in clean_df.columns:
                continue

            strat_str = str(
                overrides.get(col, overrides.get(str(col), detection.recommended_strategy))
            ).lower()
            try:
                strategy = MaskingStrategy(strat_str)
            except ValueError:
                return TransformResult.failure(
                    f"Unsupported masking strategy '{strat_str}' for column '{col}'.",
                    audit_log,
                )

            # Apply transformation
            masked_series, code_snippet = self.mask_series(
                series=clean_df[col],
                pii_type=detection.pii_type,
                strategy=strategy,
                col_name=col,
            )

            if detection.detection_method == "regex":
                matched_indices = [
                    index
                    for index in detection.details.get("matched_indices", [])
                    if index in clean_df.index
                ]
                original_column = clean_df[col].copy()
                selected_mask = clean_df.index.isin(matched_indices)
                masked_series = original_column.where(~selected_mask, masked_series)
                column_ref = repr(col)
                code_snippet = (
                    f"_privacy_original_column = df[{column_ref}].copy()\n"
                    f"{code_snippet}\n"
                    f"_privacy_selected_indices = {matched_indices!r}\n"
                    f"_privacy_restore_mask = ~df.index.isin(_privacy_selected_indices)\n"
                    f"df.loc[_privacy_restore_mask, {column_ref}] = "
                    f"_privacy_original_column.loc[_privacy_restore_mask]"
                )

            # Assign to clean copy (non-mutating)
            clean_df[col] = masked_series

            original_series = df[col]
            unchanged = original_series.eq(masked_series) | (
                original_series.isna() & masked_series.isna()
            )
            changed_cells = int((~unchanged).sum())
            if changed_cells == 0 and original_series.notna().any():
                return TransformResult.failure(
                    f"Privacy masking made no changes to detected PII column '{col}'.",
                    audit_log,
                )

            action = TransformationAction(
                step_id=step_id,
                module="privacy",
                action_type=f"mask_{strategy.value}",
                target_columns=[str(col)],
                parameters={
                    "pii_type": detection.pii_type.value,
                    "strategy": strategy.value,
                    "changed_cells": changed_cells,
                    "salt_hash": (
                        hashlib.sha256(self.session_salt.encode("utf-8")).hexdigest()[:12]
                        if strategy == MaskingStrategy.HASH
                        else None
                    ),
                },
                code_snippet=code_snippet,
            )
            actions.append(action)

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            audit_log.append(
                f"[{now_iso}] [PRIVACY] Sanitized '{col}' ({detection.pii_type.value}) "
                f"using {strategy.value.upper()}; changed {changed_cells} cells."
            )
            step_id += 1

        return TransformResult(
            status=ExecutionStatus.SUCCESS,
            transformed_df=clean_df,
            applied_actions=actions,
            audit_log=audit_log,
            error_message=None,
        )

    def apply_masking(
        self,
        df: pd.DataFrame,
        alerts: Optional[Any] = None,
        findings: Optional[Any] = None,
        strategy: str = "redact",
        start_step_id: int = 1,
        **kwargs,
    ) -> TransformResult:
        """
        Primary PII transformation routine supporting generic alert/finding objects or list inputs.
        Robustly absorbs any unexpected keyword arguments (e.g. mode='...', pii_alerts=...).
        """
        # Resolve positional or keyword variations for target alerts
        input_items = (
            alerts
            if alerts is not None
            else (findings if findings is not None else kwargs.get("pii_alerts", []))
        )

        # Resolve strategy vs mode parameter naming
        effective_strategy = kwargs.get("mode", strategy)

        if df is None:
            return TransformResult(
                status=ExecutionStatus.FAILED,
                transformed_df=None,
                applied_actions=[],
                audit_log=["[PRIVACY] Error: Input DataFrame is None."],
                error_message="Cannot mask None DataFrame.",
            )

        detections: Dict[Any, PIIDetectionResult] = {}
        strategy_overrides: Dict[Any, str] = {}

        if isinstance(input_items, Mapping):
            items = list(input_items.items())
        else:
            items = [(None, item) for item in input_items]

        for mapping_key, item in items:
            if isinstance(item, PIIDetectionResult):
                col_name = item.column
                detection = item
            else:
                col_name = getattr(item, "column", None)
                if isinstance(item, Mapping):
                    col_name = item.get("column", mapping_key)
                elif col_name is None:
                    col_name = mapping_key if mapping_key is not None else item

                raw_pii_type = getattr(item, "pii_type", PIIType.GENERIC_PII)
                if isinstance(item, Mapping):
                    raw_pii_type = item.get("pii_type", PIIType.GENERIC_PII)
                if isinstance(raw_pii_type, str):
                    try:
                        pii_type_enum = PIIType(raw_pii_type)
                    except ValueError:
                        pii_type_enum = PIIType.GENERIC_PII
                elif isinstance(raw_pii_type, PIIType):
                    pii_type_enum = raw_pii_type
                else:
                    pii_type_enum = PIIType.GENERIC_PII

                rec_strat = getattr(item, "recommended_strategy", effective_strategy)
                confidence = getattr(item, "confidence", 1.0)
                detection_method = getattr(item, "detection_method", "manual")
                if isinstance(item, Mapping):
                    rec_strat = item.get("recommended_strategy", effective_strategy)
                    confidence = item.get("confidence", 1.0)
                    detection_method = item.get("detection_method", "manual")

                detection = PIIDetectionResult(
                    column=col_name,
                    pii_type=pii_type_enum,
                    confidence=float(confidence),
                    detection_method=str(detection_method),
                    recommended_strategy=str(rec_strat),
                )

            if col_name not in df.columns:
                continue

            detections[col_name] = detection
            strategy_overrides[col_name] = effective_strategy

        return self.mask_dataframe(
            df=df,
            detections=detections,
            strategy_overrides=strategy_overrides,
            start_step_id=start_step_id,
        )

    def mask(self, df: pd.DataFrame, *args, **kwargs) -> TransformResult:
        """UI Integration Alias: maps masker.mask(...) directly to apply_masking(...)."""
        return self.apply_masking(df, *args, **kwargs)

    # -----------------------------------------------------------------------
    # Specific Masking Strategies
    # -----------------------------------------------------------------------

    def _apply_redaction(
        self,
        series: pd.Series,
        pii_type: PIIType,
        col_name: Any,
    ) -> Tuple[pd.Series, str]:
        """Apply composite, fail-closed redaction to every detected value."""
        source = series.copy()
        masked = source.astype(str)
        for pattern, replacement in _REDACTION_RULES:
            masked = masked.str.replace(pattern, replacement, regex=True)

        full_pattern = _FULL_VALUE_PATTERNS.get(pii_type)
        if full_pattern is None:
            masked = masked.mask(source.notna(), "[REDACTED]")
        else:
            recognized = source.astype(str).str.fullmatch(full_pattern)
            masked = masked.mask(source.notna() & ~recognized, "[REDACTED]")

        unchanged = source.notna() & masked.eq(source.astype(str))
        masked = masked.mask(unchanged, "[REDACTED]").mask(source.isna(), np.nan)

        column_ref = repr(col_name)
        code = (
            f"_privacy_source = df[{column_ref}]\n"
            f"_privacy_masked = _privacy_source.astype(str)\n"
            f"for _privacy_pattern, _privacy_replacement in {repr(_REDACTION_RULES)}:\n"
            f"    _privacy_masked = _privacy_masked.str.replace("
            f"_privacy_pattern, _privacy_replacement, regex=True)\n"
            f"_privacy_full_pattern = {full_pattern!r}\n"
            f"if _privacy_full_pattern is None:\n"
            f"    _privacy_masked = _privacy_masked.mask(_privacy_source.notna(), '[REDACTED]')\n"
            f"else:\n"
            f"    _privacy_recognized = _privacy_source.astype(str).str.fullmatch("
            f"_privacy_full_pattern)\n"
            f"    _privacy_masked = _privacy_masked.mask("
            f"_privacy_source.notna() & ~_privacy_recognized, '[REDACTED]')\n"
            f"_privacy_unchanged = _privacy_source.notna() & "
            f"_privacy_masked.eq(_privacy_source.astype(str))\n"
            f"df[{column_ref}] = _privacy_masked.mask("
            f"_privacy_unchanged, '[REDACTED]').mask(_privacy_source.isna(), np.nan)"
        )
        return masked, code

    def _apply_hashing(
        self,
        series: pd.Series,
        col_name: Any,
        pii_type: PIIType,
    ) -> Tuple[pd.Series, str]:
        """Hashes values using SHA-256 with unique session salt."""
        salt = self.session_salt

        def hash_val(val: Any) -> Any:
            if pd.isna(val):
                return np.nan
            canonical = (
                pd.to_datetime(val).isoformat()
                if pii_type == PIIType.DOB
                else str(val)
            )
            s = f"{canonical}{salt}"
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        masked = series.map(hash_val)
        column_ref = repr(col_name)
        canonical_expression = (
            "pd.to_datetime(x).isoformat()"
            if pii_type == PIIType.DOB
            else "str(x)"
        )
        code = (
            f"import hashlib\n"
            f"salt = {salt!r}\n"
            f"df[{column_ref}] = df[{column_ref}].map("
            f"lambda x: hashlib.sha256(f'{{{canonical_expression}}}{{salt}}'.encode('utf-8')).hexdigest() if pd.notna(x) else np.nan"
            f")"
        )
        return masked, code

    def _apply_anonymization(
        self,
        series: pd.Series,
        pii_type: PIIType,
        col_name: Any,
    ) -> Tuple[pd.Series, str]:
        """Replaces values with deterministic synthetic pseudonyms."""
        prefix = pii_type.value.upper()

        if pii_type == PIIType.NAME:
            prefix = "Anon_User"
        elif pii_type == PIIType.EMAIL:
            prefix = "user"
        elif pii_type == PIIType.ADDRESS:
            prefix = "Address"
        elif pii_type == PIIType.DOB:
            prefix = "DOB"

        suffix = "@anonymized.local" if pii_type == PIIType.EMAIL else ""
        codes = pd.factorize(series, sort=False)[0]
        masked = pd.Series(
            [
                f"{prefix}_{int(code) + 1:04d}{suffix}" if code >= 0 else np.nan
                for code in codes
            ],
            index=series.index,
            dtype="object",
        )

        column_ref = repr(col_name)
        code = (
            f"_privacy_source = df[{column_ref}]\n"
            f"_privacy_codes = pd.factorize(_privacy_source, sort=False)[0]\n"
            f"_privacy_prefix = {prefix!r}\n"
            f"_privacy_suffix = {suffix!r}\n"
            f"df[{column_ref}] = pd.Series(["
            f"_privacy_prefix + '_' + str(int(_privacy_code) + 1).zfill(4) + _privacy_suffix "
            f"if _privacy_code >= 0 else np.nan for _privacy_code in _privacy_codes], "
            f"index=df.index, dtype='object')"
        )
        return masked, code


# ---------------------------------------------------------------------------
# Module Orchestrator API
# ---------------------------------------------------------------------------

def apply_privacy_firewall(
    df: pd.DataFrame,
    strategy_overrides: Optional[Dict[str, str]] = None,
    session_salt: Optional[str] = None,
    start_step_id: int = 1,
) -> TransformResult:
    """
    Executes the complete local Privacy Firewall (Module 4):
    1. Scans DataFrame for PII (Regex + Semantic Header Classifier).
    2. Masks identified features via Redaction, Salted Hashing, or Anonymization.
    3. Emits pure TransformResult contract with exact reproducible Pandas code strings.
    """
    detector = PIIDetector()
    detections = detector.scan_dataframe(df)

    masker = PIIMasker(session_salt=session_salt)
    return masker.mask_dataframe(
        df=df,
        detections=detections,
        strategy_overrides=strategy_overrides,
        start_step_id=start_step_id,
    )
