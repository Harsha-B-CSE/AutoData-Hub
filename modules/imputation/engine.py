"""Compatibility façade for the integrated multi-model imputation engine."""

from modules.imputation.categorical_imputer import CategoricalClassifierImputer
from modules.imputation.imputer import (
    BenchmarkResult,
    ImputationStrategy,
    MissingValueImputer,
    impute_missing_values,
)

ImputationEngine = MissingValueImputer

__all__ = [
    "BenchmarkResult",
    "CategoricalClassifierImputer",
    "ImputationEngine",
    "ImputationStrategy",
    "MissingValueImputer",
    "impute_missing_values",
]
