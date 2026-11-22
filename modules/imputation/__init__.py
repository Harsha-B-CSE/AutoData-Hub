"""
AutoData-Hub Missing Value Imputation Module (Module 2).
Exposes MissingValueImputer, ImputationStrategy, and impute_missing_values.
"""

from modules.imputation.imputer import (
    BenchmarkResult,
    ImputationStrategy,
    MissingValueImputer,
    impute_missing_values,
)
from modules.imputation.categorical_imputer import CategoricalClassifierImputer

ImputationEngine = MissingValueImputer

__all__ = [
    "ImputationStrategy",
    "BenchmarkResult",
    "MissingValueImputer",
    "impute_missing_values",
    "CategoricalClassifierImputer",
    "ImputationEngine",
]
