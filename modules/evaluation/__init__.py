"""Empirical validation utilities for AutoData-Hub."""


def run_imputation_accuracy_experiment(*args, **kwargs):
    from modules.evaluation.benchmarker import run_imputation_accuracy_experiment as run

    return run(*args, **kwargs)


def benchmark_downstream_ml(*args, **kwargs):
    from modules.evaluation.benchmarker import benchmark_downstream_ml as run

    return run(*args, **kwargs)

__all__ = [
    "benchmark_downstream_ml",
    "run_imputation_accuracy_experiment",
]
