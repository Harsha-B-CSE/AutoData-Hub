"""Empirical imputation and downstream-model validation experiments."""

from __future__ import annotations

from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from modules.imputation.categorical_imputer import CategoricalClassifierImputer


IMPUTATION_METHODS = (
    "SimpleImputer",
    "KNNImputer",
    "MICE",
    "CategoricalClassifierImputer",
)


def _inject_mcar(
    frame: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[object, np.ndarray]]:
    corrupted = frame.copy(deep=True)
    masks: dict[object, np.ndarray] = {}
    for column in frame.columns:
        count = max(1, int(round(len(frame) * rate)))
        positions = rng.choice(len(frame), size=min(count, len(frame)), replace=False)
        mask = np.zeros(len(frame), dtype=bool)
        mask[positions] = True
        corrupted.iloc[positions, corrupted.columns.get_loc(column)] = np.nan
        masks[column] = mask
    return corrupted, masks


def _impute_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    method: str,
    numeric_columns: list,
    categorical_columns: list,
    random_state: int,
) -> pd.DataFrame:
    output = validation.copy(deep=True)
    prepared_train = train.copy(deep=True)

    if numeric_columns:
        if method in {"SimpleImputer", "CategoricalClassifierImputer"}:
            numeric_imputer = SimpleImputer(strategy="median")
        elif method == "KNNImputer":
            numeric_imputer = KNNImputer(n_neighbors=5)
        elif method == "MICE":
            numeric_imputer = IterativeImputer(
                max_iter=5,
                random_state=random_state,
                sample_posterior=False,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            numeric_imputer.fit(prepared_train[numeric_columns])
            prepared_train[numeric_columns] = numeric_imputer.transform(
                prepared_train[numeric_columns]
            )
            output[numeric_columns] = numeric_imputer.transform(
                output[numeric_columns]
            )

    if categorical_columns:
        if method == "CategoricalClassifierImputer":
            for column in categorical_columns:
                classifier = CategoricalClassifierImputer(
                    target_column=column,
                    random_state=random_state,
                    n_jobs=1,
                )
                classifier.fit(prepared_train)
                output = classifier.transform(output)
        else:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            categorical_imputer.fit(prepared_train[categorical_columns])
            output[categorical_columns] = categorical_imputer.transform(
                output[categorical_columns]
            )

    return output


def run_imputation_accuracy_experiment(
    complete_data: pd.DataFrame,
    missing_rates: Sequence[float] = (0.10, 0.20),
    *,
    n_splits: int = 5,
    random_state: int = 42,
    methods: Sequence[str] = IMPUTATION_METHODS,
) -> pd.DataFrame:
    """Evaluate numerical and categorical reconstruction under MCAR masking."""
    if complete_data is None or complete_data.empty:
        raise ValueError("complete_data must contain rows and columns.")
    if complete_data.isna().any().any():
        raise ValueError("complete_data must be fully observed ground truth.")
    if len(complete_data) < n_splits:
        raise ValueError("complete_data must contain at least n_splits rows.")

    numeric_columns = [
        column
        for column in complete_data.columns
        if pd.api.types.is_numeric_dtype(complete_data[column])
        and not pd.api.types.is_bool_dtype(complete_data[column])
    ]
    categorical_columns = [
        column for column in complete_data.columns if column not in numeric_columns
    ]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    for rate in missing_rates:
        if not 0.0 < rate < 1.0:
            raise ValueError("missing rates must be between 0 and 1.")
        accumulators = {
            method: {
                "numeric_true": [],
                "numeric_pred": [],
                "categorical_true": [],
                "categorical_pred": [],
            }
            for method in methods
        }

        for fold_index, (train_indices, validation_indices) in enumerate(
            folds.split(complete_data)
        ):
            train_truth = complete_data.iloc[train_indices].reset_index(drop=True)
            validation_truth = complete_data.iloc[validation_indices].reset_index(drop=True)
            train_corrupted, _ = _inject_mcar(
                train_truth,
                rate,
                np.random.default_rng(random_state + fold_index * 101),
            )
            validation_corrupted, validation_masks = _inject_mcar(
                validation_truth,
                rate,
                np.random.default_rng(random_state + fold_index * 101 + 1),
            )

            for method in methods:
                reconstructed = _impute_fold(
                    train_corrupted,
                    validation_corrupted,
                    method,
                    numeric_columns,
                    categorical_columns,
                    random_state + fold_index,
                )
                bucket = accumulators[method]
                for column in numeric_columns:
                    mask = validation_masks[column]
                    bucket["numeric_true"].extend(
                        validation_truth.loc[mask, column].astype(float).tolist()
                    )
                    bucket["numeric_pred"].extend(
                        reconstructed.loc[mask, column].astype(float).tolist()
                    )
                for column in categorical_columns:
                    mask = validation_masks[column]
                    bucket["categorical_true"].extend(
                        [
                            f"{column}::{value}"
                            for value in validation_truth.loc[mask, column].astype(str)
                        ]
                    )
                    bucket["categorical_pred"].extend(
                        [
                            f"{column}::{value}"
                            for value in reconstructed.loc[mask, column].astype(str)
                        ]
                    )

        for method, bucket in accumulators.items():
            if bucket["numeric_true"]:
                numeric_true = np.asarray(bucket["numeric_true"], dtype=float)
                numeric_pred = np.asarray(bucket["numeric_pred"], dtype=float)
                rmse = float(np.sqrt(mean_squared_error(numeric_true, numeric_pred)))
                mae = float(mean_absolute_error(numeric_true, numeric_pred))
                cv_rmse = rmse / max(float(np.mean(np.abs(numeric_true))), 1e-12)
            else:
                rmse = mae = cv_rmse = np.nan

            macro_f1 = (
                float(
                    f1_score(
                        bucket["categorical_true"],
                        bucket["categorical_pred"],
                        average="macro",
                        zero_division=0,
                    )
                )
                if bucket["categorical_true"]
                else np.nan
            )
            records.append({
                "Missingness": f"{int(rate * 100)}%",
                "Method": method,
                "CV-RMSE": cv_rmse,
                "RMSE": rmse,
                "MAE": mae,
                "Categorical Macro-F1": macro_f1,
                "Numeric Masked Cells": len(bucket["numeric_true"]),
                "Categorical Masked Cells": len(bucket["categorical_true"]),
                "Folds": n_splits,
            })

    return pd.DataFrame(records)


def _classification_pipeline(frame: pd.DataFrame, target_column: str, seed: int):
    feature_frame = frame.drop(columns=[target_column])
    numeric_columns = [
        column
        for column in feature_frame.columns
        if pd.api.types.is_numeric_dtype(feature_frame[column])
        and not pd.api.types.is_bool_dtype(feature_frame[column])
    ]
    categorical_columns = [
        column for column in feature_frame.columns if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]),
            categorical_columns,
        ))
    processor = ColumnTransformer(transformers, remainder="drop")
    model = HistGradientBoostingClassifier(
        random_state=seed,
        max_iter=100,
        max_leaf_nodes=15,
        early_stopping=False,
    )
    return Pipeline([("preprocess", processor), ("classifier", model)])


def benchmark_downstream_ml(
    v0_raw: pd.DataFrame,
    v3_anomaly: pd.DataFrame,
    target_column: str,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Compare five-fold ROC-AUC/F1 using the same downstream estimator."""
    results = {}
    for label, frame in (("v0_raw", v0_raw), ("v3_anomaly", v3_anomaly)):
        if target_column not in frame.columns:
            raise KeyError(f"{target_column!r} is missing from {label}.")
        target = frame[target_column]
        minimum_class = int(target.value_counts().min())
        if minimum_class < n_splits:
            raise ValueError("Each target class must contain at least n_splits rows.")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        scores = cross_validate(
            _classification_pipeline(frame, target_column, random_state),
            frame.drop(columns=[target_column]),
            target,
            cv=splitter,
            scoring={"roc_auc": "roc_auc", "f1": "f1"},
            n_jobs=1,
        )
        results[label] = {
            "roc_auc": float(np.mean(scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "f1": float(np.mean(scores["test_f1"])),
            "f1_std": float(np.std(scores["test_f1"])),
            "n_samples": len(frame),
            "folds": n_splits,
        }

    results["delta"] = {
        "roc_auc": results["v3_anomaly"]["roc_auc"] - results["v0_raw"]["roc_auc"],
        "f1": results["v3_anomaly"]["f1"] - results["v0_raw"]["f1"],
    }
    return results


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _synthetic_data(random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, target = make_classification(
        n_samples=500,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=random_state,
    )
    complete = pd.DataFrame(features + 5.0, columns=[f"num_{i}" for i in range(6)])
    complete["segment"] = np.where(complete["num_0"] > 5.0, "high", "low")
    complete["region"] = np.where(complete["num_1"] > 5.0, "north", "south")

    raw = complete.copy(deep=True)
    raw["target"] = target
    rng = np.random.default_rng(random_state)
    for column in [f"num_{i}" for i in range(6)]:
        mask = rng.random(len(raw)) < 0.20
        raw.loc[mask, column] = np.nan

    cleaned = raw.copy(deep=True)
    numeric = [f"num_{i}" for i in range(6)]
    cleaned[numeric] = SimpleImputer(strategy="median").fit_transform(cleaned[numeric])
    for column in numeric:
        cleaned[column] = cleaned[column].clip(
            cleaned[column].quantile(0.01),
            cleaned[column].quantile(0.99),
        )
    return complete, raw, cleaned


def main() -> None:
    complete, raw, cleaned = _synthetic_data()
    imputation_report = run_imputation_accuracy_experiment(complete)
    downstream_report = benchmark_downstream_ml(raw, cleaned, "target")

    print("# Imputation Accuracy Proof")
    print(_markdown_table(imputation_report))
    downstream_frame = pd.DataFrame([
        {"Version": version, **metrics}
        for version, metrics in downstream_report.items()
        if version != "delta"
    ])
    print("\n# Downstream ML Performance")
    print(_markdown_table(downstream_frame))
    print(
        "\nDelta ROC-AUC: "
        f"{downstream_report['delta']['roc_auc']:+.4f}; "
        f"Delta F1: {downstream_report['delta']['f1']:+.4f}"
    )


if __name__ == "__main__":
    main()
