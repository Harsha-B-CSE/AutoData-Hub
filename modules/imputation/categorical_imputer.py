"""Classifier-based reconstruction for categorical target columns."""

from __future__ import annotations

from typing import Any, Hashable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


class CategoricalClassifierImputer:
    """Impute one categorical target with RF classification or a mode fallback."""

    def __init__(
        self,
        target_column: Hashable,
        *,
        min_train_rows: int = 10,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.target_column = target_column
        self.min_train_rows = min_train_rows
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.strategy_: Optional[str] = None
        self.fallback_reason_: Optional[str] = None
        self.predictor_columns_: list[Hashable] = []
        self.numeric_columns_: list[Hashable] = []
        self.categorical_columns_: list[Hashable] = []
        self.numeric_fill_values_: dict[Hashable, float] = {}
        self.encoded_columns_: list[str] = []
        self.n_train_: int = 0
        self.fill_value_: Any = "Missing"
        self.model_: Optional[RandomForestClassifier] = None
        self.fallback_imputer_: Optional[SimpleImputer] = None

    def _base_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [
            column for column in self.predictor_columns_ if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing predictor columns: {missing_columns}")

        predictors = df[self.predictor_columns_].copy()
        for column in self.numeric_columns_:
            numeric = pd.to_numeric(predictors[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            predictors[column] = numeric.fillna(self.numeric_fill_values_[column])
        for column in self.categorical_columns_:
            predictors[column] = (
                predictors[column].astype("object").where(
                    predictors[column].notna(), "__MISSING__"
                ).astype(str)
            )
        return predictors

    def _encode_predictors(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        encoded = pd.get_dummies(
            self._base_predictors(df),
            columns=self.categorical_columns_,
            dtype=float,
        )
        encoded.columns = [str(column) for column in encoded.columns]
        if fit:
            self.encoded_columns_ = list(encoded.columns)
        return encoded.reindex(columns=self.encoded_columns_, fill_value=0.0)

    def fit(self, df: pd.DataFrame) -> CategoricalClassifierImputer:
        if self.target_column not in df.columns:
            raise KeyError(f"Target column {self.target_column!r} was not found.")
        if not df.columns.is_unique:
            raise ValueError("Categorical imputation requires unique column labels.")

        target = df[self.target_column]
        observed = target.notna()
        self.n_train_ = int(observed.sum())
        self.predictor_columns_ = [
            column
            for column in df.columns
            if column != self.target_column
            and str(column) != f"{self.target_column}_is_missing"
        ]
        self.numeric_columns_ = [
            column
            for column in self.predictor_columns_
            if pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_bool_dtype(df[column])
        ]
        self.categorical_columns_ = [
            column
            for column in self.predictor_columns_
            if column not in self.numeric_columns_
        ]
        self.numeric_fill_values_ = {}
        for column in self.numeric_columns_:
            numeric = pd.to_numeric(df[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            median = numeric.loc[observed].median()
            self.numeric_fill_values_[column] = (
                0.0 if pd.isna(median) else float(median)
            )

        unique_classes = int(target.loc[observed].nunique(dropna=True))
        fallback_reason = None
        if self.n_train_ < self.min_train_rows:
            fallback_reason = f"n_train<{self.min_train_rows}"
        elif unique_classes < 2:
            fallback_reason = "single_class"
        elif not self.predictor_columns_:
            fallback_reason = "no_predictors"

        if fallback_reason is not None:
            self.strategy_ = "most_frequent"
            self.fallback_reason_ = fallback_reason
            if self.n_train_ == 0:
                self.fill_value_ = "Missing"
                self.fallback_reason_ = "all_null_target"
            else:
                self.fallback_imputer_ = SimpleImputer(strategy="most_frequent")
                observed_values = target.loc[observed].astype("object").to_numpy().reshape(-1, 1)
                self.fallback_imputer_.fit(observed_values)
                self.fill_value_ = self.fallback_imputer_.statistics_[0]
            return self

        self.strategy_ = "random_forest"
        encoded = self._encode_predictors(df, fit=True)
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced_subsample",
        )
        self.model_.fit(encoded.loc[observed], target.loc[observed])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.strategy_ is None:
            raise RuntimeError("CategoricalClassifierImputer must be fitted first.")
        if self.target_column not in df.columns:
            raise KeyError(f"Target column {self.target_column!r} was not found.")

        result = df.copy(deep=True)
        missing = result[self.target_column].isna()
        if not missing.any():
            return result

        if self.strategy_ == "most_frequent":
            result.loc[missing, self.target_column] = self.fill_value_
            return result

        encoded = self._encode_predictors(result, fit=False)
        result.loc[missing, self.target_column] = self.model_.predict(
            encoded.loc[missing]
        )
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def replay_code(self) -> str:
        """Return standalone replay code matching the fitted strategy."""
        target = repr(self.target_column)
        if self.strategy_ == "most_frequent":
            return f"df[{target}] = df[{target}].fillna({self.fill_value_!r})"

        lines = [
            "from sklearn.ensemble import RandomForestClassifier",
            f"_cat_target = {target}",
            f"_cat_predictors = {self.predictor_columns_!r}",
            f"_cat_numeric = {self.numeric_columns_!r}",
            f"_cat_categorical = {self.categorical_columns_!r}",
            f"_cat_fill_values = {self.numeric_fill_values_!r}",
            "_cat_X = df[_cat_predictors].copy()",
            "for _cat_column in _cat_numeric:",
            "    _cat_X[_cat_column] = pd.to_numeric(_cat_X[_cat_column], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(_cat_fill_values[_cat_column])",
            "for _cat_column in _cat_categorical:",
            "    _cat_X[_cat_column] = _cat_X[_cat_column].astype('object').where(_cat_X[_cat_column].notna(), '__MISSING__').astype(str)",
            "_cat_X = pd.get_dummies(_cat_X, columns=_cat_categorical, dtype=float)",
            "_cat_X.columns = [str(_cat_column) for _cat_column in _cat_X.columns]",
            f"_cat_X = _cat_X.reindex(columns={self.encoded_columns_!r}, fill_value=0.0)",
            "_cat_train = df[_cat_target].notna()",
            "_cat_missing = ~_cat_train",
            f"_cat_model = RandomForestClassifier(n_estimators={self.n_estimators}, random_state={self.random_state}, n_jobs={self.n_jobs}, class_weight='balanced_subsample')",
            "_cat_model.fit(_cat_X.loc[_cat_train], df.loc[_cat_train, _cat_target])",
            "df.loc[_cat_missing, _cat_target] = _cat_model.predict(_cat_X.loc[_cat_missing])",
        ]
        return "\n".join(lines)
