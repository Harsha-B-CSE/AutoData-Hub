"""
AutoData-Hub Multi-Model Missing Value Imputation Engine (Module 2).

Provides missing value reconstruction across MCAR, MAR, and MNAR regimes:
- Imputer Suite: KNNImputer, IterativeImputer (MICE via BayesianRidge or RandomForestRegressor),
  and SimpleImputer (Mean/Median/Mode).
- Mixed-Type Handling: Automatic numerical/categorical splitting, OrdinalEncoder encoding,
  and post-imputation category decoding.
- Dynamic Cross-Validation Benchmark: Injects artificial missingness to evaluate CV RMSE
  across algorithms and automatically recommends/applies the best model per feature.
- MNAR Safeguards: Flags features with >95% missingness for dropping/exclusion and generates
  explicit binary indicators (col_is_missing) for suspected MNAR columns.

Emits pure TransformResult contracts with exact reproducible Scikit-Learn/Pandas code snippets.
Strictly decoupled from UI frameworks (zero Streamlit dependencies) and zero in-place mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

from core.schema_inference import ExecutionStatus, TransformationAction, TransformResult
from modules.imputation.categorical_imputer import CategoricalClassifierImputer


class ImputationStrategy(str, Enum):
    """Supported imputation algorithm strategies."""
    AUTO_CV = "auto_cv"            # Dynamic Cross-Validation RMSE selection
    KNN = "knn"                    # K-Nearest Neighbors
    MICE_RIDGE = "mice_ridge"      # Iterative MICE with Bayesian Ridge
    MICE_RF = "mice_rf"            # Iterative MICE with Random Forest
    MEAN = "mean"                  # Univariate Mean (numeric)
    MEDIAN = "median"              # Univariate Median (numeric)
    MOST_FREQUENT = "most_frequent"  # Mode (categorical/numeric)


@dataclass(frozen=True)
class BenchmarkResult:
    """Evaluation metrics for an imputer on an artificially masked feature."""
    column: str
    imputer: str
    rmse: float
    mae: float
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            "column": self.column,
            "imputer": self.imputer,
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "is_best": self.is_best,
        }


class MissingValueImputer:
    """
    Automated multi-model missing value reconstruction engine.
    """

    def __init__(
        self,
        default_strategy: ImputationStrategy = ImputationStrategy.AUTO_CV,
        n_neighbors: int = 5,
        mice_max_iter: int = 10,
        mnar_threshold: float = 0.15,
        high_missing_threshold: float = 0.95,
        drop_high_missing: bool = False,
        random_state: int = 42,
    ):
        self.default_strategy = default_strategy
        self.n_neighbors = n_neighbors
        self.mice_max_iter = mice_max_iter
        self.mnar_threshold = mnar_threshold
        self.high_missing_threshold = high_missing_threshold
        self.drop_high_missing = drop_high_missing
        self.random_state = random_state

    # -----------------------------------------------------------------------
    # Cross-Validation Benchmark & Optimal Selection
    # -----------------------------------------------------------------------

    def benchmark_column(
        self,
        df: pd.DataFrame,
        target_col: str,
        candidate_strategies: Optional[List[str]] = None,
        test_ratio: float = 0.20,
    ) -> Dict[str, float]:
        """
        Artificially injects missingness into observed rows of target_col
        to calculate Cross-Validation RMSE across algorithms.

        Returns a dictionary mapping imputer name to RMSE score.
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

        series = df[target_col]
        non_null_mask = series.notna()
        non_null_indices = df[non_null_mask].index.to_numpy()

        # If too few observed values to benchmark reliably, return fallback
        if len(non_null_indices) < 8 or not pd.api.types.is_numeric_dtype(series):
            return {"median": 0.0, "mean": 0.0}

        strategies = candidate_strategies or ["mean", "median", "knn", "mice_ridge"]

        # Sample indices to artificially mask
        rng = np.random.RandomState(self.random_state)
        n_mask = max(2, int(len(non_null_indices) * test_ratio))
        mask_indices = rng.choice(non_null_indices, size=n_mask, replace=False)
        y_true = series.loc[mask_indices].to_numpy(dtype=float)

        # Prepare encoded numerical matrix for candidate models
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        if target_col not in num_cols:
            num_cols.append(target_col)

        sub_df = df[num_cols].copy()
        sub_df.loc[mask_indices, target_col] = np.nan

        rmse_scores: Dict[str, float] = {}

        for strat in strategies:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if strat == "mean":
                        imp = SimpleImputer(strategy="mean")
                        pred = imp.fit_transform(sub_df[[target_col]])
                        y_pred = pred[sub_df.index.get_indexer(mask_indices), 0]

                    elif strat == "median":
                        imp = SimpleImputer(strategy="median")
                        pred = imp.fit_transform(sub_df[[target_col]])
                        y_pred = pred[sub_df.index.get_indexer(mask_indices), 0]

                    elif strat == "knn":
                        k = min(self.n_neighbors, max(1, len(sub_df) - 1))
                        imp = KNNImputer(n_neighbors=k)
                        pred_matrix = imp.fit_transform(sub_df)
                        t_idx = num_cols.index(target_col)
                        y_pred = pred_matrix[sub_df.index.get_indexer(mask_indices), t_idx]

                    elif strat == "mice_ridge":
                        imp = IterativeImputer(
                            estimator=BayesianRidge(),
                            max_iter=min(self.mice_max_iter, 5),
                            random_state=self.random_state,
                        )
                        pred_matrix = imp.fit_transform(sub_df)
                        t_idx = num_cols.index(target_col)
                        y_pred = pred_matrix[sub_df.index.get_indexer(mask_indices), t_idx]

                    elif strat == "mice_rf":
                        rf = RandomForestRegressor(n_estimators=10, max_depth=6, random_state=self.random_state, n_jobs=1)
                        imp = IterativeImputer(
                            estimator=rf,
                            max_iter=3,
                            random_state=self.random_state,
                        )
                        pred_matrix = imp.fit_transform(sub_df)
                        t_idx = num_cols.index(target_col)
                        y_pred = pred_matrix[sub_df.index.get_indexer(mask_indices), t_idx]

                    else:
                        continue

                    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    rmse_scores[strat] = round(rmse, 4)

            except Exception:
                # If a complex model fails on small sample, skip it
                pass

        return rmse_scores

    def benchmark_dataset(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs Cross-Validation benchmarks across all features exhibiting missing values.
        """
        cols_to_test = target_cols or [c for c in df.columns if df[c].isna().sum() > 0]
        results: Dict[str, Dict[str, float]] = {}
        recommendations: Dict[str, str] = {}

        for col in cols_to_test:
            scores = self.benchmark_column(df, col)
            results[str(col)] = scores
            if scores:
                best_imputer = min(scores.items(), key=lambda item: item[1])[0]
                recommendations[str(col)] = best_imputer
            else:
                recommendations[str(col)] = "median" if pd.api.types.is_numeric_dtype(df[col]) else "most_frequent"

        return {
            "scores": results,
            "recommendations": recommendations,
        }

    # -----------------------------------------------------------------------
    # Main Imputation Transformation Pipeline
    # -----------------------------------------------------------------------

    def impute(
        self,
        df: pd.DataFrame,
        strategy: Optional[Union[str, ImputationStrategy]] = None,
        column_strategies: Optional[Dict[str, str]] = None,
        add_mnar_indicators: bool = True,
        dataset_version: str = "v1_privacy",
        start_step_id: int = 1,
    ) -> TransformResult:
        """
        Executes multi-model missing value reconstruction with zero in-place mutations.

        Parameters
        ----------
        df : pd.DataFrame
            Sanitized dataset from Module 4 (Privacy Firewall).
        strategy : Optional[Union[str, ImputationStrategy]]
            Global strategy override ('auto_cv', 'knn', 'mice_ridge', 'mice_rf', 'median', 'mean').
        column_strategies : Optional[Dict[str, str]]
            Per-column imputer assignments.
        add_mnar_indicators : bool
            Whether to add binary indicators for suspected MNAR features.
        dataset_version : str
            Active version identifier.
        start_step_id : int
            Step ID sequence counter.

        Returns
        -------
        TransformResult
            Contract wrapping the newly assigned DataFrame (v2_imputed),
            exact Scikit-Learn code strings, and audit trail.
        """
        if df is None:
            return TransformResult.failure("Cannot impute None DataFrame.")

        pre_duplicate_count = int(df.duplicated().sum())

        def duplicate_metadata(output_df: pd.DataFrame) -> Dict[str, int]:
            post_duplicate_count = int(output_df.duplicated().sum())
            return {
                "pre_imputation_duplicate_count": pre_duplicate_count,
                "post_imputation_duplicate_count": post_duplicate_count,
                "new_duplicates": max(
                    0, post_duplicate_count - pre_duplicate_count
                ),
            }

        clean_df = df.copy(deep=True)
        actions: List[TransformationAction] = []
        audit_log: List[str] = []
        benchmark_metadata: Dict[str, Any] = {}
        step_id = start_step_id

        if clean_df.empty or len(clean_df.columns) == 0:
            return TransformResult.success(
                transformed_df=clean_df,
                applied_actions=[],
                audit_log=["[IMPUTATION] Dataset is empty. No imputation applied."],
                metadata=duplicate_metadata(clean_df),
            )

        row_count = len(clean_df)
        col_strategies = column_strategies or {}
        chosen_strategy = ImputationStrategy(strategy) if strategy else self.default_strategy
        if chosen_strategy == ImputationStrategy.AUTO_CV and row_count < 2:
            audit_log.append(
                "[IMPUTATION] Single-row dataset: skipped cross-validation and used safe fallbacks."
            )

        # -------------------------------------------------------------------
        # Step 1: High-Missingness (>95%) Safeguard
        # -------------------------------------------------------------------
        high_missing_cols: List[str] = []
        for col in list(clean_df.columns):
            null_ratio = float(clean_df[col].isna().sum() / row_count)
            if null_ratio >= self.high_missing_threshold:
                high_missing_cols.append(col)
                null_pct = round(null_ratio * 100.0, 1)
                audit_log.append(
                    f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] [IMPUTATION] "
                    f"WARNING: Feature '{col}' exhibits {null_pct}% missingness (>{int(self.high_missing_threshold*100)}%)."
                )

        if high_missing_cols and self.drop_high_missing:
            clean_df = clean_df.drop(columns=high_missing_cols)
            drop_code = f"df = df.drop(columns={repr(high_missing_cols)})"
            actions.append(
                TransformationAction(
                    step_id=step_id,
                    module="imputation",
                    action_type="drop_high_missing",
                    target_columns=high_missing_cols,
                    parameters={"threshold": self.high_missing_threshold},
                    code_snippet=drop_code,
                )
            )
            audit_log.append(
                f"[IMPUTATION] Dropped {len(high_missing_cols)} column(s) exceeding {int(self.high_missing_threshold*100)}% missingness: {high_missing_cols}"
            )
            step_id += 1

        # -------------------------------------------------------------------
        # Step 2: MNAR Binary Indicator Flagging
        # -------------------------------------------------------------------
        generated_indicators = set()
        if add_mnar_indicators:
            for col in list(clean_df.columns):
                # Don't flag indicator columns themselves
                if str(col).endswith("_is_missing"):
                    continue
                null_ratio = float(clean_df[col].isna().sum() / row_count)
                # Flag if missingness exceeds MNAR suspicion threshold and not 0% or 100%
                if 0.0 < null_ratio and null_ratio >= self.mnar_threshold:
                    indicator_name = f"{col}_is_missing"
                    if indicator_name in clean_df.columns:
                        indicator_name = f"{indicator_name}_generated"
                        suffix = 2
                        while indicator_name in clean_df.columns:
                            indicator_name = f"{col}_is_missing_generated_{suffix}"
                            suffix += 1
                    clean_df[indicator_name] = clean_df[col].isna().astype(int)
                    generated_indicators.add(indicator_name)

                    ind_code = (
                        f"df[{indicator_name!r}] = df[{col!r}].isna().astype(int)"
                    )
                    actions.append(
                        TransformationAction(
                            step_id=step_id,
                            module="imputation",
                            action_type="mnar_indicator",
                            target_columns=[indicator_name],
                            parameters={"source_column": col, "missing_ratio": round(null_ratio, 4)},
                            code_snippet=ind_code,
                        )
                    )
                    audit_log.append(
                        f"[IMPUTATION] Added MNAR indicator column '{indicator_name}' ({round(null_ratio*100, 1)}% missing)."
                    )
                    step_id += 1

        # -------------------------------------------------------------------
        # Step 3: Identify Columns Requiring Imputation
        # -------------------------------------------------------------------
        cols_with_nulls = [
            c for c in clean_df.columns
            if clean_df[c].isna().sum() > 0 and c not in generated_indicators
        ]
        if not cols_with_nulls:
            audit_log.append("[IMPUTATION] No missing values found in working dataset.")
            return TransformResult.success(
                transformed_df=clean_df,
                applied_actions=actions,
                audit_log=audit_log,
                metadata=duplicate_metadata(clean_df),
            )

        # -------------------------------------------------------------------
        # Step 4: Mixed-Type Splitting & Encoding
        # -------------------------------------------------------------------
        num_cols = [c for c in clean_df.columns if pd.api.types.is_numeric_dtype(clean_df[c]) and not pd.api.types.is_bool_dtype(clean_df[c])]
        cat_cols = [c for c in clean_df.columns if c not in num_cols and c not in generated_indicators]

        # Classifier-based categorical reconstruction precedes numeric models.
        categorical_targets = [
            column for column in cols_with_nulls if column in cat_cols
        ]
        for column in categorical_targets:
            if clean_df[column].notna().sum() == 0:
                fill_value = "Missing"
                clean_df[column] = clean_df[column].fillna(fill_value)
                actions.append(
                    TransformationAction(
                        step_id=step_id,
                        module="imputation",
                        action_type="fill_all_null_categorical",
                        target_columns=[str(column)],
                        parameters={"fill_value": fill_value},
                        code_snippet=(
                            f"df[{column!r}] = df[{column!r}].fillna({fill_value!r})"
                        ),
                    )
                )
                audit_log.append(
                    f"[IMPUTATION] Filled all-null categorical column '{column}' "
                    f"with {fill_value!r}."
                )
                step_id += 1
                continue

            requested_strategy = col_strategies.get(
                column, col_strategies.get(str(column))
            )
            if (
                chosen_strategy == ImputationStrategy.MOST_FREQUENT
                or requested_strategy == ImputationStrategy.MOST_FREQUENT.value
            ):
                continue

            classifier = CategoricalClassifierImputer(
                target_column=column,
                random_state=self.random_state,
                n_jobs=1,
            )
            missing_before = int(clean_df[column].isna().sum())
            clean_df = classifier.fit_transform(clean_df)
            filled_count = missing_before - int(clean_df[column].isna().sum())
            if filled_count <= 0:
                continue

            action_type = (
                "impute_categorical_rf"
                if classifier.strategy_ == "random_forest"
                else "impute_categorical_most_frequent"
            )
            actions.append(
                TransformationAction(
                    step_id=step_id,
                    module="imputation",
                    action_type=action_type,
                    target_columns=[str(column)],
                    parameters={
                        "strategy": classifier.strategy_,
                        "n_train": classifier.n_train_,
                        "fallback_reason": classifier.fallback_reason_,
                        "filled_count": filled_count,
                    },
                    code_snippet=classifier.replay_code(),
                )
            )
            audit_log.append(
                f"[IMPUTATION] Imputed categorical column '{column}' using "
                f"{classifier.strategy_} ({filled_count} values)."
            )
            step_id += 1

        cols_with_nulls = [
            column
            for column in clean_df.columns
            if clean_df[column].isna().sum() > 0
            and column not in generated_indicators
        ]
        for column in list(cols_with_nulls):
            if column in num_cols and clean_df[column].notna().sum() == 0:
                clean_df[column] = clean_df[column].fillna(0.0)
                actions.append(
                    TransformationAction(
                        step_id=step_id,
                        module="imputation",
                        action_type="fill_all_null_numeric",
                        target_columns=[str(column)],
                        parameters={"fill_value": 0.0},
                        code_snippet=f"df[{column!r}] = df[{column!r}].fillna(0.0)",
                    )
                )
                audit_log.append(
                    f"[IMPUTATION] Filled all-null numeric column '{column}' with 0.0."
                )
                step_id += 1

        cols_with_nulls = [
            column
            for column in clean_df.columns
            if clean_df[column].isna().sum() > 0
            and column not in generated_indicators
        ]
        if not cols_with_nulls:
            metadata = duplicate_metadata(clean_df)
            metadata["benchmark_scores"] = benchmark_metadata
            if metadata["new_duplicates"] > 0:
                audit_log.append(
                    "[IMPUTATION] WARNING: Imputation introduced "
                    f"{metadata['new_duplicates']} new exact duplicate rows."
                )
            return TransformResult.success(
                transformed_df=clean_df,
                applied_actions=actions,
                audit_log=audit_log,
                metadata=metadata,
            )

        # Encode categoricals using OrdinalEncoder while preserving NaNs
        encoders: Dict[str, OrdinalEncoder] = {}
        encoded_df = clean_df.copy(deep=True)

        for c_col in cat_cols:
            if clean_df[c_col].notna().sum() > 0:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                # Reshape for 2D fitting
                vals_2d = clean_df[[c_col]].astype(object)
                encoded_df[c_col] = oe.fit_transform(vals_2d)
                encoders[c_col] = oe
            else:
                # 100% missing categorical column
                encoded_df[c_col] = np.nan

        # -------------------------------------------------------------------
        # Step 5: Execute Imputation Strategy
        # -------------------------------------------------------------------
        # If strategy is AUTO_CV, run benchmark
        best_recommendations: Dict[str, str] = {}
        if chosen_strategy == ImputationStrategy.AUTO_CV:
            if row_count < 2:
                best_recommendations = {
                    str(col): "median" if col in num_cols else "most_frequent"
                    for col in cols_with_nulls
                }
                audit_log.append(
                    "[IMPUTATION] Single-row dataset: skipped cross-validation and used safe fallbacks."
                )
            else:
                bench_res = self.benchmark_dataset(encoded_df, target_cols=cols_with_nulls)
                benchmark_metadata = bench_res
                best_recommendations = bench_res.get("recommendations", {})
                audit_log.append(f"[IMPUTATION] Benchmark selections: {best_recommendations}")

        # Case A: If user specified KNN across the dataset
        if chosen_strategy == ImputationStrategy.KNN or (not col_strategies and chosen_strategy == ImputationStrategy.AUTO_CV and all(best_recommendations.get(c) == "knn" for c in cols_with_nulls)):
            k = min(self.n_neighbors, max(1, len(encoded_df) - 1))
            knn = KNNImputer(n_neighbors=k)
            # Impute on all numeric + encoded columns
            target_features = [c for c in num_cols if c not in generated_indicators and encoded_df[c].notna().sum() > 0]
            imputed_mat = knn.fit_transform(encoded_df[target_features])
            encoded_df[target_features] = imputed_mat

            code_snippet = (
                f"from sklearn.impute import KNNImputer\n"
                f"knn = KNNImputer(n_neighbors={k})\n"
                f"df[{repr(target_features)}] = knn.fit_transform(df[{repr(target_features)}])"
            )
            actions.append(
                TransformationAction(
                    step_id=step_id,
                    module="imputation",
                    action_type="impute_knn",
                    target_columns=target_features,
                    parameters={"n_neighbors": k},
                    code_snippet=code_snippet,
                )
            )
            audit_log.append(f"[IMPUTATION] Imputed {len(target_features)} features via KNNImputer (k={k}).")
            step_id += 1

        # Case B: If user specified MICE (BayesianRidge or RandomForest)
        elif chosen_strategy in (ImputationStrategy.MICE_RIDGE, ImputationStrategy.MICE_RF):
            use_rf = (chosen_strategy == ImputationStrategy.MICE_RF)
            estimator = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=self.random_state, n_jobs=-1) if use_rf else BayesianRidge()
            target_features = [c for c in num_cols if c not in generated_indicators and encoded_df[c].notna().sum() > 0]

            mice = IterativeImputer(
                estimator=estimator,
                max_iter=self.mice_max_iter,
                random_state=self.random_state,
            )
            imputed_mat = mice.fit_transform(encoded_df[target_features])
            encoded_df[target_features] = imputed_mat

            est_name = "RandomForestRegressor" if use_rf else "BayesianRidge"
            est_import = "from sklearn.ensemble import RandomForestRegressor" if use_rf else "from sklearn.linear_model import BayesianRidge"
            estimator_code = (
                f"RandomForestRegressor(n_estimators=15, max_depth=6, "
                f"random_state={self.random_state}, n_jobs=-1)"
                if use_rf
                else "BayesianRidge()"
            )
            code_snippet = (
                f"from sklearn.experimental import enable_iterative_imputer\n"
                f"from sklearn.impute import IterativeImputer\n"
                f"{est_import}\n"
                f"mice = IterativeImputer(estimator={estimator_code}, max_iter={self.mice_max_iter}, random_state={self.random_state})\n"
                f"df[{repr(target_features)}] = mice.fit_transform(df[{repr(target_features)}])"
            )
            actions.append(
                TransformationAction(
                    step_id=step_id,
                    module="imputation",
                    action_type=f"impute_{chosen_strategy.value}",
                    target_columns=target_features,
                    parameters={"max_iter": self.mice_max_iter, "estimator": est_name},
                    code_snippet=code_snippet,
                )
            )
            audit_log.append(f"[IMPUTATION] Imputed {len(target_features)} features via IterativeImputer ({est_name}).")
            step_id += 1

        # Case C: Column-by-Column (Mean, Median, Most Frequent, or Auto-CV Per-Feature)
        else:
            for col in cols_with_nulls:
                strat_name = col_strategies.get(col, best_recommendations.get(col, chosen_strategy.value))
                is_num = col in num_cols

                if strat_name in ("mean", "median") and is_num:
                    observed_values = clean_df[col].dropna()
                    if observed_values.empty:
                        stat_val = 0.0
                    else:
                        stat_val = float(
                            observed_values.median()
                            if strat_name == "median"
                            else observed_values.mean()
                        )
                    encoded_df[col] = encoded_df[col].fillna(stat_val)

                    code_snippet = (
                        f"df[{col!r}] = df[{col!r}].fillna(df[{col!r}].{strat_name}())"
                    )
                    actions.append(
                        TransformationAction(
                            step_id=step_id,
                            module="imputation",
                            action_type=f"impute_{strat_name}",
                            target_columns=[col],
                            parameters={"strategy": strat_name, "fill_value": round(stat_val, 4)},
                            code_snippet=code_snippet,
                        )
                    )
                    audit_log.append(f"[IMPUTATION] Imputed column '{col}' using {strat_name.upper()} ({round(stat_val, 4)}).")
                    step_id += 1

                elif strat_name == "knn" and is_num:
                    k = min(self.n_neighbors, max(1, len(encoded_df) - 1))
                    knn = KNNImputer(n_neighbors=k)
                    # Use all available numeric features as predictors
                    pred_features = [c for c in num_cols if encoded_df[c].notna().sum() > 0]
                    res = knn.fit_transform(encoded_df[pred_features])
                    col_idx = pred_features.index(col)
                    encoded_df[col] = res[:, col_idx]

                    code_snippet = (
                        f"from sklearn.impute import KNNImputer\n"
                        f"knn = KNNImputer(n_neighbors={k})\n"
                        f"_knn_values = knn.fit_transform(df[{repr(pred_features)}])\n"
                        f"df[{col!r}] = _knn_values[:, {col_idx}]"
                    )
                    actions.append(
                        TransformationAction(
                            step_id=step_id,
                            module="imputation",
                            action_type="impute_knn",
                            target_columns=[col],
                            parameters={"n_neighbors": k},
                            code_snippet=code_snippet,
                        )
                    )
                    audit_log.append(f"[IMPUTATION] Imputed column '{col}' using KNNImputer (k={k}).")
                    step_id += 1

                else:
                    # Categorical or fallback: Most Frequent (Mode)
                    non_null_vals = clean_df[col].dropna()
                    if len(non_null_vals) > 0:
                        mode_val = non_null_vals.mode().iloc[0]
                        # If encoded, find mode of encoded
                        mode_encoded = encoded_df[col].dropna().mode().iloc[0]
                        encoded_df[col] = encoded_df[col].fillna(mode_encoded)
                        code_snippet = (
                            f"df[{col!r}] = df[{col!r}].fillna(df[{col!r}].mode()[0])"
                        )
                        actions.append(
                            TransformationAction(
                                step_id=step_id,
                                module="imputation",
                                action_type="impute_mode",
                                target_columns=[col],
                                parameters={"strategy": "most_frequent", "fill_value": str(mode_val)},
                                code_snippet=code_snippet,
                            )
                        )
                        audit_log.append(f"[IMPUTATION] Imputed column '{col}' using Mode ({mode_val}).")
                        step_id += 1
                    else:
                        # 100% missing column, fill with placeholder
                        fill_placeholder = 0.0 if is_num else "Missing"
                        encoded_df[col] = encoded_df[col].fillna(fill_placeholder)
                        clean_df[col] = clean_df[col].fillna(fill_placeholder)
                        actions.append(
                            TransformationAction(
                                step_id=step_id,
                                module="imputation",
                                action_type="fill_all_null_constant",
                                target_columns=[str(col)],
                                parameters={"fill_value": fill_placeholder},
                                code_snippet=(
                                    f"df[{col!r}] = df[{col!r}].fillna({fill_placeholder!r})"
                                ),
                            )
                        )
                        audit_log.append(f"[IMPUTATION] 100% missing column '{col}' filled with constant placeholder '{fill_placeholder}'.")
                        step_id += 1

        # -------------------------------------------------------------------
        # Step 6: Decode Categoricals Back to Original Domain
        # -------------------------------------------------------------------
        final_df = encoded_df.copy(deep=True)

        for c_col, oe in encoders.items():
            if c_col in final_df.columns:
                # Clip and round continuous imputed numbers to valid integer category indices
                max_cat_idx = len(oe.categories_[0]) - 1
                rounded_indices = np.clip(
                    np.round(final_df[c_col].to_numpy(dtype=float, na_value=0.0)),
                    0,
                    max_cat_idx,
                ).astype(int)
                # Inverse transform back to original string classes
                decoded_vals = oe.categories_[0][rounded_indices]
                final_df[c_col] = decoded_vals

        metadata = duplicate_metadata(final_df)
        metadata["benchmark_scores"] = benchmark_metadata
        if metadata["new_duplicates"] > 0:
            audit_log.append(
                "[IMPUTATION] WARNING: Imputation introduced "
                f"{metadata['new_duplicates']} new exact duplicate rows."
            )

        return TransformResult(
            status=ExecutionStatus.SUCCESS,
            transformed_df=final_df,
            applied_actions=actions,
            audit_log=audit_log,
            error_message=None,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Convenience Module API
# ---------------------------------------------------------------------------

def impute_missing_values(
    df: pd.DataFrame,
    strategy: Union[str, ImputationStrategy] = ImputationStrategy.AUTO_CV,
    column_strategies: Optional[Dict[str, str]] = None,
    add_mnar_indicators: bool = True,
    dataset_version: str = "v1_privacy",
    drop_high_missing: bool = False,
    start_step_id: int = 1,
) -> TransformResult:
    """
    Convenience orchestrator for Missing Value Imputation (Module 2).
    """
    strat = ImputationStrategy(strategy) if isinstance(strategy, str) else strategy
    imputer = MissingValueImputer(
        default_strategy=strat,
        drop_high_missing=drop_high_missing,
    )
    return imputer.impute(
        df=df,
        strategy=strat,
        column_strategies=column_strategies,
        add_mnar_indicators=add_mnar_indicators,
        dataset_version=dataset_version,
        start_step_id=start_step_id,
    )
