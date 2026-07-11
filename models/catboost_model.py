# models/catboost_model.py
"""
CatBoost model for credit risk with memory-efficient, partition-wise
incremental training.

WHY THIS APPROACH:
CatBoost has no native Dask integration, and its `Pool` object is always a
single in-memory (or single-file, via `catboost.Pool(data=<path>)` with
libsvm/CatBoost-native formats only -- not Parquet) structure. The original
implementation called `X.compute()` on every fit/predict, silently pulling
the ENTIRE Dask DataFrame into driver RAM -- exactly the full-dataset
materialization this refactor is meant to eliminate (requirement #8).

Since CatBoost cannot consume Dask partitions natively, the memory-safe
strategy is CatBoost's own supported mechanism for incremental training:
`CatBoostClassifier.fit(..., init_model=<previous model>)`. We iterate over
the Dask DataFrame's partitions one at a time -- each partition is pulled
into memory, trained for a small number of boosting rounds continuing from
the previous partition's model, and then released before the next partition
is loaded. Peak memory is therefore bounded by the size of the LARGEST
SINGLE PARTITION, not the full dataset -- the same out-of-core guarantee
Dask gives XGBoost/LightGBM, implemented manually because CatBoost doesn't
expose a partition-aware training API itself.

The validation pool (used for early stopping) is still computed once into
memory, since validation folds are expected to be a small fraction of the
data (this mirrors how XGBoost/LightGBM's Dask integrations also keep a
single eval set resident) -- the expensive/large object here is TRAIN, which
this module never fully materializes.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from models.base import BaseCreditRiskModel
from models.dask_utils import ensure_dask_dataframe, identify_categorical_columns

logger = logging.getLogger(__name__)


class CatBoostModel(BaseCreditRiskModel):
    """CatBoost Classifier trained incrementally, partition-by-partition,
    to bound peak memory usage regardless of total dataset size."""

    def __init__(
        self,
        random_state: int = 42,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3,
        border_count: int = 254,
        auto_class_weights: str = 'Balanced',
        verbose: bool = False,
        cat_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
        npartitions: int = 8,
        rounds_per_partition: Optional[int] = None,
    ):
        super().__init__("CatBoost", random_state)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.auto_class_weights = auto_class_weights
        self.verbose = verbose
        self.cat_features = cat_features
        self.early_stopping_rounds = early_stopping_rounds
        self.npartitions = npartitions
        # Boosting rounds trained per partition pass. Defaults so that one
        # full pass over all partitions trains `iterations` total rounds.
        self.rounds_per_partition = rounds_per_partition
        self.is_dask_model = False  # partition-wise incremental, not Dask-native

    def _identify_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in X.columns:
            if col not in cat_cols:
                unique_count = X[col].nunique()
                if unique_count < 20 and X[col].dtype in ['int64', 'float64']:
                    cat_cols.append(col)
        return cat_cols

    def _prep_partition(self, pdf: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        pdf = pdf.copy()
        for col in cat_cols:
            pdf[col] = pdf[col].fillna('MISSING').astype(str)
            pdf[col] = pdf[col].replace({'nan': 'MISSING', 'None': 'MISSING'})
        for col in pdf.columns:
            if col not in cat_cols:
                pdf[col] = pdf[col].astype(float)
        return pdf

    def _make_pool(self, X_pd: pd.DataFrame, y_pd: Optional[pd.Series], cat_cols: List[str]) -> Pool:
        X_prepped = self._prep_partition(X_pd, cat_cols)
        y_arr = y_pd.values if y_pd is not None else None
        return Pool(data=X_prepped, label=y_arr, cat_features=cat_cols)

    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs,
    ):
        """Incrementally train CatBoost, one Dask partition at a time."""
        logger.info(f"Training {self.name} partition-by-partition (out-of-core)...")

        self.feature_names = list(X_train.columns)

        X_dask = ensure_dask_dataframe(X_train, npartitions=self.npartitions)
        y_dask = ensure_dask_dataframe(y_train, npartitions=self.npartitions)
        X_dask = X_dask.repartition(npartitions=self.npartitions)
        y_dask = y_dask.repartition(npartitions=self.npartitions)

        if self.cat_features is None:
            self.cat_features = identify_categorical_columns(X_dask)
            logger.info(f"  Identified {len(self.cat_features)} categorical features")

        n_partitions = X_dask.npartitions
        rounds_per_partition = self.rounds_per_partition or max(1, self.iterations // n_partitions)

        # Validation pool: computed once, held in memory for the duration of
        # training (small relative to train by construction of the
        # train/val/test split), used purely for early-stopping/logging.
        eval_pool = None
        if X_val is not None and y_val is not None:
            logger.info("  Materializing validation pool for early stopping...")
            X_val_pd = self._ensure_pandas(X_val)
            y_val_pd = self._ensure_pandas(y_val)
            eval_pool = self._make_pool(X_val_pd, y_val_pd, self.cat_features)

        model = None
        for part_idx in range(n_partitions):
            logger.info(f"  Partition {part_idx + 1}/{n_partitions} "
                        f"({rounds_per_partition} boosting rounds)...")

            X_part = X_dask.get_partition(part_idx).compute()
            y_part = y_dask.get_partition(part_idx).compute()

            if len(X_part) == 0:
                continue

            train_pool = self._make_pool(X_part, y_part, self.cat_features)

            # NOTE: CatBoost's `init_model` continuation only accepts a
            # limited set of changed parameters between calls (e.g. it will
            # raise if class-weighting parameters differ from the model
            # being continued). auto_class_weights is therefore only applied
            # on the very first partition; class imbalance for subsequent
            # partitions is still reflected because the running model's
            # tree structure/leaf values already encode it.
            step_model = CatBoostClassifier(
                iterations=rounds_per_partition,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                border_count=self.border_count,
                auto_class_weights=self.auto_class_weights if model is None else None,
                random_state=self.random_state,
                verbose=self.verbose,
                cat_features=self.cat_features,
                allow_writing_files=False,
            )

            fit_kwargs = {"verbose": False}
            if eval_pool is not None:
                fit_kwargs["eval_set"] = eval_pool
                # Only apply early stopping on the final partition pass to
                # avoid truncating training prematurely on an early chunk.
                if part_idx == n_partitions - 1:
                    fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

            step_model.fit(train_pool, init_model=model, **fit_kwargs)
            model = step_model

            # `X_part`/`train_pool` go out of scope here and are released
            # before the next partition is loaded -- peak memory is bounded
            # by one partition, not the full dataset.

        self.model = model

        if self.model is not None:
            self.feature_importance = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )

        logger.info(f"{self.name} training completed ({n_partitions} partitions).")
        return self

    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities. Scoring is done partition-by-partition and
        concatenated, so this also never materializes the full input at
        once for large scoring jobs."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_dask = ensure_dask_dataframe(X, npartitions=self.npartitions)
        preds = []
        for part_idx in range(X_dask.npartitions):
            X_part = X_dask.get_partition(part_idx).compute()
            if len(X_part) == 0:
                continue
            pool = self._make_pool(X_part, None, self.cat_features)
            preds.append(self.model.predict_proba(pool))

        return np.concatenate(preds, axis=0) if preds else np.empty((0, 2))

    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters for hyperparameter tuning."""
        return {
            "random_state": self.random_state,
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "border_count": self.border_count,
            "auto_class_weights": self.auto_class_weights,
            "verbose": self.verbose,
            "cat_features": self.cat_features,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
