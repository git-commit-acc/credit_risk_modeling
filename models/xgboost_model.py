# models/xgboost_model.py
"""
XGBoost model for credit risk with native Dask-distributed training.

FIX 1: the original module created a brand-new
`dask.distributed.Client(n_workers=4, threads_per_worker=2)` inside every
single `fit()` call, wrapped in a bare `try/except: pass` that silently
swallowed the "a client is already running" error. Across base-model
training, the stacking ensemble, and hyperparameter tuning (n_trials x
cv_folds calls), this could spin up dozens of redundant local clusters and
was a major source of unbounded RAM/CPU growth. It now reuses the single
shared client from `models.dask_utils`.

FIX 2 (correctness bug, not just an optimization): the dataset's
categorical columns (PROPERTY_STATE, CHANNEL, OCCUPANCY_STATUS, etc. -- see
config.features.categorical_features) come out of the Spark feature
pipeline as raw strings, and this module previously handed them to
`xgb_dask.DaskDMatrix` completely unprocessed. XGBoost's DMatrix rejects
object/string columns outright ("DataFrame.dtypes for data must be int,
float, bool or category") -- so training would fail the moment a
categorical column was present, i.e. on essentially every real run of this
dataset. Categorical columns are now cast to pandas `category` dtype via
the shared `LazyCategoricalEncoder(ordinal_encode=False)` (fit once, lazily,
partition-wise -- no full-dataset materialization), and
`enable_categorical=True` is passed to `DaskDMatrix` so XGBoost consumes
them with its native (split-search-based) categorical handling rather than
a lossy hand-rolled integer encoding.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import dask as xgb_dask

from models.base import BaseCreditRiskModel
from models.dask_utils import LazyCategoricalEncoder, ensure_dask_dataframe, get_dask_client

logger = logging.getLogger(__name__)


class XGBoostModel(BaseCreditRiskModel):
    """XGBoost Classifier with native Dask-distributed training (DaskDMatrix)
    and native (non-ordinal-encoded) categorical support."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 10.0,
        early_stopping_rounds: int = 50,
        tree_method: str = 'hist',
        eval_metric: str = 'logloss',
        npartitions: int = 8,
        categorical_columns: Optional[List[str]] = None,
    ):
        super().__init__("XGBoost", random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.tree_method = tree_method
        self.eval_metric = eval_metric
        self.npartitions = npartitions
        self.categorical_columns = categorical_columns
        self.cat_typer: Optional[LazyCategoricalEncoder] = None
        self.is_dask_model = True

    def _preprocess(self, X: Union[dd.DataFrame, pd.DataFrame], fit: bool) -> dd.DataFrame:
        """Cast categorical columns to native pandas `category` dtype
        (never ordinal-integer-encoded -- see module docstring). Purely
        numeric columns pass through untouched. Lazy/partition-wise; never
        triggers a full `.compute()`."""
        X_dask = ensure_dask_dataframe(X, npartitions=self.npartitions)

        if fit:
            self.cat_typer = LazyCategoricalEncoder(
                categorical_columns=self.categorical_columns, ordinal_encode=False
            )
            X_typed = self.cat_typer.fit_transform(X_dask)
            self.categorical_columns = self.cat_typer.categorical_columns
        else:
            if self.cat_typer is None:
                raise ValueError("Model not trained. Call fit() first.")
            X_typed = self.cat_typer.transform(X_dask)

        return X_typed

    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs,
    ):
        """Train XGBoost model with native Dask-distributed training."""
        logger.info(f"Training {self.name} with Dask...")

        self.feature_names = list(X_train.columns)

        client = get_dask_client()

        X_train_dask = self._preprocess(X_train, fit=True)
        y_train_dask = ensure_dask_dataframe(y_train, npartitions=self.npartitions)

        dtrain = xgb_dask.DaskDMatrix(client, X_train_dask, y_train_dask, enable_categorical=True)

        evals = [(dtrain, 'train')]
        early_stopping = None
        if X_val is not None and y_val is not None:
            X_val_dask = self._preprocess(X_val, fit=False)
            y_val_dask = ensure_dask_dataframe(y_val, npartitions=self.npartitions)
            dval = xgb_dask.DaskDMatrix(client, X_val_dask, y_val_dask, enable_categorical=True)
            evals = [(dtrain, 'train'), (dval, 'valid')]
            early_stopping = self.early_stopping_rounds

        params = {
            'objective': 'binary:logistic',
            'eval_metric': self.eval_metric,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'scale_pos_weight': self.scale_pos_weight,
            # 'hist'/'approx' are the only tree methods that support native
            # categorical splits; anything else would silently ignore
            # enable_categorical and error out on a category-dtype column.
            'tree_method': self.tree_method if self.tree_method in ('hist', 'approx') else 'hist',
            'seed': self.random_state,
        }

        self.model = xgb_dask.train(
            client,
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )

        try:
            importance = self.model['booster'].get_score(importance_type='weight')
            if importance:
                self.feature_importance = {
                    self.feature_names[int(k[1:])] if k.startswith('f') else k: v
                    for k, v in importance.items()
                }
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"  Could not extract feature importance: {e}")

        logger.info(f"{self.name} training completed.")
        return self

    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities. Only the (n_samples,) score array is
        computed to the client -- X itself stays distributed across workers."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        client = get_dask_client()
        X_dask = self._preprocess(X, fit=False)
        dtest = xgb_dask.DaskDMatrix(client, X_dask, enable_categorical=True)

        preds = xgb_dask.predict(client, self.model, dtest)
        result = preds.compute()

        return np.column_stack([1 - result, result])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters for hyperparameter tuning."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'scale_pos_weight': self.scale_pos_weight,
            'tree_method': self.tree_method,
            'eval_metric': self.eval_metric,
            'random_state': self.random_state,
        }
