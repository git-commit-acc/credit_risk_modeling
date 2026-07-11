# models/lightgbm_model.py
"""
LightGBM model for credit risk with native Dask-distributed training.

FIX 1: same client-lifecycle fix as models/xgboost_model.py -- see that
module's docstring for details. Reuses the shared client from
models.dask_utils instead of creating (and leaking) a new LocalCluster on
every fit() call.

FIX 2 (correctness bug): same root cause as XGBoostModel -- categorical
columns arrive as raw strings from the Spark feature pipeline, and
`lightgbm.dask.DaskLGBMClassifier` also rejects object/string dtype columns
outright. LightGBM's native categorical support requires pandas `category`
dtype columns (it reads the category codes directly rather than one-hot/
ordinal encoding them). Categorical columns are now cast via the shared
`LazyCategoricalEncoder(ordinal_encode=False)` (fit once, lazily,
partition-wise) before every fit/predict call, and `categorical_feature`
is passed explicitly so LightGBM's optimal-split categorical algorithm is
used rather than treating them as numeric.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import dask as lgb_dask

from models.base import BaseCreditRiskModel
from models.dask_utils import LazyCategoricalEncoder, ensure_dask_dataframe, get_dask_client

logger = logging.getLogger(__name__)


class LightGBMModel(BaseCreditRiskModel):
    """LightGBM Classifier with native Dask-distributed training
    (DaskLGBMClassifier) and native (non-ordinal-encoded) categorical support."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        is_unbalance: bool = True,
        verbosity: int = -1,
        early_stopping_rounds: int = 50,
        npartitions: int = 8,
        categorical_columns: Optional[List[str]] = None,
    ):
        super().__init__("LightGBM", random_state)
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.is_unbalance = is_unbalance
        self.verbosity = verbosity
        self.early_stopping_rounds = early_stopping_rounds
        self.npartitions = npartitions
        self.categorical_columns = categorical_columns
        self.cat_typer: Optional[LazyCategoricalEncoder] = None
        self.is_dask_model = True

    def _preprocess(self, X: Union[dd.DataFrame, pd.DataFrame], fit: bool) -> dd.DataFrame:
        """Cast categorical columns to native pandas `category` dtype so
        LightGBM's own categorical split-finding is used (never a lossy
        ordinal integer encoding). Lazy/partition-wise; no full compute."""
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
        """Train LightGBM model with native Dask-distributed training."""
        logger.info(f"Training {self.name} with Dask...")

        self.feature_names = list(X_train.columns)

        client = get_dask_client()

        X_train_dask = self._preprocess(X_train, fit=True)
        y_train_dask = ensure_dask_dataframe(y_train, npartitions=self.npartitions)

        cat_feature_arg = self.categorical_columns if self.categorical_columns else 'auto'

        dask_model = lgb_dask.DaskLGBMClassifier(
            client=client,
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            is_unbalance=self.is_unbalance,
            random_state=self.random_state,
            verbosity=self.verbosity,
        )

        if X_val is not None and y_val is not None:
            X_val_dask = self._preprocess(X_val, fit=False)
            y_val_dask = ensure_dask_dataframe(y_val, npartitions=self.npartitions)
            # LightGBM >= 4.0 moved early stopping from the `early_stopping_
            # rounds` fit() kwarg (removed) to an explicit callback.
            dask_model.fit(
                X_train_dask, y_train_dask,
                eval_set=[(X_val_dask, y_val_dask)],
                eval_metric='logloss',
                categorical_feature=cat_feature_arg,
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
        else:
            dask_model.fit(X_train_dask, y_train_dask, categorical_feature=cat_feature_arg)

        self.model = dask_model

        try:
            importance = dask_model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importance))
        except AttributeError as e:
            logger.warning(f"  Could not extract feature importance: {e}")

        logger.info(f"{self.name} training completed.")
        return self

    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities. Only the (n_samples, 2) prediction array is
        computed -- X stays distributed across workers as a Dask array."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_dask = self._preprocess(X, fit=False)
        result = self.model.predict_proba(X_dask)
        return result.compute() if hasattr(result, "compute") else np.asarray(result)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters for hyperparameter tuning."""
        return {
            'n_estimators': self.n_estimators,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'is_unbalance': self.is_unbalance,
            'random_state': self.random_state,
        }
