# models/lightgbm_model.py
"""
LightGBM model for credit risk with Dask support.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import lightgbm as lgb
from lightgbm import dask as lgb_dask
import logging
from typing import Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseCreditRiskModel):
    """LightGBM Classifier with Dask support."""
    
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
        npartitions: int = 4
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
        self.is_dask_model = True
        
        self._client = None
        
    def _ensure_dask(self, data: Union[dd.DataFrame, pd.DataFrame]) -> dd.DataFrame:
        """Convert to Dask if needed."""
        if isinstance(data, pd.DataFrame):
            return dd.from_pandas(data, npartitions=self.npartitions)
        return data
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs
    ):
        """Train LightGBM model with Dask."""
        logger.info(f"Training {self.name} with Dask...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Convert to Dask
        X_train_dask = self._ensure_dask(X_train)
        y_train_dask = self._ensure_dask(y_train) if isinstance(y_train, (pd.Series, dd.Series)) else y_train
        
        if X_val is not None:
            X_val_dask = self._ensure_dask(X_val)
            y_val_dask = self._ensure_dask(y_val) if isinstance(y_val, (pd.Series, dd.Series)) else y_val
        
        # Create Dask client if not exists
        try:
            from dask.distributed import Client
            self._client = Client(n_workers=4, threads_per_worker=2)
        except:
            pass
        
        # Prepare Dask data
        dtrain = lgb_dask.DaskLGBMClassifier(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            is_unbalance=self.is_unbalance,
            random_state=self.random_state,
            verbosity=self.verbosity
        )
        
        # Train with validation
        if X_val is not None:
            dtrain.fit(
                X_train_dask, y_train_dask,
                eval_set=[(X_val_dask, y_val_dask)],
                eval_metric='logloss',
                early_stopping_rounds=self.early_stopping_rounds
            )
        else:
            dtrain.fit(X_train_dask, y_train_dask)
        
        self.model = dtrain
        
        # Store feature importance
        try:
            importance = dtrain.feature_importances_
            self.feature_importance = dict(
                zip(self.feature_names, importance)
            )
        except:
            pass
        
        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_dask = self._ensure_dask(X)
        result = self.model.predict_proba(X_dask)
        return result.compute()
    
    def get_params(self, deep=True):
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
            'random_state': self.random_state
        }