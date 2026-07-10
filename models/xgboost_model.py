# models/xgboost_model.py
"""
XGBoost model for credit risk with Dask support.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import xgboost as xgb
from xgboost import dask as xgb_dask
import logging
from typing import Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseCreditRiskModel):
    """XGBoost Classifier with Dask support."""
    
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
        npartitions: int = 4
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
        self.is_dask_model = True
        
        self.eval_set = None
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
        """Train XGBoost model with Dask."""
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
        dtrain = xgb_dask.DaskDMatrix(self._client, X_train_dask, y_train_dask)
        
        if X_val is not None:
            dval = xgb_dask.DaskDMatrix(self._client, X_val_dask, y_val_dask)
            evals = [(dtrain, 'train'), (dval, 'valid')]
            early_stopping = self.early_stopping_rounds
        else:
            evals = [(dtrain, 'train')]
            early_stopping = None
        
        # Create model parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': self.eval_metric,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'scale_pos_weight': self.scale_pos_weight,
            'tree_method': self.tree_method,
            'seed': self.random_state
        }
        
        # Train with Dask XGBoost
        self.model = xgb_dask.train(
            self._client,
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=False
        )
        
        # Store feature importance
        try:
            importance = self.model['booster'].get_score(importance_type='weight')
            # Map to feature names
            if importance:
                self.feature_importance = {
                    self.feature_names[int(k[1:])] if k.startswith('f') else k: v
                    for k, v in importance.items()
                }
        except:
            pass
        
        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_dask = self._ensure_dask(X)
        dtest = xgb_dask.DaskDMatrix(self._client, X_dask)
        
        # Predict
        preds = xgb_dask.predict(self._client, self.model, dtest)
        result = preds.compute()
        
        # Return as 2-column array
        return np.column_stack([1 - result, result])
    
    def get_params(self, deep=True):
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
            'random_state': self.random_state
        }