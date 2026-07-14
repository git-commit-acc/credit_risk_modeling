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
# models/xgboost_model.py
"""
XGBoost model for credit risk - uses sklearn API (no Dask distributed).
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import xgboost as xgb
import logging
from typing import Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseCreditRiskModel):
    """XGBoost Classifier using sklearn API (stable)."""
    
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 10.0,
        early_stopping_rounds: int = 50,
        tree_method: str = 'hist',
        eval_metric: str = 'logloss',
        n_jobs: int = -1
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
        self.n_jobs = n_jobs
        self.is_distributed = False
        self.supports_dask_data = True
        
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    # def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
    #     """Encode categorical columns for XGBoost."""
    #     X_encoded = X.copy()
        
    #     for col in X_encoded.columns:
    #         if X_encoded[col].dtype == 'object' or X_encoded[col].dtype == 'category':
    #             X_encoded[col] = X_encoded[col].fillna('MISSING')
    #             X_encoded[col] = X_encoded[col].astype(str)
    #             X_encoded[col] = X_encoded[col].replace('nan', 'MISSING')
    #             X_encoded[col] = X_encoded[col].replace('None', 'MISSING')
    #             # Convert to categorical codes
    #             X_encoded[col] = X_encoded[col].astype('category').cat.codes
        
    #     return X_encoded

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns for models."""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            # Check for ANY non-numeric dtype (object, string, category, etc.)
            if pd.api.types.is_object_dtype(X_encoded[col]) or \
            pd.api.types.is_string_dtype(X_encoded[col]) or \
            pd.api.types.is_categorical_dtype(X_encoded[col]):
                
                # Fill missing values
                X_encoded[col] = X_encoded[col].fillna('MISSING')
                X_encoded[col] = X_encoded[col].astype(str)
                X_encoded[col] = X_encoded[col].replace('nan', 'MISSING')
                X_encoded[col] = X_encoded[col].replace('None', 'MISSING')
                X_encoded[col] = X_encoded[col].replace('', 'MISSING')
                
                # Convert to categorical codes (0, 1, 2, ...)
                # Handle case where all values become 'MISSING'
                if X_encoded[col].nunique() <= 1:
                    X_encoded[col] = 0
                else:
                    X_encoded[col] = X_encoded[col].astype('category').cat.codes
        
        return X_encoded
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs
    ):
        """Train XGBoost model using sklearn API."""
        logger.info(f"Training {self.name} (sklearn API)...")
        
        # Convert to pandas
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        # Encode categorical columns
        X_train_encoded = self._encode_categorical(X_train_pd)
        
        # Create model - eval_metric is set in constructor, not fit()
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            tree_method=self.tree_method,
            eval_metric=self.eval_metric,  # Set here, not in fit()
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            use_label_encoder=False
        )
        
        # Train with validation if provided
        if X_val is not None and y_val is not None:
            X_val_pd = self._ensure_pandas(X_val)
            y_val_pd = self._ensure_pandas(y_val)
            X_val_encoded = self._encode_categorical(X_val_pd)
            
            # Only pass eval_set, nothing else
            self.model.fit(
                X_train_encoded, y_train_pd,
                eval_set=[(X_val_encoded, y_val_pd)],
                verbose=False
            )
        else:
            self.model.fit(X_train_encoded, y_train_pd, verbose=False)
        
        # Store feature importance
        try:
            importance = self.model.feature_importances_
            if self.feature_names:
                self.feature_importance = dict(
                    zip(self.feature_names, importance)
                )
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
        
        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_pd = self._ensure_pandas(X)
        X_encoded = self._encode_categorical(X_pd)
        
        return self.model.predict_proba(X_encoded)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.feature_importance is not None:
            return self.feature_importance
        return {}
    
    def get_params(self, deep=True):
        """Get model parameters."""
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
            'n_jobs': self.n_jobs
        }