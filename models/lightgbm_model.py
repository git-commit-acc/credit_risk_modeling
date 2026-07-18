# models/lightgbm_model.py
"""
LightGBM model for credit risk with GPU support.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import lightgbm as lgb
import logging
from typing import Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel
import os
# Suppress Boost filesystem warnings on Windows
os.environ['BOOST_LOG_DISABLE'] = '1'

logger = logging.getLogger(__name__)


class LightGBMModel(BaseCreditRiskModel):
    """LightGBM Classifier with GPU acceleration support."""
    
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 200,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        is_unbalance: bool = True,
        verbosity: int = -1,
        early_stopping_rounds: int = 50,
        n_jobs: int = -1,
        use_gpu: bool = True,
        gpu_platform_id: int = 0,
        gpu_device_id: int = 0
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
        self.n_jobs = n_jobs
        
        # GPU Configuration
        self.use_gpu = use_gpu
        self.gpu_platform_id = gpu_platform_id
        self.gpu_device_id = gpu_device_id
        
        if use_gpu:
            logger.info("LightGBM: GPU acceleration enabled")
        
        self.is_distributed = False
        self.supports_dask_data = True
    
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns for LightGBM."""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if pd.api.types.is_object_dtype(X_encoded[col]) or \
               pd.api.types.is_string_dtype(X_encoded[col]) or \
               pd.api.types.is_categorical_dtype(X_encoded[col]):
                
                X_encoded[col] = X_encoded[col].fillna('MISSING')
                X_encoded[col] = X_encoded[col].astype(str)
                X_encoded[col] = X_encoded[col].replace('nan', 'MISSING')
                X_encoded[col] = X_encoded[col].replace('None', 'MISSING')
                
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
        """Train LightGBM model with GPU support."""
        logger.info(f"Training {self.name}...")
        
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        logger.info(f"  Training with {len(X_train_pd):,} samples, {len(self.feature_names)} features")
        
        X_train_encoded = self._encode_categorical(X_train_pd)
        X_train_encoded = X_train_encoded.fillna(0)
        
        # Create model with GPU parameters
        model_params = {
            'n_estimators': self.n_estimators,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'is_unbalance': self.is_unbalance,
            'random_state': self.random_state,
            'verbosity': self.verbosity,
            'n_jobs': self.n_jobs,
        }
        
        # Add GPU parameters
        if self.use_gpu:
            model_params['device'] = 'gpu'
            model_params['gpu_platform_id'] = self.gpu_platform_id
            model_params['gpu_device_id'] = self.gpu_device_id
        
        self.model = lgb.LGBMClassifier(**model_params)
        
        if X_val is not None and y_val is not None:
            X_val_pd = self._ensure_pandas(X_val)
            y_val_pd = self._ensure_pandas(y_val)
            X_val_encoded = self._encode_categorical(X_val_pd)
            X_val_encoded = X_val_encoded.fillna(0)
            
            self.model.fit(
                X_train_encoded, y_train_pd,
                eval_set=[(X_val_encoded, y_val_pd)],
                eval_metric='logloss',
                callbacks=[lgb.early_stopping(self.early_stopping_rounds)]
            )
        else:
            self.model.fit(X_train_encoded, y_train_pd)
        
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
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_pd = self._ensure_pandas(X)
        X_encoded = self._encode_categorical(X_pd)
        X_encoded = X_encoded.fillna(0)
        return self.model.predict_proba(X_encoded)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        return self.feature_importance if self.feature_importance else {}
    
    def get_params(self, deep=True):
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
            'verbosity': self.verbosity,
            'n_jobs': self.n_jobs,
            'use_gpu': self.use_gpu,
            'gpu_platform_id': self.gpu_platform_id,
            'gpu_device_id': self.gpu_device_id
        }