# models/xgboost_model.py
"""
XGBoost model for credit risk with GPU support.
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
    """XGBoost Classifier with GPU acceleration support."""
    
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 10.0,
        early_stopping_rounds: int = 50,
        tree_method: str = 'hist',  # Use 'hist' for GPU with device='cuda'
        eval_metric: str = 'logloss',
        n_jobs: int = -1,
        use_gpu: bool = True,
        gpu_id: int = 0
    ):
        super().__init__("XGBoost", random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        
        # GPU Configuration
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Use new XGBoost 2.0+ GPU parameters
        if use_gpu:
            self.tree_method = 'hist'
            self.device = 'cuda'
            logger.info("XGBoost: GPU acceleration enabled (device='cuda')")
        else:
            self.tree_method = tree_method
            self.device = 'cpu'
            
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.is_distributed = False
        self.supports_dask_data = True
        
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns for XGBoost."""
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
        """Train XGBoost model with GPU support."""
        logger.info(f"Training {self.name}...")
        
        # Convert to pandas
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        logger.info(f"  Training with {len(X_train_pd):,} samples, {len(self.feature_names)} features")
        
        # Encode categorical columns
        X_train_encoded = self._encode_categorical(X_train_pd)
        X_train_encoded = X_train_encoded.fillna(0)
        
        # Create model with GPU parameters
        model_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'scale_pos_weight': self.scale_pos_weight,
            'tree_method': self.tree_method,
            'device': self.device,  # NEW: Use device instead of gpu_id
            'eval_metric': self.eval_metric,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }
        
        self.model = xgb.XGBClassifier(**model_params)
        
        if X_val is not None and y_val is not None:
            X_val_pd = self._ensure_pandas(X_val)
            y_val_pd = self._ensure_pandas(y_val)
            X_val_encoded = self._encode_categorical(X_val_pd)
            X_val_encoded = X_val_encoded.fillna(0)
            
            try:
                self.model.fit(
                    X_train_encoded, y_train_pd,
                    eval_set=[(X_val_encoded, y_val_pd)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
            except TypeError:
                logger.warning("  early_stopping not supported, training without it...")
                self.model.fit(X_train_encoded, y_train_pd, verbose=False)
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
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'scale_pos_weight': self.scale_pos_weight,
            'tree_method': self.tree_method,
            'device': self.device,
            'eval_metric': self.eval_metric,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'use_gpu': self.use_gpu,
            'gpu_id': self.gpu_id
        }