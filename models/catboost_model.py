# models/catboost_model.py
"""
CatBoost model for credit risk with GPU support.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from catboost import CatBoostClassifier
import logging
from typing import Dict, Any, Optional, List, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseCreditRiskModel):
    """CatBoost Classifier with GPU acceleration support."""
    
    def __init__(
        self,
        random_state: int = 42,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3,
        border_count: int = 254,
        auto_class_weights: str = 'Balanced',
        verbose: bool = False,
        early_stopping_rounds: int = 50,
        cat_features: Optional[List[str]] = None,
        use_gpu: bool = True,
        devices: str = '0'
    ):
        super().__init__("CatBoost", random_state)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.auto_class_weights = auto_class_weights
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features = cat_features
        
        # GPU Configuration
        self.use_gpu = use_gpu
        self.devices = devices
        if use_gpu:
            logger.info("CatBoost: GPU acceleration enabled")
        
        self.is_distributed = False
        self.supports_dask_data = True
    
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    def _identify_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify categorical columns."""
        cat_cols = []
        
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]) or \
               pd.api.types.is_string_dtype(X[col]) or \
               pd.api.types.is_categorical_dtype(X[col]):
                cat_cols.append(col)
            else:
                unique_count = X[col].nunique()
                if unique_count < 20 and unique_count > 1:
                    cat_cols.append(col)
        
        return cat_cols
    
    def _prepare_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare categorical columns for CatBoost."""
        X_prepared = X.copy()
        
        if self.cat_features:
            for col in self.cat_features:
                if col in X_prepared.columns:
                    X_prepared[col] = X_prepared[col].fillna('MISSING')
                    X_prepared[col] = X_prepared[col].astype(str)
                    X_prepared[col] = X_prepared[col].replace('nan', 'MISSING')
                    X_prepared[col] = X_prepared[col].replace('None', 'MISSING')
                    X_prepared[col] = X_prepared[col].replace('', 'MISSING')
        
        for col in X_prepared.columns:
            if col not in (self.cat_features or []):
                X_prepared[col] = pd.to_numeric(X_prepared[col], errors='coerce')
                X_prepared[col] = X_prepared[col].fillna(0)
        
        return X_prepared
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs
    ):
        """Train CatBoost model with GPU support."""
        logger.info(f"Training {self.name}...")
        
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        logger.info(f"  Training with {len(X_train_pd):,} samples, {len(self.feature_names)} features")
        
        if self.cat_features is None:
            self.cat_features = self._identify_categorical_columns(X_train_pd)
            logger.info(f"  Identified {len(self.cat_features)} categorical features")
        
        X_train_prepared = self._prepare_categoricals(X_train_pd)
        
        # Create model with GPU parameters
        model_params = {
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'l2_leaf_reg': self.l2_leaf_reg,
            'border_count': self.border_count,
            'auto_class_weights': self.auto_class_weights,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'cat_features': self.cat_features,
        }
        
        # Add GPU parameters
        if self.use_gpu:
            model_params['task_type'] = 'GPU'
            model_params['devices'] = self.devices
        
        self.model = CatBoostClassifier(**model_params)
        
        if X_val is not None and y_val is not None:
            X_val_pd = self._ensure_pandas(X_val)
            y_val_pd = self._ensure_pandas(y_val)
            X_val_prepared = self._prepare_categoricals(X_val_pd)
            
            self.model.fit(
                X_train_prepared, y_train_pd,
                eval_set=[(X_val_prepared, y_val_pd)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X_train_prepared, y_train_pd, verbose=False)
        
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
        X_prepared = self._prepare_categoricals(X_pd)
        return self.model.predict_proba(X_prepared)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        return self.feature_importance if self.feature_importance else {}
    
    def get_params(self, deep=True):
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
            "use_gpu": self.use_gpu,
            "devices": self.devices
        }