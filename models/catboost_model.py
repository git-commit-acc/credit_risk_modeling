# models/catboost_model.py
"""
CatBoost model for credit risk with disk-backed support.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from catboost import CatBoostClassifier, Pool
import logging
import tempfile
import os
from typing import Dict, Any, Optional, List, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseCreditRiskModel):
    """CatBoost Classifier with disk-backed support."""
    
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
        early_stopping_rounds: int = 50
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
        self.is_dask_model = False  # CatBoost uses disk-backed data
    
    def _identify_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify categorical columns that should be treated as such by CatBoost."""
        cat_cols = []
        
        # Columns with object/category dtype
        cat_cols.extend(X.select_dtypes(include=['object', 'category']).columns.tolist())
        
        # Columns with few unique values (likely categorical)
        for col in X.columns:
            if col not in cat_cols:
                unique_count = X[col].nunique()
                if unique_count < 20 and X[col].dtype in ['int64', 'float64']:
                    cat_cols.append(col)
        
        return cat_cols
    
    def _ensure_string_for_cat_features(self, X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Convert categorical features to string for CatBoost compatibility."""
        X_processed = X.copy()
        
        for col in cat_cols:
            # Convert to string and handle missing
            X_processed[col] = X_processed[col].fillna('MISSING')
            X_processed[col] = X_processed[col].astype(str)
            
            # Replace 'nan' and 'None' with 'MISSING'
            X_processed[col] = X_processed[col].replace('nan', 'MISSING')
            X_processed[col] = X_processed[col].replace('None', 'MISSING')
        
        return X_processed
    
    def _prepare_pool(
        self,
        X: Union[dd.DataFrame, pd.DataFrame],
        y: Optional[Union[dd.Series, pd.Series]] = None,
        is_train: bool = False
    ) -> Pool:
        """Prepare CatBoost Pool from Dask or Pandas data."""
        # Convert to pandas if Dask
        if isinstance(X, dd.DataFrame):
            X = X.compute()
        
        # Ensure numeric columns are float
        for col in X.columns:
            if col not in self.cat_features and pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(float)
        
        # Identify categorical features if not provided
        if self.cat_features is None:
            self.cat_features = self._identify_categorical_columns(X)
            logger.info(f"  Identified {len(self.cat_features)} categorical features")
        
        # Ensure categorical features are strings
        X_processed = self._ensure_string_for_cat_features(X, self.cat_features)
        
        # Handle y
        if y is not None:
            if isinstance(y, dd.Series):
                y = y.compute()
            # Ensure y is numpy array
            y_array = y.values if hasattr(y, 'values') else np.array(y)
        else:
            y_array = None
        
        # Create Pool
        pool = Pool(
            data=X_processed,
            label=y_array,
            cat_features=self.cat_features
        )
        
        return pool
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs
    ):
        """Train CatBoost model."""
        logger.info(f"Training {self.name}...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Prepare training pool
        logger.info("  Preparing training data...")
        train_pool = self._prepare_pool(X_train, y_train, is_train=True)
        
        # Prepare validation pool if provided
        eval_pool = None
        if X_val is not None and y_val is not None:
            logger.info("  Preparing validation data...")
            eval_pool = self._prepare_pool(X_val, y_val, is_train=False)
        
        # Create model
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            auto_class_weights=self.auto_class_weights,
            random_state=self.random_state,
            verbose=self.verbose,
            cat_features=self.cat_features
        )
        
        # Train
        if eval_pool is not None:
            self.model.fit(
                train_pool,
                eval_set=eval_pool,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(train_pool, verbose=False)
        
        # Store feature importance
        self.feature_importance = dict(
            zip(self.feature_names, self.model.feature_importances_)
        )
        
        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare pool
        pool = self._prepare_pool(X, None, is_train=False)
        
        return self.model.predict_proba(pool)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_params(self, deep=True):
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