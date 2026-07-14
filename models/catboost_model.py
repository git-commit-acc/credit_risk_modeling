# models/catboost_model.py
"""
CatBoost model for credit risk - uses sklearn API (no Dask distributed).
CatBoost handles categorical features natively.
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
    """CatBoost Classifier using sklearn API (stable, native categorical support)."""
    
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
        cat_features: Optional[List[str]] = None
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
        self.is_distributed = False
        self.supports_dask_data = True
    
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    def _identify_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Identify categorical columns.
        
        FIX: Detect ALL non-numeric dtypes including:
        - object, category (legacy)
        - string (pandas StringDtype)
        - columns with low cardinality
        """
        cat_cols = []
        
        # FIX: Use pandas API to detect ALL string-like types
        for col in X.columns:
            # Check for ANY non-numeric dtype
            if pd.api.types.is_object_dtype(X[col]) or \
               pd.api.types.is_string_dtype(X[col]) or \
               pd.api.types.is_categorical_dtype(X[col]):
                cat_cols.append(col)
            else:
                # Check for low-cardinality numeric columns (categorical in spirit)
                unique_count = X[col].nunique()
                if unique_count < 20 and unique_count > 1:
                    # These are likely categorical flags/status codes
                    cat_cols.append(col)
        
        return cat_cols
    
    def _prepare_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare categorical columns for CatBoost.
        CatBoost requires categorical columns as strings.
        """
        X_prepared = X.copy()
        
        if self.cat_features:
            for col in self.cat_features:
                if col in X_prepared.columns:
                    # Convert to string, fill missing
                    X_prepared[col] = X_prepared[col].fillna('MISSING')
                    X_prepared[col] = X_prepared[col].astype(str)
                    X_prepared[col] = X_prepared[col].replace('nan', 'MISSING')
                    X_prepared[col] = X_prepared[col].replace('None', 'MISSING')
                    X_prepared[col] = X_prepared[col].replace('', 'MISSING')
        
        # Ensure numeric columns are actually numeric
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
        """Train CatBoost model using sklearn API."""
        logger.info(f"Training {self.name} (sklearn API)...")
        
        # Convert to pandas
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        # Identify categorical features
        if self.cat_features is None:
            self.cat_features = self._identify_categorical_columns(X_train_pd)
            logger.info(f"  Identified {len(self.cat_features)} categorical features")
            if self.cat_features:
                logger.info(f"  Categorical features: {self.cat_features[:10]}...")
        
        # Prepare data for CatBoost
        X_train_prepared = self._prepare_categoricals(X_train_pd)
        
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
        
        # Train with validation if provided
        if X_val is not None and y_val is not None:
            X_val_pd = self._ensure_pandas(X_val)
            y_val_pd = self._ensure_pandas(y_val)
            
            # Prepare validation data
            X_val_prepared = self._prepare_categoricals(X_val_pd)
            
            self.model.fit(
                X_train_prepared, y_train_pd,
                eval_set=[(X_val_prepared, y_val_pd)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X_train_prepared, y_train_pd, verbose=False)
        
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
        X_prepared = self._prepare_categoricals(X_pd)
        return self.model.predict_proba(X_prepared)
    
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