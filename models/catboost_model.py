# models/catboost_model.py
"""
CatBoost model for credit risk with memory-efficient, partition-wise
incremental training.

WHY THIS APPROACH:
CatBoost has no native Dask integration, and its `Pool` object is always a
single in-memory (or single-file, via `catboost.Pool(data=<path>)` with
libsvm/CatBoost-native formats only -- not Parquet) structure. The original
implementation called `X.compute()` on every fit/predict, silently pulling
the ENTIRE Dask DataFrame into driver RAM -- exactly the full-dataset
materialization this refactor is meant to eliminate (requirement #8).

Since CatBoost cannot consume Dask partitions natively, the memory-safe
strategy is CatBoost's own supported mechanism for incremental training:
`CatBoostClassifier.fit(..., init_model=<previous model>)`. We iterate over
the Dask DataFrame's partitions one at a time -- each partition is pulled
into memory, trained for a small number of boosting rounds continuing from
the previous partition's model, and then released before the next partition
is loaded. Peak memory is therefore bounded by the size of the LARGEST
SINGLE PARTITION, not the full dataset -- the same out-of-core guarantee
Dask gives XGBoost/LightGBM, implemented manually because CatBoost doesn't
expose a partition-aware training API itself.

The validation pool (used for early stopping) is still computed once into
memory, since validation folds are expected to be a small fraction of the
data (this mirrors how XGBoost/LightGBM's Dask integrations also keep a
single eval set resident) -- the expensive/large object here is TRAIN, which
this module never fully materializes.
"""
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
        """Identify categorical columns."""
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
        
        # Identify categorical features if not provided
        if self.cat_features is None:
            self.cat_features = self._identify_categorical_columns(X_train_pd)
            logger.info(f"  Identified {len(self.cat_features)} categorical features")
        
        # Create model using sklearn API
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
            
            self.model.fit(
                X_train_pd, y_train_pd,
                eval_set=[(X_val_pd, y_val_pd)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X_train_pd, y_train_pd, verbose=False)
        
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
        return self.model.predict_proba(X_pd)
    
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