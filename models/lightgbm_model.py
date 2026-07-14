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
# models/lightgbm_model.py
"""
LightGBM model for credit risk - uses sklearn API (no Dask distributed).
This avoids socket pickling issues.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import lightgbm as lgb
import logging
from typing import Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseCreditRiskModel):
    """LightGBM Classifier using sklearn API (stable, no socket pickling issues)."""
    
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,  # Reduced for stability
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        is_unbalance: bool = True,
        verbosity: int = -1,
        early_stopping_rounds: int = 50,
        n_jobs: int = -1
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
        self.is_distributed = False  # Not using Dask distributed
        self.supports_dask_data = True  # Will convert to pandas
    
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    # def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
    #     """Encode categorical columns for LightGBM."""
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
        """Encode categorical columns for LightGBM."""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            # Use pandas API to detect ALL string-like types
            if pd.api.types.is_object_dtype(X_encoded[col]) or \
            pd.api.types.is_string_dtype(X_encoded[col]) or \
            pd.api.types.is_categorical_dtype(X_encoded[col]):
                
                X_encoded[col] = X_encoded[col].fillna('MISSING')
                X_encoded[col] = X_encoded[col].astype(str)
                X_encoded[col] = X_encoded[col].replace('nan', 'MISSING')
                X_encoded[col] = X_encoded[col].replace('None', 'MISSING')
                X_encoded[col] = X_encoded[col].replace('', 'MISSING')
                
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
        """Train LightGBM model using sklearn API."""
        logger.info(f"Training {self.name} (sklearn API)...")
        
        # Convert to pandas
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        # Encode categorical columns
        X_train_encoded = self._encode_categorical(X_train_pd)
        
        # Create model - verbosity controls output
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            is_unbalance=self.is_unbalance,
            random_state=self.random_state,
            verbosity=-1,  # -1 = silent, 0 = warning, 1 = info
            n_jobs=self.n_jobs
        )
        
        # Train with validation if provided
        if X_val is not None and y_val is not None:
            X_val_pd = self._ensure_pandas(X_val)
            y_val_pd = self._ensure_pandas(y_val)
            X_val_encoded = self._encode_categorical(X_val_pd)
            
            self.model.fit(
                X_train_encoded, y_train_pd,
                eval_set=[(X_val_encoded, y_val_pd)],
                eval_metric='logloss',
                callbacks=[lgb.early_stopping(self.early_stopping_rounds)]
                # FIX: removed 'verbose' from here - use verbosity in constructor
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
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Convert to pandas and encode
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
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'is_unbalance': self.is_unbalance,
            'random_state': self.random_state,
            'verbosity': self.verbosity,
            'n_jobs': self.n_jobs
        }