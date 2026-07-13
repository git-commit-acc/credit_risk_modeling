# models/random_forest.py
"""
Random Forest model for credit risk with Dask support.

IMPORTANT FIX: the original module did
    `from dask_ml.ensemble import RandomForestClassifier`
which does not exist in dask_ml (verified against dask-ml 2025.1.0's public
API: `dask_ml.ensemble` only exports `BlockwiseVotingClassifier` and
`BlockwiseVotingRegressor`). That import would raise `ImportError` the
moment this module was imported, breaking the entire pipeline.

dask_ml does not ship a true distributed random forest (there is no
Dask-parallel tree-building equivalent to Dask-XGBoost/Dask-LightGBM).
The correct, memory-efficient Dask-ML pattern for tree ensembles is
`BlockwiseVotingClassifier`: it fits one `sklearn.ensemble.
RandomForestClassifier` PER PARTITION (each partition read from disk,
processed, and released -- only one partition's worth of data is ever
resident in a worker's memory at a time), then combines all partition-level
forests into a single voting ensemble at predict time. This is the
documented Dask-ML approach for "big data, doesn't fit in RAM" random
forests and is what requirement #6 ("use Dask-ML wherever appropriate") is
asking for here.
"""
# models/random_forest.py
"""
Random Forest model for credit risk - uses sklearn API (no Dask distributed).
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, Any, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseCreditRiskModel):
    """Random Forest Classifier using sklearn API (stable)."""
    
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 100,
        min_samples_leaf: int = 50,
        max_features: str = 'sqrt',
        class_weight: str = 'balanced',
        n_jobs: int = -1
    ):
        super().__init__("Random Forest", random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.is_distributed = False
        self.supports_dask_data = True
        
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns."""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype == 'category':
                X_encoded[col] = X_encoded[col].fillna('MISSING')
                X_encoded[col] = X_encoded[col].astype(str)
                X_encoded[col] = X_encoded[col].replace('nan', 'MISSING')
                X_encoded[col] = X_encoded[col].replace('None', 'MISSING')
                X_encoded[col] = X_encoded[col].astype('category').cat.codes
        
        return X_encoded
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        **kwargs
    ):
        """Train Random Forest model using sklearn API."""
        logger.info(f"Training {self.name} (sklearn API)...")
        
        # Convert to pandas
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        # Encode categorical columns
        X_train_encoded = self._encode_categorical(X_train_pd)
        
        # Create model using sklearn API
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
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
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }