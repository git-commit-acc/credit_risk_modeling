# models/random_forest.py
"""
Random Forest model for credit risk using sklearn API.
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
    """Random Forest Classifier using sklearn API."""
    
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
    
    def _ensure_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns are numeric."""
        X_numeric = X.copy()
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'object':
                X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
        return X_numeric.fillna(0)
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        **kwargs
    ):
        """Train Random Forest model."""
        logger.info(f"Training {self.name}...")
        
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        X_train_clean = self._ensure_numeric(X_train_pd)
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X_train_clean, y_train_pd)

        self.feature_importance = dict(
            zip(self.feature_names, self.model.feature_importances_)
        )

        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_pd = self._ensure_pandas(X)
        X_clean = self._ensure_numeric(X_pd)
        return self.model.predict_proba(X_clean)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        return self.feature_importance if self.feature_importance else {}
    
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