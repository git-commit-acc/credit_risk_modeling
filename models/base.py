# models/base.py
"""
Base model class for credit risk modeling with Dask support.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, Union
import dask.dataframe as dd
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseCreditRiskModel(ABC):
    """Base class for credit risk models with Dask support."""
    
    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        self.is_dask_model = False
    
    @abstractmethod
    def fit(self, X_train: Union[dd.DataFrame, pd.DataFrame], y_train: Union[dd.Series, pd.Series], **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        pass
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if self.feature_importance is not None:
            return self.feature_importance
        return {}
    
    def set_feature_names(self, feature_names: list):
        """Set feature names for the model."""
        self.feature_names = feature_names
    
    def _ensure_pandas(self, data: Union[dd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """Convert Dask to Pandas if needed for non-Dask models."""
        if isinstance(data, dd.DataFrame):
            return data.compute()
        return data
    
    def _ensure_dask(self, data: Union[dd.DataFrame, pd.DataFrame]) -> dd.DataFrame:
        """Convert Pandas to Dask if needed."""
        if isinstance(data, pd.DataFrame):
            return dd.from_pandas(data, npartitions=1)
        return data
    
    def __str__(self):
        return f"{self.name} (Random State: {self.random_state})"