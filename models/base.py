# models/base.py
"""
Base model class for credit risk modeling.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Union
import dask.dataframe as dd
import pandas as pd
import numpy as np

from models.dask_utils import ensure_dask_dataframe

logger = logging.getLogger(__name__)


class BaseCreditRiskModel(ABC):
    """Base class for credit risk models."""

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        self.is_dask_model = False

    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Predict probabilities."""
        pass

    def predict(self, X):
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        return self.feature_importance if self.feature_importance else {}

    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data

    def _ensure_dask(self, data, npartitions: int = 8):
        """Convert pandas to Dask if needed."""
        return ensure_dask_dataframe(data, npartitions=npartitions)

    def __str__(self):
        return f"{self.name} (Random State: {self.random_state})"