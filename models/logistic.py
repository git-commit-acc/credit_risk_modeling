# models/logistic.py
"""
Logistic Regression model for credit risk using sklearn API.
No Dask distributed - loads data once.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseCreditRiskModel):
    """Logistic Regression using sklearn API (stable, no Dask-ML issues)."""

    def __init__(
        self,
        random_state: int = 42,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        solver: str = 'lbfgs',
        n_jobs: int = -1
    ):
        super().__init__("Logistic Regression", random_state)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver if solver in ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga') else 'lbfgs'
        self.n_jobs = n_jobs
        self.scaler = None
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
        """Train Logistic Regression using sklearn."""
        logger.info(f"Training {self.name}...")
        
        # Convert to pandas
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        logger.info(f"  Training with {len(X_train_pd):,} samples, {len(self.feature_names)} features")
        
        # Encode categorical columns
        X_train_encoded = self._encode_categorical(X_train_pd)
        
        # Remove constant columns
        constant_cols = []
        for col in X_train_encoded.columns:
            if X_train_encoded[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            logger.info(f"  Removing {len(constant_cols)} constant columns: {constant_cols[:5]}...")
            X_train_encoded = X_train_encoded.drop(columns=constant_cols)
            self.feature_names = [c for c in self.feature_names if c not in constant_cols]
        
        # FIX: Handle NaN values - fill with 0
        if X_train_encoded.isna().any().any():
            logger.info("  Filling NaN values with 0...")
            X_train_encoded = X_train_encoded.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_encoded)
        
        # Train model
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X_train_scaled, y_train_pd)

        # Feature importance (coefficients)
        self.feature_importance = dict(
            zip(self.feature_names, np.abs(self.model.coef_[0]))
        )

        logger.info(f"{self.name} training completed.")
        return self

    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_pd = self._ensure_pandas(X)
        X_encoded = self._encode_categorical(X_pd)
        
        # Remove constant columns (same as during fit)
        if hasattr(self, 'feature_names'):
            X_encoded = X_encoded[self.feature_names]
        
        # Fill NaN with 0
        X_encoded = X_encoded.fillna(0)
        
        X_scaled = self.scaler.transform(X_encoded)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        return self.feature_importance if self.feature_importance else {}

    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if self.model is None or not hasattr(self.model, "coef_"):
            return {}
        return dict(zip(self.feature_names, self.model.coef_[0]))

    def get_params(self, deep=True) -> Dict[str, Any]:
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'solver': self.solver,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }