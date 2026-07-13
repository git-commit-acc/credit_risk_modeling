# models/logistic.py
"""
Logistic Regression model for credit risk with Dask support.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression
from dask_ml.preprocessing import StandardScaler

from models.base import BaseCreditRiskModel
from models.dask_utils import LazyCategoricalEncoder, ensure_dask_dataframe

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseCreditRiskModel):
    """Logistic Regression with Dask support (fully lazy feature pipeline)."""

    def __init__(
        self,
        random_state: int = 42,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        solver: str = 'lbfgs',
        npartitions: int = 8,
    ):
        super().__init__("Logistic Regression", random_state)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver if solver in ("admm", "lbfgs", "gradient_descent", "proximal_grad") else "lbfgs"
        self.npartitions = npartitions
        self.scaler: Optional[StandardScaler] = None
        self.cat_encoder: Optional[LazyCategoricalEncoder] = None
        self.constant_cols_: List[str] = []
        self.model = None
        self.is_dask_model = True

    def _preprocess(self, X: Union[dd.DataFrame, pd.DataFrame], fit: bool) -> dd.DataFrame:
        """Lazy categorical encoding + scaling with constant column removal."""
        X_dask = ensure_dask_dataframe(X, npartitions=self.npartitions)

        if fit:
            self.cat_encoder = LazyCategoricalEncoder()
            X_encoded = self.cat_encoder.fit_transform(X_dask)
            
            # Identify constant columns from a sample
            logger.info("  Checking for constant columns...")
            sample = X_encoded.head(1000)
            self.constant_cols_ = []
            for col in X_encoded.columns:
                if col in sample.columns:
                    # Check if column has only one unique value or all NaN
                    unique_vals = sample[col].dropna().unique()
                    if len(unique_vals) <= 1:
                        self.constant_cols_.append(col)
            
            if self.constant_cols_:
                logger.info(f"  Removing {len(self.constant_cols_)} constant columns: {self.constant_cols_[:5]}...")
                X_encoded = X_encoded.drop(columns=self.constant_cols_)
        else:
            X_encoded = self.cat_encoder.transform(X_dask)
            if self.constant_cols_:
                X_encoded = X_encoded.drop(columns=self.constant_cols_, errors='ignore')

        # Ensure everything is numeric before scaling
        X_encoded = X_encoded.astype("float64")
        X_encoded = X_encoded.fillna(0.0)

        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_encoded)
        else:
            X_scaled = self.scaler.transform(X_encoded)

        return X_scaled

    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs,
    ):
        """Train logistic regression model with Dask (out-of-core)."""
        logger.info(f"Training {self.name} with Dask...")

        self.feature_names = list(X_train.columns)

        X_scaled = self._preprocess(X_train, fit=True)
        # Remove constant columns
        X_scaled = self._remove_constant_columns(X_scaled)
        y_dask = ensure_dask_dataframe(y_train, npartitions=self.npartitions)
        y_dask = y_dask.astype("int64")

        # Ensure X and y have consistent partitions
        target_partitions = min(X_scaled.npartitions, y_dask.npartitions)
        X_scaled = X_scaled.repartition(npartitions=target_partitions)
        y_dask = y_dask.repartition(npartitions=target_partitions)

        # Update feature names after removing constant columns
        self.feature_names = [c for c in self.feature_names if c not in self.constant_cols_]

        logger.info(f"  Training with {len(self.feature_names)} features...")

        self.model = DaskLogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
        )
        
        # Convert to dask arrays with consistent lengths
        X_array = X_scaled.to_dask_array(lengths=True)
        y_array = y_dask.to_dask_array(lengths=True)
        
        self.model.fit(X_array, y_array)

        if hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_)
            coef = coef.ravel()
            if len(coef) == len(self.feature_names):
                self.feature_importance = dict(zip(self.feature_names, np.abs(coef)))

        logger.info(f"{self.name} training completed.")
        return self

    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self._preprocess(X, fit=False)
        X_array = X_scaled.to_dask_array(lengths=True)
        proba = self.model.predict_proba(X_array)
        proba = np.asarray(proba.compute()) if hasattr(proba, "compute") else np.asarray(proba)

        if proba.ndim == 1:
            return np.column_stack([1 - proba, proba])
        return proba

    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.feature_importance is not None:
            return self.feature_importance
        return {}

    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if self.model is None or not hasattr(self.model, "coef_"):
            return {}
        coef = np.asarray(self.model.coef_).ravel()
        return dict(zip(self.feature_names, coef))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters for hyperparameter tuning."""
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'solver': self.solver,
            'random_state': self.random_state,
        }
    # models/logistic.py - Add this method to handle constant columns

    def _remove_constant_columns(self, X: dd.DataFrame, threshold: float = 0.999) -> dd.DataFrame:
        """
        Remove columns that are constant or near-constant.
        Uses Dask's variance computation (lazy, no full materialization).
        """
        # Compute variance for each column (lazy operation)
        variances = X.var().compute()
        
        # Find columns with zero or near-zero variance
        constant_cols = []
        for col, var in variances.items():
            if var == 0 or var < 1e-10:
                constant_cols.append(col)
                logger.info(f"  Removing constant column: {col} (variance={var})")
        
        if constant_cols:
            X = X.drop(columns=constant_cols)
            logger.info(f"  Removed {len(constant_cols)} constant columns")
        
        return X