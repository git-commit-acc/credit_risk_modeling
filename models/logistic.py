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
        solver: str = 'admm',
        npartitions: int = 8,
    ):
        super().__init__("Logistic Regression", random_state)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        # dask_ml.linear_model.LogisticRegression is solved with dask-glm
        # optimizers; 'admm' and 'lbfgs' are supported out-of-core solvers.
        # sklearn solver names like 'liblinear'/'newton-cg' are NOT valid
        # here -- silently mapped to 'lbfgs' to stay backward compatible
        # with configs/tuned_params.json written against the old sklearn-
        # flavored search space.
        self.solver = solver if solver in ("admm", "lbfgs", "gradient_descent", "proximal_grad") else "lbfgs"
        self.npartitions = npartitions
        self.scaler: Optional[StandardScaler] = None
        self.cat_encoder: Optional[LazyCategoricalEncoder] = None
        self.model = None
        self.is_dask_model = True

    def _preprocess(self, X: Union[dd.DataFrame, pd.DataFrame], fit: bool) -> dd.DataFrame:
        """Lazy categorical encoding + scaling. Never materializes the full
        feature matrix to pandas -- both the Categorizer/OrdinalEncoder
        (dask_ml) and StandardScaler (dask_ml) operate partition-wise."""
        X_dask = ensure_dask_dataframe(X, npartitions=self.npartitions)

        if fit:
            self.cat_encoder = LazyCategoricalEncoder()
            X_encoded = self.cat_encoder.fit_transform(X_dask)
        else:
            X_encoded = self.cat_encoder.transform(X_dask)

        # Ensure everything is numeric before scaling; any leftover object
        # columns become NaN -> 0, matching the original semantics without
        # an eager compute (astype/fillna are lazy, partition-wise ops).
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
        y_dask = ensure_dask_dataframe(y_train, npartitions=self.npartitions)
        y_dask = y_dask.astype("int64")

        self.model = DaskLogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
        )
        self.model.fit(X_scaled.to_dask_array(lengths=True), y_dask.to_dask_array(lengths=True))

        if hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_)
            coef = coef.ravel()
            if len(coef) == len(self.feature_names):
                self.feature_importance = dict(zip(self.feature_names, np.abs(coef)))

        logger.info(f"{self.name} training completed.")
        return self

    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities. Computes only the (n_samples, 2) output
        array, never the input feature matrix."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self._preprocess(X, fit=False)
        proba = self.model.predict_proba(X_scaled.to_dask_array(lengths=True))
        proba = np.asarray(proba.compute()) if hasattr(proba, "compute") else np.asarray(proba)

        # dask_ml's GLM-based LogisticRegression.predict_proba already
        # returns shape (n_samples, 2); guard for older/edge-case builds
        # that return the positive-class probability as a 1-D vector.
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
