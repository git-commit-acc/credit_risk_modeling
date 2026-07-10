# models/logistic.py
"""
Logistic Regression model for credit risk with Dask support.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression
from dask_ml.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    """Logistic Regression with Dask support."""
    
    def __init__(
        self,
        random_state: int = 42,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        solver: str = 'lbfgs'
    ):
        self.name = "Logistic Regression"
        self.random_state = random_state
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver
        self.scaler = None
        self.label_encoders = {}
        self.categorical_columns = None
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.is_dask_model = True
        
    def _identify_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify categorical columns in the dataset."""
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Also identify columns with few unique values
        for col in X.columns:
            if col not in cat_cols:
                unique_count = X[col].nunique()
                if unique_count < 20 and X[col].dtype in ['int64', 'float64']:
                    cat_cols.append(col)
        
        return cat_cols
    
    def _encode_categorical(self, X: Union[dd.DataFrame, pd.DataFrame], fit: bool = True) -> Union[dd.DataFrame, pd.DataFrame]:
        """Encode categorical columns with proper handling of unseen labels."""
        is_dask = isinstance(X, dd.DataFrame)
        
        if is_dask:
            # Compute to pandas for categorical encoding (small enough after sampling)
            X_pd = X.compute()
        else:
            X_pd = X.copy()
        
        X_encoded = X_pd.copy()
        
        if self.categorical_columns is None:
            self.categorical_columns = self._identify_categorical_columns(X_pd)
        
        for col in self.categorical_columns:
            if fit:
                # Fit label encoder
                self.label_encoders[col] = LabelEncoder()
                # Convert to string and handle missing
                X_encoded[col] = X_encoded[col].fillna('MISSING').astype(str)
                # Fit and transform
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col])
            else:
                # Transform using existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Convert to string and handle missing
                    X_encoded[col] = X_encoded[col].fillna('MISSING').astype(str)
                    
                    # Get known labels
                    known_labels = set(le.classes_)
                    
                    # Map unseen labels to the first known label
                    first_label = le.classes_[0] if len(le.classes_) > 0 else 'MISSING'
                    
                    def map_label(x):
                        if x in known_labels:
                            return x
                        return first_label
                    
                    X_encoded[col] = X_encoded[col].apply(map_label)
                    X_encoded[col] = le.transform(X_encoded[col])
        
        # Convert to numeric and handle NaN
        X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
        X_encoded = X_encoded.fillna(0)
        
        if is_dask:
            return dd.from_pandas(X_encoded, npartitions=1)
        return X_encoded
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        **kwargs
    ):
        """Train logistic regression model with Dask."""
        logger.info(f"Training {self.name} with Dask...")
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        # Convert to Dask if pandas
        if isinstance(X_train, pd.DataFrame):
            X_train = dd.from_pandas(X_train, npartitions=4)
        if isinstance(y_train, pd.Series):
            y_train = dd.from_pandas(y_train, npartitions=4)
        
        # Encode categorical columns
        X_encoded = self._encode_categorical(X_train, fit=True)
        
        # Scale features with Dask
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_encoded)
        
        # Train Dask Logistic Regression
        self.model = DaskLogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver
        )
        self.model.fit(X_scaled, y_train)
        
        # Store feature importance (coefficients)
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_.compute()
            if len(coef) > 0:
                self.feature_importance = dict(
                    zip(self.feature_names, np.abs(coef[0]))
                )
        
        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Convert to Dask if pandas
        if isinstance(X, pd.DataFrame):
            X = dd.from_pandas(X, npartitions=1)
        
        # Encode categorical columns
        X_encoded = self._encode_categorical(X, fit=False)
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        # Predict and compute
        result = self.model.predict_proba(X_scaled)
        return result.compute()
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.feature_importance is not None:
            return self.feature_importance
        return {}
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if self.model is None:
            return {}
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_.compute()
            return dict(zip(self.feature_names, coef[0]))
        return {}
    
    def get_params(self, deep=True):
        """Get model parameters for hyperparameter tuning."""
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'solver': self.solver,
            'random_state': self.random_state
        }