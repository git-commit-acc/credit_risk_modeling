# models/random_forest.py
"""
Random Forest model for credit risk with Dask support.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask_ml.ensemble import RandomForestClassifier as DaskRandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Dict, Any, List, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseCreditRiskModel):
    """Random Forest Classifier with Dask support."""
    
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 100,
        min_samples_leaf: int = 50,
        max_features: str = 'sqrt',
        class_weight: str = 'balanced',
        n_jobs: int = -1,
        npartitions: int = 4
    ):
        super().__init__("Random Forest", random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.npartitions = npartitions
        self.label_encoders = {}
        self.categorical_columns = None
        self.is_dask_model = True
        
    def _identify_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify categorical columns."""
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in X.columns:
            if col not in cat_cols:
                unique_count = X[col].nunique()
                if unique_count < 20 and X[col].dtype in ['int64', 'float64']:
                    cat_cols.append(col)
        return cat_cols
    
    def _encode_categorical(self, X: Union[dd.DataFrame, pd.DataFrame], fit: bool = True) -> Union[dd.DataFrame, pd.DataFrame]:
        """Encode categorical columns."""
        is_dask = isinstance(X, dd.DataFrame)
        
        if is_dask:
            X_pd = X.compute()
        else:
            X_pd = X.copy()
        
        X_encoded = X_pd.copy()
        
        if self.categorical_columns is None:
            self.categorical_columns = self._identify_categorical_columns(X_pd)
        
        for col in self.categorical_columns:
            X_encoded[col] = X_encoded[col].fillna('MISSING').astype(str)
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(X_encoded[col])
                X_encoded[col] = self.label_encoders[col].transform(X_encoded[col])
            else:
                le = self.label_encoders[col]
                unique_vals = X_encoded[col].unique()
                known_labels = set(le.classes_)
                
                def map_value(x):
                    if x in known_labels:
                        return x
                    if 'MISSING' in known_labels:
                        return 'MISSING'
                    return list(known_labels)[0]
                
                X_encoded[col] = X_encoded[col].apply(map_value)
                X_encoded[col] = le.transform(X_encoded[col])
        
        X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
        X_encoded = X_encoded.fillna(0)
        
        if is_dask:
            return dd.from_pandas(X_encoded, npartitions=self.npartitions)
        return X_encoded
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        **kwargs
    ):
        """Train random forest model with Dask."""
        logger.info(f"Training {self.name} with Dask...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Convert to Dask if pandas
        if isinstance(X_train, pd.DataFrame):
            X_train = dd.from_pandas(X_train, npartitions=self.npartitions)
        if isinstance(y_train, pd.Series):
            y_train = dd.from_pandas(y_train, npartitions=self.npartitions)
        
        # Encode categorical columns
        X_encoded = self._encode_categorical(X_train, fit=True)
        
        # Create Dask Random Forest
        self.model = DaskRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.model.fit(X_encoded, y_train)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_.compute()
            self.feature_importance = dict(
                zip(self.feature_names, importances)
            )
        
        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Convert to Dask if pandas
        if isinstance(X, pd.DataFrame):
            X = dd.from_pandas(X, npartitions=self.npartitions)
        
        X_encoded = self._encode_categorical(X, fit=False)
        result = self.model.predict_proba(X_encoded)
        return result.compute()
    
    def get_params(self, deep=True):
        """Get model parameters for hyperparameter tuning."""
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