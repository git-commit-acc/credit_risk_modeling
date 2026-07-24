# models/ensemble.py
"""
Stacking ensemble for credit risk modeling using sklearn.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import logging
from typing import List, Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseCreditRiskModel):
    """Stacking ensemble using sklearn's StackingClassifier."""
    
    def __init__(
        self,
        base_models: List[BaseCreditRiskModel],
        meta_model: Optional[BaseCreditRiskModel] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        use_proba: bool = True
    ):
        super().__init__("Stacking Ensemble", random_state)
        
        self.sklearn_models = []
        
        for model in base_models:
            model_name = model.name.lower().replace(' ', '_')
            
            if 'catboost' in model_name:
                logger.info(f"  Skipping {model.name} for ensemble")
                continue
            
            if hasattr(model, 'model') and model.model is not None:
                self.sklearn_models.append((model_name, model.model))
        
        self.cv_folds = cv_folds
        self.use_proba = use_proba
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
        """Train stacking ensemble."""
        logger.info(f"Training {self.name}...")
        
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        X_train_encoded = self._encode_categorical(X_train_pd)
        X_train_encoded = X_train_encoded.fillna(0)
        
        if len(self.sklearn_models) < 2:
            logger.warning(f"Need at least 2 models for ensemble (have {len(self.sklearn_models)})")
            if len(self.sklearn_models) == 1:
                logger.info("Using single model as fallback...")
                self.model = self.sklearn_models[0][1]
                self.model.fit(X_train_encoded, y_train_pd)
            return self
        
        final_estimator = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        
        self.model = StackingClassifier(
            estimators=self.sklearn_models,
            final_estimator=final_estimator,
            cv=self.cv_folds,
            stack_method='predict_proba' if self.use_proba else 'predict',
            n_jobs=-1
        )
        
        self.model.fit(X_train_encoded, y_train_pd)
        
        logger.info(f"{self.name} completed with {len(self.sklearn_models)} base models.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Ensemble not trained. Call fit() first.")
        
        X_pd = self._ensure_pandas(X)
        X_encoded = self._encode_categorical(X_pd)
        X_encoded = X_encoded.fillna(0)
        
        if not hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_encoded)
        
        return self.model.predict_proba(X_encoded)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if hasattr(self.model, 'final_estimator_'):
            if hasattr(self.model.final_estimator_, 'coef_'):
                coef = self.model.final_estimator_.coef_[0]
                feature_names = [name for name, _ in self.model.estimators]
                if len(coef) == len(feature_names):
                    return dict(zip(feature_names, np.abs(coef)))
        return {}