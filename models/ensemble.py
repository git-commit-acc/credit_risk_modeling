# models/ensemble.py
"""
Stacking ensemble for credit risk modeling.
Uses sklearn's StackingClassifier with sklearn-compatible models only.
Excludes CatBoost (not cloneable by sklearn due to cat_features).
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import logging
from typing import List, Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


# List of model names that are sklearn-compatible (can be cloned)
SKLEARN_COMPATIBLE_MODELS = ['logistic', 'random_forest', 'xgboost', 'lightgbm']


class StackingEnsemble(BaseCreditRiskModel):
    """
    Stacking ensemble using sklearn's StackingClassifier.
    Only uses sklearn-compatible models (excludes CatBoost).
    """
    
    def __init__(
        self,
        base_models: List[BaseCreditRiskModel],
        meta_model: Optional[BaseCreditRiskModel] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        use_proba: bool = True
    ):
        super().__init__("Stacking Ensemble", random_state)
        
        # Filter to only sklearn-compatible models
        self.sklearn_models = []
        self.base_model_names = []
        
        for model in base_models:
            model_name = model.name.lower().replace(' ', '_')
            
            # Skip CatBoost (not cloneable)
            if 'catboost' in model_name:
                logger.info(f"  Skipping {model.name} for ensemble (not sklearn-cloneable)")
                continue
            
            # Check if model has a fitted sklearn model
            if hasattr(model, 'model') and model.model is not None:
                self.sklearn_models.append((model_name, model.model))
                self.base_model_names.append(model_name)
                logger.info(f"  Adding {model.name} to ensemble")
        
        self.cv_folds = cv_folds
        self.use_proba = use_proba
        self.is_distributed = False
        self.supports_dask_data = True
    
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs
    ):
        """Train stacking ensemble using sklearn API."""
        logger.info(f"Training {self.name} (sklearn API)...")
        
        # Convert to pandas
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        self.feature_names = X_train_pd.columns.tolist()
        
        # Ensure all columns are numeric
        for col in self.feature_names:
            if X_train_pd[col].dtype == 'object':
                X_train_pd[col] = pd.to_numeric(X_train_pd[col], errors='coerce')
        
        X_train_pd = X_train_pd.fillna(0)
        
        if len(self.sklearn_models) < 2:
            logger.warning(f"Need at least 2 sklearn-compatible models for ensemble (have {len(self.sklearn_models)})")
            if len(self.sklearn_models) == 1:
                # Just use the single model
                logger.info("  Using single model as fallback...")
                self.model = self.sklearn_models[0][1]
                
                # Convert to numpy to avoid Dask-ML / Pandas clashes
                self.model.fit(X_train_pd.to_numpy(), y_train_pd.to_numpy())
                logger.info(f"{self.name} training completed (fallback mode).")
                return self
            return self
        
        # Use sklearn's LogisticRegression as meta-learner
        final_estimator = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        
        # Create stacking ensemble
        self.model = StackingClassifier(
            estimators=self.sklearn_models,
            final_estimator=final_estimator,
            cv=self.cv_folds,
            stack_method='predict_proba' if self.use_proba else 'predict',
            n_jobs=-1
        )
        
        # Fit ensemble (Convert to NumPy arrays here!)
        X_train_np = X_train_pd.to_numpy()
        y_train_np = y_train_pd.to_numpy()
        self.model.fit(X_train_np, y_train_np)
        
        logger.info(f"{self.name} training completed with {len(self.sklearn_models)} base models.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Ensemble not trained. Call fit() first.")
        
        X_pd = self._ensure_pandas(X)
        
        # Ensure all columns are numeric
        for col in X_pd.columns:
            if X_pd[col].dtype == 'object':
                X_pd[col] = pd.to_numeric(X_pd[col], errors='coerce')
        
        X_pd = X_pd.fillna(0)
        
        # Convert to numpy to avoid Dask-ML signature errors during prediction
        X_np = X_pd.to_numpy()
        
        # If model is a single estimator (fallback)
        if not hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_np)
        
        return self.model.predict_proba(X_np)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-learner."""
        if hasattr(self.model, 'final_estimator_'):
            if hasattr(self.model.final_estimator_, 'coef_'):
                coef = self.model.final_estimator_.coef_[0]
                feature_names = [name for name, _ in self.model.estimators]
                if len(coef) == len(feature_names):
                    return dict(zip(feature_names, np.abs(coef)))
        return {}
    
    def get_params(self, deep=True):
        """Get model parameters."""
        return {
            'cv_folds': self.cv_folds,
            'use_proba': self.use_proba,
            'random_state': self.random_state
        }