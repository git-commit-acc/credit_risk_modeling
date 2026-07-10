# models/ensemble.py
"""
Stacking ensemble for credit risk modeling with Dask support.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split as dask_train_test_split
import logging
from typing import List, Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel
from models.logistic import LogisticRegressionModel
from models.lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseCreditRiskModel):
    """
    Stacking ensemble with multiple base models and a meta-learner with Dask support.
    """
    
    def __init__(
        self,
        base_models: List[BaseCreditRiskModel],
        meta_model: Optional[BaseCreditRiskModel] = None,
        use_cv_predictions: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
        npartitions: int = 4
    ):
        super().__init__("Stacking Ensemble", random_state)
        self.base_models = base_models
        self.meta_model = meta_model or LightGBMModel(
            random_state=random_state,
            is_unbalance=True,
            npartitions=npartitions
        )
        self.use_cv_predictions = use_cv_predictions
        self.cv_folds = cv_folds
        self.npartitions = npartitions
        self.is_dask_model = True
        
        self.base_model_predictions = None
        
    def _ensure_dask(self, data: Union[dd.DataFrame, pd.DataFrame]) -> dd.DataFrame:
        """Convert to Dask if needed."""
        if isinstance(data, pd.DataFrame):
            return dd.from_pandas(data, npartitions=self.npartitions)
        return data
    
    def _ensure_pandas(self, data: Union[dd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """Convert to pandas if needed."""
        if isinstance(data, dd.DataFrame):
            return data.compute()
        return data
    
    def _get_meta_features(
        self,
        X: Union[dd.DataFrame, pd.DataFrame],
        y: Optional[Union[dd.Series, pd.Series]] = None
    ) -> pd.DataFrame:
        """Generate meta features from base model predictions."""
        # Convert to Dask
        X_dask = self._ensure_dask(X)
        
        meta_features = []
        
        for model in self.base_models:
            # Get predictions from base model
            preds = model.predict_proba(X_dask)[:, 1]
            meta_features.append(pd.Series(preds, name=model.name))
        
        meta_df = pd.concat(meta_features, axis=1)
        meta_df.columns = [f"{m.name}_pred" for m in self.base_models]
        return meta_df
    
    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs
    ):
        """Train stacking ensemble with Dask."""
        logger.info(f"Training {self.name} with Dask...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Train base models
        for i, model in enumerate(self.base_models):
            logger.info(f"  Training base model {i+1}/{len(self.base_models)}: {model.name}...")
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)
        
        # Generate meta features
        logger.info("  Generating meta features...")
        meta_X_train = self._get_meta_features(X_train, y_train)
        meta_y_train = self._ensure_pandas(y_train) if isinstance(y_train, (dd.Series, pd.Series)) else y_train
        
        # Train meta-learner
        logger.info("  Training meta-learner...")
        if X_val is not None and y_val is not None:
            meta_X_val = self._get_meta_features(X_val)
            meta_y_val = self._ensure_pandas(y_val) if isinstance(y_val, (dd.Series, pd.Series)) else y_val
            self.meta_model.fit(meta_X_train, meta_y_train, meta_X_val, meta_y_val)
        else:
            self.meta_model.fit(meta_X_train, meta_y_train)
        
        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.meta_model is None:
            raise ValueError("Ensemble not trained. Call fit() first.")
        
        # Get base model predictions
        meta_features = self._get_meta_features(X)
        
        # Get meta-learner predictions
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-learner."""
        if hasattr(self.meta_model, 'feature_importance'):
            imp = self.meta_model.feature_importance
            if imp:
                return {
                    f"{model.name}_pred": list(imp.values())[i] 
                    for i, model in enumerate(self.base_models)
                }
        return {}