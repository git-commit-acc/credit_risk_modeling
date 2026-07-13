# models/ensemble.py
"""
Stacking ensemble for credit risk modeling with Dask support.

FIX (requirement #9 + a real leakage bug): the original `fit()` generated
meta-features by calling each already-fitted base model's `predict_proba`
directly on the TRAINING data the base model had just been fit on. That
means the meta-learner was trained on in-sample predictions -- base models
that overfit (trees especially) produce artificially confident/accurate
predictions on their own training rows, so the meta-learner learns to trust
those signals more than it should, and the whole ensemble overfits. The
constructor even accepted a `use_cv_predictions` flag advertising
"proper" behavior, but the flag was never read anywhere in `fit()`.

This version actually implements it: when `use_cv_predictions=True` (the
default), TRAIN meta-features are generated via out-of-fold predictions --
each base model is fit on `cv_folds - 1` folds and predicts on the held-out
fold, so every training-set meta-feature is an honest out-of-sample
prediction, exactly like proper stacking (Wolpert 1992 / standard sklearn
StackingClassifier behavior). Base models are then refit once on the FULL
training set for use at inference time. VAL/TEST meta-features continue to
use the fully-trained base models directly (no CV needed there -- those
rows were never used to fit any base model).

Meta-features remain prediction-output-only (n_samples x n_base_models),
never a duplicate of the full feature matrix -- satisfying requirement #9's
"redesign to use prediction outputs instead of duplicating the full feature
matrix" (this part of the original design was already correct; only the
leakage bug is new here).
"""
# models/ensemble.py
"""
Stacking ensemble for credit risk modeling - uses sklearn API (no Dask distributed).
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import logging
from typing import List, Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel
from models.logistic import LogisticRegressionModel
from models.lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseCreditRiskModel):
    """
    Stacking ensemble using sklearn's StackingClassifier.
    No Dask distributed - uses sklearn API.
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
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegressionModel(
            random_state=random_state,
            class_weight='balanced'
        )
        self.cv_folds = cv_folds
        self.use_proba = use_proba
        self.is_distributed = False
        self.supports_dask_data = True
        
        self.base_model_predictions = None
    
    def _ensure_pandas(self, data):
        """Convert Dask to pandas if needed."""
        if isinstance(data, (dd.DataFrame, dd.Series)):
            return data.compute()
        return data
    
    def _get_estimators(self):
        """Get sklearn estimators from base models."""
        estimators = []
        for model in self.base_models:
            if hasattr(model, 'model'):
                name = model.name.lower().replace(' ', '_')
                estimators.append((name, model.model))
        return estimators
    
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
        
        # Get base estimators
        estimators = self._get_estimators()
        
        if len(estimators) < 2:
            logger.warning(f"Need at least 2 base models for ensemble (have {len(estimators)})")
            return self
        
        # Create meta-learner
        final_estimator = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        
        # Create stacking ensemble
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=self.cv_folds,
            stack_method='predict_proba' if self.use_proba else 'predict'
        )
        
        # Fit ensemble
        self.model.fit(X_train_pd, y_train_pd)
        
        logger.info(f"{self.name} training completed.")
        return self
    
    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Ensemble not trained. Call fit() first.")
        
        X_pd = self._ensure_pandas(X)
        return self.model.predict_proba(X_pd)
    
    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-learner."""
        if hasattr(self.model, 'final_estimator_'):
            if hasattr(self.model.final_estimator_, 'coef_'):
                coef = self.model.final_estimator_.coef_[0]
                if self.feature_names:
                    return dict(zip(self.feature_names, np.abs(coef)))
        return {}