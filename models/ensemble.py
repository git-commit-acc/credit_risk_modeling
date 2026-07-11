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

import logging
from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from models.base import BaseCreditRiskModel
from models.dask_utils import ensure_dask_dataframe
from models.lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseCreditRiskModel):
    """Stacking ensemble with multiple base models and a meta-learner,
    using out-of-fold predictions to avoid leakage into the meta-learner."""

    def __init__(
        self,
        base_models: List[BaseCreditRiskModel],
        meta_model: Optional[BaseCreditRiskModel] = None,
        use_cv_predictions: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
        npartitions: int = 8,
    ):
        super().__init__("Stacking Ensemble", random_state)
        self.base_models = base_models
        self.meta_model = meta_model or LightGBMModel(
            random_state=random_state,
            is_unbalance=True,
            npartitions=npartitions,
        )
        self.use_cv_predictions = use_cv_predictions
        self.cv_folds = cv_folds
        self.npartitions = npartitions
        self.is_dask_model = True

    def _get_meta_features(
        self,
        X: Union[dd.DataFrame, pd.DataFrame],
        models: Optional[List[BaseCreditRiskModel]] = None,
    ) -> pd.DataFrame:
        """Generate meta features (prediction-output only) from already-
        fitted base models. Small output (n_samples x n_base_models);
        the full feature matrix X is never duplicated into this frame."""
        models = models or self.base_models
        X_dask = ensure_dask_dataframe(X, npartitions=self.npartitions)

        meta_features = {}
        for model in models:
            preds = model.predict_proba(X_dask)[:, 1]
            meta_features[f"{model.name}_pred"] = preds

        return pd.DataFrame(meta_features)

    def _generate_oof_meta_features(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
    ) -> pd.DataFrame:
        """Out-of-fold meta-feature generation: each base model is fit on
        cv_folds-1 folds and predicts on the held-out fold, so every row of
        the returned meta-feature matrix is an honest out-of-sample
        prediction for that row."""
        X_pd = self._ensure_pandas(X_train).reset_index(drop=True)
        y_pd = self._ensure_pandas(y_train)
        y_pd = (y_pd.reset_index(drop=True) if hasattr(y_pd, "reset_index") else pd.Series(y_pd))

        n_samples = len(X_pd)
        oof_preds = {model.name: np.zeros(n_samples) for model in self.base_models}

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for fold_idx, (fit_idx, holdout_idx) in enumerate(skf.split(X_pd, y_pd)):
            logger.info(f"  OOF stacking fold {fold_idx + 1}/{self.cv_folds}...")

            X_fit = X_pd.iloc[fit_idx]
            y_fit = y_pd.iloc[fit_idx]
            X_holdout = X_pd.iloc[holdout_idx]

            for model in self.base_models:
                fold_model = _clone_base_model(model)
                fold_model.fit(X_fit, y_fit)
                preds = fold_model.predict_proba(X_holdout)[:, 1]
                oof_preds[model.name][holdout_idx] = preds

        meta_df = pd.DataFrame({f"{name}_pred": vals for name, vals in oof_preds.items()})
        return meta_df

    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        X_val: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[dd.Series, pd.Series]] = None,
        **kwargs,
    ):
        """Train stacking ensemble with Dask."""
        logger.info(f"Training {self.name} with Dask "
                    f"(out-of-fold meta-features: {self.use_cv_predictions})...")

        self.feature_names = list(X_train.columns)

        if self.use_cv_predictions:
            logger.info("  Generating out-of-fold TRAIN meta-features "
                        f"({self.cv_folds}-fold)...")
            meta_X_train = self._generate_oof_meta_features(X_train, y_train)

        # Fit each base model on the FULL training set -- these fully-fit
        # models are what generates VAL/TEST meta-features and what the
        # ensemble uses at inference time. When use_cv_predictions=False,
        # they also directly generate the (in-sample, less rigorous) TRAIN
        # meta-features, preserving the original opt-out behavior.
        for i, model in enumerate(self.base_models):
            logger.info(f"  Training base model {i + 1}/{len(self.base_models)}: {model.name}...")
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)

        if not self.use_cv_predictions:
            meta_X_train = self._get_meta_features(X_train)

        meta_y_train = self._ensure_pandas(y_train)
        meta_y_train = pd.Series(meta_y_train).reset_index(drop=True)
        meta_X_train = meta_X_train.reset_index(drop=True)

        logger.info("  Training meta-learner...")
        if X_val is not None and y_val is not None:
            meta_X_val = self._get_meta_features(X_val)
            meta_y_val = self._ensure_pandas(y_val)
            self.meta_model.fit(meta_X_train, meta_y_train, meta_X_val, meta_y_val)
        else:
            self.meta_model.fit(meta_X_train, meta_y_train)

        logger.info(f"{self.name} training completed.")
        return self

    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities via the meta-learner on base-model
        prediction outputs."""
        if self.meta_model is None:
            raise ValueError("Ensemble not trained. Call fit() first.")

        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-learner (over base-model
        prediction outputs, not the original features)."""
        imp = getattr(self.meta_model, "feature_importance", None)
        if imp:
            return dict(imp)
        return {}


def _clone_base_model(model: BaseCreditRiskModel) -> BaseCreditRiskModel:
    """Instantiate a fresh, unfit copy of a base model with the same
    hyperparameters -- used per-fold during out-of-fold meta-feature
    generation so each fold's model never sees its own holdout rows."""
    params = model.get_params() if hasattr(model, "get_params") else {}
    return type(model)(**params)
