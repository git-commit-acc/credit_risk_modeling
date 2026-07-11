# models/random_forest.py
"""
Random Forest model for credit risk with Dask support.

IMPORTANT FIX: the original module did
    `from dask_ml.ensemble import RandomForestClassifier`
which does not exist in dask_ml (verified against dask-ml 2025.1.0's public
API: `dask_ml.ensemble` only exports `BlockwiseVotingClassifier` and
`BlockwiseVotingRegressor`). That import would raise `ImportError` the
moment this module was imported, breaking the entire pipeline.

dask_ml does not ship a true distributed random forest (there is no
Dask-parallel tree-building equivalent to Dask-XGBoost/Dask-LightGBM).
The correct, memory-efficient Dask-ML pattern for tree ensembles is
`BlockwiseVotingClassifier`: it fits one `sklearn.ensemble.
RandomForestClassifier` PER PARTITION (each partition read from disk,
processed, and released -- only one partition's worth of data is ever
resident in a worker's memory at a time), then combines all partition-level
forests into a single voting ensemble at predict time. This is the
documented Dask-ML approach for "big data, doesn't fit in RAM" random
forests and is what requirement #6 ("use Dask-ML wherever appropriate") is
asking for here.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.ensemble import BlockwiseVotingClassifier
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier

from models.base import BaseCreditRiskModel
from models.dask_utils import LazyCategoricalEncoder, ensure_dask_dataframe

logger = logging.getLogger(__name__)


class RandomForestModel(BaseCreditRiskModel):
    """Random Forest Classifier, distributed across Dask partitions via
    blockwise voting (one sub-forest trained per partition, out-of-core)."""

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
        npartitions: int = 8,
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
        self.cat_encoder: Optional[LazyCategoricalEncoder] = None
        self.is_dask_model = True

    def _preprocess(self, X: Union[dd.DataFrame, pd.DataFrame], fit: bool) -> dd.DataFrame:
        """Lazy categorical encoding shared with logistic.py -- see
        models/dask_utils.py. Never calls `.compute()` on the full matrix."""
        X_dask = ensure_dask_dataframe(X, npartitions=self.npartitions)

        if fit:
            self.cat_encoder = LazyCategoricalEncoder()
            X_encoded = self.cat_encoder.fit_transform(X_dask)
        else:
            X_encoded = self.cat_encoder.transform(X_dask)

        X_encoded = X_encoded.astype("float64").fillna(0.0)
        return X_encoded

    def fit(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
        **kwargs,
    ):
        """Train blockwise-voting random forest with Dask (out-of-core:
        each worker only ever holds one partition of X_train in memory)."""
        logger.info(f"Training {self.name} with Dask (blockwise voting)...")

        self.feature_names = list(X_train.columns)

        X_encoded = self._preprocess(X_train, fit=True)
        y_dask = ensure_dask_dataframe(y_train, npartitions=self.npartitions).astype("int64")

        # Re-partition X and y consistently so each partition pair aligns
        # (BlockwiseVotingClassifier fits one estimator per aligned
        # partition pair of X, y).
        X_encoded = X_encoded.repartition(npartitions=self.npartitions)
        y_dask = y_dask.repartition(npartitions=self.npartitions)

        base_estimator = SkRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model = BlockwiseVotingClassifier(
            base_estimator,
            voting="soft",
            classes=[0, 1],
        )
        self.model.fit(X_encoded, y_dask)

        # Average feature_importances_ across the per-partition forests --
        # BlockwiseVotingClassifier exposes the fitted sub-estimators via
        # `.estimators_`.
        try:
            importances = np.mean(
                [est.feature_importances_ for est in self.model.estimators_], axis=0
            )
            self.feature_importance = dict(zip(self.feature_names, importances))
        except Exception as e:
            logger.warning(f"  Could not aggregate feature importances: {e}")

        logger.info(f"{self.name} training completed "
                    f"({self.npartitions} partition-level forests).")
        return self

    def predict_proba(self, X: Union[dd.DataFrame, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities. Only the (n_samples, 2) prediction array
        is computed -- the input feature matrix stays partitioned."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_encoded = self._preprocess(X, fit=False)
        result = self.model.predict_proba(X_encoded)
        if hasattr(result, "compute"):
            result = result.compute()
        return np.asarray(result)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters for hyperparameter tuning."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }
