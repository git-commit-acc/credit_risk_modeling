# models/hyperparameter_tuning.py
"""
Hyperparameter tuning for credit risk models using Optuna with Dask.

FIX (requirement #13, "remove duplication and simplify preprocessing" +
a real train/tune skew bug): the original module hand-rolled its own
categorical-encoding routine (`_preprocess_for_sklearn`, built on
`sklearn.preprocessing.LabelEncoder` per column) that was completely
separate from -- and inconsistent with -- the preprocessing each model
class now does internally:

  - LogisticRegressionModel / RandomForestModel ordinal-encode via the
    shared `LazyCategoricalEncoder` (models/dask_utils.py).
  - XGBoostModel / LightGBMModel cast to native pandas `category` dtype
    (LazyCategoricalEncoder(ordinal_encode=False)) and use each library's
    native categorical split-finding -- NOT integer/label encoding.
  - CatBoostModel keeps categoricals as strings and passes them through
    CatBoost's own `cat_features` mechanism.

Tuning against a hand-rolled LabelEncoder representation while production
training uses a completely different representation for 3 of the 5 models
means Optuna could easily select hyperparameters that are optimal for a
feature representation the model never actually trains on -- an invisible
train/tune skew. This version removes the duplicate preprocessing entirely
and evaluates every trial through the SAME model class `fit()` /
`predict_proba()` that main.py's training module calls, so tuning always
measures what will actually be deployed.

Also fixes a real (if silent) bug: `tune_logistic_regression`'s solver
search space suggested `'lbfgs' | 'liblinear' | 'newton-cg'` -- valid
scikit-learn solver names, but `LogisticRegressionModel` wraps
`dask_ml.linear_model.LogisticRegression`, whose supported out-of-core
solvers are `'admm' | 'lbfgs' | 'gradient_descent' | 'proximal_grad'`.
`'liblinear'`/`'newton-cg'` were silently remapped to `'lbfgs'` inside the
model (see `LogisticRegressionModel.__init__`), so 2 of every 3 solver
choices Optuna explored were actually identical trials -- wasted budget and
a misleading "best solver" result. The search space here now only offers
values `LogisticRegressionModel` actually respects.

Tuning still operates on a bounded, sampled slice of the data
(`sample_fraction`, default 10%) -- never the full 47M-row panel -- via
`_sample_data()`, which is the one place a `.compute()` remains: it is
computing a fraction-of-a-percent SAMPLE for Optuna's in-memory CV loop,
not the full feature matrix, matching the "minimal RAM usage" requirement.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
import dask.dataframe as dd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from models.catboost_model import CatBoostModel
from models.lightgbm_model import LightGBMModel
from models.logistic import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# Small partition count for tuning trials -- CV folds during tuning operate
# on an already-sampled, in-memory pandas slice, so there is no benefit to
# the default 8-way partitioning used for full training; fewer partitions
# means less Dask scheduling overhead per (trial x fold) fit.
_TUNING_NPARTITIONS = 2


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna. Every trial fits/evaluates the
    real `models/*.py` model classes on a sampled, in-memory slice of the
    data, so tuned hyperparameters always match the preprocessing the
    model will actually use in full training."""

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        timeout: int = 3600,
        direction: str = 'maximize',
        sample_fraction: float = 0.1,  # Sample fraction for tuning
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.timeout = timeout
        self.direction = direction
        self.sample_fraction = sample_fraction

        self.best_params: Dict[str, Any] = {}
        self.best_score: Optional[float] = None
        self.study: Optional[optuna.Study] = None

    def _sample_data(self, X: Union[dd.DataFrame, pd.DataFrame], y: Union[dd.Series, pd.Series]):
        """Sample a bounded slice of the data for tuning. This is the only
        place `.compute()` is called in this module, and it is computing a
        SAMPLE (sample_fraction, default 10%), never the full dataset."""
        if isinstance(X, dd.DataFrame):
            X_sampled = X.sample(frac=self.sample_fraction, random_state=self.random_state)
            y_sampled = y.sample(frac=self.sample_fraction, random_state=self.random_state)
            X_sampled = X_sampled.compute()
            y_sampled = y_sampled.compute()
        else:
            n_samples = int(len(X) * self.sample_fraction)
            indices = np.random.RandomState(self.random_state).choice(
                len(X), size=min(n_samples, len(X)), replace=False
            )
            X_sampled = X.iloc[indices]
            y_sampled = y.iloc[indices]

        return X_sampled.reset_index(drop=True), pd.Series(y_sampled).reset_index(drop=True)

    def _cross_val_score(self, model_cls, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> float:
        """K-fold CV using the model's own fit()/predict_proba(), so tuning
        always evaluates the exact preprocessing + training path that
        main.py's training module will use for the full dataset."""
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            try:
                X_fit, y_fit = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                model = model_cls(random_state=self.random_state, npartitions=_TUNING_NPARTITIONS, **params)
                model.fit(X_fit, y_fit)
                y_pred = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, y_pred))
            except Exception as e:
                logger.warning(f"  CV fold {fold_idx} failed: {e}")
                continue

        return float(np.mean(scores)) if scores else 0.0

    def _run_study(self, study_name: str, objective) -> Dict[str, Any]:
        self.study = optuna.create_study(direction=self.direction, study_name=study_name)
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Best {study_name} params: {self.best_params}")
        logger.info(f"Best CV score: {self.best_score:.4f}")
        return self.best_params

    def tune_logistic_regression(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
    ) -> Dict[str, Any]:
        """Tune Logistic Regression hyperparameters."""
        logger.info("Tuning Logistic Regression...")
        X_sample, y_sample = self._sample_data(X_train, y_train)

        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                # Valid out-of-core solvers for dask_ml's GLM-based
                # LogisticRegression -- see module docstring.
                'solver': trial.suggest_categorical(
                    'solver', ['admm', 'lbfgs', 'gradient_descent', 'proximal_grad']
                ),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            }
            return self._cross_val_score(LogisticRegressionModel, params, X_sample, y_sample)

        return self._run_study('logistic_regression', objective)

    def tune_random_forest(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
    ) -> Dict[str, Any]:
        """Tune Random Forest (blockwise-voting) hyperparameters."""
        logger.info("Tuning Random Forest...")
        X_sample, y_sample = self._sample_data(X_train, y_train)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 200, step=10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100, step=5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical(
                    'class_weight', ['balanced', 'balanced_subsample', None]
                ),
            }
            return self._cross_val_score(RandomForestModel, params, X_sample, y_sample)

        return self._run_study('random_forest', objective)

    def tune_xgboost(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
    ) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters."""
        logger.info("Tuning XGBoost...")
        X_sample, y_sample = self._sample_data(X_train, y_train)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
                # 'exact' does not support native categorical splits (see
                # xgboost_model.py); only offer tree methods that do.
                'tree_method': trial.suggest_categorical('tree_method', ['hist', 'approx']),
            }
            return self._cross_val_score(XGBoostModel, params, X_sample, y_sample)

        return self._run_study('xgboost', objective)

    def tune_lightgbm(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
    ) -> Dict[str, Any]:
        """Tune LightGBM hyperparameters."""
        logger.info("Tuning LightGBM...")
        X_sample, y_sample = self._sample_data(X_train, y_train)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
            }
            return self._cross_val_score(LightGBMModel, params, X_sample, y_sample)

        return self._run_study('lightgbm', objective)

    def tune_catboost(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series],
    ) -> Dict[str, Any]:
        """Tune CatBoost hyperparameters."""
        logger.info("Tuning CatBoost...")
        X_sample, y_sample = self._sample_data(X_train, y_train)

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500, step=50),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 50, 255),
                'auto_class_weights': trial.suggest_categorical(
                    'auto_class_weights', ['Balanced', 'SqrtBalanced', None]
                ),
                'verbose': False,
            }
            return self._cross_val_score(CatBoostModel, params, X_sample, y_sample)

        return self._run_study('catboost', objective)
