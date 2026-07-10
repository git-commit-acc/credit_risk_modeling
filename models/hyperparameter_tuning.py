# models/hyperparameter_tuning.py
"""
Hyperparameter tuning for credit risk models using Optuna with Dask.
"""

import optuna
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split as dask_train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Dict, Any, Optional, List, Union
import json

from models.logistic import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna with Dask support."""
    
    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        timeout: int = 3600,
        direction: str = 'maximize',
        sample_fraction: float = 0.1  # Sample fraction for tuning
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.timeout = timeout
        self.direction = direction
        self.sample_fraction = sample_fraction
        
        self.best_params = {}
        self.best_score = None
        self.study = None
        self.categorical_columns = None
    
    def _sample_data(self, X: Union[dd.DataFrame, pd.DataFrame], y: Union[dd.Series, pd.Series]):
        """Sample data for tuning to reduce computation."""
        if isinstance(X, dd.DataFrame):
            # Sample from Dask
            X_sampled = X.sample(frac=self.sample_fraction, random_state=self.random_state)
            y_sampled = y.sample(frac=self.sample_fraction, random_state=self.random_state)
            # Compute to pandas for Optuna
            X_sampled = X_sampled.compute()
            y_sampled = y_sampled.compute()
        else:
            # Already pandas, sample directly
            n_samples = int(len(X) * self.sample_fraction)
            indices = np.random.RandomState(self.random_state).choice(
                len(X), size=min(n_samples, len(X)), replace=False
            )
            X_sampled = X.iloc[indices]
            y_sampled = y.iloc[indices]
        
        return X_sampled, y_sampled
    
    def _get_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify categorical columns in the dataset."""
        if self.categorical_columns is not None:
            return self.categorical_columns
        
        # Find columns with object/category dtype
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Also find columns with few unique values (likely categorical)
        for col in X.columns:
            if col not in cat_cols:
                unique_count = X[col].nunique()
                if unique_count < 20 and X[col].dtype in ['int64', 'float64']:
                    cat_cols.append(col)
        
        self.categorical_columns = cat_cols
        return cat_cols
    
    def _preprocess_for_sklearn(self, X: pd.DataFrame, 
                                cat_columns: List[str] = None) -> pd.DataFrame:
        """Preprocess DataFrame for sklearn models by encoding categorical variables."""
        if cat_columns is None:
            cat_columns = self._get_categorical_columns(X)
        
        X_processed = X.copy()
        
        # Encode categorical columns
        for col in cat_columns:
            le = LabelEncoder()
            # Handle NaN values
            X_processed[col] = X_processed[col].fillna('MISSING')
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Convert all to numeric and handle remaining NaN
        X_processed = X_processed.apply(pd.to_numeric, errors='coerce')
        X_processed = X_processed.fillna(0)
        
        return X_processed
    
    def tune_logistic_regression(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series]
    ) -> Dict[str, Any]:
        """Tune Logistic Regression hyperparameters."""
        logger.info("Tuning Logistic Regression...")
        
        # Sample data for tuning
        X_sample, y_sample = self._sample_data(X_train, y_train)
        
        # Preprocess categorical columns
        cat_cols = self._get_categorical_columns(X_sample)
        X_processed = self._preprocess_for_sklearn(X_sample, cat_cols)
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
            
            model = LogisticRegressionModel(
                random_state=self.random_state,
                **params
            )
            
            score = self._cross_val_score_sklearn(model, X_processed, y_sample)
            return score
        
        self.study = optuna.create_study(
            direction=self.direction,
            study_name='logistic_regression'
        )
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Best Logistic Regression params: {self.best_params}")
        logger.info(f"Best CV score: {self.best_score:.4f}")
        
        return self.best_params
    
    def tune_random_forest(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series]
    ) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters."""
        logger.info("Tuning Random Forest...")
        
        # Sample data for tuning
        X_sample, y_sample = self._sample_data(X_train, y_train)
        
        # Preprocess categorical columns
        cat_cols = self._get_categorical_columns(X_sample)
        X_processed = self._preprocess_for_sklearn(X_sample, cat_cols)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 200, step=10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100, step=5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
            
            model = RandomForestModel(
                random_state=self.random_state,
                **params
            )
            
            score = self._cross_val_score_sklearn(model, X_processed, y_sample)
            return score
        
        self.study = optuna.create_study(
            direction=self.direction,
            study_name='random_forest'
        )
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Best Random Forest params: {self.best_params}")
        logger.info(f"Best CV score: {self.best_score:.4f}")
        
        return self.best_params
    
    def tune_xgboost(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series]
    ) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters."""
        logger.info("Tuning XGBoost...")
        
        # Sample data for tuning
        X_sample, y_sample = self._sample_data(X_train, y_train)
        
        # XGBoost needs strictly numeric data
        cat_cols = self._get_categorical_columns(X_sample)
        X_processed = self._preprocess_for_sklearn(X_sample, cat_cols)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
                'tree_method': trial.suggest_categorical('tree_method', ['hist', 'approx'])
            }
            
            model = XGBoostModel(
                random_state=self.random_state,
                **params
            )
            
            score = self._cross_val_score_sklearn(model, X_processed, y_sample)
            return score
        
        self.study = optuna.create_study(
            direction=self.direction,
            study_name='xgboost'
        )
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Best XGBoost params: {self.best_params}")
        logger.info(f"Best CV score: {self.best_score:.4f}")
        
        return self.best_params
    
    def tune_lightgbm(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series]
    ) -> Dict[str, Any]:
        """Tune LightGBM hyperparameters."""
        logger.info("Tuning LightGBM...")
        
        # Sample data for tuning
        X_sample, y_sample = self._sample_data(X_train, y_train)
        
        # LightGBM needs numeric data
        cat_cols = self._get_categorical_columns(X_sample)
        X_processed = self._preprocess_for_sklearn(X_sample, cat_cols)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False])
            }
            
            model = LightGBMModel(
                random_state=self.random_state,
                **params
            )
            
            score = self._cross_val_score_sklearn(model, X_processed, y_sample)
            return score
        
        self.study = optuna.create_study(
            direction=self.direction,
            study_name='lightgbm'
        )
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Best LightGBM params: {self.best_params}")
        logger.info(f"Best CV score: {self.best_score:.4f}")
        
        return self.best_params
    
    def tune_catboost(
        self,
        X_train: Union[dd.DataFrame, pd.DataFrame],
        y_train: Union[dd.Series, pd.Series]
    ) -> Dict[str, Any]:
        """Tune CatBoost hyperparameters."""
        logger.info("Tuning CatBoost...")
        
        # Sample data for tuning
        X_sample, y_sample = self._sample_data(X_train, y_train)
        
        # Identify categorical columns
        cat_cols = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in X_sample.columns:
            if col not in cat_cols:
                unique_count = X_sample[col].nunique()
                if unique_count < 20 and X_sample[col].dtype in ['int64', 'float64']:
                    cat_cols.append(col)
        
        logger.info(f"  Categorical features: {len(cat_cols)}")
        
        # Prepare data for CatBoost (convert categorical to strings)
        def prepare_for_catboost(df):
            df_prep = df.copy()
            for col in cat_cols:
                df_prep[col] = df_prep[col].fillna('MISSING').astype(str)
                df_prep[col] = df_prep[col].replace('nan', 'MISSING')
                df_prep[col] = df_prep[col].replace('None', 'MISSING')
            return df_prep
        
        X_sample_prep = prepare_for_catboost(X_sample)
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500, step=50),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 50, 255),
                'auto_class_weights': trial.suggest_categorical(
                    'auto_class_weights', 
                    ['Balanced', 'SqrtBalanced', None]
                )
            }
            
            model = CatBoostModel(
                random_state=self.random_state,
                **params,
                cat_features=cat_cols if cat_cols else None,
                verbose=False
            )
            
            score = self._cross_val_score_catboost(model, X_sample_prep, y_sample)
            return score
        
        self.study = optuna.create_study(
            direction=self.direction,
            study_name='catboost'
        )
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Best CatBoost params: {self.best_params}")
        logger.info(f"Best CV score: {self.best_score:.4f}")
        
        return self.best_params
    
    def _cross_val_score_sklearn(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """Perform cross-validation for sklearn-compatible models."""
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            try:
                model_clone = self._clone_model(model)
                model_clone.fit(X_train_fold, y_train_fold)
                y_pred = model_clone.predict_proba(X_val_fold)[:, 1]
                score = roc_auc_score(y_val_fold, y_pred)
                scores.append(score)
            except Exception as e:
                logger.warning(f"  CV fold failed: {e}")
                continue
        
        if len(scores) == 0:
            return 0.0
        
        return np.mean(scores)
    
    def _cross_val_score_catboost(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """Perform cross-validation for CatBoost."""
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            try:
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Clone model with same parameters
                model_clone = CatBoostModel(
                    random_state=self.random_state,
                    iterations=model.iterations,
                    depth=model.depth,
                    learning_rate=model.learning_rate,
                    l2_leaf_reg=model.l2_leaf_reg,
                    border_count=model.border_count,
                    auto_class_weights=model.auto_class_weights,
                    cat_features=model.cat_features,
                    verbose=False
                )
                
                model_clone.fit(X_train_fold, y_train_fold)
                y_pred = model_clone.predict_proba(X_val_fold)[:, 1]
                score = roc_auc_score(y_val_fold, y_pred)
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"  CV fold {fold} failed: {e}")
                continue
        
        if len(scores) == 0:
            return 0.0
        
        return np.mean(scores)
    
    def _clone_model(self, model):
        """Clone a model instance."""
        from models.logistic import LogisticRegressionModel
        from models.random_forest import RandomForestModel
        from models.xgboost_model import XGBoostModel
        from models.lightgbm_model import LightGBMModel
        from models.catboost_model import CatBoostModel
        
        if isinstance(model, LogisticRegressionModel):
            params = model.get_params()
            return LogisticRegressionModel(**params)
        elif isinstance(model, RandomForestModel):
            params = model.get_params()
            return RandomForestModel(**params)
        elif isinstance(model, XGBoostModel):
            params = model.get_params()
            return XGBoostModel(**params)
        elif isinstance(model, LightGBMModel):
            params = model.get_params()
            return LightGBMModel(**params)
        elif isinstance(model, CatBoostModel):
            params = model.get_params()
            return CatBoostModel(**params)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")