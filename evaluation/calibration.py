# evaluation/calibration.py
"""
Probability calibration for credit risk models with Dask support.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import logging
from typing import Dict, Any, Optional, Union

from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrates model probabilities using isotonic regression or Platt scaling with Dask support.
    """
    
    def __init__(
        self,
        method: str = 'isotonic',
        cv_folds: int = 5
    ):
        """
        Initialize calibrator.
        
        Args:
            method: 'isotonic' or 'sigmoid' (Platt scaling)
            cv_folds: Number of CV folds for calibration
        """
        self.method = method
        self.cv_folds = cv_folds
        self.calibrator = None
        self.base_model = None
    
    def _ensure_pandas(self, data: Union[np.ndarray, pd.Series, dd.Series, pd.DataFrame, dd.DataFrame]):
        """Convert to pandas/numpy if needed."""
        if isinstance(data, dd.Series):
            return data.compute()
        if isinstance(data, dd.DataFrame):
            return data.compute()
        if isinstance(data, pd.Series):
            return data
        if isinstance(data, pd.DataFrame):
            return data
        return data
    
    def calibrate(
        self,
        model: BaseCreditRiskModel,
        X_train: Union[pd.DataFrame, dd.DataFrame],
        y_train: Union[pd.Series, dd.Series],
        X_val: Optional[Union[pd.DataFrame, dd.DataFrame]] = None,
        y_val: Optional[Union[pd.Series, dd.Series]] = None
    ) -> BaseCreditRiskModel:
        """
        Calibrate model probabilities.
        
        Args:
            model: Base model to calibrate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Calibrated model wrapper
        """
        logger.info(f"Calibrating {model.name} using {self.method} method...")
        
        self.base_model = model
        
        # Convert to pandas for sklearn calibration
        X_train_pd = self._ensure_pandas(X_train)
        y_train_pd = self._ensure_pandas(y_train)
        
        # If validation data is provided, use it for calibration
        if X_val is not None and y_val is not None:
            X_val_pd = self._ensure_pandas(X_val)
            y_val_pd = self._ensure_pandas(y_val)
            
            # Get uncalibrated predictions
            y_pred_uncal = model.predict_proba(X_val_pd)[:, 1]
            
            # Fit calibrator
            if self.method == 'isotonic':
                self.calibrator = IsotonicRegression(
                    y_min=0.0,
                    y_max=1.0,
                    out_of_bounds='clip'
                )
                self.calibrator.fit(y_pred_uncal, y_val_pd)
            else:  # sigmoid
                from sklearn.linear_model import LogisticRegression
                self.calibrator = LogisticRegression(
                    random_state=42,
                    C=1e6  # High C for Platt scaling
                )
                self.calibrator.fit(y_pred_uncal.reshape(-1, 1), y_val_pd)
        else:
            # Use cross-validated calibration
            self.calibrator = CalibratedClassifierCV(
                model.model,
                method=self.method,
                cv=self.cv_folds
            )
            self.calibrator.fit(X_train_pd, y_train_pd)
        
        logger.info("Calibration completed.")
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, dd.DataFrame]) -> np.ndarray:
        """
        Predict calibrated probabilities.
        
        Args:
            X: Features
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call calibrate() first.")
        
        X_pd = self._ensure_pandas(X)
        
        if isinstance(self.calibrator, (IsotonicRegression, LogisticRegression)):
            # Get uncalibrated predictions
            y_pred_uncal = self.base_model.predict_proba(X_pd)[:, 1]
            
            # Calibrate
            if self.method == 'isotonic':
                y_pred_cal = self.calibrator.transform(y_pred_uncal)
            else:
                y_pred_cal = self.calibrator.predict_proba(y_pred_uncal.reshape(-1, 1))[:, 1]
            
            # Return as 2-column array for compatibility
            return np.column_stack([1 - y_pred_cal, y_pred_cal])
        else:
            # CalibratedClassifierCV
            return self.calibrator.predict_proba(X_pd)
    
    def evaluate_calibration(
        self,
        y_true: Union[np.ndarray, pd.Series, dd.Series],
        y_proba: Union[np.ndarray, pd.Series, dd.Series],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate calibration quality.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        from sklearn.calibration import calibration_curve
        
        y_true_np = self._ensure_pandas(y_true)
        y_proba_np = self._ensure_pandas(y_proba)
        
        prob_true, prob_pred = calibration_curve(y_true_np, y_proba_np, n_bins=n_bins)
        
        # Calculate calibration metrics
        brier = brier_score_loss(y_true_np, y_proba_np)
        
        # ECE: Expected Calibration Error
        bin_counts = np.histogram(y_proba_np, bins=n_bins, range=(0, 1))[0]
        ece = np.sum(
            np.abs(prob_true - prob_pred) * bin_counts / len(y_true_np)
        )
        
        # MCE: Maximum Calibration Error
        mce = np.max(np.abs(prob_true - prob_pred))
        
        return {
            'brier_score': brier,
            'ece': ece,
            'mce': mce,
            'calibration_curve': (prob_true, prob_pred),
            'bin_counts': bin_counts
        }