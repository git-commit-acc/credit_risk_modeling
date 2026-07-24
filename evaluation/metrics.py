# evaluation/metrics.py
"""
Evaluation metrics for credit risk models with Dask support.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, log_loss,
    confusion_matrix, precision_recall_curve
)
from scipy.stats import ks_2samp
import logging
from typing import Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)


class CreditRiskMetrics:
    """Comprehensive metrics for credit risk model evaluation with Dask support."""
    
    def __init__(self):
        self.metrics = {}
    
    def _ensure_numpy(self, data: Union[np.ndarray, pd.Series, dd.Series]) -> np.ndarray:
        """Convert to numpy array."""
        if isinstance(data, dd.Series):
            return data.compute().values
        if isinstance(data, pd.Series):
            return data.values
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)
    
    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.Series, dd.Series],
        y_pred: Union[np.ndarray, pd.Series, dd.Series],
        y_proba: Union[np.ndarray, pd.Series, dd.Series]
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (class 1)
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        y_true_np = self._ensure_numpy(y_true)
        y_pred_np = self._ensure_numpy(y_pred)
        y_proba_np = self._ensure_numpy(y_proba)
        
        metrics = {}
        
        # Classification metrics
        metrics['roc_auc'] = roc_auc_score(y_true_np, y_proba_np)
        metrics['pr_auc'] = average_precision_score(y_true_np, y_proba_np)
        metrics['f1'] = f1_score(y_true_np, y_pred_np)
        metrics['precision'] = precision_score(y_true_np, y_pred_np)
        metrics['recall'] = recall_score(y_true_np, y_pred_np)
        metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_np, y_pred_np)
        metrics['mcc'] = matthews_corrcoef(y_true_np, y_pred_np)
        
        # Probability quality
        metrics['brier_score'] = brier_score_loss(y_true_np, y_proba_np)
        metrics['log_loss'] = log_loss(y_true_np, y_proba_np)
        
        # Ranking metrics
        metrics['ks_statistic'] = self._compute_ks(y_true_np, y_proba_np)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np).ravel()
        metrics['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
        
        self.metrics = metrics
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: Union[np.ndarray, pd.Series, dd.Series],
        y_proba: Union[np.ndarray, pd.Series, dd.Series],
        beta: float = 1.0
    ) -> Dict[str, Any]:
        """
        Find the classification threshold that maximizes the F-beta score,
        instead of using a fixed 0.5 cutoff.

        Under class imbalance, 0.5 is rarely where precision and recall
        actually balance -- it's just an arbitrary point on the
        probability scale that depends on how each model happens to be
        calibrated. Two models with identical ranking ability (same AUC)
        can report wildly different F1 at a fixed 0.5 threshold simply
        because one pushes probabilities closer to 0.5 than the other.
        Searching the precision-recall curve for the actual best threshold
        makes F1 comparable across models again.

        beta > 1 (e.g. 2.0) weights recall more heavily than precision --
        appropriate for credit risk, where a missed default (false
        negative) is typically far costlier than a false-positive review
        (false positive just costs a loss-mitigation team's time).

        Returns the threshold, the F-beta score achieved there, and the
        precision/recall/F1 at that same threshold for context.
        """
        y_true_np = self._ensure_numpy(y_true)
        y_proba_np = self._ensure_numpy(y_proba)

        precisions, recalls, thresholds = precision_recall_curve(y_true_np, y_proba_np)

        # precision_recall_curve returns one more precision/recall pair
        # than thresholds (it appends the (1, 0) point for threshold=inf);
        # drop that last pair so arrays align.
        precisions, recalls = precisions[:-1], recalls[:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            beta_sq = beta ** 2
            f_beta = (1 + beta_sq) * (precisions * recalls) / (beta_sq * precisions + recalls)
        f_beta = np.nan_to_num(f_beta, nan=0.0)

        best_idx = np.argmax(f_beta)
        best_threshold = float(thresholds[best_idx])
        y_pred_at_best = (y_proba_np >= best_threshold).astype(int)

        return {
            'threshold': best_threshold,
            'beta': beta,
            'f_beta': float(f_beta[best_idx]),
            'f1': float(f1_score(y_true_np, y_pred_at_best)),
            'precision': float(precision_score(y_true_np, y_pred_at_best, zero_division=0)),
            'recall': float(recall_score(y_true_np, y_pred_at_best, zero_division=0)),
        }

    def evaluate_with_optimal_thresholds(
        self,
        y_true: Union[np.ndarray, pd.Series, dd.Series],
        y_proba: Union[np.ndarray, pd.Series, dd.Series]
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: F1-optimal threshold (clean model-to-model
        comparison) and F2-optimal threshold (recall-weighted, closer to
        a credit-risk deployment cutoff), computed from the same curve.
        """
        f1_result = self.find_optimal_threshold(y_true, y_proba, beta=1.0)
        f2_result = self.find_optimal_threshold(y_true, y_proba, beta=2.0)
        return {
            'f1_optimal_threshold': f1_result['threshold'],
            'f1_optimal': f1_result['f1'],
            'precision_at_f1_optimal': f1_result['precision'],
            'recall_at_f1_optimal': f1_result['recall'],
            'f2_optimal_threshold': f2_result['threshold'],
            'f2_optimal': f2_result['f_beta'],
            'precision_at_f2_optimal': f2_result['precision'],
            'recall_at_f2_optimal': f2_result['recall'],
        }
    
    def _compute_ks(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        return ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic
    
    def compute_lift_gain(
        self,
        y_true: Union[np.ndarray, pd.Series, dd.Series],
        y_proba: Union[np.ndarray, pd.Series, dd.Series],
        n_deciles: int = 10
    ) -> Dict[str, Any]:
        """
        Compute lift and gain metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_deciles: Number of deciles
            
        Returns:
            Dictionary with lift and gain data
        """
        # Convert to numpy
        y_true_np = self._ensure_numpy(y_true)
        y_proba_np = self._ensure_numpy(y_proba)
        
        df = pd.DataFrame({
            'y_true': y_true_np,
            'y_proba': y_proba_np
        })
        
        df = df.sort_values('y_proba', ascending=False)
        df['decile'] = pd.qcut(
            range(len(df)), n_deciles, labels=False
        ) + 1
        
        decile_metrics = []
        total_defaults = y_true_np.sum()
        
        for decile in range(1, n_deciles + 1):
            decile_data = df[df['decile'] == decile]
            n_obs = len(decile_data)
            n_defaults = decile_data['y_true'].sum()
            default_rate = n_defaults / n_obs if n_obs > 0 else 0
            
            cumulative_defaults = df[df['decile'] <= decile]['y_true'].sum()
            cumulative_obs = df[df['decile'] <= decile].shape[0]
            
            decile_metrics.append({
                'decile': decile,
                'n_obs': n_obs,
                'n_defaults': n_defaults,
                'default_rate': default_rate,
                'lift': default_rate / (total_defaults / len(df)),
                'cumulative_defaults': cumulative_defaults,
                'cumulative_obs': cumulative_obs,
                'gain': cumulative_defaults / total_defaults if total_defaults > 0 else 0
            })
        
        return {
            'decile_metrics': decile_metrics,
            'top_decile_capture': decile_metrics[0]['cumulative_defaults'] / total_defaults if total_defaults > 0 else 0
        }
    
    def compute_risk_band_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series, dd.Series],
        y_proba: Union[np.ndarray, pd.Series, dd.Series],
        bands: Dict[str, Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Compute metrics by risk band.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            bands: Dictionary of band definitions
            
        Returns:
            Metrics by risk band
        """
        # Convert to numpy
        y_true_np = self._ensure_numpy(y_true)
        y_proba_np = self._ensure_numpy(y_proba)
        
        if bands is None:
            bands = {
                'Low': (0.00, 0.02),
                'Medium': (0.02, 0.05),
                'High': (0.05, 0.10),
                'Very High': (0.10, 1.00)
            }
        
        df = pd.DataFrame({
            'y_true': y_true_np,
            'y_proba': y_proba_np
        })
        
        band_metrics = {}
        for band_name, (lower, upper) in bands.items():
            band_data = df[(df['y_proba'] >= lower) & (df['y_proba'] < upper)]
            n_obs = len(band_data)
            n_defaults = band_data['y_true'].sum()
            default_rate = n_defaults / n_obs if n_obs > 0 else 0
            
            band_metrics[band_name] = {
                'n_obs': n_obs,
                'n_defaults': n_defaults,
                'default_rate': default_rate,
                'pct_of_portfolio': n_obs / len(df) if len(df) > 0 else 0
            }
        
        return band_metrics
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        if not self.metrics:
            return {}
        
        return {
            'roc_auc': self.metrics.get('roc_auc'),
            'pr_auc': self.metrics.get('pr_auc'),
            'f1': self.metrics.get('f1'),
            'precision': self.metrics.get('precision'),
            'recall': self.metrics.get('recall'),
            'balanced_accuracy': self.metrics.get('balanced_accuracy'),
            'mcc': self.metrics.get('mcc'),
            'brier_score': self.metrics.get('brier_score'),
            'log_loss': self.metrics.get('log_loss'),
            'ks_statistic': self.metrics.get('ks_statistic')
        }