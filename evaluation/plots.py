# evaluation/plots.py
"""
Visualization utilities for credit risk models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import logging
from typing import Dict, Any, List, Optional, Tuple
import os

from scoring.score_generator import CreditScoreGenerator

logger = logging.getLogger(__name__)


class CreditRiskVisualizer:
    """Visualization tools for credit risk models."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
        
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        ax: Optional[plt.Axes] = None,
        color: Optional[str] = None
    ) -> plt.Axes:
        """Plot ROC curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', 
                linewidth=2, color=color)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        return ax
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        ax: Optional[plt.Axes] = None,
        color: Optional[str] = None
    ) -> plt.Axes:
        """Plot Precision-Recall curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        ax.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})', 
                linewidth=2, color=color)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        return ax
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        n_bins: int = 10,
        ax: Optional[plt.Axes] = None,
        color: Optional[str] = None
    ) -> plt.Axes:
        """Plot calibration curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        
        ax.plot(prob_pred, prob_true, 'o-', label=model_name, 
                linewidth=2, markersize=8, color=color)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Observed Probability', fontsize=12)
        ax.set_title(f'Calibration Curve - {model_name}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        return ax
    
    def plot_lift_curve(
        self,
        lift_data: Dict[str, Any],
        model_name: str,
        ax: Optional[plt.Axes] = None,
        color: Optional[str] = None
    ) -> plt.Axes:
        """Plot lift curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        decile_metrics = lift_data['decile_metrics']
        deciles = [d['decile'] for d in decile_metrics]
        gains = [d['gain'] for d in decile_metrics]
        
        ax.plot(deciles, gains, 'o-', label=model_name, 
                linewidth=2, markersize=8, color=color)
        ax.plot([0, len(deciles)], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('Decile', fontsize=12)
        ax.set_ylabel('Cumulative % of Defaults', fontsize=12)
        ax.set_title(f'Gain Curve - {model_name}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, len(deciles)])
        ax.set_ylim([0, 1])
        
        return ax
    
    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 20,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot feature importance."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by importance
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, values = zip(*sorted_imp)
        
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(features)))
        ax.barh(range(len(features)), values, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        return ax
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'roc_auc',
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot comparison of multiple models."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(results.keys())
        values = [results[m].get(metric, 0) for m in models]
        
        colors = sns.color_palette("husl", len(models))
        bars = ax.bar(models, values, color=colors)
        
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return ax
    
    def plot_risk_distribution(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        title: str = "Risk Score Distribution",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot risk score distribution by default status."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate by default status
        default_scores = scores[labels == 1]
        non_default_scores = scores[labels == 0]
        
        # Plot histograms
        ax.hist(non_default_scores, bins=30, alpha=0.6, 
                label=f'Non-Default (n={len(non_default_scores):,})', 
                density=True, color='steelblue')
        ax.hist(default_scores, bins=30, alpha=0.6, 
                label=f'Default (n={len(default_scores):,})', 
                density=True, color='coral')
        
        ax.set_xlabel('Credit Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for risk bands
        ax.axvline(x=700, color='green', linestyle='--', alpha=0.5, label='Low Risk (>700)')
        ax.axvline(x=600, color='orange', linestyle='--', alpha=0.5, label='Medium Risk (600-700)')
        ax.axvline(x=300, color='red', linestyle='--', alpha=0.5, label='High Risk (<600)')
        
        return ax
    
    def plot_risk_band_metrics(
        self,
        risk_metrics: Dict[str, Any],
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot risk band metrics."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        bands = list(risk_metrics.keys())
        default_rates = [risk_metrics[b]['default_rate'] for b in bands]
        portfolio_pcts = [risk_metrics[b]['pct_of_portfolio'] for b in bands]
        
        x = np.arange(len(bands))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, default_rates, width, 
                      label='Default Rate', color='coral')
        bars2 = ax.bar(x + width/2, portfolio_pcts, width, 
                      label='% of Portfolio', color='steelblue')
        
        ax.set_xlabel('Risk Band', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_title('Risk Band Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        return ax
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot confusion matrix."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Default', 'Default'],
                   yticklabels=['Non-Default', 'Default'],
                   ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
        
        return ax
    
    def plot_ks_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot Kolmogorov-Smirnov curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Sort by probability
        df = pd.DataFrame({
            'y_true': y_true,
            'y_proba': y_proba
        }).sort_values('y_proba', ascending=False)
        
        # Calculate cumulative distributions
        n_pos = df['y_true'].sum()
        n_neg = len(df) - n_pos
        
        df['cum_pos'] = df['y_true'].cumsum() / n_pos
        df['cum_neg'] = (1 - df['y_true']).cumsum() / n_neg
        df['ks'] = df['cum_pos'] - df['cum_neg']
        
        ax.plot(range(len(df)), df['cum_pos'], label='Default', linewidth=2)
        ax.plot(range(len(df)), df['cum_neg'], label='Non-Default', linewidth=2)
        ax.plot(range(len(df)), df['ks'], label='KS Curve', linewidth=2, color='green')
        
        # Mark maximum KS
        max_ks_idx = df['ks'].argmax()
        max_ks = df['ks'].max()
        ax.axvline(x=max_ks_idx, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=max_ks, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Population (sorted by score)', fontsize=12)
        ax.set_ylabel('Cumulative %', fontsize=12)
        ax.set_title(f'KS Curve - {model_name} (KS = {max_ks:.3f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, len(df)])
        ax.set_ylim([0, 1])
        
        return ax
    
    def create_model_evaluation_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        feature_importance: Optional[Dict[str, float]] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive model evaluation report.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            y_pred: Predicted labels
            model_name: Name of the model
            feature_importance: Feature importance dictionary
            save_name: Filename to save the report
            
        Returns:
            Matplotlib figure
        """
        from evaluation.metrics import CreditRiskMetrics
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        
        # Calculate metrics
        metric_calc = CreditRiskMetrics()
        metrics = metric_calc.evaluate(y_true, y_pred, y_proba)
        lift_data = metric_calc.compute_lift_gain(y_true, y_proba)
        risk_metrics = metric_calc.compute_risk_band_metrics(y_true, y_proba)
        
        # 1. ROC Curve
        ax1 = plt.subplot(2, 3, 1)
        self.plot_roc_curve(y_true, y_proba, model_name, ax=ax1, color=self.colors[0])
        
        # 2. PR Curve
        ax2 = plt.subplot(2, 3, 2)
        self.plot_pr_curve(y_true, y_proba, model_name, ax=ax2, color=self.colors[1])
        
        # 3. Calibration Curve
        ax3 = plt.subplot(2, 3, 3)
        self.plot_calibration_curve(y_true, y_proba, model_name, ax=ax3, color=self.colors[2])
        
        # 4. KS Curve
        ax4 = plt.subplot(2, 3, 4)
        self.plot_ks_curve(y_true, y_proba, model_name, ax=ax4)
        
        # 5. Risk Distribution
        ax5 = plt.subplot(2, 3, 5)
        score_gen = CreditScoreGenerator()
        scores = score_gen.probability_to_score(y_proba)
        self.plot_risk_distribution(scores, y_true, ax=ax5)
        
        # 6. Risk Band Metrics
        ax6 = plt.subplot(2, 3, 6)
        self.plot_risk_band_metrics(risk_metrics, ax=ax6)
        
        # Add metrics summary
        metrics_text = (
            f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
            f"PR-AUC: {metrics['pr_auc']:.4f}\n"
            f"KS: {metrics['ks_statistic']:.4f}\n"
            f"F1: {metrics['f1']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall: {metrics['recall']:.4f}\n"
            f"Brier: {metrics['brier_score']:.4f}"
        )
        fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'Model Evaluation Report - {model_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save if requested
        if self.save_dir and save_name:
            fig.savefig(f"{self.save_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved evaluation report to {self.save_dir}/{save_name}.png")
        
        return fig
    
    def create_ensemble_comparison_plot(
        self,
        results: Dict[str, Dict[str, float]],
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create ensemble model comparison plot.
        
        Args:
            results: Dictionary of model results
            save_name: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        
        # 1. ROC-AUC Comparison
        ax1 = plt.subplot(2, 2, 1)
        self.plot_model_comparison(results, 'roc_auc', ax=ax1)
        ax1.set_title('ROC-AUC Comparison', fontsize=14)
        
        # 2. PR-AUC Comparison
        ax2 = plt.subplot(2, 2, 2)
        self.plot_model_comparison(results, 'pr_auc', ax=ax2)
        ax2.set_title('PR-AUC Comparison', fontsize=14)
        
        # 3. F1 Score Comparison
        ax3 = plt.subplot(2, 2, 3)
        self.plot_model_comparison(results, 'f1', ax=ax3)
        ax3.set_title('F1 Score Comparison', fontsize=14)
        
        # 4. KS Statistic Comparison
        ax4 = plt.subplot(2, 2, 4)
        self.plot_model_comparison(results, 'ks_statistic', ax=ax4)
        ax4.set_title('KS Statistic Comparison', fontsize=14)
        
        plt.suptitle('Model Comparison - All Metrics', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if self.save_dir and save_name:
            fig.savefig(f"{self.save_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {self.save_dir}/{save_name}.png")
        
        return fig