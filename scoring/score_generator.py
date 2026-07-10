# scoring/score_generator.py
"""
Credit score generator for behavioral risk scoring.
Converts probability of delinquency to a 300-900 credit score.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CreditScoreGenerator:
    """
    Generates credit scores from delinquency probabilities.
    Uses logistic scorecard transformation.
    """
    
    def __init__(
        self,
        min_score: int = 300,
        max_score: int = 900,
        target_default_rate: float = 0.05,
        pdo: float = 20
    ):
        """
        Initialize score generator.
        
        Args:
            min_score: Minimum score (default: 300)
            max_score: Maximum score (default: 900)
            target_default_rate: Target default rate at score midpoint
            pdo: Points to Double Odds (default: 20)
        """
        self.min_score = min_score
        self.max_score = max_score
        self.target_default_rate = target_default_rate
        self.pdo = pdo
        
        # Calculate odds at midpoint
        self.target_odds = self.target_default_rate / (1 - self.target_default_rate)
        self.factor = self.pdo / np.log(2)
        self.offset = self._calculate_offset()
        
    def _calculate_offset(self) -> float:
        """Calculate score offset."""
        midpoint = (self.max_score + self.min_score) / 2
        return midpoint - self.factor * np.log(self.target_odds)
    
    def probability_to_score(self, prob: np.ndarray) -> np.ndarray:
        """
        Convert probability to credit score.
        
        Args:
            prob: Probability of delinquency (0-1)
            
        Returns:
            Credit score (300-900)
        """
        # Avoid division by zero
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        
        # Calculate odds
        odds = prob / (1 - prob)
        
        # Calculate score
        score = self.offset + self.factor * np.log(odds)
        
        # Clip to valid range
        score = np.clip(score, self.min_score, self.max_score)
        
        # Round to nearest integer
        score = np.round(score).astype(int)
        
        return score
    
    def score_to_probability(self, score: np.ndarray) -> np.ndarray:
        """
        Convert credit score to probability.
        
        Args:
            score: Credit score (300-900)
            
        Returns:
            Probability of delinquency
        """
        odds = np.exp((score - self.offset) / self.factor)
        prob = odds / (1 + odds)
        return prob
    
    def generate_score_band(self, score: int) -> str:
        """
        Assign risk band based on credit score.
        
        Args:
            score: Credit score
            
        Returns:
            Risk band (Low, Medium, High)
        """
        if score >= 700:
            return 'Low'
        elif score >= 600:
            return 'Medium'
        else:
            return 'High'
    
    def generate_all_scores(
        self,
        df: pd.DataFrame,
        prob_col: str = 'probability',
        score_col: str = 'credit_score'
    ) -> pd.DataFrame:
        """
        Generate credit scores for all observations.
        
        Args:
            df: DataFrame with probabilities
            prob_col: Column name for probabilities
            score_col: Column name for scores
            
        Returns:
            DataFrame with added score and risk band
        """
        df = df.copy()
        
        # Generate scores
        df[score_col] = self.probability_to_score(df[prob_col].values)
        
        # Add risk band
        df['risk_band'] = df[score_col].apply(self.generate_score_band)
        
        return df
    
    def get_score_distribution(
        self,
        df: pd.DataFrame,
        score_col: str = 'credit_score'
    ) -> Dict[str, Any]:
        """
        Get credit score distribution statistics.
        
        Args:
            df: DataFrame with scores
            score_col: Column name for scores
            
        Returns:
            Dictionary with distribution statistics
        """
        scores = df[score_col]
        
        return {
            'min': scores.min(),
            'max': scores.max(),
            'mean': scores.mean(),
            'median': scores.median(),
            'std': scores.std(),
            'percentiles': {
                '5th': scores.quantile(0.05),
                '10th': scores.quantile(0.10),
                '25th': scores.quantile(0.25),
                '50th': scores.quantile(0.50),
                '75th': scores.quantile(0.75),
                '90th': scores.quantile(0.90),
                '95th': scores.quantile(0.95)
            }
        }