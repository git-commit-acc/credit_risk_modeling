# config/config.py
"""
Configuration module for the Behavioral Credit Risk Scoring System.
Centralizes all paths, parameters, and feature definitions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import os


@dataclass
class PathConfig:
    """Configuration for data paths."""
    raw_dir: str = "D:/Projects/credit_risk_scoring/data/raw"
    data_dir: str = "D:/Projects/credit_risk_scoring/data"
    bronze_dir: str = "D:/Projects/credit_risk_scoring/data/bronze"
    silver_dir: str = "D:/Projects/credit_risk_scoring/data/silver"
    features_dir: str = "D:/Projects/credit_risk_scoring/data/features"
    models_dir: str = "D:/Projects/credit_risk_scoring/models"
    results_dir: str = "D:/Projects/credit_risk_scoring/results"
    eda_dir: str = "D:/Projects/credit_risk_scoring/results/eda"
    
    @property
    def origination_bronze(self) -> str:
        return f"{self.bronze_dir}/origination_bronze.parquet"
    
    @property
    def performance_bronze(self) -> str:
        return f"{self.bronze_dir}/performance_bronze.parquet"
    
    @property
    def feature_dataset(self) -> str:
        return f"{self.features_dir}/feature_dataset.parquet"
    
    @property
    def train_data(self) -> str:
        return f"{self.features_dir}/train_data.parquet"
    
    @property
    def val_data(self) -> str:
        return f"{self.features_dir}/val_data.parquet"
    
    @property
    def test_data(self) -> str:
        return f"{self.features_dir}/test_data.parquet"


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # Target definition
    default_threshold: int = 30  # 30+ DPD
    lookahead_months: int = 12
    
    # Data split
    train_start_year: int = 1999
    train_end_year: int = 2008
    test_start_year: int = 2009
    test_end_year: int = 2012
    val_frac: float = 0.2
    
    # Cross-validation
    cv_folds: int = 5
    n_jobs: int = -1
    random_state: int = 42
    
    # Class imbalance
    scale_pos_weight: Optional[float] = None
    
    # Hyperparameter tuning
    n_trials: int = 50
    timeout: int = 3600


@dataclass
class FeatureConfig:
    """Feature definitions for the model."""
    # Static features from origination
    static_features: List[str] = field(default_factory=lambda: [
        'CREDIT_SCORE',
        'FIRST_TIME_HOMEBUYER_FLAG',
        'ORIGINAL_LTV',
        'ORIGINAL_CLTV',
        'ORIGINAL_DTI',
        'ORIGINAL_UPB',
        'ORIGINAL_INTEREST_RATE',
        'OCCUPANCY_STATUS',
        'PROPERTY_TYPE',
        'LOAN_PURPOSE',
        'NUMBER_OF_BORROWERS',
        'MI_PERCENTAGE',
        'ORIGINAL_LOAN_TERM',
        'PROPERTY_STATE',
        'CHANNEL',
        'SUPER_CONFORMING_FLAG',
        'RELIEF_REFINANCE_INDICATOR',
        'AMORTIZATION_TYPE'
    ])
    
    # Behavioral features from performance
    behavioral_features: List[str] = field(default_factory=lambda: [
        # Current state
        'CURRENT_ACTUAL_UPB',
        'CURRENT_INTEREST_RATE',
        'CURRENT_LOAN_DELINQUENCY_STATUS',
        'LOAN_AGE',
        'REMAINING_MONTHS_TO_LEGAL_MATURITY',
        'MODIFICATION_FLAG',
        'PAYMENT_DEFERRAL_FLAG',
        'INTEREST_BEARING_UPB',
        'CURRENT_NON_INTEREST_BEARING_UPB',
        'BORROWER_ASSISTANCE_STATUS_CODE',
        'ELTV',
        'DELINQUENCY_DUE_TO_DISASTER',
        
        # Behavioral features
        'max_delinquency_3m',
        'max_delinquency_6m',
        'max_delinquency_12m',
        'rolling_mean_delinquency_6m',
        'num_delinquent_months_12m',
        'consecutive_delinquent_months',
        'months_since_last_delinquency',
        'delinquency_trend_6m',
        'remaining_balance_pct',
        'principal_paid_pct',
        'balance_change_3m',
        'balance_change_6m',
        'balance_change_12m',
        'rolling_avg_balance_6m',
        'rate_change_since_origination',
        'rate_reduction_after_mod',
        'ever_modified',
        'num_modifications',
        'months_since_modification',
        'payment_deferral_count',
        'remaining_term_pct',
        'observation_month',
        'observation_quarter',
        'observation_year',
        'loan_age_squared',
        'loan_age_cubic',
        'seasonality_sin',
        'seasonality_cos',
        'dti_ltv_interaction',
        'credit_dti_interaction',
        'balance_delinquency_interaction',
        'age_balance_interaction'
    ])
    
    # Features to drop
    drop_features: List[str] = field(default_factory=lambda: [
        'LOAN_SEQUENCE_NUMBER',
        'MONTHLY_REPORTING_PERIOD',
        'FIRST_PAYMENT_DATE',
        'MATURITY_DATE',
        'POSTAL_CODE',
        'SELLER_NAME',
        'SERVICER_NAME',
        'ingestion_year',
        'ingestion_timestamp',
        'reporting_year',
        'delinquency_numeric',
        'is_delinquent',
        'is_terminated',
        'future_termination',
        'future_delinquency_max',
        'delinquency_days'
    ])
    
    # Categorical features
    categorical_features: List[str] = field(default_factory=lambda: [
        'FIRST_TIME_HOMEBUYER_FLAG',
        'OCCUPANCY_STATUS',
        'PROPERTY_TYPE',
        'LOAN_PURPOSE',
        'PROPERTY_STATE',
        'CHANNEL',
        'SUPER_CONFORMING_FLAG',
        'RELIEF_REFINANCE_INDICATOR',
        'AMORTIZATION_TYPE',
        'MODIFICATION_FLAG',
        'PAYMENT_DEFERRAL_FLAG',
        'BORROWER_ASSISTANCE_STATUS_CODE',
        'DELINQUENCY_DUE_TO_DISASTER',
        'CURRENT_LOAN_DELINQUENCY_STATUS'
    ])
    
    # Numerical features
    numerical_features: List[str] = field(default_factory=lambda: [
        'CREDIT_SCORE',
        'ORIGINAL_LTV',
        'ORIGINAL_CLTV',
        'ORIGINAL_DTI',
        'ORIGINAL_UPB',
        'ORIGINAL_INTEREST_RATE',
        'NUMBER_OF_BORROWERS',
        'MI_PERCENTAGE',
        'ORIGINAL_LOAN_TERM',
        'CURRENT_ACTUAL_UPB',
        'CURRENT_INTEREST_RATE',
        'LOAN_AGE',
        'REMAINING_MONTHS_TO_LEGAL_MATURITY',
        'INTEREST_BEARING_UPB',
        'CURRENT_NON_INTEREST_BEARING_UPB',
        'ELTV',
        'max_delinquency_3m',
        'max_delinquency_6m',
        'max_delinquency_12m',
        'rolling_mean_delinquency_6m',
        'num_delinquent_months_12m',
        'consecutive_delinquent_months',
        'months_since_last_delinquency',
        'delinquency_trend_6m',
        'remaining_balance_pct',
        'principal_paid_pct',
        'balance_change_3m',
        'balance_change_6m',
        'balance_change_12m',
        'rolling_avg_balance_6m',
        'rate_change_since_origination',
        'rate_reduction_after_mod',
        'ever_modified',
        'num_modifications',
        'months_since_modification',
        'payment_deferral_count',
        'remaining_term_pct',
        'observation_month',
        'observation_quarter',
        'observation_year',
        'loan_age_squared',
        'loan_age_cubic',
        'seasonality_sin',
        'seasonality_cos',
        'dti_ltv_interaction',
        'credit_dti_interaction',
        'balance_delinquency_interaction',
        'age_balance_interaction'
    ])


@dataclass
class DaskConfig:
    """Configuration for the shared Dask distributed cluster used by every
    models/*.py module (see models/dask_utils.py::get_dask_client). Centralizing
    this here means memory_limit in particular -- the main lever for keeping
    the pipeline within RAM budget on the full 47M-row servicing panel -- is
    tuned in one place rather than hardcoded inside individual model files."""
    n_workers: int = 4
    threads_per_worker: int = 2
    memory_limit: str = "4GB"
    npartitions: int = 8


# Global configuration instance
paths = PathConfig()
model = ModelConfig()
features = FeatureConfig()
dask_cfg = DaskConfig()

config = {
    'paths': paths,
    'model': model,
    'features': features,
    'dask': dask_cfg,
}