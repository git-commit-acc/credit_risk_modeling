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
    
    # NEW: Sampled features directory
    sampled_features_dir: str = "D:/Projects/credit_risk_scoring/data/features_sampled"
    
    @property
    def sampled_train_data(self) -> str:
        return f"{self.sampled_features_dir}/train_data.parquet"
    
    @property
    def sampled_val_data(self) -> str:
        return f"{self.sampled_features_dir}/val_data.parquet"
    
    @property
    def sampled_test_data(self) -> str:
        return f"{self.sampled_features_dir}/test_data.parquet"
    
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
    
    # @property
    # def train_data(self) -> str:
    #     return f"{self.features_dir}_sampled/train_data.parquet"
    
    # @property
    # def val_data(self) -> str:
    #     return f"{self.features_dir}_sampled/val_data.parquet"
    
    # @property
    # def test_data(self) -> str:
    #     return f"{self.features_dir}_sampled/test_data.parquet"


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # Target definition
    default_threshold: int = 90
    lookahead_months: int = 12
    
    # Data split (out-of-time validation)
    train_start_year: int = 1999
    train_end_year: int = 2008
    test_start_year: int = 2009
    test_end_year: int = 2012
    val_frac: float = 0.2
    
    # Cross-validation
    cv_folds: int = 5
    n_jobs: int = -1
    sample_size: int = 500000  # 500K training samples
    random_state: int = 42
    
    # Class imbalance
    scale_pos_weight: Optional[float] = None
    
    # Hyperparameter tuning
    n_trials: int = 50
    timeout: int = 3600

    # GPU Configuration
    use_gpu: bool = True  # Enable GPU acceleration
    gpu_id: int = 0       # GPU device ID (0 for single GPU)

    # Sampling Configuration
    use_sample: bool = True
    sample_size: int = 500000
    sample_frac: float = 0.1
    random_seed: int = 42



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
    ])
    
    # Behavioral features from performance
    behavioral_features: List[str] = field(default_factory=lambda: [
        # Current state
        'CURRENT_ACTUAL_UPB',
        'CURRENT_INTEREST_RATE',
        'CURRENT_LOAN_DELINQUENCY_STATUS',
        'LOAN_AGE',
        'REMAINING_MONTHS_TO_LEGAL_MATURITY',
        'ELTV',
        'INTEREST_BEARING_UPB',
        
        # Delinquency history
        'max_delinquency_3m',
        'max_delinquency_6m',
        'max_delinquency_12m',
        'rolling_mean_delinquency_6m',
        'num_delinquent_months_12m',
        'consecutive_delinquent_months',
        'months_since_last_delinquency',
        'delinquency_trend_6m',
        
        # Balance behavior
        'remaining_balance_pct',
        'principal_paid_pct',
        'balance_change_3m',
        'balance_change_6m',
        'balance_change_12m',
        'rolling_avg_balance_6m',
        
        # Rate behavior
        'rate_change_since_origination',
        
        # Modification history
        'ever_modified',
        'months_since_modification',
        
        # Time features
        'observation_month',
        'observation_quarter',
        'observation_year',
        'loan_age_squared',
        'loan_age_cubic',
        'seasonality_sin',
        'seasonality_cos',
        
        # Interactions
        'dti_ltv_interaction',
        'credit_dti_interaction',
        'balance_delinquency_interaction',
        'age_balance_interaction'
    ])
    
    # Features to drop (identifiers, constant columns, derived targets)
    drop_features: List[str] = field(default_factory=lambda: [
        # Identifiers
        'LOAN_SEQUENCE_NUMBER',
        'MONTHLY_REPORTING_PERIOD',
        'FIRST_PAYMENT_DATE',
        'MATURITY_DATE',
        'POSTAL_CODE',
        'SELLER_NAME',
        'SERVICER_NAME',
        
        # Empty/constant performance columns
        'DEFECT_SETTLEMENT_DATE',
        'ZERO_BALANCE_CODE',
        'ZERO_BALANCE_EFFECTIVE_DATE',
        'DDLPI',
        'MI_RECOVERIES',
        'NET_SALE_PROCEEDS',
        'NON_MI_RECOVERIES',
        'TOTAL_EXPENSES',
        'LEGAL_COSTS',
        'MAINTENANCE_AND_PRESERVATION_COSTS',
        'TAXES_AND_INSURANCE',
        'MISCELLANEOUS_EXPENSES',
        'ACTUAL_LOSS_CALCULATION',
        'CUMULATIVE_MODIFICATION_COST',
        'INTEREST_RATE_STEP_INDICATOR',
        'PAYMENT_DEFERRAL_FLAG',
        'ZERO_BALANCE_REMOVAL_UPB',
        'DELINQUENT_ACCRUED_INTEREST',
        'CURRENT_MONTH_MODIFICATION_COST',
        'BORROWER_ASSISTANCE_STATUS_CODE',
        'DELINQUENCY_DUE_TO_DISASTER',
        
        # Constant/near-constant origination columns
        'AMORTIZATION_TYPE',
        'PROPERTY_VALUATION_METHOD',
        'IO_INDICATOR',
        'MI_CANCELLATION_INDICATOR',
        'SUPER_CONFORMING_FLAG',
        'RELIEF_REFINANCE_INDICATOR',
        'PRE_RELIEF_REFINANCE_LSN',
        'SPECIAL_ELIGIBILITY_PROGRAM',
        
        # Derived columns that shouldn't be features
        'vintage_year',
        'origination_year',
        'reporting_year',
        'delinquency_numeric',
        'is_delinquent',
        'is_terminated',
        'future_termination',
        'future_delinquency_max',
        'delinquency_days',
        'row_num',
        'cumulative_delinquency',
        'last_delinquent_month',
        'last_modification_month',
        'delinquency_streak_id',
        'num_modifications',
        'payment_deferral_count',
        'rate_reduction_after_mod',
        'CURRENT_NON_INTEREST_BEARING_UPB',
        'prev_interest_rate',
        
        # Rolling statistics with high null rates
        'rolling_std_balance_6m',
        'rolling_std_rate_6m',
        'rolling_avg_eltv_6m',
        'rolling_std_eltv_6m',
        'rolling_min_eltv_6m',
        'rolling_max_eltv_6m',
    ])
    
    # Categorical features (encoded as numeric in cleaning)
    categorical_features: List[str] = field(default_factory=lambda: [
        'FIRST_TIME_HOMEBUYER_FLAG',
        'OCCUPANCY_STATUS',
        'PROPERTY_TYPE',
        'LOAN_PURPOSE',
        'CHANNEL',
        'CURRENT_LOAN_DELINQUENCY_STATUS',
        'PROPERTY_STATE',
        'NUMBER_OF_BORROWERS',
        'NUMBER_OF_UNITS',
        'MODIFICATION_FLAG',
        'PPM_FLAG',
    ])
    
    # Numerical features
    numerical_features: List[str] = field(default_factory=lambda: [
        'CREDIT_SCORE',
        'ORIGINAL_LTV',
        'ORIGINAL_CLTV',
        'ORIGINAL_DTI',
        'ORIGINAL_UPB',
        'ORIGINAL_INTEREST_RATE',
        'MI_PERCENTAGE',
        'ORIGINAL_LOAN_TERM',
        'CURRENT_ACTUAL_UPB',
        'CURRENT_INTEREST_RATE',
        'LOAN_AGE',
        'REMAINING_MONTHS_TO_LEGAL_MATURITY',
        'ELTV',
        'INTEREST_BEARING_UPB',
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
        'ever_modified',
        'months_since_modification',
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


# @dataclass
# class DaskConfig:
#     """Configuration for the shared Dask distributed cluster."""
#     n_workers: int = 4
#     threads_per_worker: int = 2
#     memory_limit: str = "4GB"
#     npartitions: int = 8

@dataclass
class DaskConfig:
    n_workers: int = 4
    threads_per_worker: int = 4
    memory_limit: str = "6GB"
    npartitions: int = 32


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