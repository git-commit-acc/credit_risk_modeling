# config/config.py
"""
Configuration module for the Behavioral Credit Risk Scoring System.
Centralizes all paths, parameters, and feature definitions.

EDA VALIDATION (column_profiling_eda.ipynb, run against the live SFLLD
feature dataset): `categorical_features` below was cross-checked column by
column against real dtypes and left UNCHANGED -- every string-dtype column
in `static_features`/`behavioral_features` is already present in this list,
and nothing in it is numeric. The earlier "88 categorical columns detected"
issue (see models/dask_utils.py, models/logistic.py) was never about this
list being wrong; it was the auto-detection heuristic (used only as a
fallback when no explicit list is given) misclassifying numeric columns
that shouldn't have reached the model at all, because the on-disk
`train_data.parquet` at the time predated `DatasetCreator._select_model_
features()`'s allow-list filtering and still carried ~70 extra raw
columns -- including terminal/liquidation-only fields (ZERO_BALANCE_CODE,
NET_SALE_PROCEEDS, TOTAL_EXPENSES, ACTUAL_LOSS_CALCULATION, etc.) that are
only populated after a loan has already defaulted/terminated, i.e. label
leakage. That is a stale-data problem, fixed by regenerating the dataset
(`python main.py --module dataset_creation`), not a config problem.
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
    default_threshold: int = 90  # 30+ DPD
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


# @dataclass
# class FeatureConfig:
#     """Feature definitions for the model."""
#     # Static features from origination
#     static_features: List[str] = field(default_factory=lambda: [
#         'CREDIT_SCORE',
#         'FIRST_TIME_HOMEBUYER_FLAG',
#         'ORIGINAL_LTV',
#         'ORIGINAL_CLTV',
#         'ORIGINAL_DTI',
#         'ORIGINAL_UPB',
#         'ORIGINAL_INTEREST_RATE',
#         'OCCUPANCY_STATUS',
#         'PROPERTY_TYPE',
#         'LOAN_PURPOSE',
#         'NUMBER_OF_BORROWERS',
#         'MI_PERCENTAGE',
#         'ORIGINAL_LOAN_TERM',
#         'PROPERTY_STATE',
#         'CHANNEL',
#         'SUPER_CONFORMING_FLAG',
#         'RELIEF_REFINANCE_INDICATOR',
#         'AMORTIZATION_TYPE'
#     ])
    
#     # Behavioral features from performance
#     behavioral_features: List[str] = field(default_factory=lambda: [
#         # Current state
#         'CURRENT_ACTUAL_UPB',
#         'CURRENT_INTEREST_RATE',
#         'CURRENT_LOAN_DELINQUENCY_STATUS',
#         'LOAN_AGE',
#         'REMAINING_MONTHS_TO_LEGAL_MATURITY',
#         'MODIFICATION_FLAG',
#         'PAYMENT_DEFERRAL_FLAG',
#         'INTEREST_BEARING_UPB',
#         'CURRENT_NON_INTEREST_BEARING_UPB',
#         'BORROWER_ASSISTANCE_STATUS_CODE',
#         'ELTV',
#         'DELINQUENCY_DUE_TO_DISASTER',
        
#         # Behavioral features
#         'max_delinquency_3m',
#         'max_delinquency_6m',
#         'max_delinquency_12m',
#         'rolling_mean_delinquency_6m',
#         'num_delinquent_months_12m',
#         'consecutive_delinquent_months',
#         'months_since_last_delinquency',
#         'delinquency_trend_6m',
#         'remaining_balance_pct',
#         'principal_paid_pct',
#         'balance_change_3m',
#         'balance_change_6m',
#         'balance_change_12m',
#         'rolling_avg_balance_6m',
#         'rate_change_since_origination',
#         'rate_reduction_after_mod',
#         'ever_modified',
#         'num_modifications',
#         'months_since_modification',
#         'payment_deferral_count',
#         'remaining_term_pct',
#         'observation_month',
#         'observation_quarter',
#         'observation_year',
#         'loan_age_squared',
#         'loan_age_cubic',
#         'seasonality_sin',
#         'seasonality_cos',
#         'dti_ltv_interaction',
#         'credit_dti_interaction',
#         'balance_delinquency_interaction',
#         'age_balance_interaction'
#     ])
    
#     # Features to drop
#     drop_features: List[str] = field(default_factory=lambda: [
#         'LOAN_SEQUENCE_NUMBER',
#         'MONTHLY_REPORTING_PERIOD',
#         'FIRST_PAYMENT_DATE',
#         'MATURITY_DATE',
#         'POSTAL_CODE',
#         'SELLER_NAME',
#         'SERVICER_NAME',
#         'ingestion_year',
#         'ingestion_timestamp',
#         'reporting_year',
#         'delinquency_numeric',
#         'is_delinquent',
#         'is_terminated',
#         'future_termination',
#         'future_delinquency_max',
#         'delinquency_days',
#         # Previously only excluded via a hardcoded inline list inside
#         # DatasetCreator._select_model_features() (`drop_features +
#         # ['row_num', 'cumulative_delinquency']`), invisible from this
#         # config file. Made explicit here so the full drop list lives in
#         # one place; dataset_creation.py's inline addition is now
#         # redundant but harmless (list union, not a hard requirement).
#         'row_num',
#         'cumulative_delinquency',
#     ])
    
#     # Categorical features -- validated against column_profiling_eda.ipynb
#     # (dtype cross-check against static_features + behavioral_features).
#     # Every string-dtype allow-listed column is present here; unchanged.
#     categorical_features: List[str] = field(default_factory=lambda: [
#         'FIRST_TIME_HOMEBUYER_FLAG',
#         'OCCUPANCY_STATUS',
#         'PROPERTY_TYPE',
#         'LOAN_PURPOSE',
#         'PROPERTY_STATE',
#         'CHANNEL',
#         'SUPER_CONFORMING_FLAG',
#         'RELIEF_REFINANCE_INDICATOR',
#         'AMORTIZATION_TYPE',
#         'MODIFICATION_FLAG',
#         'PAYMENT_DEFERRAL_FLAG',
#         'BORROWER_ASSISTANCE_STATUS_CODE',
#         'DELINQUENCY_DUE_TO_DISASTER',
#         'CURRENT_LOAN_DELINQUENCY_STATUS'
#     ])
    
#     # Numerical features
#     numerical_features: List[str] = field(default_factory=lambda: [
#         'CREDIT_SCORE',
#         'ORIGINAL_LTV',
#         'ORIGINAL_CLTV',
#         'ORIGINAL_DTI',
#         'ORIGINAL_UPB',
#         'ORIGINAL_INTEREST_RATE',
#         'NUMBER_OF_BORROWERS',
#         'MI_PERCENTAGE',
#         'ORIGINAL_LOAN_TERM',
#         'CURRENT_ACTUAL_UPB',
#         'CURRENT_INTEREST_RATE',
#         'LOAN_AGE',
#         'REMAINING_MONTHS_TO_LEGAL_MATURITY',
#         'INTEREST_BEARING_UPB',
#         'CURRENT_NON_INTEREST_BEARING_UPB',
#         'ELTV',
#         'max_delinquency_3m',
#         'max_delinquency_6m',
#         'max_delinquency_12m',
#         'rolling_mean_delinquency_6m',
#         'num_delinquent_months_12m',
#         'consecutive_delinquent_months',
#         'months_since_last_delinquency',
#         'delinquency_trend_6m',
#         'remaining_balance_pct',
#         'principal_paid_pct',
#         'balance_change_3m',
#         'balance_change_6m',
#         'balance_change_12m',
#         'rolling_avg_balance_6m',
#         'rate_change_since_origination',
#         'rate_reduction_after_mod',
#         'ever_modified',
#         'num_modifications',
#         'months_since_modification',
#         'payment_deferral_count',
#         'remaining_term_pct',
#         'observation_month',
#         'observation_quarter',
#         'observation_year',
#         'loan_age_squared',
#         'loan_age_cubic',
#         'seasonality_sin',
#         'seasonality_cos',
#         'dti_ltv_interaction',
#         'credit_dti_interaction',
#         'balance_delinquency_interaction',
#         'age_balance_interaction'
#     ])

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
    
    # Features to drop - INCLUDING ALL CONSTANT COLUMNS FROM EDA
    drop_features: List[str] = field(default_factory=lambda: [
        # Identifiers (always drop)
        'LOAN_SEQUENCE_NUMBER',
        'MONTHLY_REPORTING_PERIOD',
        'FIRST_PAYMENT_DATE',
        'MATURITY_DATE',
        'POSTAL_CODE',
        'SELLER_NAME',
        'SERVICER_NAME',
        'MSA',  # High cardinality, mostly null
        
        # EMPTY/CONSTANT COLUMNS FROM PERFORMANCE DATA (99%+ null)
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
        'ELTV',
        'ZERO_BALANCE_REMOVAL_UPB',
        'DELINQUENT_ACCRUED_INTEREST',
        'DELINQUENCY_DUE_TO_DISASTER',
        'BORROWER_ASSISTANCE_STATUS_CODE',
        'CURRENT_MONTH_MODIFICATION_COST',
        
        # CONSTANT/NON-INFORMATIVE COLUMNS FROM ORIGINATION
        'AMORTIZATION_TYPE',  # Only 'FRM' - constant
        'PROPERTY_VALUATION_METHOD',  # Only 7 - constant
        'IO_INDICATOR',  # Only 'N' - constant
        'MI_CANCELLATION_INDICATOR',  # Only '9' - constant
        'SPECIAL_ELIGIBILITY_PROGRAM',  # Only '9' - constant
        'PRE_RELIEF_REFINANCE_LSN',  # Mostly null
        'SUPER_CONFORMING_FLAG',  # 99% null
        'RELIEF_REFINANCE_INDICATOR',  # 92% null
        
        # FEATURES THAT BECAME CONSTANT AFTER ENGINEERING
        'vintage_year',
        'origination_year',
        'CURRENT_NON_INTEREST_BEARING_UPB',
        'delinquency_streak_id',
        'rate_change_since_origination',
        'rate_reduction_after_mod',
        'ever_modified',
        'num_modifications',
        'last_modification_month',
        'months_since_modification',
        'payment_deferral_count',
        'rolling_std_balance_6m',
        'rolling_std_rate_6m',
        'rolling_avg_eltv_6m',
        'rolling_std_eltv_6m',
        'rolling_min_eltv_6m',
        'rolling_max_eltv_6m',
        'dti_ltv_interaction',
        'is_terminated',
        'cumulative_delinquency',
        
        # Already in drop_features
        'ingestion_year',
        'ingestion_timestamp',
        'reporting_year',
        'delinquency_numeric',
        'is_delinquent',
        'future_termination',
        'future_delinquency_max',
        'delinquency_days',
        'row_num',
    ])
    
    # Categorical features (ONLY non-constant, meaningful columns)
    categorical_features: List[str] = field(default_factory=lambda: [
        'FIRST_TIME_HOMEBUYER_FLAG',  # 3 values: N, Y, 9
        'OCCUPANCY_STATUS',            # 3 values: P, S, I
        'PROPERTY_TYPE',               # 6 values: CP, SF, PU, MH, CO
        'LOAN_PURPOSE',                # 4 values: N, C, P, 9
        'CHANNEL',                     # 5 values: B, C, R, T, 9
        'CURRENT_LOAN_DELINQUENCY_STATUS',  # 12 values: 0-9, RA
        'PROPERTY_STATE',              # 54 values: US states
        'NUMBER_OF_BORROWERS',         # 3 values: 1, 2, 3
        'NUMBER_OF_UNITS',             # 5 values: 1-4, 9
        'MODIFICATION_FLAG',           # 3 values: P, Y, null
        'ZERO_BALANCE_CODE',           # 8 values: 01, 02, 03, 09, 96
        'PPM_FLAG',                    # 2 values: N, Y
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
    memory_limit: str = "8GB"
    npartitions: int = 4


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