# =============================================================================
# BEHAVIORAL CREDIT RISK SCORING SYSTEM
# Module-by-Module Execution Notebook
# =============================================================================

"""
This notebook allows you to run each module of the credit risk scoring system
independently. Each section has:
1. A commented section to LOAD pre-processed data (skip processing)
2. An active section to PROCESS data from scratch

Use this to debug specific modules or run the pipeline incrementally.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from datetime import datetime
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# SETUP & CONFIGURATION
# =============================================================================

# Import configuration
from config.config import config
paths = config['paths']
model_config = config['model']
feature_config = config['features']

# Create directories
for d in [paths.bronze_dir, paths.silver_dir, paths.features_dir, 
          paths.models_dir, paths.results_dir, paths.eda_dir]:
    os.makedirs(d, exist_ok=True)

print("=" * 80)
print("BEHAVIORAL CREDIT RISK SCORING SYSTEM")
print(f"Started at: {datetime.now()}")
print("=" * 80)
print(f"Data Directory: {paths.data_dir}")
print(f"Bronze Directory: {paths.bronze_dir}")
print(f"Silver Directory: {paths.silver_dir}")
print(f"Features Directory: {paths.features_dir}")
print(f"Models Directory: {paths.models_dir}")
print(f"Results Directory: {paths.results_dir}")
print("=" * 80)

# =============================================================================
# SPARK SESSION CREATION
# =============================================================================

from data_ingestion.data_ingestion import SFLLDDataIngestion, create_spark_session

# Create Spark session
spark = create_spark_session()
print(f"Spark Session created: {spark.version}")

# Initialize components
ingestor = SFLLDDataIngestion(spark)

# =============================================================================
# MODULE 1: DATA INGESTION
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 1: DATA INGESTION")
print("=" * 80)

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING BRONZE DATA (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING DATA - Use this if you already have bronze data
print("\nLoading existing bronze data...")
origination_df = spark.read.parquet(paths.origination_bronze)
performance_df = spark.read.parquet(paths.performance_bronze)
print(f"Origination: {origination_df.count():,} loans")
print(f"Performance: {performance_df.count():,} records")
"""

# -----------------------------------------------------------------------------
# OPTION B: PROCESS DATA FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# PROCESS FROM SCRATCH - Use this for first time or to re-ingest
print("\nIngesting data from scratch...")

raw_dir = paths.raw_dir
years = list(range(1999, 2013))

# Ingest origination
print("Ingesting origination data...")
origination_df = ingestor.ingest_all_years(
    raw_dir=raw_dir,
    years=years,
    bronze_dir=paths.bronze_dir,
    file_prefix="sample",
    data_type="origination"
)

# Ingest performance
print("Ingesting performance data...")
performance_df = ingestor.ingest_all_years(
    raw_dir=raw_dir,
    years=years,
    bronze_dir=paths.bronze_dir,
    file_prefix="sample",
    data_type="performance"
)

print(f"Origination: {origination_df.count():,} loans")
print(f"Performance: {performance_df.count():,} records")
"""

# =============================================================================
# MODULE 2: DATA CLEANING
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 2: DATA CLEANING")
print("=" * 80)

from preprocessing.cleaning import SFLLDDataCleaner

# Initialize cleaner
cleaner = SFLLDDataCleaner(spark)

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING SILVER DATA (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING CLEANED DATA
print("\nLoading existing cleaned data...")
orig_cleaned = spark.read.parquet(f"{paths.silver_dir}/origination_cleaned.parquet")
perf_cleaned = spark.read.parquet(f"{paths.silver_dir}/performance_cleaned.parquet")
print(f"Cleaned Origination: {orig_cleaned.count():,} loans")
print(f"Cleaned Performance: {perf_cleaned.count():,} records")
"""

# -----------------------------------------------------------------------------
# OPTION B: PROCESS DATA FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# PROCESS FROM SCRATCH - Clean bronze data
print("\nCleaning data from scratch...")

# Load bronze data if not already loaded
if 'origination_df' not in dir():
    origination_df = spark.read.parquet(paths.origination_bronze)
if 'performance_df' not in dir():
    performance_df = spark.read.parquet(paths.performance_bronze)

# Clean data
orig_cleaned, perf_cleaned = cleaner.clean_both_datasets(
    origination_df, performance_df
)

# Save to silver layer
orig_cleaned.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(f"{paths.silver_dir}/origination_cleaned.parquet")

perf_cleaned.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(f"{paths.silver_dir}/performance_cleaned.parquet")

print(f"Cleaned Origination: {orig_cleaned.count():,} loans")
print(f"Cleaned Performance: {perf_cleaned.count():,} records")
"""

# =============================================================================
# MODULE 3: FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 3: FEATURE ENGINEERING")
print("=" * 80)

from features.behavioral_features import BehavioralFeatureEngineer

# Initialize feature engineer
feature_engineer = BehavioralFeatureEngineer(spark)

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING FEATURES (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING FEATURES
print("\nLoading existing feature data...")
feature_df = spark.read.parquet(paths.feature_dataset)
print(f"Features loaded: {feature_df.count():,} records")
print(f"Feature count: {len(feature_df.columns)}")
"""

# -----------------------------------------------------------------------------
# OPTION B: PROCESS DATA FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# PROCESS FROM SCRATCH - Create features
print("\nCreating features from scratch...")

# Load cleaned data if not already loaded
if 'orig_cleaned' not in dir():
    orig_cleaned = spark.read.parquet(f"{paths.silver_dir}/origination_cleaned.parquet")
if 'perf_cleaned' not in dir():
    perf_cleaned = spark.read.parquet(f"{paths.silver_dir}/performance_cleaned.parquet")

# Create features
feature_df = feature_engineer.create_all_features(
    orig_cleaned, perf_cleaned
)

# Remove duplicate columns
seen = set()
cols_to_keep = []
for col_name in feature_df.columns:
    if col_name not in seen:
        seen.add(col_name)
        cols_to_keep.append(col_name)
feature_df = feature_df.select(*cols_to_keep)

# Save feature dataset
feature_df.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(paths.feature_dataset)

print(f"Features created: {feature_df.count():,} records")
print(f"Feature count: {len(feature_df.columns)}")
"""

# =============================================================================
# MODULE 4: TARGET CREATION
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 4: TARGET CREATION")
print("=" * 80)

from target.target_creation import TargetCreator

# Initialize target creator
target_creator = TargetCreator(
    spark=spark,
    default_threshold=model_config.default_threshold,
    lookahead_months=model_config.lookahead_months
)

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING DATASET WITH TARGET (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING DATASET WITH TARGET
print("\nLoading existing dataset with target...")
dataset_df = spark.read.parquet(f"{paths.features_dir}/dataset_with_target.parquet")
print(f"Dataset loaded: {dataset_df.count():,} records")
target_creator.analyze_target_distribution(dataset_df)
"""

# -----------------------------------------------------------------------------
# OPTION B: PROCESS DATA FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# PROCESS FROM SCRATCH - Create target
print("\nCreating target from scratch...")

# Load feature data if not already loaded
if 'feature_df' not in dir():
    feature_df = spark.read.parquet(paths.feature_dataset)

# Create target
dataset_df = target_creator.create_target(feature_df)

# Analyze target distribution
target_creator.analyze_target_distribution(dataset_df)

# Save dataset
dataset_df.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(f"{paths.features_dir}/dataset_with_target.parquet")

print(f"Dataset with target saved: {dataset_df.count():,} records")
"""

# =============================================================================
# MODULE 5: DATASET CREATION & SPLIT
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 5: DATASET CREATION & SPLIT")
print("=" * 80)

from validation.splitter import DataSplitter

# Initialize splitter
splitter = DataSplitter(spark)

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING SPLITS (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING SPLITS
print("\nLoading existing data splits...")

train_df = spark.read.parquet(paths.train_data)
val_df = spark.read.parquet(paths.val_data)
test_df = spark.read.parquet(paths.test_data)

print(f"Train: {train_df.count():,} records")
print(f"Validation: {val_df.count():,} records")
print(f"Test: {test_df.count():,} records")
"""

# -----------------------------------------------------------------------------
# OPTION B: PROCESS DATA FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# PROCESS FROM SCRATCH - Split data
print("\nSplitting data from scratch...")

# Load dataset with target if not already loaded
if 'dataset_df' not in dir():
    dataset_df = spark.read.parquet(f"{paths.features_dir}/dataset_with_target.parquet")

# Split data
train_df, val_df, test_df = splitter.split_data(
    dataset_df,
    train_start_year=model_config.train_start_year,
    train_end_year=model_config.train_end_year,
    test_start_year=model_config.test_start_year,
    test_end_year=model_config.test_end_year,
    val_frac=model_config.val_frac
)

# Save splits
train_df.write.mode("overwrite").parquet(paths.train_data)
val_df.write.mode("overwrite").parquet(paths.val_data)
test_df.write.mode("overwrite").parquet(paths.test_data)

print(f"Train: {train_df.count():,} records")
print(f"Validation: {val_df.count():,} records")
print(f"Test: {test_df.count():,} records")
"""

# =============================================================================
# MODULE 6: PREPARE PANDAS DATA FOR MODELING
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 6: PREPARE PANDAS DATA")
print("=" * 80)

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING PANDAS DATA (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING PANDAS DATA
print("\nLoading existing pandas data...")

pkl_dir = f"{paths.features_dir}/pandas"
X_train = pd.read_pickle(f"{pkl_dir}/X_train.pkl")
y_train = pd.read_pickle(f"{pkl_dir}/y_train.pkl")
X_val = pd.read_pickle(f"{pkl_dir}/X_val.pkl")
y_val = pd.read_pickle(f"{pkl_dir}/y_val.pkl")
X_test = pd.read_pickle(f"{pkl_dir}/X_test.pkl")
y_test = pd.read_pickle(f"{pkl_dir}/y_test.pkl")

print(f"Train: {len(X_train):,} records")
print(f"Validation: {len(X_val):,} records")
print(f"Test: {len(X_test):,} records")
print(f"Features: {X_train.shape[1]}")
"""

# -----------------------------------------------------------------------------
# OPTION B: PROCESS DATA FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# PROCESS FROM SCRATCH - Prepare pandas data
print("\nPreparing pandas data from scratch...")

# Load splits if not already loaded
if 'train_df' not in dir():
    train_df = spark.read.parquet(paths.train_data)
    val_df = spark.read.parquet(paths.val_data)
    test_df = spark.read.parquet(paths.test_data)

def prepare_features(df):
    """Prepare features from Spark DataFrame."""
    feature_cols = [c for c in df.columns if c not in 
                   ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'target']]
    pdf = df.select(*feature_cols).toPandas()
    pdf = pdf.replace([np.inf, -np.inf], np.nan)
    pdf.fillna(0, inplace=True)
    return pdf

X_train = prepare_features(train_df)
y_train = train_df.select("target").toPandas()["target"]
X_val = prepare_features(val_df)
y_val = val_df.select("target").toPandas()["target"]
X_test = prepare_features(test_df)
y_test = test_df.select("target").toPandas()["target"]

# Save pandas data
pkl_dir = f"{paths.features_dir}/pandas"
os.makedirs(pkl_dir, exist_ok=True)
pd.to_pickle(X_train, f"{pkl_dir}/X_train.pkl")
pd.to_pickle(y_train, f"{pkl_dir}/y_train.pkl")
pd.to_pickle(X_val, f"{pkl_dir}/X_val.pkl")
pd.to_pickle(y_val, f"{pkl_dir}/y_val.pkl")
pd.to_pickle(X_test, f"{pkl_dir}/X_test.pkl")
pd.to_pickle(y_test, f"{pkl_dir}/y_test.pkl")

print(f"Train: {len(X_train):,} records")
print(f"Validation: {len(X_val):,} records")
print(f"Test: {len(X_test):,} records")
print(f"Features: {X_train.shape[1]}")
"""

# =============================================================================
# MODULE 7: HYPERPARAMETER TUNING
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 7: HYPERPARAMETER TUNING")
print("=" * 80)

from models.hyperparameter_tuning import HyperparameterTuner

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING TUNED PARAMS (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING TUNED PARAMS
print("\nLoading existing tuned parameters...")
with open(f"{paths.results_dir}/tuned_params.json", 'r') as f:
    tuned_params = json.load(f)
print("Loaded tuned parameters:")
for model, params in tuned_params.items():
    print(f"  {model}: {params}")
"""

# -----------------------------------------------------------------------------
# OPTION B: RUN TUNING FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# RUN TUNING FROM SCRATCH
print("\nRunning hyperparameter tuning...")

# Ensure data is loaded
if 'X_train' not in dir():
    pkl_dir = f"{paths.features_dir}/pandas"
    X_train = pd.read_pickle(f"{pkl_dir}/X_train.pkl")
    y_train = pd.read_pickle(f"{pkl_dir}/y_train.pkl")

# Create tuner
tuner = HyperparameterTuner(
    n_trials=20,  # Reduce for faster execution
    cv_folds=3,
    random_state=model_config.random_state
)

tuned_params = {}

# Tune each model
models_to_tune = [
    ('logistic', tuner.tune_logistic_regression),
    ('random_forest', tuner.tune_random_forest),
    ('xgboost', tuner.tune_xgboost),
    ('lightgbm', tuner.tune_lightgbm),
    ('catboost', tuner.tune_catboost)
]

for name, tune_func in models_to_tune:
    print(f"\nTuning {name}...")
    try:
        params = tune_func(X_train, y_train)
        tuned_params[name] = params
    except Exception as e:
        print(f"Could not tune {name}: {e}")
        tuned_params[name] = {}

# Save tuned parameters
with open(f"{paths.results_dir}/tuned_params.json", 'w') as f:
    json.dump(tuned_params, f, indent=2)

print("\nTuning completed!")
for model, params in tuned_params.items():
    print(f"  {model}: {params}")
"""

# =============================================================================
# MODULE 8: MODEL TRAINING
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 8: MODEL TRAINING")
print("=" * 80)

from models.logistic import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.ensemble import StackingEnsemble

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING MODELS (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING MODELS
print("\nLoading existing models...")

models = {}
model_names = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'ensemble']
for name in model_names:
    try:
        with open(f"{paths.models_dir}/model_{name}.pkl", 'rb') as f:
            models[name] = pickle.load(f)
        print(f"Loaded {name} model")
    except Exception as e:
        print(f"Could not load {name} model: {e}")

print(f"Loaded {len(models)} models")
"""

# -----------------------------------------------------------------------------
# OPTION B: TRAIN MODELS FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# TRAIN MODELS FROM SCRATCH
print("\nTraining models from scratch...")

# Ensure data is loaded
if 'X_train' not in dir():
    pkl_dir = f"{paths.features_dir}/pandas"
    X_train = pd.read_pickle(f"{pkl_dir}/X_train.pkl")
    y_train = pd.read_pickle(f"{pkl_dir}/y_train.pkl")
    X_val = pd.read_pickle(f"{pkl_dir}/X_val.pkl")
    y_val = pd.read_pickle(f"{pkl_dir}/y_val.pkl")

# Load tuned parameters if available
try:
    with open(f"{paths.results_dir}/tuned_params.json", 'r') as f:
        tuned_params = json.load(f)
    print("Loaded tuned parameters")
except:
    tuned_params = {}
    print("No tuned parameters found, using defaults")

models = {}

# 1. Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegressionModel(
    random_state=model_config.random_state,
    **tuned_params.get('logistic', {})
)
lr.fit(X_train, y_train)
models['logistic'] = lr

# 2. Random Forest
print("\nTraining Random Forest...")
rf = RandomForestModel(
    random_state=model_config.random_state,
    **tuned_params.get('random_forest', {})
)
rf.fit(X_train, y_train)
models['random_forest'] = rf

# 3. XGBoost
print("\nTraining XGBoost...")
xgb = XGBoostModel(
    random_state=model_config.random_state,
    **tuned_params.get('xgboost', {})
)
xgb.fit(X_train, y_train, X_val, y_val)
models['xgboost'] = xgb

# 4. LightGBM
print("\nTraining LightGBM...")
lgb = LightGBMModel(
    random_state=model_config.random_state,
    **tuned_params.get('lightgbm', {})
)
lgb.fit(X_train, y_train, X_val, y_val)
models['lightgbm'] = lgb

# 5. CatBoost
print("\nTraining CatBoost...")
cat = CatBoostModel(
    random_state=model_config.random_state,
    **tuned_params.get('catboost', {})
)
cat.fit(X_train, y_train, X_val, y_val)
models['catboost'] = cat

# 6. Ensemble
print("\nTraining Stacking Ensemble...")
ensemble = StackingEnsemble(
    base_models=list(models.values()),
    meta_model=LightGBMModel(
        random_state=model_config.random_state,
        is_unbalance=True
    ),
    cv_folds=3,
    random_state=model_config.random_state
)
ensemble.fit(X_train, y_train, X_val, y_val)
models['ensemble'] = ensemble

# Save models
for name, model in models.items():
    with open(f"{paths.models_dir}/model_{name}.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name} model")

print(f"\nTraining completed! {len(models)} models trained.")
"""

# =============================================================================
# MODULE 9: MODEL EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 9: MODEL EVALUATION")
print("=" * 80)

from evaluation.metrics import CreditRiskMetrics

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING RESULTS (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING RESULTS
print("\nLoading existing evaluation results...")
results_df = pd.read_csv(f"{paths.results_dir}/model_results.csv")
print(results_df.to_string())
"""

# -----------------------------------------------------------------------------
# OPTION B: RUN EVALUATION FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# RUN EVALUATION FROM SCRATCH
print("\nRunning model evaluation...")

# Ensure data is loaded
if 'X_test' not in dir():
    pkl_dir = f"{paths.features_dir}/pandas"
    X_test = pd.read_pickle(f"{pkl_dir}/X_test.pkl")
    y_test = pd.read_pickle(f"{pkl_dir}/y_test.pkl")

# Ensure models are loaded
if 'models' not in dir() or not models:
    model_names = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'ensemble']
    models = {}
    for name in model_names:
        try:
            with open(f"{paths.models_dir}/model_{name}.pkl", 'rb') as f:
                models[name] = pickle.load(f)
            print(f"Loaded {name} model")
        except Exception as e:
            print(f"Could not load {name} model: {e}")

results = {}
metrics_calculator = CreditRiskMetrics()

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = metrics_calculator.evaluate(y_test, y_pred, y_proba)
    metrics['name'] = name
    metrics['y_proba'] = y_proba
    metrics['y_pred'] = y_pred
    
    # Get feature importance
    if hasattr(model, 'get_feature_importance'):
        metrics['feature_importance'] = model.get_feature_importance()
    
    # Get lift/gain
    lift_data = metrics_calculator.compute_lift_gain(y_test, y_proba)
    metrics['lift_data'] = lift_data
    
    results[name] = metrics
    
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  KS: {metrics['ks_statistic']:.4f}")

# Save results
results_df = pd.DataFrame([
    {
        'model': name,
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'mcc': metrics['mcc'],
        'brier_score': metrics['brier_score'],
        'log_loss': metrics['log_loss'],
        'ks_statistic': metrics['ks_statistic']
    }
    for name, metrics in results.items()
])
results_df.to_csv(f"{paths.results_dir}/model_results.csv", index=False)

print("\nMODEL PERFORMANCE SUMMARY:")
print("-" * 80)
print(f"{'Model':<20} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<10} {'KS':<10}")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['model']:<20} {row['roc_auc']:<10.4f} {row['pr_auc']:<10.4f} "
          f"{row['f1']:<10.4f} {row['ks_statistic']:<10.4f}")
print("-" * 80)
"""

# =============================================================================
# MODULE 10: PROBABILITY CALIBRATION
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 10: PROBABILITY CALIBRATION")
print("=" * 80)

from evaluation.calibration import ProbabilityCalibrator

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING CALIBRATED MODELS (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING CALIBRATED MODELS
print("\nLoading existing calibrated models...")
calibrated_models = {}
model_names = ['logistic_calibrated', 'random_forest_calibrated', 'xgboost_calibrated',
               'lightgbm_calibrated', 'catboost_calibrated', 'ensemble_calibrated']
for name in model_names:
    try:
        with open(f"{paths.models_dir}/model_{name}.pkl", 'rb') as f:
            calibrated_models[name] = pickle.load(f)
        print(f"Loaded {name}")
    except Exception as e:
        print(f"Could not load {name}: {e}")
"""

# -----------------------------------------------------------------------------
# OPTION B: RUN CALIBRATION FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# RUN CALIBRATION FROM SCRATCH
print("\nRunning probability calibration...")

# Ensure data is loaded
if 'X_train' not in dir():
    pkl_dir = f"{paths.features_dir}/pandas"
    X_train = pd.read_pickle(f"{pkl_dir}/X_train.pkl")
    y_train = pd.read_pickle(f"{pkl_dir}/y_train.pkl")
    X_val = pd.read_pickle(f"{pkl_dir}/X_val.pkl")
    y_val = pd.read_pickle(f"{pkl_dir}/y_val.pkl")

# Ensure models are loaded
if 'models' not in dir() or not models:
    model_names = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'ensemble']
    models = {}
    for name in model_names:
        try:
            with open(f"{paths.models_dir}/model_{name}.pkl", 'rb') as f:
                models[name] = pickle.load(f)
            print(f"Loaded {name} model")
        except Exception as e:
            print(f"Could not load {name} model: {e}")

calibrated_models = {}

for name, model in models.items():
    print(f"\nCalibrating {name}...")
    
    calibrator = ProbabilityCalibrator(method='isotonic')
    calibrated_model = calibrator.calibrate(
        model, X_train, y_train, X_val, y_val
    )
    calibrated_models[f"{name}_calibrated"] = calibrated_model

# Save calibrated models
for name, model in calibrated_models.items():
    with open(f"{paths.models_dir}/model_{name}.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name}")

print("\nCalibration completed!")
"""

# =============================================================================
# MODULE 11: CREDIT SCORE GENERATION
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 11: CREDIT SCORE GENERATION")
print("=" * 80)

from scoring.score_generator import CreditScoreGenerator

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING SCORES (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING SCORES
print("\nLoading existing credit scores...")
score_results = pd.read_csv(f"{paths.results_dir}/scores.csv")
print(f"Loaded {len(score_results):,} scores")
print("\nScore Distribution:")
print(score_results['credit_score'].describe())
"""

# -----------------------------------------------------------------------------
# OPTION B: GENERATE SCORES FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# GENERATE SCORES FROM SCRATCH
print("\nGenerating credit scores...")

# Ensure data is loaded
if 'X_test' not in dir():
    pkl_dir = f"{paths.features_dir}/pandas"
    X_test = pd.read_pickle(f"{pkl_dir}/X_test.pkl")
    y_test = pd.read_pickle(f"{pkl_dir}/y_test.pkl")

# Load ensemble model
try:
    with open(f"{paths.models_dir}/model_ensemble.pkl", 'rb') as f:
        ensemble = pickle.load(f)
    print("Loaded ensemble model")
except:
    with open(f"{paths.models_dir}/model_lightgbm.pkl", 'rb') as f:
        ensemble = pickle.load(f)
    print("Loaded LightGBM model (ensemble not available)")

# Get probabilities
y_proba = ensemble.predict_proba(X_test)[:, 1]

# Create score generator
score_generator = CreditScoreGenerator(
    min_score=300,
    max_score=900,
    target_default_rate=0.05,
    pdo=20
)

# Generate scores
score_results = pd.DataFrame({
    'probability': y_proba,
    'true_label': y_test
})

score_results = score_generator.generate_all_scores(
    score_results,
    prob_col='probability',
    score_col='credit_score'
)

# Score distribution
print("\nScore Distribution:")
print(score_results['credit_score'].describe())

# Risk band analysis
print("\nRisk Band Analysis:")
for band in ['Low', 'Medium', 'High']:
    band_data = score_results[score_results['risk_band'] == band]
    if len(band_data) > 0:
        default_rate = band_data['true_label'].mean()
        print(f"  {band}:")
        print(f"    Observations: {len(band_data):,}")
        print(f"    Default Rate: {default_rate:.2%}")
        print(f"    % of Portfolio: {len(band_data)/len(score_results):.2%}")

# Save score results
score_results.to_csv(f"{paths.results_dir}/scores.csv", index=False)
print(f"\nScores saved to {paths.results_dir}/scores.csv")
"""

# =============================================================================
# MODULE 12: VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("MODULE 12: VISUALIZATION")
print("=" * 80)

from evaluation.plots import CreditRiskVisualizer

# -----------------------------------------------------------------------------
# OPTION A: LOAD EXISTING VISUALIZATIONS (Skip Processing)
# -----------------------------------------------------------------------------
"""
# LOAD EXISTING VISUALIZATIONS
print("\nExisting visualizations are in:")
print(f"  {paths.results_dir}/plots/")
import os
if os.path.exists(f"{paths.results_dir}/plots"):
    for f in os.listdir(f"{paths.results_dir}/plots"):
        print(f"    {f}")
"""

# -----------------------------------------------------------------------------
# OPTION B: CREATE VISUALIZATIONS FROM SCRATCH
# -----------------------------------------------------------------------------
"""
# CREATE VISUALIZATIONS FROM SCRATCH
print("\nCreating visualizations...")

# Ensure data is loaded
if 'X_test' not in dir():
    pkl_dir = f"{paths.features_dir}/pandas"
    X_test = pd.read_pickle(f"{pkl_dir}/X_test.pkl")
    y_test = pd.read_pickle(f"{pkl_dir}/y_test.pkl")

# Ensure results are available
if 'results' not in dir() or not results:
    # Load evaluation results
    results_df = pd.read_csv(f"{paths.results_dir}/model_results.csv")
    results = {}
    for _, row in results_df.iterrows():
        name = row['model']
        # Load predictions if available
        try:
            preds = pd.read_csv(f"{paths.results_dir}/predictions_{name}.csv")
            results[name] = {
                'name': name,
                'roc_auc': row['roc_auc'],
                'pr_auc': row['pr_auc'],
                'f1': row['f1'],
                'precision': row['precision'],
                'recall': row['recall'],
                'balanced_accuracy': row['balanced_accuracy'],
                'mcc': row['mcc'],
                'brier_score': row['brier_score'],
                'log_loss': row['log_loss'],
                'ks_statistic': row['ks_statistic'],
                'y_proba': preds['y_proba'].values,
                'y_pred': preds['y_pred'].values
            }
        except:
            # Use sample predictions if not available
            model_name = f"model_{name}.pkl"
            try:
                with open(f"{paths.models_dir}/{model_name}", 'rb') as f:
                    model = pickle.load(f)
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                results[name] = {
                    'name': name,
                    'roc_auc': row['roc_auc'],
                    'pr_auc': row['pr_auc'],
                    'f1': row['f1'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'balanced_accuracy': row['balanced_accuracy'],
                    'mcc': row['mcc'],
                    'brier_score': row['brier_score'],
                    'log_loss': row['log_loss'],
                    'ks_statistic': row['ks_statistic'],
                    'y_proba': y_proba,
                    'y_pred': y_pred
                }
            except:
                pass

visualizer = CreditRiskVisualizer(save_dir=f"{paths.results_dir}/plots")

# Create individual model evaluation reports
for name, metrics in results.items():
    if 'y_proba' in metrics and 'feature_importance' in metrics:
        visualizer.create_model_evaluation_report(
            y_test,
            metrics['y_proba'],
            metrics['y_pred'],
            name,
            metrics.get('feature_importance'),
            save_name=f"{name}_report"
        )

# Create ensemble comparison plot
visualizer.create_ensemble_comparison_plot(
    results,
    save_name="ensemble_comparison"
)

print(f"\nVisualizations saved to {paths.results_dir}/plots/")
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_data_exists():
    """Check which data files exist."""
    files = {
        'Bronze Origination': paths.origination_bronze,
        'Bronze Performance': paths.performance_bronze,
        'Silver Origination': f"{paths.silver_dir}/origination_cleaned.parquet",
        'Silver Performance': f"{paths.silver_dir}/performance_cleaned.parquet",
        'Features': paths.feature_dataset,
        'Dataset with Target': f"{paths.features_dir}/dataset_with_target.parquet",
        'Train Data': paths.train_data,
        'Validation Data': paths.val_data,
        'Test Data': paths.test_data,
        'Tuned Parameters': f"{paths.results_dir}/tuned_params.json",
        'Models': paths.models_dir,
        'Results': f"{paths.results_dir}/model_results.csv",
        'Scores': f"{paths.results_dir}/scores.csv"
    }
    
    print("\n" + "=" * 80)
    print("DATA AVAILABILITY CHECK")
    print("=" * 80)
    
    for name, path in files.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
    
    # Check model files
    model_names = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'ensemble']
    print("\nModels:")
    for name in model_names:
        path = f"{paths.models_dir}/model_{name}.pkl"
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}")

def print_module_usage():
    """Print usage instructions for this notebook."""
    print("\n" + "=" * 80)
    print("MODULE USAGE INSTRUCTIONS")
    print("=" * 80)
    print("""
    To run a specific module:
    1. Uncomment the OPTION A (LOAD) section to use existing data
    2. OR uncomment the OPTION B (PROCESS) section to process from scratch
    3. Comment out the other option
    4. Run the cell
    
    Recommended workflow:
    1. First time: Run all modules with OPTION B (PROCESS)
    2. Subsequent runs: Use OPTION A (LOAD) for modules you don't want to reprocess
    
    To check what data exists:
    check_data_exists()
    
    Module Dependencies:
    - Module 1 (Ingestion) -> Bronze data
    - Module 2 (Cleaning) -> Silver data
    - Module 3 (Feature Engineering) -> Features
    - Module 4 (Target Creation) -> Dataset with target
    - Module 5 (Dataset Creation & Split) -> Train/Val/Test splits
    - Module 6 (Pandas Data) -> Pandas DataFrames
    - Module 7 (Tuning) -> Tuned parameters
    - Module 8 (Training) -> Trained models
    - Module 9 (Evaluation) -> Evaluation results
    - Module 10 (Calibration) -> Calibrated models
    - Module 11 (Scoring) -> Credit scores
    - Module 12 (Visualization) -> Plots
    """)

# Run this to check what data exists
# check_data_exists()

# Run this to see usage instructions
# print_module_usage()

print("\n" + "=" * 80)
print("NOTEBOOK READY")
print("=" * 80)
print("""
To use this notebook:
1. Check what data exists: check_data_exists()
2. Uncomment the desired option (LOAD or PROCESS) for each module
3. Run each cell sequentially
4. Use LOAD options for modules where you already have processed data
5. Use PROCESS options for modules you want to run from scratch
""")