# main.py
"""
Main pipeline for behavioral credit risk scoring system.
Supports module-by-module execution with Dask for modeling.
"""

import os
import sys
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
from pyspark.sql import SparkSession
from datetime import datetime
import pickle
import json
from typing import Optional, Dict, Any
import time
import threading

# Import project modules
from config.config import config
from data_ingestion.data_ingestion import SFLLDDataIngestion, create_spark_session
from preprocessing.cleaning import SFLLDDataCleaner
from features.behavioral_features import BehavioralFeatureEngineer
from target.target_creation import TargetCreator
from datasets.dataset_creation import DatasetCreator
from validation.splitter import DataSplitter

# Shared Dask client + lazy-loading utilities (see models/dask_utils.py) --
# this is the single client every model module reuses instead of each
# spinning up its own LocalCluster.
from models.dask_utils import get_dask_client, close_dask_client

# Import Dask-based models
from models.logistic import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.ensemble import StackingEnsemble
from models.hyperparameter_tuning import HyperparameterTuner

# Import evaluation modules
from evaluation.metrics import CreditRiskMetrics
from evaluation.calibration import ProbabilityCalibrator
from evaluation.plots import CreditRiskVisualizer

# Import scoring
from scoring.score_generator import CreditScoreGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('credit_risk_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CreditRiskPipeline:
    """
    Main pipeline orchestrator for credit risk modeling.
    Supports module-by-module execution with Dask for modeling.
    """
    
    def __init__(
        self, 
        spark: Optional[SparkSession] = None, 
        skip_data: bool = False,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        test_mode: bool = False 
    ):
        """
        Initialize the pipeline.
        
        Args:
            spark: Optional SparkSession. If None, creates a new one.
            skip_data: If True, skip all data preparation and use existing data.
            n_workers: Number of Dask workers
            threads_per_worker: Threads per Dask worker
            test_mode: If True, run in test mode with limited data
        """
        self.spark = spark or create_spark_session()
        self.paths = config['paths']
        self.model_config = config['model']
        self.feature_config = config['features']
        self.skip_data = skip_data
        self.test_mode = test_mode 
        
        # Initialize Spark components
        self.ingestor = SFLLDDataIngestion(self.spark)
        self.cleaner = SFLLDDataCleaner(self.spark)
        self.feature_engineer = BehavioralFeatureEngineer(self.spark)
        self.target_creator = TargetCreator(
            spark=self.spark,
            default_threshold=self.model_config.default_threshold,
            lookahead_months=self.model_config.lookahead_months
        )
        self.splitter = DataSplitter(self.spark)
        
        # Initialize Dask client.
        # FIX: this used to call `Client(...)` directly here, while EVERY
        # individual model in models/*.py ALSO called `Client(...)` inside
        # its own fit() -- meaning a run of this pipeline could spin up
        # 1 (here) + up to 6 (one per base model + ensemble) + many more
        # (one per hyperparameter-tuning trial) separate local Dask
        # clusters. Routing through `models.dask_utils.get_dask_client()`
        # makes this THE single shared client for the whole process; every
        # model module now reuses it instead of creating its own.
        logger.info("Initializing shared Dask client...")
        try:
            self.dask_client = get_dask_client(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=config['dask'].memory_limit,
            )
            logger.info(f"Dask client initialized with {n_workers} workers "
                        f"(dashboard: {self.dask_client.dashboard_link})")
        except Exception as e:
            logger.warning(f"Could not initialize Dask client: {e}. Using single-threaded mode.")
            self.dask_client = None
        
        # State tracking
        self.state = {
            'origination_df': None,
            'performance_df': None,
            'orig_cleaned': None,
            'perf_cleaned': None,
            'feature_df': None,
            'dataset_df': None,
            'train_df': None,
            'val_df': None,
            'test_df': None,
            'X_train': None,      # Dask DataFrame
            'y_train': None,      # Dask Series
            'X_val': None,        # Dask DataFrame
            'y_val': None,        # Dask Series
            'X_test': None,       # Dask DataFrame
            'y_test': None,       # Dask Series
            'models': {},
            'tuned_params': {},
            'results': {},
            'calibrated_models': {},
            'data_loaded': False,
            'feature_names': None
        }
        
        # Track which modules have been completed
        self.completed_modules = set()
        
        # Create directories
        self._create_directories()
        
        # If skip_data is True, try to load existing data
        if skip_data:
            self._load_existing_data()
        
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.paths.bronze_dir,
            self.paths.silver_dir,
            self.paths.features_dir,
            self.paths.models_dir,
            self.paths.results_dir,
            self.paths.eda_dir
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            
            # Verify write permission
            test_file = os.path.join(d, ".write_test")
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                logger.error(f"Cannot write to {d}: {e}")
                raise
    
    # Non-feature columns present in the Spark-written train/val/test parquet
    # (identifiers + target) that must be excluded from the X feature matrix.
    _NON_FEATURE_COLUMNS = ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'target']

    # =============================================================================
    # DATA LOADING - Uses sampled data when enabled
    # =============================================================================

    def _load_existing_data(self) -> bool:
        """Load existing data with sampling."""
        logger.info("Loading train/val/test splits...")

        try:
            if self.model_config.use_sample:
                train_path = self.paths.sampled_train_data
                val_path = self.paths.sampled_val_data
                test_path = self.paths.sampled_test_data
                logger.info(f"  Using sampled data from: {self.paths.sampled_features_dir}")
            else:
                train_path = self.paths.train_data
                val_path = self.paths.val_data
                test_path = self.paths.test_data
            
            train_ddf = dd.read_parquet(train_path)
            test_ddf = dd.read_parquet(test_path)
            
            target_partitions = 16
            train_ddf = train_ddf.repartition(npartitions=target_partitions)
            test_ddf = test_ddf.repartition(npartitions=target_partitions)

            self.state['X_train'], self.state['y_train'] = self._split_features_target(train_ddf)
            self.state['X_test'], self.state['y_test'] = self._split_features_target(test_ddf)

            if os.path.exists(val_path):
                val_ddf = dd.read_parquet(val_path)
                val_ddf = val_ddf.repartition(npartitions=target_partitions)
                self.state['X_val'], self.state['y_val'] = self._split_features_target(val_ddf)

            self.state['feature_names'] = list(self.state['X_train'].columns)
            self.state['data_loaded'] = True

            logger.info(f"  Train: {len(self.state['X_train']):,} records")
            logger.info(f"  Test: {len(self.state['X_test']):,} records")
            if self.state.get('X_val') is not None:
                logger.info(f"  Val: {len(self.state['X_val']):,} records")

            return True
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")
            return False
        

    def _split_features_target(self, ddf: dd.DataFrame):
        """Lazily split a combined (features + identifiers + target) Dask
        DataFrame into X (feature columns only) and y (target Series).
        Pure column selection -- no compute, no shuffle, no row sampling."""
        feature_cols = [c for c in ddf.columns if c not in self._NON_FEATURE_COLUMNS]
        X = ddf[feature_cols]
        y = ddf['target'].astype('int64')
        return X, y

    def _ensure_data_loaded(self, required_for: str = "modeling"):
        """
        Ensure data is loaded as Dask DataFrames.
        """
        if self.state['data_loaded']:
            return

        if self.skip_data:
            if self._load_existing_data():
                return
            else:
                raise RuntimeError(
                    f"Cannot proceed with {required_for}. Data not found and --skip_data is True. "
                    f"Please run data preparation first or remove --skip_data flag."
                )

        # Run dataset creation if needed -- this writes train/val/test
        # Parquet to disk via Spark (datasets/dataset_creation.py).
        if self.state['train_df'] is None:
            self._run_dataset_creation()

        # Read the Parquet Spark just wrote back in as lazy Dask
        # DataFrames. We deliberately re-read from disk here (rather than
        # trying to hand off Spark's in-memory DataFrame objects directly)
        # because that's the documented hand-off point in the target
        # architecture: "Spark ... Save Train/Val/Test as Parquet" ->
        # "Dask: Read Parquet lazily". It also means `--skip_data` runs
        # later reuse the exact same code path.
        if not self._load_existing_data():
            raise RuntimeError(
                "Dataset creation completed but the resulting Parquet splits "
                "could not be read back as Dask DataFrames. Check "
                f"{self.paths.train_data}, {self.paths.val_data}, {self.paths.test_data}."
            )
    
    def run_module(self, module: str, **kwargs):
        """
        Run a specific module of the pipeline.
        
        Args:
            module: Module name
            **kwargs: Additional arguments for the module
        """
        module_map = {
            'ingestion': self._run_ingestion,
            'cleaning': self._run_cleaning,
            'feature_engineering': self._run_feature_engineering,
            'target_creation': self._run_target_creation,
            'dataset_creation': self._run_dataset_creation,
            'tuning': self._run_tuning,
            'training': self._run_training,
            'evaluation': self._run_evaluation,
            'calibration': self._run_calibration,
            'scoring': self._run_scoring,
            'visualization': self._run_visualization,
            'full': self._run_full_pipeline
        }
        
        if module not in module_map:
            raise ValueError(f"Unknown module: {module}. Available: {list(module_map.keys())}")
        
        logger.info("=" * 80)
        logger.info(f"RUNNING MODULE: {module.upper()}")
        if self.skip_data:
            logger.info("NOTE: --skip_data is enabled. Using existing data only.")
        logger.info("=" * 80)
        
        module_map[module](**kwargs)
        
        self.completed_modules.add(module)
        
        logger.info("=" * 80)
        logger.info(f"MODULE {module.upper()} COMPLETED")
        logger.info("=" * 80)
    
    # =========================================================================
    # SPARK MODULES (Unchanged - kept as reference)
    # =========================================================================
    
    def _run_ingestion(self, **kwargs):
        """Run data ingestion module."""
        if self.skip_data:
            logger.info("SKIPPING ingestion (--skip_data is enabled)")
            return
        
        logger.info("Running Data Ingestion Module...")
        raw_dir = kwargs.get('raw_dir', self.paths.raw_dir)
        years = kwargs.get('years', list(range(1999, 2013)))
        force = kwargs.get('force', False)
        
        if not force and self._check_bronze_exists():
            logger.info("Bronze data already exists. Loading...")
            self.state['origination_df'] = self.spark.read.parquet(self.paths.origination_bronze)
            self.state['performance_df'] = self.spark.read.parquet(self.paths.performance_bronze)
            return
        
        self.state['origination_df'] = self.ingestor.ingest_all_years(
            raw_dir=raw_dir, years=years, bronze_dir=self.paths.bronze_dir,
            file_prefix="sample", data_type="origination"
        )
        self.state['performance_df'] = self.ingestor.ingest_all_years(
            raw_dir=raw_dir, years=years, bronze_dir=self.paths.bronze_dir,
            file_prefix="sample", data_type="performance"
        )
    
    def _run_cleaning(self, **kwargs):
        """Run data cleaning module."""
        if self.skip_data:
            logger.info("SKIPPING cleaning (--skip_data is enabled)")
            return
        
        logger.info("Running Data Cleaning Module...")
        force = kwargs.get('force', False)
        
        if self.state['origination_df'] is None:
            self.state['origination_df'] = self.spark.read.parquet(self.paths.origination_bronze)
        if self.state['performance_df'] is None:
            self.state['performance_df'] = self.spark.read.parquet(self.paths.performance_bronze)
        
        self.state['orig_cleaned'], self.state['perf_cleaned'] = self.cleaner.clean_both_datasets(
            self.state['origination_df'], self.state['performance_df']
        )
        
        self.state['orig_cleaned'].write.mode("overwrite").option("compression", "snappy") \
            .parquet(f"{self.paths.silver_dir}/origination_cleaned.parquet")
        self.state['perf_cleaned'].write.mode("overwrite").option("compression", "snappy") \
            .parquet(f"{self.paths.silver_dir}/performance_cleaned.parquet")
    
    def _run_feature_engineering(self, **kwargs):
        """Run feature engineering module."""
        if self.skip_data:
            logger.info("SKIPPING feature engineering (--skip_data is enabled)")
            return
        
        logger.info("Running Feature Engineering Module...")
        force = kwargs.get('force', False)
        
        if self.state['orig_cleaned'] is None:
            self.state['orig_cleaned'] = self.spark.read.parquet(f"{self.paths.silver_dir}/origination_cleaned.parquet")
            self.state['perf_cleaned'] = self.spark.read.parquet(f"{self.paths.silver_dir}/performance_cleaned.parquet")
        
        feature_df = self.feature_engineer.create_all_features(
            self.state['orig_cleaned'], self.state['perf_cleaned']
        )
        
        seen = set()
        cols_to_keep = []
        for col_name in feature_df.columns:
            if col_name not in seen:
                seen.add(col_name)
                cols_to_keep.append(col_name)
        feature_df = feature_df.select(*cols_to_keep)
        
        self.state['feature_df'] = feature_df
        
        feature_df.write.mode("overwrite").option("compression", "snappy") \
            .parquet(self.paths.feature_dataset)
    
    def _run_target_creation(self, **kwargs):
        """Run target creation module."""
        if self.skip_data:
            logger.info("SKIPPING target creation (--skip_data is enabled)")
            return
        
        logger.info("Running Target Creation Module...")
        threshold = kwargs.get('threshold', self.model_config.default_threshold)
        
        if self.state['feature_df'] is None:
            self.state['feature_df'] = self.spark.read.parquet(self.paths.feature_dataset)
        
        self.target_creator.default_threshold = threshold
        self.state['dataset_df'] = self.target_creator.create_target(
            self.state['feature_df'], threshold=threshold
        )
        
        self.target_creator.analyze_target_distribution(self.state['dataset_df'])
        
        self.state['dataset_df'].write.mode("overwrite").option("compression", "snappy") \
            .parquet(f"{self.paths.features_dir}/dataset_with_target.parquet")
    
    def _run_dataset_creation(self, **kwargs):
        """Run dataset creation module."""
        if self.skip_data:
            logger.info("SKIPPING dataset creation (--skip_data is enabled)")
            if self._load_existing_data():
                return
            raise RuntimeError(
                "Cannot run dataset creation with --skip_data. "
                "Data not found in parquet files."
            )
        
        logger.info("Running Dataset Creation Module...")
        
        if self.state['dataset_df'] is None:
            self.state['dataset_df'] = self.spark.read.parquet(f"{self.paths.features_dir}/dataset_with_target.parquet")
        
        dataset_creator = DatasetCreator(self.spark)
        self.state['train_df'], self.state['val_df'], self.state['test_df'] = dataset_creator.create_dataset()
        # `create_dataset()` already wrote train/val/test to
        # self.paths.train_data / val_data / test_data via Spark
        # (datasets/dataset_creation.py::_save_splits). Read that Parquet
        # back lazily as Dask -- no toPandas, no row sampling, no full
        # in-memory conversion.
        if not self._load_existing_data():
            raise RuntimeError(
                "Dataset creation completed but the resulting Parquet splits "
                "could not be read back as Dask DataFrames."
            )
    
    def _check_bronze_exists(self) -> bool:
        """Check if bronze data already exists."""
        try:
            self.spark.read.parquet(self.paths.origination_bronze)
            self.spark.read.parquet(self.paths.performance_bronze)
            return True
        except:
            return False
    
    # =========================================================================
    # DASK MODELING MODULES
    # =========================================================================
    
    def _run_tuning(self, **kwargs):
        """Run hyperparameter tuning module with Dask."""
        logger.info("Running Hyperparameter Tuning Module with Dask...")
        
        # FIX: Load SAMPLED data if use_sample is True
        if self.model_config.use_sample:
            logger.info("  Using SAMPLED data for tuning...")
            # Load sampled data directly
            train_path = self.paths.sampled_train_data
            val_path = self.paths.sampled_val_data
            test_path = self.paths.sampled_test_data
            
            X_train = dd.read_parquet(train_path)
            y_train = X_train['target']
            X_train = X_train.drop(columns=['target', 'LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD'])
            
            # Set state so other methods work
            self.state['X_train'] = X_train
            self.state['y_train'] = y_train
            self.state['data_loaded'] = True
        else:
            self._ensure_data_loaded("tuning")
        
        n_trials = kwargs.get('n_trials', self.model_config.n_trials)
        
        # Check if tuning already done
        tuned_params_path = f"{self.paths.results_dir}/tuned_params.json"
        if os.path.exists(tuned_params_path) and not kwargs.get('force', False):
            logger.info("Tuned parameters already exist. Loading...")
            with open(tuned_params_path, 'r') as f:
                self.state['tuned_params'] = json.load(f)
            return
        
        # Create tuner with sampling
        tuner = HyperparameterTuner(
            n_trials=min(n_trials, 30),
            cv_folds=min(3, self.model_config.cv_folds),
            random_state=self.model_config.random_state,
            sample_fraction=0.1  # Use 10% for tuning
        )
        
        # ... rest of tuning code ...
        
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
            logger.info(f"\nTuning {name}...")
            try:
                params = tune_func(self.state['X_train'], self.state['y_train'])
                tuned_params[name] = params
                logger.info(f"  === {name} tuning completed")
            except Exception as e:
                logger.warning(f"Could not tune {name}: {e}")
                tuned_params[name] = {}
        
        self.state['tuned_params'] = tuned_params
        
        with open(tuned_params_path, 'w') as f:
            json.dump(tuned_params, f, indent=2)
        
        logger.info("Hyperparameter tuning completed.")
    
    def _log_progress(self, stop_event, interval=300):
        """
        Log progress every `interval` seconds.
        
        Args:
            stop_event: threading.Event to signal stop
            interval: seconds between logs (default 300 = 5 min)
        """
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60
            
            # Check if Dask client is running
            dask_status = "Unknown"
            if hasattr(self, 'dask_client') and self.dask_client:
                try:
                    if self.dask_client.status == 'running':
                        # Get worker info
                        workers = self.dask_client.scheduler_info()['workers']
                        dask_status = f"Running ({len(workers)} workers)"
                    else:
                        dask_status = self.dask_client.status
                except:
                    dask_status = "Not available"
            
            # Get model training status if available
            model_status = "Training in progress..."
            if hasattr(self.state, 'current_model'):
                model_status = f"Training: {self.state.current_model}"
            
            logger.info(f"   PROGRESS UPDATE (Elapsed: {elapsed_min:.1f} min)")
            logger.info(f"   Dask Status: {dask_status}")
            logger.info(f"   Model Status: {model_status}")
            logger.info(f"   Models Completed: {list(self.state.get('models_completed', []))}")
            
            # Wait for next interval
            stop_event.wait(interval)
        
        logger.info("Progress logger stopped")


    # =============================================================================
    # MODEL TRAINING - All 6 models with error handling
    # =============================================================================

    def _run_training(self, **kwargs):
        """Run model training module with Dask."""
        import time
        import threading
        
        logger.info("Running Model Training Module with Dask...")
        
        self._ensure_data_loaded("training")
        
        # Check for sample fraction
        sample_fraction = kwargs.get('sample_fraction', 1.0)
        if sample_fraction < 1.0:
            logger.info(f"  Using {sample_fraction*100:.0f}% of training data for testing...")
            X_train_sampled = self.state['X_train'].sample(frac=sample_fraction, random_state=42)
            y_train_sampled = self.state['y_train'].sample(frac=sample_fraction, random_state=42)
            X_val_sampled = self.state['X_val'].sample(frac=sample_fraction, random_state=42)
            y_val_sampled = self.state['y_val'].sample(frac=sample_fraction, random_state=42)
        else:
            X_train_sampled = self.state['X_train']
            y_train_sampled = self.state['y_train']
            X_val_sampled = self.state['X_val']
            y_val_sampled = self.state['y_val']
        
        specific_model = kwargs.get('specific_model', 'all')
        force = kwargs.get('force', False)
        use_tuned_params = kwargs.get('use_tuned_params', True)

        # Initialize progress logging
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=self._log_progress,
            args=(stop_event,),
            kwargs={'interval': 300}  # 5 minutes
        )
        progress_thread.daemon = True
        progress_thread.start()
        
        logger.info(f"Target model(s) to train: {specific_model.upper()}")
        
        # Load existing models
        self._load_models()
        models = self.state.get('models', {})
        
        # Initialize tracking for progress
        self.state['models_completed'] = list(models.keys())
        self.state['current_model'] = None
        
        if models:
            logger.info(f"Currently available models: {list(models.keys())}")
        
        # Load tuned parameters
        tuned_params = {}
        if use_tuned_params:
            try:
                with open(f"{self.paths.results_dir}/tuned_params.json", 'r') as f:
                    tuned_params = json.load(f)
                    logger.info("Loaded tuned parameters")
            except:
                logger.warning("No tuned parameters found, using defaults")
        
        failed_models = []

        # ============================================================
        # 1. LOGISTIC REGRESSION
        # ============================================================
        if specific_model in ['all', 'logistic']:
            if 'logistic' not in models or force:
                logger.info("\n" + "=" * 60)
                logger.info("Training Logistic Regression (sklearn API)...")
                logger.info("=" * 60)
                self.state['current_model'] = 'Logistic Regression'
                start_time = time.time()
                try:
                    lr_params = tuned_params.get('logistic', {})
                    if 'random_state' not in lr_params:
                        lr_params['random_state'] = self.model_config.random_state
                    
                    # Use sklearn LogisticRegression
                    lr = LogisticRegressionModel(**lr_params)
                    lr.fit(X_train_sampled, y_train_sampled)
                    models['logistic'] = lr
                    elapsed = time.time() - start_time
                    logger.info(f"Logistic Regression completed in {elapsed/60:.1f} min")
                    if 'logistic' not in self.state['models_completed']:
                        self.state['models_completed'].append('logistic')
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.warning(f"Logistic Regression failed: {e}")
                    failed_models.append('logistic')
            else:
                logger.info("Skipping Logistic Regression (Already trained)")
                
        # ============================================================
        # 2. RANDOM FOREST (10% sample)
        # ============================================================
        # if specific_model in ['all', 'random_forest']:
        #     if 'random_forest' not in models or force:
        #         logger.info("\n" + "=" * 60)
        #         logger.info("Training Random Forest with Dask (10% sample)...")
        #         logger.info("=" * 60)
        #         self.state['current_model'] = 'Random Forest'
        #         start_time = time.time()
        #         try:
        #             logger.info("  Downsampling to 10% specifically for Random Forest...")
                    
        #             # FIX: Convert to pandas first, then sample (works reliably)
        #             logger.info("  Converting training data to pandas (for sampling)...")
        #             X_train_pd = self.state['X_train'].compute()
        #             y_train_pd = self.state['y_train'].compute()
                    
        #             total_rows = len(X_train_pd)
        #             sample_frac = 0.1
        #             sample_size = int(total_rows * sample_frac)
        #             logger.info(f"  Total rows: {total_rows:,}, sampling {sample_size:,} rows")
                    
        #             # Sample using pandas (reliable)
        #             import numpy as np
        #             np.random.seed(42)
        #             indices = np.random.choice(total_rows, size=sample_size, replace=False)
        #             indices = sorted(indices)
                    
        #             X_rf_sample = X_train_pd.iloc[indices]
        #             y_rf_sample = y_train_pd.iloc[indices]
                    
        #             # Reset indices
        #             X_rf_sample = X_rf_sample.reset_index(drop=True)
        #             y_rf_sample = y_rf_sample.reset_index(drop=True)
                    
        #             logger.info(f"  Sampled X: {len(X_rf_sample):,}, Sampled y: {len(y_rf_sample):,}")
                    
        #             if len(X_rf_sample) != len(y_rf_sample):
        #                 raise ValueError(f"Sample mismatch: X={len(X_rf_sample)}, y={len(y_rf_sample)}")
                    
        #             rf_params = tuned_params.get('random_forest', {})
        #             rf_params.update({
        #                 'random_state': self.model_config.random_state,
        #                 'n_estimators': 50,
        #                 'max_depth': 8,
        #             })
                    
        #             rf = RandomForestModel(**rf_params)
        #             rf.fit(X_rf_sample, y_rf_sample)
        #             models['random_forest'] = rf
        #             elapsed = time.time() - start_time
        #             logger.info(f"Random Forest completed in {elapsed/60:.1f} min")
        #             if 'random_forest' not in self.state['models_completed']:
        #                 self.state['models_completed'].append('random_forest')
        #         except Exception as e:
        #             elapsed = time.time() - start_time
        #             import traceback
        #             logger.warning(f"Random Forest failed after {elapsed/60:.1f} min: {e}")
        #             logger.debug(traceback.format_exc())
        #             failed_models.append('random_forest')
        #     else:
        #         logger.info("Skipping Random Forest (Already trained)")
        if specific_model in ['all', 'random_forest']:
            if 'random_forest' not in models or force:
                logger.info("\n" + "=" * 60)
                logger.info("Training Random Forest...")
                logger.info("=" * 60)
                self.state['current_model'] = 'Random Forest'
                start_time = time.time()
                try:
                    # REMOVED: Downsampling to 10% (not needed with sampled data)
                    # Use the same sampled data as other models
                    X_rf_sample = X_train_sampled
                    y_rf_sample = y_train_sampled
                    
                    logger.info(f"  Training with {len(X_rf_sample):,} samples, {len(X_rf_sample.columns)} features")
                    
                    rf_params = tuned_params.get('random_forest', {})
                    rf_params.update({
                        'random_state': self.model_config.random_state,
                        'n_estimators': 100,  # Increase back to 100 (was 50 due to downsampling)
                        'max_depth': 10,
                    })
                    
                    rf = RandomForestModel(**rf_params)
                    rf.fit(X_rf_sample, y_rf_sample)
                    models['random_forest'] = rf
                    elapsed = time.time() - start_time
                    logger.info(f"Random Forest completed in {elapsed/60:.1f} min")
                    if 'random_forest' not in self.state['models_completed']:
                        self.state['models_completed'].append('random_forest')
                except Exception as e:
                    elapsed = time.time() - start_time
                    import traceback
                    logger.warning(f"Random Forest failed after {elapsed/60:.1f} min: {e}")
                    logger.debug(traceback.format_exc())
                    failed_models.append('random_forest')
            else:
                logger.info("Skipping Random Forest (Already trained)")


        # 3. XGBoost
        if specific_model in ['all', 'xgboost']:
            if 'xgboost' not in models or force:
                logger.info("\n" + "=" * 60)
                logger.info("Training XGBoost with GPU...")
                logger.info("=" * 60)
                self.state['current_model'] = 'XGBoost'
                start_time = time.time()
                try:
                    xgb_params = tuned_params.get('xgboost', {})
                    xgb_params.update({
                        'random_state': self.model_config.random_state,
                        'n_estimators': 200,
                        'use_gpu': self.model_config.use_gpu,
                        'gpu_id': self.model_config.gpu_id,
                    })
                    
                    xgb = XGBoostModel(**xgb_params)
                    xgb.fit(X_train_sampled, y_train_sampled, X_val_sampled, y_val_sampled)
                    models['xgboost'] = xgb
                    
                    # FIX: Save immediately after training
                    with open(f"{self.paths.models_dir}/model_xgboost.pkl", 'wb') as f:
                        pickle.dump(xgb, f)
                    logger.info(f"Saved xgboost model")
                    
                    elapsed = time.time() - start_time
                    logger.info(f"XGBoost completed in {elapsed/60:.1f} min")
                    if 'xgboost' not in self.state['models_completed']:
                        self.state['models_completed'].append('xgboost')
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.warning(f"XGBoost failed after {elapsed/60:.1f} min: {e}")
                    failed_models.append('xgboost')
            else:
                logger.info("Skipping XGBoost (Already trained)")
        # ============================================================
        # 4. LightGBM
        # ============================================================
        if specific_model in ['all', 'lightgbm']:
            if 'lightgbm' not in models or force:
                logger.info("\n" + "=" * 60)
                logger.info("Training LightGBM (CPU fallback)...")  # Changed from GPU
                logger.info("=" * 60)
                self.state['current_model'] = 'LightGBM'
                start_time = time.time()
                try:
                    lgb_params = tuned_params.get('lightgbm', {})
                    lgb_params.update({
                        'random_state': self.model_config.random_state,
                        'n_estimators': 200,
                        'use_gpu': False,  # Force CPU
                        # 'gpu_device_id': self.model_config.gpu_id,
                    })
                    
                    lgb = LightGBMModel(**lgb_params)
                    lgb.fit(X_train_sampled, y_train_sampled, X_val_sampled, y_val_sampled)
                    models['lightgbm'] = lgb
                    elapsed = time.time() - start_time
                    logger.info(f"LightGBM completed in {elapsed/60:.1f} min")
                    if 'lightgbm' not in self.state['models_completed']:
                        self.state['models_completed'].append('lightgbm')
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.warning(f"LightGBM failed after {elapsed/60:.1f} min: {e}")
                    failed_models.append('lightgbm')
            else:
                logger.info("Skipping LightGBM (Already trained)")
                
        # ============================================================
        # 5. CatBoost
        # ============================================================
        
        if specific_model in ['all', 'catboost']:
            if 'catboost' not in models or force:
                logger.info("\n" + "=" * 60)
                logger.info("Training CatBoost with GPU...")
                logger.info("=" * 60)
                self.state['current_model'] = 'CatBoost'
                start_time = time.time()
                try:
                    cat_params = tuned_params.get('catboost', {})
                    cat_params.update({
                        'random_state': self.model_config.random_state,
                        'iterations': 100,
                        'use_gpu': self.model_config.use_gpu,
                        'devices': str(self.model_config.gpu_id),
                    })
                    
                    cat = CatBoostModel(**cat_params)
                    cat.fit(X_train_sampled, y_train_sampled, X_val_sampled, y_val_sampled)
                    models['catboost'] = cat
                    elapsed = time.time() - start_time
                    logger.info(f"CatBoost completed in {elapsed/60:.1f} min")
                    if 'catboost' not in self.state['models_completed']:
                        self.state['models_completed'].append('catboost')
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.warning(f"CatBoost failed after {elapsed/60:.1f} min: {e}")
                    failed_models.append('catboost')
            else:
                logger.info("Skipping CatBoost (Already trained)")
        
        # ============================================================
        # 6. ENSEMBLE
        # ============================================================
        # In _run_training(), Ensemble section:

        # 6. ENSEMBLE (with Logistic Regression)
        if specific_model in ['all', 'ensemble']:
            if 'ensemble' not in models or force:
                # Include Logistic Regression in ensemble
                valid_base_models = [
                    m for name, m in models.items() 
                    if name not in ['catboost', 'ensemble']  # Only exclude CatBoost
                ]
                
                if len(valid_base_models) >= 2:
                    logger.info("\n" + "=" * 60)
                    logger.info(f"Training Ensemble with {[m.name for m in valid_base_models]}...")
                    logger.info("=" * 60)
                    self.state['current_model'] = 'Ensemble'
                    start_time = time.time()
                    try:
                        ensemble = StackingEnsemble(
                            base_models=valid_base_models,
                            meta_model=LightGBMModel(
                                random_state=self.model_config.random_state, 
                                is_unbalance=True
                            ),
                            cv_folds=min(3, self.model_config.cv_folds),
                            random_state=self.model_config.random_state
                        )
                        ensemble.fit(X_train_sampled, y_train_sampled, X_val_sampled, y_val_sampled)
                        models['ensemble'] = ensemble
                        elapsed = time.time() - start_time
                        logger.info(f"Ensemble completed in {elapsed/60:.1f} min")
                        if 'ensemble' not in self.state['models_completed']:
                            self.state['models_completed'].append('ensemble')
                    except Exception as e:
                        elapsed = time.time() - start_time
                        logger.warning(f"Ensemble failed: {e}")
                        failed_models.append('ensemble')
                else:
                    logger.warning(f"Skipping Ensemble: Need at least 2 models, found {len(valid_base_models)}")
            else:
                logger.info("Skipping Ensemble (Already trained)")
        
        # ============================================================
        # SAVE MODELS
        # ============================================================
        self.state['models'] = models
        self._save_models(models)
        
        # Stop progress logging
        stop_event.set()
        progress_thread.join(timeout=2)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Training execution complete. Models ready: {list(models.keys())}")
        if failed_models:
            logger.warning(f"Failed models during this run: {failed_models}")
        logger.info("=" * 60)


    def _run_evaluation(self, **kwargs):
        """Run model evaluation module with Dask."""
        logger.info("Running Model Evaluation Module with Dask...")
        
        self._ensure_data_loaded("evaluation")
        
        if not self.state['models']:
            if self.skip_data:
                self._load_models()
            else:
                logger.warning("Models not trained. Running training first...")
                self._run_training()
        
        results = {}
        metrics_calculator = CreditRiskMetrics()

        # y_test never changes across models -- compute it once here rather
        # than once per model inside the loop below (small label vector,
        # not the feature matrix, but no reason to redo it per model).
        y_test_np = self.state['y_test'].compute().values

        for name, model in self.state['models'].items():
            logger.info(f"\nEvaluating {name}...")
            
            # Get predictions (returns numpy arrays)
            y_proba = model.predict_proba(self.state['X_test'])[:, 1]
            y_pred = model.predict(self.state['X_test'])

            # Compute metrics
            metrics = metrics_calculator.evaluate(y_test_np, y_pred, y_proba)
            metrics['name'] = name

            optimal = metrics_calculator.evaluate_with_optimal_thresholds(y_test_np, y_proba)
            metrics.update(optimal)
            
            if hasattr(model, 'get_feature_importance'):
                metrics['feature_importance'] = model.get_feature_importance()
            
            lift_data = metrics_calculator.compute_lift_gain(y_test_np, y_proba)
            metrics['lift_data'] = lift_data
            
            results[name] = metrics
            
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            logger.info(f"  F1 (0.5 threshold): {metrics['f1']:.4f}")
            logger.info(f"  F1 (optimal threshold={metrics['f1_optimal_threshold']:.3f}): {metrics['f1_optimal']:.4f}")
            logger.info(f"  F2 (optimal threshold={metrics['f2_optimal_threshold']:.3f}): {metrics['f2_optimal']:.4f}"
                        f"  [precision={metrics['precision_at_f2_optimal']:.3f}, recall={metrics['recall_at_f2_optimal']:.3f}]")
            logger.info(f"  KS: {metrics['ks_statistic']:.4f}")

        self.state['results'] = results
        
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
                'ks_statistic': metrics['ks_statistic'],
                'f1_optimal_threshold': metrics['f1_optimal_threshold'],
                'f1_optimal': metrics['f1_optimal'],
                'f2_optimal_threshold': metrics['f2_optimal_threshold'],
                'f2_optimal': metrics['f2_optimal'],
                'precision_at_f2_optimal': metrics['precision_at_f2_optimal'],
                'recall_at_f2_optimal': metrics['recall_at_f2_optimal'],
            }
            for name, metrics in results.items()
        ])
        results_df.to_csv(f"{self.paths.results_dir}/model_results.csv", index=False)
        
        # Summary table
        logger.info("\nMODEL PERFORMANCE SUMMARY:")
        logger.info("-" * 80)
        logger.info(f"{'Model':<20} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<10} {'KS':<10}")
        logger.info("-" * 80)
        for _, row in results_df.iterrows():
            logger.info(
                f"{row['model']:<20} {row['roc_auc']:<10.4f} {row['pr_auc']:<10.4f} "
                f"{row['f1']:<10.4f} {row['ks_statistic']:<10.4f}"
            )
        logger.info("-" * 80)
    
    def _run_calibration(self, **kwargs):
        """Run probability calibration module with Dask."""
        logger.info("Running Probability Calibration Module with Dask...")
        
        self._ensure_data_loaded("calibration")
        
        if not self.state['models']:
            if self.skip_data:
                self._load_models()
            else:
                logger.warning("Models not trained. Running training first...")
                self._run_training()
        
        calibrated_models = {}
        
        for name, model in self.state['models'].items():
            logger.info(f"\nCalibrating {name}...")
            
            calibrator = ProbabilityCalibrator(method='isotonic')
            calibrated_model = calibrator.calibrate(
                model,
                self.state['X_train'],
                self.state['y_train'],
                self.state['X_val'],
                self.state['y_val']
            )
            calibrated_models[f"{name}_calibrated"] = calibrated_model
        
        self.state['calibrated_models'] = calibrated_models
        
        for name, model in calibrated_models.items():
            with open(f"{self.paths.models_dir}/model_{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} model")
    
    def _run_scoring(self, **kwargs):
        """Run credit score generation module."""
        logger.info("Running Credit Score Generation Module with Dask...")
        
        self._ensure_data_loaded("scoring")
        
        if not self.state['models']:
            if self.skip_data:
                self._load_models()
            else:
                logger.warning("Models not trained. Running training first...")
                self._run_training()
        
        ensemble = self.state['models'].get('ensemble')
        if ensemble is None:
            ensemble = self.state['models'].get('lightgbm')
        
        if ensemble is None:
            raise RuntimeError("No model available for scoring")
        
        # Get probabilities (returns numpy)
        y_proba = ensemble.predict_proba(self.state['X_test'])[:, 1]
        y_test_np = self.state['y_test'].compute().values
        
        score_generator = CreditScoreGenerator(
            min_score=300, max_score=900,
            target_default_rate=0.05, pdo=20
        )
        
        results_df = pd.DataFrame({
            'probability': y_proba,
            'true_label': y_test_np
        })
        
        results_df = score_generator.generate_all_scores(
            results_df, prob_col='probability', score_col='credit_score'
        )
        
        distribution = score_generator.get_score_distribution(results_df)
        logger.info("\nScore Distribution:")
        for key, value in distribution.items():
            if key != 'percentiles':
                logger.info(f"  {key}: {value:.2f}")
        
        results_df.to_csv(f"{self.paths.results_dir}/scores.csv", index=False)
        self.state['score_results'] = results_df
    
    def _run_visualization(self, **kwargs):
        """Run visualization module to generate plots and reports."""
        logger.info("Running Visualization Module...")
        
        # Ensure models are loaded
        if not self.state['models']:
            if self.skip_data:
                self._load_models()
            else:
                logger.warning("Models not trained. Running training first...")
                self._run_training()
        
        # Ensure results are available
        if not self.state['results']:
            if self.skip_data:
                try:
                    results_df = pd.read_csv(f"{self.paths.results_dir}/model_results.csv")
                    self.state['results'] = {}
                    for _, row in results_df.iterrows():
                        self.state['results'][row['model']] = row.to_dict()
                    logger.info("Loaded results from CSV")
                except Exception as e:
                    logger.warning(f"Could not load results: {e}")
                    logger.info("Running evaluation first...")
                    self._run_evaluation()
            else:
                logger.info("Running evaluation first...")
                self._run_evaluation()
        
        # Get test data - compute to pandas to avoid Dask issues
        self._ensure_data_loaded("visualization")
        
        # FIX: Convert to pandas for prediction to avoid Dask scheduling issues
        logger.info("Loading test data as pandas (for visualization)...")
        X_test_pd = self.state['X_test'].compute()
        y_test_np = self.state['y_test'].compute().values
        
        # Create visualizer
        visualizer = CreditRiskVisualizer(save_dir=f"{self.paths.results_dir}/plots")
        
        # Process each model
        for name, model in self.state['models'].items():
            logger.info(f"Generating visualizations for {name}...")
            
            try:
                # FIX: Convert X to pandas if needed for prediction
                if hasattr(model, 'supports_dask_data') and model.supports_dask_data:
                    # Some models can handle Dask directly
                    X_pred = self.state['X_test']
                else:
                    X_pred = X_test_pd
                
                # Get predictions
                y_proba = model.predict_proba(X_pred)[:, 1]
                
                # If prediction returned a Dask array, compute it
                if hasattr(y_proba, 'compute'):
                    y_proba = y_proba.compute()
                
                y_pred = (y_proba >= 0.5).astype(int)
                
                # Get feature importance if available
                feature_importance = {}
                if hasattr(model, 'get_feature_importance'):
                    feature_importance = model.get_feature_importance()
                
                # Create individual model report
                if feature_importance:
                    visualizer.create_model_evaluation_report(
                        y_test_np,
                        y_proba,
                        y_pred,
                        name,
                        feature_importance,
                        save_name=f"{name}_report"
                    )
                    logger.info(f"  Created report for {name}")
                else:
                    logger.warning(f"  No feature importance for {name}, skipping report")
                    
            except Exception as e:
                logger.warning(f"  Could not generate visualizations for {name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Create ensemble comparison plot (using stored results)
        if self.state['results']:
            try:
                visualizer.create_ensemble_comparison_plot(
                    self.state['results'],
                    save_name="ensemble_comparison"
                )
                logger.info("Created ensemble comparison plot")
            except Exception as e:
                logger.warning(f"Could not create ensemble comparison: {e}")
        
        # Create additional plots
        self._create_additional_plots()
        
        logger.info(f"Visualizations saved to: {self.paths.results_dir}/plots")

    def _create_additional_plots(self):
        """Create additional diagnostic plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import roc_curve, auc, confusion_matrix
            
            plots_dir = f"{self.paths.results_dir}/plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Load results
            results_df = pd.read_csv(f"{self.paths.results_dir}/model_results.csv")
            
            # 1. ROC-AUC Comparison Bar Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(results_df['model'], results_df['roc_auc'], 
                        color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'])
            ax.axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)')
            ax.set_ylim(0, 1)
            ax.set_ylabel('ROC-AUC Score')
            ax.set_title('Model ROC-AUC Comparison')
            ax.legend()
            
            # Add value labels on bars
            for bar, val in zip(bars, results_df['roc_auc']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/roc_auc_comparison.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: roc_auc_comparison.png")
            
            # 2. All Metrics Heatmap
            metrics = ['roc_auc', 'pr_auc', 'f1', 'balanced_accuracy', 'ks_statistic']
            fig, ax = plt.subplots(figsize=(10, 6))
            
            heatmap_data = results_df.set_index('model')[metrics]
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                        vmin=0, vmax=1, ax=ax)
            ax.set_title('Model Performance Heatmap')
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/performance_heatmap.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: performance_heatmap.png")
            
            # 3. Metric Distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            results_melted = results_df.melt(id_vars=['model'], 
                                            value_vars=['roc_auc', 'pr_auc', 'f1', 'ks_statistic'],
                                            var_name='metric', value_name='score')
            sns.barplot(data=results_melted, x='model', y='score', hue='metric', ax=ax)
            ax.set_title('Model Performance by Metric')
            ax.legend(bbox_to_anchor=(1.05, 1))
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/metrics_comparison.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: metrics_comparison.png")
            
            # 4. Model Ranking (if evaluation metrics available)
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_df = results_df.sort_values('roc_auc', ascending=True)
            colors = ['#e74c3c' if i < 2 else '#f39c12' if i < 4 else '#2ecc71' 
                    for i in range(len(sorted_df))]
            ax.barh(sorted_df['model'], sorted_df['roc_auc'], color=colors)
            ax.set_xlabel('ROC-AUC')
            ax.set_title('Model Ranking by ROC-AUC')
            ax.axvline(x=0.5, color='red', linestyle='--', label='Random')
            ax.legend()
            
            for i, (_, row) in enumerate(sorted_df.iterrows()):
                ax.text(row['roc_auc'] + 0.01, i, f"{row['roc_auc']:.3f}", 
                        va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/model_ranking.png", dpi=150)
            plt.close()
            logger.info(f"  Saved: model_ranking.png")
            
        except Exception as e:
            logger.warning(f"Could not create additional plots: {e}")
        
    def _run_full_pipeline(self):
        """Run the complete pipeline from ingestion to visualization."""
        logger.info("Running Full Pipeline...")
        
        modules = [
            'ingestion', 'cleaning', 'feature_engineering', 'target_creation',
            'dataset_creation', 'tuning', 'training', 'evaluation',
            'calibration', 'scoring', 'visualization'
        ]
        
        for module in modules:
            self.run_module(module)
        
        logger.info("Full pipeline completed successfully!")
    
    def _check_models_exist(self) -> bool:
        """Check if models already exist."""
        model_names = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'ensemble']
        for name in model_names:
            if not os.path.exists(f"{self.paths.models_dir}/model_{name}.pkl"):
                return False
        return True
    
    def _load_models(self):
        """Load existing models from disk."""
        model_names = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'ensemble']
        for name in model_names:
            try:
                with open(f"{self.paths.models_dir}/model_{name}.pkl", 'rb') as f:
                    self.state['models'][name] = pickle.load(f)
                logger.info(f"Loaded {name} model")
            except Exception as e:
                logger.warning(f"Could not load {name} model: {e}")
    
    def _save_models(self, models: Dict[str, Any]):
        """Save trained models."""
        for name, model in models.items():
            with open(f"{self.paths.models_dir}/model_{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} model")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current pipeline state."""
        return self.state
    
    def stop(self):
        """Stop Spark and Dask sessions."""
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session stopped.")
        # Closes the single shared Dask client/cluster used by this pipeline
        # AND by every models/*.py module (they all call get_dask_client()
        # and get back this same instance) -- one clean teardown instead of
        # each model leaking its own cluster.
        close_dask_client()

    def validate_setup(self):
        """Validate all paths, permissions, and dependencies before running."""
        checks = []
        
        # Check paths
        for attr in ['raw_dir', 'bronze_dir', 'silver_dir', 'features_dir', 
                    'models_dir', 'results_dir']:
            path = getattr(self.paths, attr)
            if not os.path.exists(path):
                checks.append(f"=== Missing directory: {path}")
            else:
                checks.append(f"=== Directory exists: {path}")
        
        # Check data files if skip_data is True
        if self.skip_data:
            for attr in ['train_data', 'val_data', 'test_data']:
                path = getattr(self.paths, attr)
                if os.path.exists(path):
                    checks.append(f"=== Data file exists: {path}")
                else:
                    checks.append(f"=== Missing data file: {path}")
        
        # Check Spark
        try:
            self.spark.version
            checks.append(f"=== Spark available: {self.spark.version}")
        except Exception as e:
            checks.append(f"=== Spark not available: {e}")
        
        # Check Dask
        try:
            import dask
            checks.append(f"=== Dask available: {dask.__version__}")
        except Exception as e:
            checks.append(f"=== Dask not available: {e}")
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            checks.append(f"=== Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
            if available_gb < 4:
                checks.append(f"=== Low memory available: {available_gb:.1f}GB (recommend > 8GB)")
        except ImportError:
            checks.append("=== psutil not installed - skipping memory check")
        
        return checks

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Behavioral Credit Risk Scoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--module', type=str,
        choices=['ingestion', 'cleaning', 'feature_engineering', 'target_creation',
                 'dataset_creation', 'tuning', 'training', 'evaluation',
                 'calibration', 'scoring', 'visualization', 'full'],
        help='Module to execute'
    )
    
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--skip_data', action='store_true', 
                       help='SKIP ALL data preparation. Use existing data only.')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-processing even if data exists')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='Number of Optuna trials for hyperparameter tuning')
    parser.add_argument('--threshold', type=int, choices=[30, 60, 90], default=30,
                       help='Delinquency threshold for target creation')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='Number of Dask workers')
    parser.add_argument('--threads_per_worker', type=int, default=2,
                       help='Threads per Dask worker')
    parser.add_argument('--dry_run', action='store_true', 
                   help='Validate pipeline setup without executing heavy operations')
    parser.add_argument('--test', action='store_true', 
                       help='Run training on 10% of data for quick testing')
    parser.add_argument('--train_model', type=str, default='all',
                       choices=['all', 'logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'ensemble'],
                       help='Specific model to train when running the training module')

    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("BEHAVIORAL CREDIT RISK SCORING SYSTEM")
    logger.info(f"Started at: {datetime.now()}")
    if args.skip_data:
        logger.info("*** SKIP DATA MODE ENABLED - Using existing data only ***")
    if args.dry_run:
        logger.info("*** DRY RUN MODE - Validating setup only ***")
    logger.info("=" * 80)
    
    # Handle dry run first - no need to initialize full pipeline for basic checks
    if args.dry_run:
        logger.info("\n DRY RUN: Validating pipeline setup...")
        logger.info("-" * 60)
        
        # Create pipeline with minimal initialization
        pipeline = CreditRiskPipeline(
            skip_data=args.skip_data,
            n_workers=1,  # Minimal workers for dry run
            threads_per_worker=1
        )
        
        try:
            checks = pipeline.validate_setup()
            logger.info("\nValidation Results:")
            for check in checks:
                logger.info(f"  {check}")
            
            # Check Python dependencies
            logger.info("\nChecking Python dependencies...")
            dependencies = [
                ('pyspark', 'spark'),
                ('dask', 'dask'),
                ('dask_ml', 'dask_ml'),
                ('xgboost', 'xgboost'),
                ('lightgbm', 'lightgbm'),
                ('catboost', 'catboost'),
                ('optuna', 'optuna'),
                ('sklearn', 'sklearn'),
                ('pandas', 'pandas'),
                ('numpy', 'numpy')
            ]
            
            for module_name, import_name in dependencies:
                try:
                    if import_name == 'spark':
                        # Spark is already checked
                        continue
                    __import__(import_name)
                    logger.info(f"=== {module_name} available")
                except ImportError:
                    logger.warning(f"=== {module_name} not found")
            
            # Check data files if not skipping data
            if not args.skip_data:
                raw_dir = pipeline.paths.raw_dir
                logger.info(f"\nChecking raw data directory: {raw_dir}")
                if os.path.exists(raw_dir):
                    files = os.listdir(raw_dir)
                    orig_files = [f for f in files if 'orig_' in f]
                    svcg_files = [f for f in files if 'svcg_' in f]
                    logger.info(f"  Found {len(orig_files)} origination files")
                    logger.info(f"  Found {len(svcg_files)} performance files")
                    
                    # Check for years 1999-2012
                    expected_years = list(range(1999, 2013))
                    found_years = []
                    for f in orig_files:
                        try:
                            year = int(f.split('_')[-1].split('.')[0])
                            found_years.append(year)
                        except:
                            pass
                    
                    missing_years = [y for y in expected_years if y not in found_years]
                    if missing_years:
                        logger.warning(f"=== Missing origination files for years: {missing_years}")
                    else:
                        logger.info(f"=== All origination files present (1999-2012)")
                else:
                    logger.warning(f"=== Raw data directory not found: {raw_dir}")
            
            logger.info("\n=== DRY RUN COMPLETED - All validations passed")
            logger.info("-" * 60)
            return
            
        except Exception as e:
            logger.error(f"\n=== DRY RUN FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
        finally:
            pipeline.stop()
    
    # Normal execution - handle module selection
    modules_to_run = []
    if args.full:
        modules_to_run = ['full']
    elif args.module:
        modules_to_run = [args.module]
    else:
        logger.info("No module specified. Use --module or --full")
        logger.info("Available modules: ingestion, cleaning, feature_engineering, target_creation, dataset_creation, tuning, training, evaluation, calibration, scoring, visualization, full")
        logger.info("Or use --dry_run to validate setup without executing heavy operations")
        return
    
    pipeline = CreditRiskPipeline(
        skip_data=args.skip_data,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        test_mode=args.test 
    )
    
    try:
        for module in modules_to_run:
            if module == 'full':
                pipeline.run_module('full')
            else:
                kwargs = {}
                if module == 'tuning':
                    kwargs['n_trials'] = args.n_trials
                elif module == 'target_creation':
                    kwargs['threshold'] = args.threshold
                elif module in ['ingestion', 'cleaning', 'feature_engineering', 'training']:
                    kwargs['force'] = args.force
                elif module == 'training':  # --- UPDATE THIS BLOCK ---
                    kwargs['force'] = args.force
                elif module == 'dataset_creation':
                    pass  # No special args needed
                
                pipeline.run_module(module, **kwargs)
        
        logger.info("=" * 80)
        logger.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("=== PIPELINE INTERRUPTED BY USER")
        logger.warning("=" * 80)
    except Exception as e:
        logger.error(f"=== Pipeline execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()