# datasets/dataset_creation.py
"""
Dataset creation pipeline for behavioral credit risk modeling.
"""

from pyspark.sql import SparkSession, DataFrame
import logging
from typing import Tuple

from config.config import config
from preprocessing.cleaning import SFLLDDataCleaner
from features.behavioral_features import BehavioralFeatureEngineer
from target.target_creation import TargetCreator
from validation.splitter import DataSplitter

logger = logging.getLogger(__name__)


class DatasetCreator:
    """Creates the complete dataset for behavioral credit risk modeling."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.paths = config['paths']
        self.model_config = config['model']
        self.feature_config = config['features']
        
        self.cleaner = SFLLDDataCleaner(spark)
        self.feature_engineer = BehavioralFeatureEngineer(spark)
        self.target_creator = TargetCreator(
            spark=spark,
            default_threshold=self.model_config.default_threshold,
            lookahead_months=self.model_config.lookahead_months
        )
        self.splitter = DataSplitter(spark)
    
    def create_dataset(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Create the full dataset with features and targets."""
        logger.info("=" * 80)
        logger.info("STARTING DATASET CREATION PIPELINE")
        logger.info("=" * 80)
        
        # Load bronze data
        logger.info("Step 1: Loading bronze data...")
        origination_df, performance_df = self._load_bronze_data()
        
        # Clean data
        logger.info("Step 2: Cleaning data...")
        orig_cleaned, perf_cleaned = self.cleaner.clean_both_datasets(
            origination_df, performance_df
        )
        
        # Create features
        logger.info("Step 3: Creating features...")
        feature_df = self.feature_engineer.create_all_features(
            orig_cleaned, perf_cleaned
        )
        
        # Remove duplicate columns before saving
        seen = set()
        cols_to_keep = []
        for col_name in feature_df.columns:
            if col_name not in seen:
                seen.add(col_name)
                cols_to_keep.append(col_name)
        feature_df = feature_df.select(*cols_to_keep)
        
        # Save intermediate feature dataset
        feature_df = self._save_feature_dataset(feature_df)
        
        # Create target
        logger.info("Step 4: Creating target...")
        dataset_df = self.target_creator.create_target(feature_df)
        self.target_creator.analyze_target_distribution(dataset_df)
        
        # Select features for modeling
        logger.info("Step 5: Selecting features for modeling...")
        dataset_df = self._select_model_features(dataset_df)
        
        # Split data
        logger.info("Step 6: Splitting data...")
        train_df, val_df, test_df = self.splitter.split_data(
            dataset_df,
            train_start_year=self.model_config.train_start_year,
            train_end_year=self.model_config.train_end_year,
            test_start_year=self.model_config.test_start_year,
            test_end_year=self.model_config.test_end_year,
            val_frac=self.model_config.val_frac
        )
        
        # Save splits
        logger.info("Step 7: Saving data splits...")
        self._save_splits(train_df, val_df, test_df)
        
        logger.info("=" * 80)
        logger.info("DATASET CREATION COMPLETED")
        logger.info("=" * 80)
        
        return train_df, val_df, test_df
    
    def _load_bronze_data(self) -> Tuple[DataFrame, DataFrame]:
        """Load bronze layer data."""
        origination_df = self.spark.read.parquet(self.paths.origination_bronze)
        performance_df = self.spark.read.parquet(self.paths.performance_bronze)
        
        logger.info(f"  Origination: {origination_df.count():,} loans")
        logger.info(f"  Performance: {performance_df.count():,} records")
        
        return origination_df, performance_df
    
    def _save_feature_dataset(self, df: DataFrame) -> DataFrame:
        """Save feature dataset to disk."""
        output_path = self.paths.feature_dataset
        
        df.write.mode("overwrite").option("compression", "snappy").parquet(output_path)
        logger.info(f"  Feature dataset saved to: {output_path}")
        
        return df
    
    def _select_model_features(self, df: DataFrame) -> DataFrame:
        """Select features for modeling."""
        # Get all columns except drop features
        drop_features = self.feature_config.drop_features + ['row_num', 'cumulative_delinquency']
        available_cols = df.columns
        
        # Get static and behavioral features
        feature_cols = (
            self.feature_config.static_features +
            self.feature_config.behavioral_features
        )
        
        # Only keep features that exist
        feature_cols = [c for c in feature_cols if c in available_cols]
        
        # Add target and loan identifier
        select_cols = ["LOAN_SEQUENCE_NUMBER", "MONTHLY_REPORTING_PERIOD", "target"] + feature_cols
        select_cols = [c for c in select_cols if c in available_cols and c not in drop_features]
        
        return df.select(*select_cols)
    
    def _save_splits(self, train_df, val_df, test_df):
        """Save data splits to disk."""
        train_df.write.mode("overwrite").parquet(self.paths.train_data)
        val_df.write.mode("overwrite").parquet(self.paths.val_data)
        test_df.write.mode("overwrite").parquet(self.paths.test_data)
        
        logger.info(f"  Train: {train_df.count():,} records")
        logger.info(f"  Validation: {val_df.count():,} records")
        logger.info(f"  Test: {test_df.count():,} records")