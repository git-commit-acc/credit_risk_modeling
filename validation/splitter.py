# validation/splitter.py
"""
Data splitter for out-of-time validation.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, rand, when
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits data into train, validation, and test sets."""
    
    def __init__(self, spark: SparkSession, random_seed: int = 42):
        self.spark = spark
        self.random_seed = random_seed
    
    def split_data(
        self,
        df: DataFrame,
        train_start_year: int = 1999,
        train_end_year: int = 2008,
        test_start_year: int = 2009,
        test_end_year: int = 2012,
        val_frac: float = 0.2
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data...")
        
        # Extract origination year
        df = df.withColumn(
            "origination_year",
            col("LOAN_SEQUENCE_NUMBER").substr(2, 2).cast("int") + 2000
        )
        
        # Filter by origination year
        train_df = df.filter(
            (col("origination_year") >= train_start_year) &
            (col("origination_year") <= train_end_year)
        )
        
        test_df = df.filter(
            (col("origination_year") >= test_start_year) &
            (col("origination_year") <= test_end_year)
        )
        
        # Split training loans for validation
        train_loans = train_df.select("LOAN_SEQUENCE_NUMBER").distinct()
        train_loans = train_loans.withColumn("rand", rand(self.random_seed))
        
        val_loans = train_loans.filter(
            col("rand") < val_frac
        ).select("LOAN_SEQUENCE_NUMBER")
        
        train_loans = train_loans.filter(
            col("rand") >= val_frac
        ).select("LOAN_SEQUENCE_NUMBER")
        
        # Filter dataframes by loan sets
        val_df = train_df.join(val_loans, on="LOAN_SEQUENCE_NUMBER", how="inner")
        train_df = train_df.join(train_loans, on="LOAN_SEQUENCE_NUMBER", how="inner")
        
        # Log split statistics
        self._log_split_stats(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _log_split_stats(self, train_df, val_df, test_df):
        """Log statistics for each split."""
        
        def get_stats(df, name):
            count = df.count()
            loans = df.select("LOAN_SEQUENCE_NUMBER").distinct().count()
            default_rate = df.filter(col("target") == 1).count() / count if count > 0 else 0
            
            logger.info(f"  {name}:")
            logger.info(f"    Observations: {count:,}")
            logger.info(f"    Loans: {loans:,}")
            logger.info(f"    Default Rate: {default_rate:.2%}")
        
        get_stats(train_df, "Train")
        get_stats(val_df, "Validation")
        get_stats(test_df, "Test")
    
    def create_cv_splits(
        self,
        df: DataFrame,
        n_folds: int = 5,
        seed: int = 42
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """Create cross-validation splits at the loan level."""
        logger.info(f"Creating {n_folds}-fold cross-validation splits...")
        
        loans = df.select("LOAN_SEQUENCE_NUMBER").distinct()
        loans = loans.withColumn("rand", rand(seed))
        loans = loans.withColumn("fold", (col("rand") * n_folds).cast("int"))
        
        splits = []
        for fold in range(n_folds):
            val_loans = loans.filter(col("fold") == fold)
            train_loans = loans.filter(col("fold") != fold)
            
            train_df = df.join(train_loans, on="LOAN_SEQUENCE_NUMBER", how="inner")
            val_df = df.join(val_loans, on="LOAN_SEQUENCE_NUMBER", how="inner")
            
            splits.append((train_df, val_df))
        
        logger.info(f"Created {n_folds} cross-validation splits")
        return splits