# validation/splitter.py
"""
Data splitter for out-of-time validation.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, rand
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
        logger.info("Splitting data...")
        
        df = df.withColumn("origination_year", col("LOAN_SEQUENCE_NUMBER").substr(2, 2).cast("int") + 2000)
        
        train_df = df.filter((col("origination_year") >= train_start_year) & (col("origination_year") <= train_end_year))
        test_df = df.filter((col("origination_year") >= test_start_year) & (col("origination_year") <= test_end_year))
        
        train_loans = train_df.select("LOAN_SEQUENCE_NUMBER").distinct()
        train_loans = train_loans.withColumn("rand", rand(self.random_seed))
        
        val_loans = train_loans.filter(col("rand") < val_frac).select("LOAN_SEQUENCE_NUMBER")
        train_loans = train_loans.filter(col("rand") >= val_frac).select("LOAN_SEQUENCE_NUMBER")
        
        val_df = train_df.join(val_loans, on="LOAN_SEQUENCE_NUMBER", how="inner")
        train_df = train_df.join(train_loans, on="LOAN_SEQUENCE_NUMBER", how="inner")
        
        self._log_split_stats(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _log_split_stats(self, train_df, val_df, test_df):
        for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            count = df.count()
            loans = df.select("LOAN_SEQUENCE_NUMBER").distinct().count()
            default_rate = df.filter(col("target") == 1).count() / count if count > 0 else 0
            logger.info(f"  {name}: Observations: {count:,}, Loans: {loans:,}, Default Rate: {default_rate:.2%}")
    
    def create_cv_splits(self, df: DataFrame, n_folds: int = 5, seed: int = 42) -> List[Tuple[DataFrame, DataFrame]]:
        logger.info(f"Creating {n_folds}-fold cross-validation splits...")
        
        loans = df.select("LOAN_SEQUENCE_NUMBER").distinct()
        loans = loans.withColumn("rand", rand(seed)).withColumn("fold", (col("rand") * n_folds).cast("int"))
        
        splits = []
        for fold in range(n_folds):
            val_loans = loans.filter(col("fold") == fold)
            train_loans = loans.filter(col("fold") != fold)
            splits.append((df.join(train_loans, on="LOAN_SEQUENCE_NUMBER", how="inner"),
                          df.join(val_loans, on="LOAN_SEQUENCE_NUMBER", how="inner")))
        
        return splits