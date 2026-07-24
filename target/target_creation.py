# target/target_creation.py
"""
Target creation for behavioral credit risk modeling.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, max as spark_max, lit, row_number
from pyspark.sql.window import Window
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class TargetCreator:
    """Creates binary target for behavioral credit risk modeling."""
    
    def __init__(self, spark: SparkSession, default_threshold: int = 30, lookahead_months: int = 12):
        self.spark = spark
        self.default_threshold = default_threshold
        self.lookahead_months = lookahead_months
    
    def create_target(self, df: DataFrame, threshold: int = None) -> DataFrame:
        threshold = threshold or self.default_threshold
        logger.info(f"Creating target with {threshold}+ DPD threshold")
        
        df = self._add_delinquency_days(df)
        df = self._lookahead_delinquency(df, threshold)
        df = self._filter_incomplete_observations(df)
        df = self._filter_after_first_delinquency(df)
        
        return df
    
    def _add_delinquency_days(self, df: DataFrame) -> DataFrame:
        return df.withColumn("delinquency_days",
            when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 0, 0)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 1, 30)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 2, 60)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 3, 90)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 4, 120)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 5, 150)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 6, 180)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 7, 210)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 8, 240)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 9, 270)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == 10, 999)
            .otherwise(0).cast("int"))
    
    def _lookahead_delinquency(self, df: DataFrame, threshold: int) -> DataFrame:
        logger.info(f"Looking ahead {self.lookahead_months} months...")
        
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        df = df.withColumn("future_delinquency_max",
            spark_max(when(col("delinquency_days") >= threshold, 1).otherwise(0))
            .over(window_spec.rowsBetween(1, self.lookahead_months))) \
            .withColumn("target",
                when(col("future_delinquency_max") == 1, 1)
                .when(col("future_delinquency_max") == 0, 0)
                .otherwise(lit(None))) \
            .withColumn("future_termination",
                spark_max(when(col("ZERO_BALANCE_CODE").isNotNull(), 1).otherwise(0))
                .over(window_spec.rowsBetween(1, self.lookahead_months))) \
            .withColumn("is_terminated",
                when(col("ZERO_BALANCE_CODE").isNotNull(), 1).otherwise(0))
        
        return df
    
    def _filter_incomplete_observations(self, df: DataFrame) -> DataFrame:
        logger.info("Filtering incomplete observations...")
        return df.filter(col("future_delinquency_max").isNotNull()) \
            .filter(col("is_terminated") == 0) \
            .filter((col("future_termination") == 0) | (col("target") == 1))
    
    def _filter_after_first_delinquency(self, df: DataFrame) -> DataFrame:
        logger.info("Filtering observations after first delinquency...")
        
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        df = df.withColumn("is_delinquent",
            when(col("delinquency_days") >= self.default_threshold, 1).otherwise(0)) \
            .withColumn("row_num",
                row_number().over(Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD"))) \
            .withColumn("cumulative_delinquency",
                spark_max("is_delinquent").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow))) \
            .filter((col("cumulative_delinquency") == 0) | ((col("is_delinquent") == 1) & (col("target") == 1))) \
            .filter(col("row_num") <= self.lookahead_months + 1)
        
        return df
    
    def analyze_target_distribution(self, df: DataFrame) -> Dict:
        total = df.count()
        pos_count = df.filter(col("target") == 1).count()
        loan_stats = df.groupBy("LOAN_SEQUENCE_NUMBER").agg(spark_max("target").alias("ever_default")) \
            .filter(col("ever_default") == 1).count()
        
        stats = {
            'total_observations': total,
            'positive_count': pos_count,
            'negative_count': total - pos_count,
            'default_rate': pos_count / total if total > 0 else 0,
            'default_loans': loan_stats
        }
        
        logger.info(f"Target distribution:")
        logger.info(f"  Total observations: {total:,}")
        logger.info(f"  Defaults: {pos_count:,} ({stats['default_rate']:.2%})")
        logger.info(f"  Loans with default: {loan_stats:,}")
        
        return stats