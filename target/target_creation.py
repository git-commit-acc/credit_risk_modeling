# target/target_creation.py
"""
Target creation for behavioral credit risk modeling.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, max as spark_max, lit, months_between,
    row_number, coalesce, count as spark_count
)
from pyspark.sql.window import Window
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class TargetCreator:
    """Creates binary target for behavioral credit risk modeling."""
    
    def __init__(
        self,
        spark: SparkSession,
        default_threshold: int ,
        lookahead_months: int 
    ):
        self.spark = spark
        self.default_threshold = default_threshold
        self.lookahead_months = lookahead_months
    
    def create_target(self, df: DataFrame, threshold: int = None) -> DataFrame:
        """Create binary target for each loan-month observation."""
        threshold = threshold or self.default_threshold
        logger.info(f"Creating target with {threshold}+ DPD threshold")
        
        # Parse delinquency status to numeric
        df = self._parse_delinquency_status(df)
        
        # Look ahead and find future delinquency
        df = self._lookahead_delinquency(df, threshold)
        
        # Remove observations without full lookahead window
        df = self._filter_incomplete_observations(df)
        
        # Remove observations after first delinquency
        df = self._filter_after_first_delinquency(df)
        
        return df
    
    def _parse_delinquency_status(self, df: DataFrame) -> DataFrame:
        """Parse delinquency status to numeric."""
        return df.withColumn(
            "delinquency_days",
            when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "0", 0)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "1", 30)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "2", 60)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "3", 90)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "4", 120)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "5", 150)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "6", 180)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "7", 210)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "8", 240)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "9", 270)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "RA", 999)
            .otherwise(0).cast("int")
        )
    
    def _lookahead_delinquency(self, df: DataFrame, threshold: int) -> DataFrame:
        """Look ahead and check if loan becomes delinquent within next 12 months."""
        logger.info(f"Looking ahead {self.lookahead_months} months...")
        
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        # Check future delinquency
        df = df.withColumn(
            "future_delinquency_max",
            spark_max(
                when(col("delinquency_days") >= threshold, 1).otherwise(0)
            ).over(
                window_spec.rowsBetween(1, self.lookahead_months)
            )
        )
        
        # Create target
        df = df.withColumn(
            "target",
            when(col("future_delinquency_max") == 1, 1)
            .when(col("future_delinquency_max") == 0, 0)
            .otherwise(lit(None))
        )
        
        # Track termination
        df = df.withColumn(
            "future_termination",
            spark_max(
                when(col("ZERO_BALANCE_CODE").isNotNull(), 1).otherwise(0)
            ).over(
                window_spec.rowsBetween(1, self.lookahead_months)
            )
        )
        
        df = df.withColumn(
            "is_terminated",
            when(col("ZERO_BALANCE_CODE").isNotNull(), 1).otherwise(0)
        )
        
        return df
    
    def _filter_incomplete_observations(self, df: DataFrame) -> DataFrame:
        """Remove observations without a full lookahead window."""
        logger.info("Filtering incomplete observations...")
        
        # Keep only observations with full lookahead window
        df = df.filter(col("future_delinquency_max").isNotNull())
        
        # Remove observations after loan termination
        df = df.filter(col("is_terminated") == 0)
        
        # Remove observations where loan terminates in lookahead window
        # (only if not already terminated due to delinquency)
        df = df.filter(
            (col("future_termination") == 0) |
            (col("target") == 1)
        )
        
        return df
    
    def _filter_after_first_delinquency(self, df: DataFrame) -> DataFrame:
        """
        Remove observations after the first delinquency event.
        We want to model the first transition into delinquency.
        """
        logger.info("Filtering observations after first delinquency...")
        
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        # Find first delinquency
        df = df.withColumn(
            "is_delinquent",
            when(col("delinquency_days") >= self.default_threshold, 1).otherwise(0)
        )
        
        # Create row number for ordering - FIX: Create row_num column
        df = df.withColumn(
            "row_num",
            row_number().over(
                Window.partitionBy("LOAN_SEQUENCE_NUMBER")
                .orderBy("MONTHLY_REPORTING_PERIOD")
            )
        )
        
        # Find cumulative delinquency
        df = df.withColumn(
            "cumulative_delinquency",
            spark_max("is_delinquent").over(
                window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        )
        
        # Keep observations before first delinquency
        # Only keep observations where cumulative delinquency is 0 OR
        # the current observation is the first delinquency AND target is 1
        df = df.filter(
            (col("cumulative_delinquency") == 0) |
            ((col("is_delinquent") == 1) & (col("target") == 1))
        )
        
        # Keep only observations within the lookahead window + 1
        df = df.filter(col("row_num") <= self.lookahead_months + 1)
        
        return df
    
    def analyze_target_distribution(self, df: DataFrame) -> Dict:
        """Analyze target distribution."""
        stats = {}
        
        total = df.count()
        stats['total_observations'] = total
        
        pos_count = df.filter(col("target") == 1).count()
        neg_count = df.filter(col("target") == 0).count()
        
        stats['positive_count'] = pos_count
        stats['negative_count'] = neg_count
        stats['default_rate'] = pos_count / total if total > 0 else 0
        
        loan_stats = df.groupBy("LOAN_SEQUENCE_NUMBER").agg(
            spark_max("target").alias("ever_default")
        ).filter(col("ever_default") == 1).count()
        
        stats['default_loans'] = loan_stats
        
        logger.info(f"Target distribution:")
        logger.info(f"  Total observations: {total:,}")
        logger.info(f"  Defaults: {pos_count:,} ({stats['default_rate']:.2%})")
        logger.info(f"  Loans with default: {loan_stats:,}")
        
        return stats