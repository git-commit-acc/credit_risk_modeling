# features/feature_engineering.py
"""
Behavioral feature engineering using Spark Window functions.
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lag, max as spark_max, min as spark_min, 
    avg, sum as spark_sum, count, months_between,
    ceil, sin, cos, pow, lit
)
from pyspark.sql.window import Window
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create behavioral features for credit risk modeling."""
    
    @staticmethod
    def create_features(orig_df: DataFrame, perf_df: DataFrame) -> DataFrame:
        """Create all features by joining origination and performance."""
        logger.info("Creating features...")
        
        # Join data
        df = perf_df.join(orig_df, on="LOAN_SEQUENCE_NUMBER", how="inner")
        
        # Parse dates and create time features
        df = df.withColumn("observation_year", col("MONTHLY_REPORTING_PERIOD").substr(1, 4).cast("int"))
        df = df.withColumn("observation_month", col("MONTHLY_REPORTING_PERIOD").substr(5, 2).cast("int"))
        df = df.withColumn("observation_quarter", ceil(col("observation_month") / 3).cast("int"))
        
        # Seasonal features
        df = df.withColumn("seasonality_sin", sin(2 * 3.14159 * col("observation_month") / 12))
        df = df.withColumn("seasonality_cos", cos(2 * 3.14159 * col("observation_month") / 12))
        
        # Loan age transformations
        df = df.withColumn("loan_age_squared", pow(col("LOAN_AGE"), 2))
        df = df.withColumn("remaining_term_pct", 
                          when(col("ORIGINAL_LOAN_TERM") > 0, 
                               col("REMAINING_MONTHS_TO_LEGAL_MATURITY") / col("ORIGINAL_LOAN_TERM"))
                          .otherwise(lit(0)))
        
        # Delinquency features
        df = FeatureEngineer._create_delinquency_features(df)
        
        # Balance features
        df = FeatureEngineer._create_balance_features(df)
        
        # Modification features
        df = FeatureEngineer._create_modification_features(df)
        
        logger.info("Feature engineering completed.")
        return df
    
    @staticmethod
    def _create_delinquency_features(df: DataFrame) -> DataFrame:
        """Create delinquency behavior features."""
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        # Parse delinquency to numeric
        df = df.withColumn(
            "delinquency_numeric",
            when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "0", 0)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "1", 1)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "2", 2)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "3", 3)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "4", 4)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "5", 5)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "6", 6)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "7", 7)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "8", 8)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "9", 9)
            .when(col("CURRENT_LOAN_DELINQUENCY_STATUS") == "RA", 10)
            .otherwise(0)
        )
        
        # Rolling max delinquency
        for months, suffix in [(3, '3m'), (6, '6m'), (12, '12m')]:
            window = window_spec.rowsBetween(-months, -1)
            df = df.withColumn(f"max_delinquency_{suffix}", 
                              spark_max("delinquency_numeric").over(window))
        
        # Rolling mean delinquency
        df = df.withColumn("rolling_mean_delinquency_6m", 
                          avg("delinquency_numeric").over(window_spec.rowsBetween(-6, -1)))
        
        # Number of delinquent months
        df = df.withColumn("is_delinquent", when(col("delinquency_numeric") > 0, 1).otherwise(0))
        df = df.withColumn("num_delinquent_months_12m", 
                          spark_sum("is_delinquent").over(window_spec.rowsBetween(-12, -1)))
        
        # Consecutive delinquent months
        df = df.withColumn(
            "consecutive_delinquent_months",
            when(col("is_delinquent") == 0, 0)
            .otherwise(
                count("is_delinquent").over(
                    Window.partitionBy("LOAN_SEQUENCE_NUMBER")
                    .orderBy("MONTHLY_REPORTING_PERIOD")
                    .rowsBetween(-9999, 0)
                ) - count("is_delinquent").over(
                    Window.partitionBy("LOAN_SEQUENCE_NUMBER")
                    .orderBy("MONTHLY_REPORTING_PERIOD")
                    .rowsBetween(-9999, -1)
                )
            )
        )
        
        # Months since last delinquency
        df = df.withColumn(
            "months_since_last_delinquency",
            when(col("is_delinquent") == 1, 0)
            .otherwise(
                when(lag("is_delinquent", 1).over(window_spec) == 1, 1)
                .otherwise(lag("months_since_last_delinquency", 1).over(window_spec) + 1)
            )
        )
        
        # Delinquency trend
        df = df.withColumn("delinquency_trend_6m",
                          when(col("max_delinquency_6m").isNotNull() & col("max_delinquency_3m").isNotNull(),
                               col("max_delinquency_6m") - col("max_delinquency_3m"))
                          .otherwise(lit(0)))
        
        return df
    
    @staticmethod
    def _create_balance_features(df: DataFrame) -> DataFrame:
        """Create balance behavior features."""
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        # Remaining balance percentage
        df = df.withColumn("remaining_balance_pct",
                          when(col("ORIGINAL_UPB") > 0, 
                               col("CURRENT_ACTUAL_UPB") / col("ORIGINAL_UPB"))
                          .otherwise(lit(0)))
        
        # Balance changes
        for months in [3, 6, 12]:
            df = df.withColumn(f"balance_change_{months}m",
                              when(lag("CURRENT_ACTUAL_UPB", months).over(window_spec).isNotNull(),
                                   col("CURRENT_ACTUAL_UPB") - lag("CURRENT_ACTUAL_UPB", months).over(window_spec))
                              .otherwise(lit(0)))
        
        # Rolling average balance
        df = df.withColumn("rolling_avg_balance_6m",
                          avg("CURRENT_ACTUAL_UPB").over(window_spec.rowsBetween(-6, -1)))
        
        # Rate change
        df = df.withColumn("rate_change_since_origination",
                          col("CURRENT_INTEREST_RATE") - col("ORIGINAL_INTEREST_RATE"))
        
        return df
    
    @staticmethod
    def _create_modification_features(df: DataFrame) -> DataFrame:
        """Create modification features."""
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        # Ever modified
        df = df.withColumn("ever_modified",
                          when(spark_max(when(col("MODIFICATION_FLAG").isin(['Y', 'P']), 1).otherwise(0))
                               .over(window_spec.rowsBetween(Window.unboundedPreceding, 0)) > 0, 1)
                          .otherwise(0))
        
        # Number of modifications
        df = df.withColumn("num_modifications",
                          spark_sum(when(col("MODIFICATION_FLAG") == "Y", 1).otherwise(0))
                          .over(window_spec.rowsBetween(Window.unboundedPreceding, 0)))
        
        # Months since modification
        df = df.withColumn(
            "months_since_modification",
            when(col("MODIFICATION_FLAG") == "Y", 0)
            .otherwise(lag("months_since_modification", 1).over(window_spec) + 1)
        )
        
        return df