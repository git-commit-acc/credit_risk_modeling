# features/behavioral_features.py
"""
Behavioral feature engineering using PySpark Window functions.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lag, lead, sum as spark_sum, max as spark_max,
    min as spark_min, avg as spark_avg, count, row_number,
    months_between, lit, ceil, sin, cos, pow, sqrt, abs,
    expr, coalesce, to_date, datediff
)
from pyspark.sql.window import Window
import logging
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BehavioralFeatureEngineer:
    """Creates behavioral features for credit risk modeling."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def create_all_features(
        self,
        origination_df: DataFrame,
        performance_df: DataFrame
    ) -> DataFrame:
        """Create all behavioral features."""
        logger.info("Starting behavioral feature engineering...")

        origination_df = origination_df.drop("ingestion_timestamp")
        performance_df = performance_df.drop("ingestion_timestamp", "vintage_year")

        # logger.info(f"Originations columns: {origination_df.columns}")
        # logger.info(f"Performance columns: {performance_df.columns}")

        # Join performance with origination
        joined_df = performance_df.join(
            origination_df,
            on="LOAN_SEQUENCE_NUMBER",
            how="inner"
        )

        # cols = joined_df.columns
        # duplicates = [c for c, n in Counter(cols).items() if n > 1]
        # print("Duplicate columns:", duplicates)
        
        # Create time features
        joined_df = self._create_time_features(joined_df)
        
        # Create delinquency behavior features
        joined_df = self._create_delinquency_features(joined_df)
        
        # Create balance behavior features
        joined_df = self._create_balance_features(joined_df)
        
        # Create interest behavior features
        joined_df = self._create_interest_features(joined_df)
        
        # Create modification features
        joined_df = self._create_modification_features(joined_df)
        
        # Create rolling statistics
        joined_df = self._create_rolling_statistics(joined_df)
        
        # Create interaction features
        joined_df = self._create_interaction_features(joined_df)
        
        logger.info("Feature engineering completed.")
        return joined_df
    
    def _create_time_features(self, df: DataFrame) -> DataFrame:
        """Create time-based features."""
        logger.info("Creating time features...")
        
        # Parse reporting period
        df = df.withColumn(
            "observation_year",
            col("MONTHLY_REPORTING_PERIOD").substr(1, 4).cast("int")
        ).withColumn(
            "observation_month",
            col("MONTHLY_REPORTING_PERIOD").substr(5, 2).cast("int")
        ).withColumn(
            "observation_quarter",
            ceil(col("observation_month") / 3).cast("int")
        )
        
        # Seasonal features
        df = df.withColumn(
            "seasonality_sin",
            sin(2 * 3.14159 * col("observation_month") / 12)
        ).withColumn(
            "seasonality_cos",
            cos(2 * 3.14159 * col("observation_month") / 12)
        )
        
        # Loan age transformations
        df = df.withColumn(
            "loan_age_squared",
            pow(col("LOAN_AGE"), 2)
        ).withColumn(
            "loan_age_cubic",
            pow(col("LOAN_AGE"), 3)
        )
        
        # Remaining term percentage
        df = df.withColumn(
            "remaining_term_pct",
            when(
                col("ORIGINAL_LOAN_TERM") > 0,
                col("REMAINING_MONTHS_TO_LEGAL_MATURITY") / col("ORIGINAL_LOAN_TERM")
            ).otherwise(lit(0))
        )
        
        return df
    
    def _create_delinquency_features(self, df: DataFrame) -> DataFrame:
        """Create delinquency behavior features."""
        logger.info("Creating delinquency features...")
        
        # Define Window specifications
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        window_3m = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD").rowsBetween(-3, -1)
        window_6m = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD").rowsBetween(-6, -1)
        window_12m = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD").rowsBetween(-12, -1)
        
        # Parse delinquency status to numeric
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
            .otherwise(0).cast("int")
        )
        
        # Rolling max delinquency
        df = df.withColumn(
            "max_delinquency_3m",
            spark_max("delinquency_numeric").over(window_3m)
        ).withColumn(
            "max_delinquency_6m",
            spark_max("delinquency_numeric").over(window_6m)
        ).withColumn(
            "max_delinquency_12m",
            spark_max("delinquency_numeric").over(window_12m)
        )
        
        # Rolling mean delinquency
        df = df.withColumn(
            "rolling_mean_delinquency_6m",
            spark_avg("delinquency_numeric").over(window_6m)
        )
        
        # Number of delinquent months
        df = df.withColumn(
            "is_delinquent",
            when(col("delinquency_numeric") > 0, 1).otherwise(0)
        )
        
        df = df.withColumn(
            "num_delinquent_months_12m",
            spark_sum("is_delinquent").over(window_12m)
        )
        
        # Consecutive delinquent months
        df = df.withColumn(
            "delinquency_streak_id",
            when(col("is_delinquent") == 0, lit(None))
            .otherwise(
                spark_sum(col("is_delinquent").isNull().cast("int")).over(
                    window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
                )
            )
        )
        
        df = df.withColumn(
            "consecutive_delinquent_months",
            when(
                col("is_delinquent") == 1,
                row_number().over(
                    Window.partitionBy("LOAN_SEQUENCE_NUMBER", "delinquency_streak_id")
                    .orderBy("MONTHLY_REPORTING_PERIOD")
                )
            ).otherwise(lit(0))
        )
        
        # Months since last delinquency
        df = df.withColumn(
            "last_delinquent_month",
            when(col("is_delinquent") == 1, col("MONTHLY_REPORTING_PERIOD"))
            .otherwise(lit(None))
        )
        
        df = df.withColumn(
            "last_delinquent_month",
            spark_max("last_delinquent_month").over(
                window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        )
        
        df = df.withColumn(
            "months_since_last_delinquency",
            when(
                col("last_delinquent_month").isNotNull(),
                (
                    (
                        col("MONTHLY_REPORTING_PERIOD").substr(1, 4).cast("int") * 12 +
                        col("MONTHLY_REPORTING_PERIOD").substr(5, 2).cast("int")
                    )
                    -
                    (
                        col("last_delinquent_month").substr(1, 4).cast("int") * 12 +
                        col("last_delinquent_month").substr(5, 2).cast("int")
                    )
                ).cast("int")
            ).otherwise(lit(-1))
        )
        
        # Delinquency trend
        df = df.withColumn(
            "delinquency_trend_6m",
            when(
                col("max_delinquency_6m").isNotNull() & col("max_delinquency_3m").isNotNull(),
                col("max_delinquency_6m") - col("max_delinquency_3m")
            ).otherwise(lit(0))
        )
        
        return df
    
    def _create_balance_features(self, df: DataFrame) -> DataFrame:
        """Create balance behavior features."""
        logger.info("Creating balance features...")
        
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        window_3m = window_spec.rowsBetween(-3, -1)
        window_6m = window_spec.rowsBetween(-6, -1)
        window_12m = window_spec.rowsBetween(-12, -1)
        
        # Remaining balance percentage
        df = df.withColumn(
            "remaining_balance_pct",
            when(
                col("ORIGINAL_UPB") > 0,
                col("CURRENT_ACTUAL_UPB") / col("ORIGINAL_UPB")
            ).otherwise(lit(0))
        )
        
        # Principal paid percentage
        df = df.withColumn(
            "principal_paid_pct",
            lit(1) - col("remaining_balance_pct")
        )
        
        # Balance changes
        df = df.withColumn(
            "balance_change_3m",
            when(
                lag("CURRENT_ACTUAL_UPB", 3).over(window_spec).isNotNull(),
                col("CURRENT_ACTUAL_UPB") - lag("CURRENT_ACTUAL_UPB", 3).over(window_spec)
            ).otherwise(lit(0))
        ).withColumn(
            "balance_change_6m",
            when(
                lag("CURRENT_ACTUAL_UPB", 6).over(window_spec).isNotNull(),
                col("CURRENT_ACTUAL_UPB") - lag("CURRENT_ACTUAL_UPB", 6).over(window_spec)
            ).otherwise(lit(0))
        ).withColumn(
            "balance_change_12m",
            when(
                lag("CURRENT_ACTUAL_UPB", 12).over(window_spec).isNotNull(),
                col("CURRENT_ACTUAL_UPB") - lag("CURRENT_ACTUAL_UPB", 12).over(window_spec)
            ).otherwise(lit(0))
        )
        
        # Rolling average balance
        df = df.withColumn(
            "rolling_avg_balance_6m",
            spark_avg("CURRENT_ACTUAL_UPB").over(window_6m)
        )
        
        return df
    
    def _create_interest_features(self, df: DataFrame) -> DataFrame:
        """Create interest rate behavior features."""
        logger.info("Creating interest features...")
        
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        # Rate change since origination
        df = df.withColumn(
            "rate_change_since_origination",
            col("CURRENT_INTEREST_RATE") - col("ORIGINAL_INTEREST_RATE")
        )
        
        # Rate reduction after modification
        df = df.withColumn(
            "prev_interest_rate",
            lag("CURRENT_INTEREST_RATE", 1).over(window_spec)
        )
        
        df = df.withColumn(
            "rate_reduction_after_mod",
            when(
                col("MODIFICATION_FLAG") == "Y",
                col("prev_interest_rate") - col("CURRENT_INTEREST_RATE")
            ).when(
                col("prev_interest_rate").isNotNull(),
                col("prev_interest_rate") - col("CURRENT_INTEREST_RATE")
            ).otherwise(lit(0))
        )
        
        return df
    
    def _create_modification_features(self, df: DataFrame) -> DataFrame:
        """Create modification and deferral features."""
        logger.info("Creating modification features...")
        
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        # Ever modified
        df = df.withColumn(
            "ever_modified",
            when(
                spark_max(
                    when(col("MODIFICATION_FLAG").isin(["Y", "P"]), 1).otherwise(0)
                ).over(
                    window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
                ) > 0,
                1
            ).otherwise(0)
        )
        
        # Number of modifications
        df = df.withColumn(
            "num_modifications",
            spark_sum(
                when(col("MODIFICATION_FLAG") == "Y", 1).otherwise(0)
            ).over(
                window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        )
        
        # Months since last modification
        df = df.withColumn(
            "last_modification_month",
            when(col("MODIFICATION_FLAG") == "Y", col("MONTHLY_REPORTING_PERIOD"))
            .otherwise(lit(None))
        )
        
        df = df.withColumn(
            "last_modification_month",
            spark_max("last_modification_month").over(
                window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        )
        
        df = df.withColumn(
            "months_since_modification",
            when(
                col("last_modification_month").isNotNull(),
                (
                    (
                        col("MONTHLY_REPORTING_PERIOD").substr(1, 4).cast("int") * 12 +
                        col("MONTHLY_REPORTING_PERIOD").substr(5, 2).cast("int")
                    )
                    -

                    (
                        col("last_modification_month").substr(1, 4).cast("int") * 12 +
                        col("last_modification_month").substr(5, 2).cast("int")
                    )
).cast("int")
            ).otherwise(lit(-1))
        )
        
        # Payment deferral count
        df = df.withColumn(
            "payment_deferral_count",
            spark_sum(
                when(col("PAYMENT_DEFERRAL_FLAG").isin(["Y", "P"]), 1).otherwise(0)
            ).over(
                window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        )
        
        return df
    
    def _create_rolling_statistics(self, df: DataFrame) -> DataFrame:
        """Create rolling statistics for key variables."""
        logger.info("Creating rolling statistics...")
        
        window_6m = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD").rowsBetween(-6, -1)
        
        rolling_vars = [
            ("CURRENT_ACTUAL_UPB", "balance"),
            ("CURRENT_INTEREST_RATE", "rate"),
            ("ELTV", "eltv")
        ]
        
        for var, suffix in rolling_vars:
            df = df.withColumn(
                f"rolling_avg_{suffix}_6m",
                spark_avg(var).over(window_6m)
            ).withColumn(
                f"rolling_std_{suffix}_6m",
                sqrt(spark_avg(pow(var - spark_avg(var).over(window_6m), 2)).over(window_6m))
            ).withColumn(
                f"rolling_min_{suffix}_6m",
                spark_min(var).over(window_6m)
            ).withColumn(
                f"rolling_max_{suffix}_6m",
                spark_max(var).over(window_6m)
            )
        
        return df
    
    def _create_interaction_features(self, df: DataFrame) -> DataFrame:
        """Create interaction features."""
        logger.info("Creating interaction features...")
        
        # DTI × LTV
        df = df.withColumn(
            "dti_ltv_interaction",
            when(
                col("ORIGINAL_DTI").isNotNull() & col("ORIGINAL_LTV").isNotNull(),
                col("ORIGINAL_DTI") * col("ORIGINAL_LTV") / 100
            ).otherwise(lit(0))
        )
        
        # Credit Score × DTI
        df = df.withColumn(
            "credit_dti_interaction",
            when(
                col("CREDIT_SCORE").isNotNull() & col("ORIGINAL_DTI").isNotNull(),
                col("CREDIT_SCORE") * col("ORIGINAL_DTI") / 100
            ).otherwise(lit(0))
        )
        
        # Balance × Delinquency
        df = df.withColumn(
            "balance_delinquency_interaction",
            when(
                col("delinquency_numeric") > 0,
                col("CURRENT_ACTUAL_UPB") * col("delinquency_numeric")
            ).otherwise(lit(0))
        )
        
        # Loan Age × Balance
        df = df.withColumn(
            "age_balance_interaction",
            col("LOAN_AGE") * col("remaining_balance_pct")
        )
        
        return df