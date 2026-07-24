# features/behavioral_features.py
"""
Behavioral feature engineering using PySpark Window functions.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, sum as spark_sum, max as spark_max, lag,
    min as spark_min, avg as spark_avg, row_number,
    lit, ceil, sin, cos, pow, sqrt, coalesce
)
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
import logging

logger = logging.getLogger(__name__)


class BehavioralFeatureEngineer:
    """Creates behavioral features for credit risk modeling."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def create_all_features(self, orig_df: DataFrame, perf_df: DataFrame) -> DataFrame:
        """Create all behavioral features."""
        logger.info("Starting behavioral feature engineering...")
        
        orig_df = orig_df.drop("ingestion_timestamp")
        perf_df = perf_df.drop("ingestion_timestamp", "vintage_year")
        
        joined_df = perf_df.join(orig_df, on="LOAN_SEQUENCE_NUMBER", how="inner")
        
        joined_df = self._create_time_features(joined_df)
        joined_df = self._create_delinquency_features(joined_df)
        joined_df = self._create_balance_features(joined_df)
        joined_df = self._create_interest_features(joined_df)
        joined_df = self._create_modification_features(joined_df)
        joined_df = self._create_rolling_statistics(joined_df)
        joined_df = self._create_interaction_features(joined_df)
        
        # Ensure numeric columns
        for col_name in joined_df.columns:
            if col_name not in ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD']:
                if str(joined_df.schema[col_name].dataType) == 'StringType':
                    joined_df = joined_df.withColumn(col_name, col(col_name).cast(DoubleType()))
        
        logger.info("Feature engineering completed.")
        return joined_df
    
    def _create_time_features(self, df: DataFrame) -> DataFrame:
        """Create time-based features."""
        return df.withColumn("observation_year", col("MONTHLY_REPORTING_PERIOD").substr(1, 4).cast("int")) \
            .withColumn("observation_month", col("MONTHLY_REPORTING_PERIOD").substr(5, 2).cast("int")) \
            .withColumn("observation_quarter", ceil(col("observation_month") / 3).cast("int")) \
            .withColumn("seasonality_sin", sin(2 * 3.14159 * col("observation_month") / 12)) \
            .withColumn("seasonality_cos", cos(2 * 3.14159 * col("observation_month") / 12)) \
            .withColumn("loan_age_squared", pow(col("LOAN_AGE"), 2)) \
            .withColumn("loan_age_cubic", pow(col("LOAN_AGE"), 3)) \
            .withColumn("remaining_term_pct", 
                when(col("ORIGINAL_LOAN_TERM") > 0, 
                    col("REMAINING_MONTHS_TO_LEGAL_MATURITY") / col("ORIGINAL_LOAN_TERM"))
                .otherwise(lit(0)))
    
    def _create_delinquency_features(self, df: DataFrame) -> DataFrame:
        """Create delinquency behavior features."""
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        w3 = window_spec.rowsBetween(-3, -1)
        w6 = window_spec.rowsBetween(-6, -1)
        w12 = window_spec.rowsBetween(-12, -1)
        
        df = df.withColumn("delinquency_numeric", coalesce(col("CURRENT_LOAN_DELINQUENCY_STATUS").cast("int"), lit(0)))
        
        df = df.withColumn("max_delinquency_3m", spark_max("delinquency_numeric").over(w3)) \
            .withColumn("max_delinquency_6m", spark_max("delinquency_numeric").over(w6)) \
            .withColumn("max_delinquency_12m", spark_max("delinquency_numeric").over(w12)) \
            .withColumn("rolling_mean_delinquency_6m", spark_avg("delinquency_numeric").over(w6)) \
            .withColumn("is_delinquent", when(col("delinquency_numeric") > 0, 1).otherwise(0)) \
            .withColumn("num_delinquent_months_12m", spark_sum("is_delinquent").over(w12))
        
        # Consecutive delinquency
        df = df.withColumn("delinquency_streak_id",
            when(col("is_delinquent") == 0, lit(None))
            .otherwise(spark_sum(col("is_delinquent").isNull().cast("int"))
                .over(window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow))))
        
        df = df.withColumn("consecutive_delinquent_months",
            when(col("is_delinquent") == 1,
                row_number().over(Window.partitionBy("LOAN_SEQUENCE_NUMBER", "delinquency_streak_id")
                    .orderBy("MONTHLY_REPORTING_PERIOD")))
            .otherwise(lit(0)))
        
        # Months since last delinquency
        df = df.withColumn("last_delinquent_month",
            when(col("is_delinquent") == 1, col("MONTHLY_REPORTING_PERIOD")).otherwise(lit(None)))
        df = df.withColumn("last_delinquent_month",
            spark_max("last_delinquent_month").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
        
        df = df.withColumn("months_since_last_delinquency",
            when(col("last_delinquent_month").isNotNull(),
                (col("MONTHLY_REPORTING_PERIOD").substr(1, 4).cast("int") * 12 +
                 col("MONTHLY_REPORTING_PERIOD").substr(5, 2).cast("int")) -
                (col("last_delinquent_month").substr(1, 4).cast("int") * 12 +
                 col("last_delinquent_month").substr(5, 2).cast("int")))
            .otherwise(lit(-1)))
        
        df = df.withColumn("delinquency_trend_6m",
            when(col("max_delinquency_6m").isNotNull() & col("max_delinquency_3m").isNotNull(),
                col("max_delinquency_6m") - col("max_delinquency_3m"))
            .otherwise(lit(0)))
        
        return df
    
    def _create_balance_features(self, df: DataFrame) -> DataFrame:
        """Create balance behavior features."""
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        w3 = window_spec.rowsBetween(-3, -1)
        w6 = window_spec.rowsBetween(-6, -1)
        w12 = window_spec.rowsBetween(-12, -1)
        
        return df.withColumn("remaining_balance_pct",
            when(col("ORIGINAL_UPB") > 0, col("CURRENT_ACTUAL_UPB") / col("ORIGINAL_UPB")).otherwise(lit(0))) \
            .withColumn("principal_paid_pct", lit(1) - col("remaining_balance_pct")) \
            .withColumn("balance_change_3m",
                when(lag("CURRENT_ACTUAL_UPB", 3).over(window_spec).isNotNull(),
                    col("CURRENT_ACTUAL_UPB") - lag("CURRENT_ACTUAL_UPB", 3).over(window_spec))
                .otherwise(lit(0))) \
            .withColumn("balance_change_6m",
                when(lag("CURRENT_ACTUAL_UPB", 6).over(window_spec).isNotNull(),
                    col("CURRENT_ACTUAL_UPB") - lag("CURRENT_ACTUAL_UPB", 6).over(window_spec))
                .otherwise(lit(0))) \
            .withColumn("balance_change_12m",
                when(lag("CURRENT_ACTUAL_UPB", 12).over(window_spec).isNotNull(),
                    col("CURRENT_ACTUAL_UPB") - lag("CURRENT_ACTUAL_UPB", 12).over(window_spec))
                .otherwise(lit(0))) \
            .withColumn("rolling_avg_balance_6m", spark_avg("CURRENT_ACTUAL_UPB").over(w6))
    
    def _create_interest_features(self, df: DataFrame) -> DataFrame:
        """Create interest rate behavior features."""
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        return df.withColumn("rate_change_since_origination",
            col("CURRENT_INTEREST_RATE") - col("ORIGINAL_INTEREST_RATE")) \
            .withColumn("prev_interest_rate", lag("CURRENT_INTEREST_RATE", 1).over(window_spec)) \
            .withColumn("rate_reduction_after_mod",
                when(col("MODIFICATION_FLAG") == 1, col("prev_interest_rate") - col("CURRENT_INTEREST_RATE"))
                .when(col("prev_interest_rate").isNotNull(), col("prev_interest_rate") - col("CURRENT_INTEREST_RATE"))
                .otherwise(lit(0)))
    
    def _create_modification_features(self, df: DataFrame) -> DataFrame:
        """Create modification features."""
        window_spec = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD")
        
        df = df.withColumn("ever_modified",
            when(spark_max(when(col("MODIFICATION_FLAG").isin([1, 2]), 1).otherwise(0))
                .over(window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)) > 0, 1)
            .otherwise(0)) \
            .withColumn("num_modifications",
                spark_sum(when(col("MODIFICATION_FLAG") == 1, 1).otherwise(0))
                .over(window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
        
        df = df.withColumn("last_modification_month",
            when(col("MODIFICATION_FLAG") == 1, col("MONTHLY_REPORTING_PERIOD")).otherwise(lit(None)))
        df = df.withColumn("last_modification_month",
            spark_max("last_modification_month").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
        
        df = df.withColumn("months_since_modification",
            when(col("last_modification_month").isNotNull(),
                (col("MONTHLY_REPORTING_PERIOD").substr(1, 4).cast("int") * 12 +
                 col("MONTHLY_REPORTING_PERIOD").substr(5, 2).cast("int")) -
                (col("last_modification_month").substr(1, 4).cast("int") * 12 +
                 col("last_modification_month").substr(5, 2).cast("int")))
            .otherwise(lit(-1))) \
            .withColumn("payment_deferral_count",
                spark_sum(when(col("PAYMENT_DEFERRAL_FLAG").isin([1, 2]), 1).otherwise(0))
                .over(window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
        
        return df
    
    def _create_rolling_statistics(self, df: DataFrame) -> DataFrame:
        """Create rolling statistics."""
        w6 = Window.partitionBy("LOAN_SEQUENCE_NUMBER").orderBy("MONTHLY_REPORTING_PERIOD").rowsBetween(-6, -1)
        
        for var, suffix in [("CURRENT_ACTUAL_UPB", "balance"), ("CURRENT_INTEREST_RATE", "rate"), ("ELTV", "eltv")]:
            df = df.withColumn(f"rolling_avg_{suffix}_6m", spark_avg(var).over(w6)) \
                .withColumn(f"rolling_std_{suffix}_6m",
                    sqrt(spark_avg(pow(var - spark_avg(var).over(w6), 2)).over(w6))) \
                .withColumn(f"rolling_min_{suffix}_6m", spark_min(var).over(w6)) \
                .withColumn(f"rolling_max_{suffix}_6m", spark_max(var).over(w6))
        return df
    
    def _create_interaction_features(self, df: DataFrame) -> DataFrame:
        """Create interaction features."""
        return df.withColumn("dti_ltv_interaction",
            when(col("ORIGINAL_DTI").isNotNull() & col("ORIGINAL_LTV").isNotNull(),
                col("ORIGINAL_DTI") * col("ORIGINAL_LTV") / 100).otherwise(lit(0))) \
            .withColumn("credit_dti_interaction",
                when(col("CREDIT_SCORE").isNotNull() & col("ORIGINAL_DTI").isNotNull(),
                    col("CREDIT_SCORE") * col("ORIGINAL_DTI") / 100).otherwise(lit(0))) \
            .withColumn("balance_delinquency_interaction",
                when(col("delinquency_numeric") > 0, col("CURRENT_ACTUAL_UPB") * col("delinquency_numeric"))
                .otherwise(lit(0))) \
            .withColumn("age_balance_interaction", col("LOAN_AGE") * col("remaining_balance_pct"))