# preprocessing/cleaning.py
"""
Data cleaning for Freddie Mac SFLLD data based on the Data Dictionary.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lit, trim, isnan, isnull, coalesce, round as spark_round,
    length, substring, concat, expr, regexp_replace
)
from pyspark.sql.types import DoubleType, IntegerType, StringType
import logging
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# CATEGORICAL MAPPING DICTIONARIES (SFLLD Data Dictionary Compliant)
# =============================================================================

# Add to the categorical mappings section
PROPERTY_STATE_MAPPING = {
    'AL': 1, 'AK': 2, 'AZ': 3, 'AR': 4, 'CA': 5, 'CO': 6, 'CT': 7, 'DE': 8,
    'FL': 9, 'GA': 10, 'HI': 11, 'ID': 12, 'IL': 13, 'IN': 14, 'IA': 15,
    'KS': 16, 'KY': 17, 'LA': 18, 'ME': 19, 'MD': 20, 'MA': 21, 'MI': 22,
    'MN': 23, 'MS': 24, 'MO': 25, 'MT': 26, 'NE': 27, 'NV': 28, 'NH': 29,
    'NJ': 30, 'NM': 31, 'NY': 32, 'NC': 33, 'ND': 34, 'OH': 35, 'OK': 36,
    'OR': 37, 'PA': 38, 'RI': 39, 'SC': 40, 'SD': 41, 'TN': 42, 'TX': 43,
    'UT': 44, 'VT': 45, 'VA': 46, 'WA': 47, 'WV': 48, 'WI': 49, 'WY': 50,
    'DC': 51, 'PR': 52, 'GU': 53, 'VI': 54
}

# Delinquency Status - Includes 'RA' for REO Acquisition
DELINQUENCY_STATUS_NUMERIC = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "RA": 10,      # REO Acquisition
}

# Origination Mappings
FIRST_TIME_HOMEBUYER_FLAG = {"Y": 1, "N": 0, "9": None}
OCCUPANCY_STATUS = {"P": 1, "S": 2, "I": 3, "9": None}
CHANNEL = {"R": 1, "B": 2, "C": 3, "T": 4, "9": None}
PROPERTY_TYPE = {"SF": 1, "CONDO": 2, "COOP": 3, "PUD": 4, "MH": 5, "9": None}
LOAN_PURPOSE = {"P": 1, "R": 2, "C": 3, "9": None}
AMORTIZATION_TYPE = {"FRM": 1, "ARM": 2, "IO": 3, "9": None}
RELIEF_REFINANCE_INDICATOR = {"Y": 1, "N": 0, " ": None}
SUPER_CONFORMING_FLAG = {"Y": 1, " ": 0}
SPECIAL_ELIGIBILITY_PROGRAM = {"H": 1, "F": 2, "R": 3, " ": None}
PPM_FLAG = {"Y": 1, "N": 0}
IO_INDICATOR = {"Y": 1, "N": 0, "9": None}
MI_CANCELLATION_INDICATOR = {"Y": 1, "N": 0, "9": None}

# Performance Mappings
ZERO_BALANCE_CODE = {
    "01": 1, "02": 2, "03": 3, "09": 4, "15": 5, "16": 6, "96": 7
}
MODIFICATION_FLAG = {"Y": 1, "P": 2}
PAYMENT_DEFERRAL_FLAG = {"Y": 1, "P": 2}
INTEREST_RATE_STEP_INDICATOR = {"Y": 1, "N": 0}
DELINQUENCY_DUE_TO_DISASTER = {"Y": 1, "N": 0}
BORROWER_ASSISTANCE_STATUS_CODE = {"T": 1, "F": 2, "R": 3}


class SFLLDDataCleaner:
    """Data cleaner for Freddie Mac SFLLD data."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
        # Valid US state abbreviations
        self.VALID_STATES = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'PR', 'GU', 'VI'
        ]
        
        # Valid delinquency statuses
        self.VALID_DELINQUENCY = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'RA']
        
        # Valid zero balance codes
        self.VALID_ZERO_BALANCE = ['01', '02', '03', '09', '15', '16', '96', '']
        
        # Valid modification flags
        self.VALID_MOD_FLAGS = ['Y', 'P', '']
        
        # Valid payment deferral flags
        self.VALID_DEFERRAL_FLAGS = ['Y', 'P', '']
    
    def _encode_categorical(self, df: DataFrame, column: str, mapping: Dict) -> DataFrame:
        """Encode categorical column using mapping dictionary."""
        if column not in df.columns:
            return df
        
        when_expr = None
        for value, code in mapping.items():
            if code is None:
                continue
            condition = col(column) == lit(value)
            if when_expr is None:
                when_expr = when(condition, lit(code))
            else:
                when_expr = when_expr.when(condition, lit(code))
        
        # WORKING VERSION - DO NOT CHANGE
        if when_expr is not None:
            df = df.withColumn(column, when_expr.otherwise(lit(None)))
        
        return df
    
    def clean_origination_data(self, df: DataFrame) -> DataFrame:
        """Clean origination data according to Data Dictionary rules."""
        logger.info("Cleaning origination data...")
        
        # Credit Score: 300-850 valid, 9999 = Not Available
        df = df.withColumn(
            "CREDIT_SCORE",
            when(
                col("CREDIT_SCORE").between(300, 850),
                col("CREDIT_SCORE")
            ).otherwise(lit(None))
        )
        
        # FIRST_PAYMENT_DATE: Validate YYYYMM format
        df = df.withColumn(
            "FIRST_PAYMENT_DATE",
            when(
                length(col("FIRST_PAYMENT_DATE")) == 6,
                col("FIRST_PAYMENT_DATE")
            ).otherwise(lit(None))
        )
        
        # FIRST_TIME_HOMEBUYER_FLAG: Encode to numeric
        df = self._encode_categorical(
            df, "FIRST_TIME_HOMEBUYER_FLAG", FIRST_TIME_HOMEBUYER_FLAG
        )
        
        # MATURITY_DATE: Validate YYYYMM format
        df = df.withColumn(
            "MATURITY_DATE",
            when(
                length(col("MATURITY_DATE")) == 6,
                col("MATURITY_DATE")
            ).otherwise(lit(None))
        )
        
        # MI_PERCENTAGE: 0-100 valid, 999 = Not Available
        df = df.withColumn(
            "MI_PERCENTAGE",
            when(
                (col("MI_PERCENTAGE").between(0, 100)) & (col("MI_PERCENTAGE") != 999),
                col("MI_PERCENTAGE")
            ).otherwise(lit(None))
        )
        
        # NUMBER_OF_UNITS: 1-4 valid, 9 = Not Available
        df = df.withColumn(
            "NUMBER_OF_UNITS",
            when(
                (col("NUMBER_OF_UNITS").between(1, 4)) & (col("NUMBER_OF_UNITS") != 9),
                col("NUMBER_OF_UNITS")
            ).otherwise(lit(None))
        )
        
        # OCCUPANCY_STATUS: Encode to numeric
        df = self._encode_categorical(
            df, "OCCUPANCY_STATUS", OCCUPANCY_STATUS
        )
        
        # ORIGINAL_CLTV: 0-200 valid, 999 = Not Available
        df = df.withColumn(
            "ORIGINAL_CLTV",
            when(
                (col("ORIGINAL_CLTV").between(0, 200)) & (col("ORIGINAL_CLTV") != 999),
                col("ORIGINAL_CLTV")
            ).otherwise(lit(None))
        )
        
        # ORIGINAL_DTI: 0-65 valid, 999 = Not Available (HARP = 999)
        df = df.withColumn(
            "ORIGINAL_DTI",
            when(
                (col("ORIGINAL_DTI") <= 65) & (col("ORIGINAL_DTI") >= 0) & (col("ORIGINAL_DTI") != 999),
                col("ORIGINAL_DTI")
            ).otherwise(lit(None))
        )
        
        # ORIGINAL_UPB: Round to nearest $1,000 (per Data Dictionary)
        df = df.withColumn(
            "ORIGINAL_UPB",
            spark_round(col("ORIGINAL_UPB") / 1000) * 1000
        )
        
        # ORIGINAL_LTV: Per Data Dictionary - HARP and non-HARP have different ranges
        df = df.withColumn(
            "ORIGINAL_LTV",
            when(
                # Non-HARP: 6-105%
                (col("RELIEF_REFINANCE_INDICATOR") != "Y") &
                (col("ORIGINAL_LTV").between(6, 105)),
                col("ORIGINAL_LTV")
            ).when(
                # HARP: 1-998%
                (col("RELIEF_REFINANCE_INDICATOR") == "Y") &
                (col("ORIGINAL_LTV").between(1, 998)),
                col("ORIGINAL_LTV")
            ).when(
                # 999 = Not Available
                col("ORIGINAL_LTV") == 999,
                lit(None)
            ).otherwise(lit(None))
        )
        
        # ORIGINAL_INTEREST_RATE: 0-30% valid
        df = df.withColumn(
            "ORIGINAL_INTEREST_RATE",
            when(
                col("ORIGINAL_INTEREST_RATE").between(0, 30),
                col("ORIGINAL_INTEREST_RATE")
            ).otherwise(lit(None))
        )
        
        # PROPERTY_STATE: Encode to numeric
        df = self._encode_categorical(df, "PROPERTY_STATE", PROPERTY_STATE_MAPPING)

        # CHANNEL: Encode to numeric
        df = self._encode_categorical(df, "CHANNEL", CHANNEL)
        
        # PPM_FLAG: Encode to numeric
        df = self._encode_categorical(df, "PPM_FLAG", PPM_FLAG)
        
        # AMORTIZATION_TYPE: Encode to numeric
        df = self._encode_categorical(df, "AMORTIZATION_TYPE", AMORTIZATION_TYPE)
        
        # PROPERTY_TYPE: Encode to numeric
        df = self._encode_categorical(df, "PROPERTY_TYPE", PROPERTY_TYPE)
        
        # POSTAL_CODE: Mask last 2 digits to "00" (per Data Dictionary)
        df = df.withColumn(
            "POSTAL_CODE",
            when(
                length(col("POSTAL_CODE")) >= 5,
                concat(
                    substring(col("POSTAL_CODE"), 1, 3),
                    lit("00")
                )
            ).otherwise(col("POSTAL_CODE"))
        )
        
        # LOAN_PURPOSE: Encode to numeric
        df = self._encode_categorical(df, "LOAN_PURPOSE", LOAN_PURPOSE)
        
        # ORIGINAL_LOAN_TERM: 60-480 months valid
        df = df.withColumn(
            "ORIGINAL_LOAN_TERM",
            when(
                col("ORIGINAL_LOAN_TERM").between(60, 480),
                col("ORIGINAL_LOAN_TERM")
            ).otherwise(lit(None))
        )
        
        # NUMBER_OF_BORROWERS: 1-10 valid, 99 = Not Available
        df = df.withColumn(
            "NUMBER_OF_BORROWERS",
            when(
                (col("NUMBER_OF_BORROWERS").between(1, 10)) & (col("NUMBER_OF_BORROWERS") != 99),
                col("NUMBER_OF_BORROWERS")
            ).otherwise(lit(None))
        )
        
        # SUPER_CONFORMING_FLAG: Encode to numeric
        df = self._encode_categorical(df, "SUPER_CONFORMING_FLAG", SUPER_CONFORMING_FLAG)
        
        # PRE_RELIEF_REFINANCE_LSN: Keep as string (link to prior loan)
        # Already string type, no cleaning needed
        
        # SPECIAL_ELIGIBILITY_PROGRAM: Encode to numeric
        df = self._encode_categorical(df, "SPECIAL_ELIGIBILITY_PROGRAM", SPECIAL_ELIGIBILITY_PROGRAM)
        
        # RELIEF_REFINANCE_INDICATOR: Encode to numeric
        df = self._encode_categorical(df, "RELIEF_REFINANCE_INDICATOR", RELIEF_REFINANCE_INDICATOR)
        
        # IO_INDICATOR: Encode to numeric
        df = self._encode_categorical(df, "IO_INDICATOR", IO_INDICATOR)
        
        # MI_CANCELLATION_INDICATOR: Encode to numeric
        df = self._encode_categorical(df, "MI_CANCELLATION_INDICATOR", MI_CANCELLATION_INDICATOR)
        
        logger.info("Origination data cleaning completed.")
        return df
    
    def clean_performance_data(self, df: DataFrame) -> DataFrame:
        """Clean performance data according to Data Dictionary rules."""
        logger.info("Cleaning performance data...")
        
        # CRITICAL: Encode delinquency status with 'RA' support
        df = self._encode_categorical(
            df, "CURRENT_LOAN_DELINQUENCY_STATUS", DELINQUENCY_STATUS_NUMERIC
        )
        
        # LOAN_AGE: Should be non-negative
        df = df.withColumn(
            "LOAN_AGE",
            when(
                col("LOAN_AGE") >= 0,
                col("LOAN_AGE")
            ).otherwise(lit(None))
        )
        
        # REMAINING_MONTHS_TO_LEGAL_MATURITY: Should be non-negative
        df = df.withColumn(
            "REMAINING_MONTHS_TO_LEGAL_MATURITY",
            when(
                col("REMAINING_MONTHS_TO_LEGAL_MATURITY") >= 0,
                col("REMAINING_MONTHS_TO_LEGAL_MATURITY")
            ).otherwise(lit(None))
        )
        
        # DEFECT_SETTLEMENT_DATE: Validate YYYYMM format
        df = df.withColumn(
            "DEFECT_SETTLEMENT_DATE",
            when(
                length(col("DEFECT_SETTLEMENT_DATE")) == 6,
                col("DEFECT_SETTLEMENT_DATE")
            ).otherwise(lit(None))
        )
        
        # MODIFICATION_FLAG: Encode to numeric
        df = self._encode_categorical(df, "MODIFICATION_FLAG", MODIFICATION_FLAG)
        
        # ZERO_BALANCE_CODE: Encode to numeric
        df = self._encode_categorical(df, "ZERO_BALANCE_CODE", ZERO_BALANCE_CODE)
        
        # ZERO_BALANCE_EFFECTIVE_DATE: Validate YYYYMM format
        df = df.withColumn(
            "ZERO_BALANCE_EFFECTIVE_DATE",
            when(
                length(col("ZERO_BALANCE_EFFECTIVE_DATE")) == 6,
                col("ZERO_BALANCE_EFFECTIVE_DATE")
            ).otherwise(lit(None))
        )
        
        # CURRENT_INTEREST_RATE: 0-30% valid
        df = df.withColumn(
            "CURRENT_INTEREST_RATE",
            when(
                col("CURRENT_INTEREST_RATE").between(0, 30),
                col("CURRENT_INTEREST_RATE")
            ).otherwise(lit(None))
        )
        
        # INTEREST_RATE_STEP_INDICATOR: Encode to numeric
        df = self._encode_categorical(df, "INTEREST_RATE_STEP_INDICATOR", INTEREST_RATE_STEP_INDICATOR)
        
        # PAYMENT_DEFERRAL_FLAG: Encode to numeric
        df = self._encode_categorical(df, "PAYMENT_DEFERRAL_FLAG", PAYMENT_DEFERRAL_FLAG)
        
        # DELINQUENCY_DUE_TO_DISASTER: Encode to numeric
        df = self._encode_categorical(df, "DELINQUENCY_DUE_TO_DISASTER", DELINQUENCY_DUE_TO_DISASTER)
        
        # BORROWER_ASSISTANCE_STATUS_CODE: Encode to numeric
        df = self._encode_categorical(df, "BORROWER_ASSISTANCE_STATUS_CODE", BORROWER_ASSISTANCE_STATUS_CODE)
        
        # Clean numeric fields: Convert to Double, invalid → NULL
        numeric_fields = [
            'CURRENT_ACTUAL_UPB', 
            'CURRENT_NON_INTEREST_BEARING_UPB',
            'MI_RECOVERIES', 
            'NET_SALE_PROCEEDS',
            'NON_MI_RECOVERIES', 
            'TOTAL_EXPENSES', 
            'LEGAL_COSTS',
            'MAINTENANCE_AND_PRESERVATION_COSTS', 
            'TAXES_AND_INSURANCE',
            'MISCELLANEOUS_EXPENSES', 
            'ACTUAL_LOSS_CALCULATION',
            'CUMULATIVE_MODIFICATION_COST', 
            'ELTV', 
            'ZERO_BALANCE_REMOVAL_UPB',
            'DELINQUENT_ACCRUED_INTEREST', 
            'CURRENT_MONTH_MODIFICATION_COST',
            'INTEREST_BEARING_UPB'
        ]
        
        for field in numeric_fields:
            if field in df.columns:
                df = df.withColumn(
                    field,
                    when(
                        col(field).isNotNull() & ~isnan(col(field)),
                        col(field).cast(DoubleType())
                    ).otherwise(lit(None))
                )
        
        # DDLPI: Validate YYYYMM format
        df = df.withColumn(
            "DDLPI",
            when(
                length(col("DDLPI")) == 6,
                col("DDLPI")
            ).otherwise(lit(None))
        )
        
        # Calculate derived fields if missing
        df = df.withColumn(
            "LOAN_AGE_CALCULATED",
            when(
                col("LOAN_AGE").isNull() & 
                col("MONTHLY_REPORTING_PERIOD").isNotNull(),
                expr("""
                    (cast(substring(MONTHLY_REPORTING_PERIOD, 1, 4) as int) - 1999) * 12 +
                    cast(substring(MONTHLY_REPORTING_PERIOD, 5, 2) as int)
                """)
            ).otherwise(col("LOAN_AGE"))
        )
        
        df = df.withColumn(
            "LOAN_AGE",
            coalesce(col("LOAN_AGE"), col("LOAN_AGE_CALCULATED"))
        ).drop("LOAN_AGE_CALCULATED")
        
        logger.info("Performance data cleaning completed.")
        return df
    
    def clean_both_datasets(
        self,
        origination_df: DataFrame,
        performance_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """Clean both origination and performance datasets."""
        logger.info("=" * 60)
        logger.info("STARTING DATA CLEANING")
        logger.info("=" * 60)
        
        orig_cleaned = self.clean_origination_data(origination_df)
        perf_cleaned = self.clean_performance_data(performance_df)
        
        logger.info("=" * 60)
        logger.info("DATA CLEANING COMPLETED")
        logger.info(f"Origination: {orig_cleaned.count():,} loans")
        logger.info(f"Performance: {perf_cleaned.count():,} records")
        logger.info("=" * 60)
        
        return orig_cleaned, perf_cleaned