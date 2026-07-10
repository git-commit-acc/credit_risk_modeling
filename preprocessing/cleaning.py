# preprocessing/cleaning.py
"""
Data cleaning for Freddie Mac SFLLD data based on the Data Dictionary.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (col, when, lit, trim, isnan, isnull, coalesce, round as spark_round, length, substring, concat, expr)
from pyspark.sql.types import DoubleType
import logging
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


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
    
    def clean_origination_data(self, df: DataFrame) -> DataFrame:
        """Clean origination data according to Data Dictionary rules."""
        logger.info("Cleaning origination data...")
        
        # Credit Score: 300-850 valid, others NULL
        df = df.withColumn(
            "CREDIT_SCORE",
            when(
                col("CREDIT_SCORE").between(300, 850),
                col("CREDIT_SCORE")
            ).otherwise(lit(None))
        )
        
        # DTI: > 65% becomes NULL
        df = df.withColumn(
            "ORIGINAL_DTI",
            when(
                (col("ORIGINAL_DTI") <= 65) & (col("ORIGINAL_DTI") >= 0),
                col("ORIGINAL_DTI")
            ).when(
                col("ORIGINAL_DTI") == 999,
                lit(None)
            ).otherwise(lit(None))
        )
        
        # LTV: Handle HARP and non-HARP
        df = df.withColumn(
            "ORIGINAL_LTV",
            when(
                (col("RELIEF_REFINANCE_INDICATOR") == "Y") &
                (col("ORIGINAL_LTV") <= 998) &
                (col("ORIGINAL_LTV") >= 1),
                col("ORIGINAL_LTV")
            ).when(
                (col("RELIEF_REFINANCE_INDICATOR") != "Y") &
                (col("ORIGINAL_LTV").between(6, 105)),
                col("ORIGINAL_LTV")
            ).when(
                col("ORIGINAL_LTV") == 999,
                lit(None)
            ).otherwise(lit(None))
        )
        
        # CLTV: 0-200 valid
        df = df.withColumn(
            "ORIGINAL_CLTV",
            when(
                col("ORIGINAL_CLTV").between(0, 200),
                col("ORIGINAL_CLTV")
            ).when(
                col("ORIGINAL_CLTV") == 999,
                lit(None)
            ).otherwise(lit(None))
        )
        
        # Number of Borrowers: 1-10 valid
        df = df.withColumn(
            "NUMBER_OF_BORROWERS",
            when(
                col("NUMBER_OF_BORROWERS").between(1, 10),
                col("NUMBER_OF_BORROWERS")
            ).when(
                col("NUMBER_OF_BORROWERS") == 99,
                lit(None)
            ).otherwise(lit(None))
        )
        
        # Loan Term: 60-480 valid
        df = df.withColumn(
            "ORIGINAL_LOAN_TERM",
            when(
                col("ORIGINAL_LOAN_TERM").between(60, 480),
                col("ORIGINAL_LOAN_TERM")
            ).otherwise(lit(None))
        )
        
        # Property Type
        valid_property = ['SF', 'CONDO', 'COOP', 'PUD', 'MH', '9']
        df = df.withColumn(
            "PROPERTY_TYPE",
            when(
                col("PROPERTY_TYPE").isin(valid_property),
                col("PROPERTY_TYPE")
            ).otherwise(lit(None))
        )
        
        # Occupancy Status
        valid_occupancy = ['P', 'S', 'I', '9']
        df = df.withColumn(
            "OCCUPANCY_STATUS",
            when(
                col("OCCUPANCY_STATUS").isin(valid_occupancy),
                col("OCCUPANCY_STATUS")
            ).otherwise(lit(None))
        )
        
        # Channel
        valid_channel = ['R', 'B', 'C', 'T', '9']
        df = df.withColumn(
            "CHANNEL",
            when(
                col("CHANNEL").isin(valid_channel),
                col("CHANNEL")
            ).otherwise(lit(None))
        )
        
        # Loan Purpose
        valid_purpose = ['P', 'R', 'C', '9']
        df = df.withColumn(
            "LOAN_PURPOSE",
            when(
                col("LOAN_PURPOSE").isin(valid_purpose),
                col("LOAN_PURPOSE")
            ).otherwise(lit(None))
        )
        
        # MI Percentage: 0-100 valid
        df = df.withColumn(
            "MI_PERCENTAGE",
            when(
                col("MI_PERCENTAGE").between(0, 100),
                col("MI_PERCENTAGE")
            ).when(
                col("MI_PERCENTAGE") == 999,
                lit(None)
            ).otherwise(lit(None))
        )
        
        # Number of Units: 1-4 valid
        df = df.withColumn(
            "NUMBER_OF_UNITS",
            when(
                col("NUMBER_OF_UNITS").between(1, 4),
                col("NUMBER_OF_UNITS")
            ).when(
                col("NUMBER_OF_UNITS") == 9,
                lit(None)
            ).otherwise(lit(None))
        )
        
        # Amortization Type
        valid_amort = ['FRM', 'ARM', 'IO', '9']
        df = df.withColumn(
            "AMORTIZATION_TYPE",
            when(
                col("AMORTIZATION_TYPE").isin(valid_amort),
                col("AMORTIZATION_TYPE")
            ).otherwise(lit(None))
        )
        
        # Super Conforming Flag
        df = df.withColumn(
            "SUPER_CONFORMING_FLAG",
            when(
                col("SUPER_CONFORMING_FLAG").isin(['Y', ' ']),
                col("SUPER_CONFORMING_FLAG")
            ).otherwise(lit(None))
        )
        
        # Relief Refinance Indicator
        df = df.withColumn(
            "RELIEF_REFINANCE_INDICATOR",
            when(
                col("RELIEF_REFINANCE_INDICATOR").isin(['Y', 'N', ' ']),
                col("RELIEF_REFINANCE_INDICATOR")
            ).otherwise(lit(None))
        )
        
        # Special Eligibility Program
        valid_programs = ['H', 'F', 'R', ' ']
        df = df.withColumn(
            "SPECIAL_ELIGIBILITY_PROGRAM",
            when(
                col("SPECIAL_ELIGIBILITY_PROGRAM").isin(valid_programs),
                col("SPECIAL_ELIGIBILITY_PROGRAM")
            ).otherwise(lit(None))
        )
        
        # Round UPB to nearest $1,000
        df = df.withColumn(
            "ORIGINAL_UPB",
            spark_round(col("ORIGINAL_UPB") / 1000) * 1000
        )
        
        # Mask Postal Code (last 2 digits -> 00)
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
        
        # Validate Property State
        df = df.withColumn(
            "PROPERTY_STATE",
            when(
                col("PROPERTY_STATE").isin(self.VALID_STATES),
                col("PROPERTY_STATE")
            ).otherwise(lit(None))
        )
        
        # IO Indicator
        df = df.withColumn(
            "IO_INDICATOR",
            when(
                col("IO_INDICATOR").isin(['Y', 'N', '9']),
                col("IO_INDICATOR")
            ).otherwise(lit(None))
        )
        
        # Validate date fields
        df = df.withColumn(
            "FIRST_PAYMENT_DATE",
            when(
                length(col("FIRST_PAYMENT_DATE")) == 6,
                col("FIRST_PAYMENT_DATE")
            ).otherwise(lit(None))
        ).withColumn(
            "MATURITY_DATE",
            when(
                length(col("MATURITY_DATE")) == 6,
                col("MATURITY_DATE")
            ).otherwise(lit(None))
        )
        
        logger.info("Origination data cleaning completed.")
        return df
    
    def clean_performance_data(self, df: DataFrame) -> DataFrame:
        """Clean performance data according to Data Dictionary rules."""
        logger.info("Cleaning performance data...")
        
        # Delinquency Status
        df = df.withColumn(
            "CURRENT_LOAN_DELINQUENCY_STATUS",
            when(
                col("CURRENT_LOAN_DELINQUENCY_STATUS").isin(self.VALID_DELINQUENCY),
                col("CURRENT_LOAN_DELINQUENCY_STATUS")
            ).otherwise(lit(None))
        )
        
        # Modification Flag
        df = df.withColumn(
            "MODIFICATION_FLAG",
            when(
                col("MODIFICATION_FLAG").isin(self.VALID_MOD_FLAGS),
                col("MODIFICATION_FLAG")
            ).otherwise(lit(None))
        )
        
        # Zero Balance Code
        df = df.withColumn(
            "ZERO_BALANCE_CODE",
            when(
                col("ZERO_BALANCE_CODE").isin(self.VALID_ZERO_BALANCE),
                col("ZERO_BALANCE_CODE")
            ).otherwise(lit(None))
        )
        
        # Payment Deferral Flag
        df = df.withColumn(
            "PAYMENT_DEFERRAL_FLAG",
            when(
                col("PAYMENT_DEFERRAL_FLAG").isin(self.VALID_DEFERRAL_FLAGS),
                col("PAYMENT_DEFERRAL_FLAG")
            ).otherwise(lit(None))
        )
        
        # Disaster Flag
        df = df.withColumn(
            "DELINQUENCY_DUE_TO_DISASTER",
            when(
                col("DELINQUENCY_DUE_TO_DISASTER").isin(['Y', 'N', '']),
                col("DELINQUENCY_DUE_TO_DISASTER")
            ).otherwise(lit(None))
        )
        
        # Current Interest Rate: 0-30 valid
        df = df.withColumn(
            "CURRENT_INTEREST_RATE",
            when(
                col("CURRENT_INTEREST_RATE").between(0, 30),
                col("CURRENT_INTEREST_RATE")
            ).otherwise(lit(None))
        )
        
        # Clean numeric fields
        numeric_fields = [
            'CURRENT_ACTUAL_UPB', 'LOAN_AGE', 'REMAINING_MONTHS_TO_LEGAL_MATURITY',
            'CURRENT_NON_INTEREST_BEARING_UPB', 'MI_RECOVERIES', 'NET_SALE_PROCEEDS',
            'NON_MI_RECOVERIES', 'TOTAL_EXPENSES', 'LEGAL_COSTS',
            'MAINTENANCE_AND_PRESERVATION_COSTS', 'TAXES_AND_INSURANCE',
            'MISCELLANEOUS_EXPENSES', 'ACTUAL_LOSS_CALCULATION',
            'CUMULATIVE_MODIFICATION_COST', 'ELTV', 'ZERO_BALANCE_REMOVAL_UPB',
            'DELINQUENT_ACCRUED_INTEREST', 'CURRENT_MONTH_MODIFICATION_COST',
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
        
        # Validate date fields
        date_fields = [
            'MONTHLY_REPORTING_PERIOD',
            'ZERO_BALANCE_EFFECTIVE_DATE',
            'DDLPI',
            'DEFECT_SETTLEMENT_DATE'
        ]
        
        for field in date_fields:
            if field in df.columns:
                df = df.withColumn(
                    field,
                    when(
                        length(col(field)) == 6,
                        col(field)
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