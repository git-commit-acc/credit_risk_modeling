# data_ingestion/data_ingestion.py
"""
Data ingestion module for Freddie Mac SFLLD data.
Handles loading, schema definition, and initial processing.
"""

import os
import logging
import sys
from typing import List, Optional
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
from pyspark.sql.functions import col, lit, current_timestamp

logger = logging.getLogger(__name__)


class SFLLDDataIngestion:
    """
    Handles ingestion of Freddie Mac SFLLD data files.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(__name__)
    
    def get_origination_schema(self) -> StructType:
        """
        Origination schema - EXACT column order from Freddie Mac documentation.
        32 columns, pipe-delimited, NO header.
        """
        return StructType([
            # Col 1-4: Basic Loan Information
            StructField("CREDIT_SCORE", IntegerType(), True),
            StructField("FIRST_PAYMENT_DATE", StringType(), True),
            StructField("FIRST_TIME_HOMEBUYER_FLAG", StringType(), True),
            StructField("MATURITY_DATE", StringType(), True),
            
            # Col 5-9: Property and Loan Details
            StructField("MSA", StringType(), True),
            StructField("MI_PERCENTAGE", IntegerType(), True),
            StructField("NUMBER_OF_UNITS", IntegerType(), True),
            StructField("OCCUPANCY_STATUS", StringType(), True),
            StructField("ORIGINAL_CLTV", IntegerType(), True),
            
            # Col 10-14: Financial Metrics
            StructField("ORIGINAL_DTI", IntegerType(), True),
            StructField("ORIGINAL_UPB", IntegerType(), True),
            StructField("ORIGINAL_LTV", IntegerType(), True),
            StructField("ORIGINAL_INTEREST_RATE", DoubleType(), True),
            StructField("CHANNEL", StringType(), True),
            
            # Col 15-19: Loan Characteristics
            StructField("PPM_FLAG", StringType(), True),
            StructField("AMORTIZATION_TYPE", StringType(), True),
            StructField("PROPERTY_STATE", StringType(), True),
            StructField("PROPERTY_TYPE", StringType(), True),
            StructField("POSTAL_CODE", StringType(), True),
            
            # Col 20-24: Identification and Terms
            StructField("LOAN_SEQUENCE_NUMBER", StringType(), False),
            StructField("LOAN_PURPOSE", StringType(), True),
            StructField("ORIGINAL_LOAN_TERM", IntegerType(), True),
            StructField("NUMBER_OF_BORROWERS", IntegerType(), True),
            StructField("SELLER_NAME", StringType(), True),
            
            # Col 25-29: Servicing Information
            StructField("SERVICER_NAME", StringType(), True),
            StructField("SUPER_CONFORMING_FLAG", StringType(), True),
            StructField("PRE_RELIEF_REFINANCE_LSN", StringType(), True),
            StructField("SPECIAL_ELIGIBILITY_PROGRAM", StringType(), True),
            StructField("RELIEF_REFINANCE_INDICATOR", StringType(), True),
            
            # Col 30-32: Additional Fields
            StructField("PROPERTY_VALUATION_METHOD", IntegerType(), True),
            StructField("IO_INDICATOR", StringType(), True),
            StructField("MI_CANCELLATION_INDICATOR", StringType(), True),
        ])
    
    def get_performance_schema(self) -> StructType:
        """
        Performance schema - EXACT column order from Freddie Mac documentation.
        32 columns, pipe-delimited, NO header.
        """
        return StructType([
            # Col 1-3: Identification and Balance
            StructField("LOAN_SEQUENCE_NUMBER", StringType(), False),
            StructField("MONTHLY_REPORTING_PERIOD", StringType(), True),
            StructField("CURRENT_ACTUAL_UPB", DoubleType(), True),
            
            # Col 4-7: Delinquency Status
            StructField("CURRENT_LOAN_DELINQUENCY_STATUS", StringType(), True),
            StructField("LOAN_AGE", IntegerType(), True),
            StructField("REMAINING_MONTHS_TO_LEGAL_MATURITY", IntegerType(), True),
            StructField("DEFECT_SETTLEMENT_DATE", StringType(), True),
            
            # Col 8-11: Modification and Termination
            StructField("MODIFICATION_FLAG", StringType(), True),
            StructField("ZERO_BALANCE_CODE", StringType(), True),
            StructField("ZERO_BALANCE_EFFECTIVE_DATE", StringType(), True),
            StructField("CURRENT_INTEREST_RATE", DoubleType(), True),
            
            # Col 12-16: Financial Details
            StructField("CURRENT_NON_INTEREST_BEARING_UPB", DoubleType(), True),
            StructField("DDLPI", StringType(), True),
            StructField("MI_RECOVERIES", DoubleType(), True),
            StructField("NET_SALE_PROCEEDS", DoubleType(), True),
            StructField("NON_MI_RECOVERIES", DoubleType(), True),
            
            # Col 17-21: Expense Details
            StructField("TOTAL_EXPENSES", DoubleType(), True),
            StructField("LEGAL_COSTS", DoubleType(), True),
            StructField("MAINTENANCE_AND_PRESERVATION_COSTS", DoubleType(), True),
            StructField("TAXES_AND_INSURANCE", DoubleType(), True),
            StructField("MISCELLANEOUS_EXPENSES", DoubleType(), True),
            
            # Col 22-26: Loss and Modification
            StructField("ACTUAL_LOSS_CALCULATION", DoubleType(), True),
            StructField("CUMULATIVE_MODIFICATION_COST", DoubleType(), True),
            StructField("INTEREST_RATE_STEP_INDICATOR", StringType(), True),
            StructField("PAYMENT_DEFERRAL_FLAG", StringType(), True),
            StructField("ELTV", DoubleType(), True),
            
            # Col 27-32: Additional Fields
            StructField("ZERO_BALANCE_REMOVAL_UPB", DoubleType(), True),
            StructField("DELINQUENT_ACCRUED_INTEREST", DoubleType(), True),
            StructField("DELINQUENCY_DUE_TO_DISASTER", StringType(), True),
            StructField("BORROWER_ASSISTANCE_STATUS_CODE", StringType(), True),
            StructField("CURRENT_MONTH_MODIFICATION_COST", DoubleType(), True),
            StructField("INTEREST_BEARING_UPB", DoubleType(), True),
        ])
    
    def read_origination_file(
        self,
        file_path: str,
        year: int
    ) -> DataFrame:
        """
        Read a single origination file.
        
        Args:
            file_path: Path to the file
            year: Vintage year
            
        Returns:
            DataFrame with origination data
        """
        schema = self.get_origination_schema()
        
        df = self.spark.read.csv(
            file_path,
            sep="|",
            schema=schema,
            header=False,
            mode="PERMISSIVE"
        )
        
        # Add metadata - only add if columns don't exist
        if 'vintage_year' not in df.columns:
            df = df.withColumn("vintage_year", lit(year))
        if 'ingestion_timestamp' not in df.columns:
            df = df.withColumn("ingestion_timestamp", current_timestamp())
        
        return df
    
    def read_performance_file(
        self,
        file_path: str,
        year: int
    ) -> DataFrame:
        """
        Read a single performance file.
        
        Args:
            file_path: Path to the file
            year: Vintage year
            
        Returns:
            DataFrame with performance data
        """
        schema = self.get_performance_schema()
        
        df = self.spark.read.csv(
            file_path,
            sep="|",
            schema=schema,
            header=False,
            mode="PERMISSIVE"
        )
        
        # Add metadata - only add if columns don't exist
        if 'vintage_year' not in df.columns:
            df = df.withColumn("vintage_year", lit(year))
        if 'ingestion_timestamp' not in df.columns:
            df = df.withColumn("ingestion_timestamp", current_timestamp())
        
        # Add reporting year for partitioning - only if column doesn't exist
        if 'reporting_year' not in df.columns:
            df = df.withColumn(
                "reporting_year",
                col("MONTHLY_REPORTING_PERIOD").substr(1, 4)
            )
        
        return df
    
    def ingest_all_years(
        self,
        raw_dir: str,
        years: List[int],
        bronze_dir: str,
        file_prefix: str = "sample",
        data_type: str = "origination"
    ) -> DataFrame:
        """
        Ingest all years for a given data type.
        
        Args:
            raw_dir: Raw data directory
            years: List of years to ingest
            bronze_dir: Output directory
            file_prefix: File prefix (sample, historical_data, etc.)
            data_type: "origination" or "performance"
            
        Returns:
            Unioned DataFrame of all years
        """
        self.logger.info(f"Ingesting {data_type} data for years {years}")
        
        all_dfs = []
        
        for year in years:
            if data_type == "origination":
                file_name = f"{file_prefix}_orig_{year}.txt"
                file_path = os.path.join(raw_dir, file_name)
                # Convert Windows path to Spark-compatible path
                file_path = file_path.replace("\\", "/")
                df = self.read_origination_file(file_path, year)
            else:
                file_name = f"{file_prefix}_svcg_{year}.txt"
                file_path = os.path.join(raw_dir, file_name)
                file_path = file_path.replace("\\", "/")
                df = self.read_performance_file(file_path, year)
            
            count = df.count()
            self.logger.info(f"  Year {year}: Loaded {count:,} records")
            all_dfs.append(df)
        
        # Union all DataFrames
        if not all_dfs:
            raise RuntimeError(f"No {data_type} files were successfully loaded")
        
        df_all = all_dfs[0]
        for df in all_dfs[1:]:
            df_all = df_all.unionByName(df, allowMissingColumns=True)
        
        # Write to bronze layer
        if data_type == "origination":
            output_path = os.path.join(bronze_dir, "origination_bronze.parquet").replace("\\", "/")
            df_all.write.mode("overwrite") \
                  .option("compression", "snappy") \
                  .parquet(output_path)
        else:
            output_path = os.path.join(bronze_dir, "performance_bronze.parquet").replace("\\", "/")
            df_all.write.mode("overwrite") \
                  .option("compression", "snappy") \
                  .partitionBy("reporting_year") \
                  .parquet(output_path)
        
        self.logger.info(f"Total {data_type} records ingested: {df_all.count():,}")
        return df_all


def get_spark_path(windows_path: str) -> str:
    """
    Convert Windows path to Spark-compatible local file path.
    
    Args:
        windows_path: Windows-style path
        
    Returns:
        Spark-compatible path
    """
    normalized = windows_path.replace("\\", "/")
    if ":" in normalized:
        return f"file:///{normalized}"
    else:
        return f"file:///{normalized}"


# def create_spark_session(app_name: str = "SFLLD_DataIngestion") -> SparkSession:
#     """
#     Creates a Spark session optimized for Windows 11 local development.
#     """
#     os.environ["PYSPARK_PYTHON"] = sys.executable
#     os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
#     # os.environ["HADOOP_HOME"] = r"C:\hadoop"
#     os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    
#     # Windows-specific optimizations
#     os.environ["PYTHONHASHSEED"] = "0"

#     warehouse_dir = r"D:\Projects\credit_risk_scoring\spark-warehouse"
#     os.makedirs(warehouse_dir, exist_ok=True)
    
#     # Clean up any leftover temp files
#     import shutil
#     temp_dir = os.path.expanduser("~/AppData/Local/Temp/spark-*")
#     try:
#         for d in glob.glob(temp_dir):
#             if os.path.isdir(d):
#                 shutil.rmtree(d, ignore_errors=True)
#     except Exception:
#         pass
    
#     spark = (
#         SparkSession.builder
#         .appName(app_name)
#         .master("local[*]")

#         # Storage
#         .config("spark.sql.warehouse.dir", get_spark_path(warehouse_dir))
#         .config("spark.local.dir", r"D:\spark\spark-temp")

#         # Memory
#         .config("spark.driver.memory", "16g")
#         .config("spark.executor.memory", "8g")
#         .config("spark.driver.maxResultSize", "4g")

#         # Shuffle
#         .config("spark.sql.shuffle.partitions", "128")
#         .config("spark.default.parallelism", "128")

#         # Adaptive Query Execution
#         .config("spark.sql.adaptive.enabled", "true")
#         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
#         .config("spark.sql.adaptive.skewJoin.enabled", "true")

#         # Serialization
#         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

#         # Compression
#         .config("spark.sql.parquet.compression.codec", "snappy")

#         # Local filesystem
#         .config("spark.hadoop.fs.defaultFS", "file:///")
#         .config("spark.hadoop.fs.default.name", "file:///")
#         .config("spark.hadoop.fs.file.impl",
#                 "org.apache.hadoop.fs.LocalFileSystem")

#         # Windows
#         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
#         .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        
#         # Timeouts to prevent connection issues
#         .config("spark.network.timeout", "600s")
#         .config("spark.executor.heartbeatInterval", "60s")
#         .config("spark.sql.broadcastTimeout", "600")

#         .getOrCreate()
#     )
    
#     conf = spark.sparkContext._jsc.hadoopConfiguration()

#     print("fs.defaultFS =", conf.get("fs.defaultFS"))
#     print("fs.default.name =", conf.get("fs.default.name"))
    
#     spark.sparkContext.setLogLevel("WARN")
#     logger.info(f"Spark Session created: {spark.version}")
#     return spark

def create_spark_session(app_name: str = "SFLLD_DataIngestion") -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    
    warehouse_dir = r"D:\Projects\credit_risk_scoring\spark-warehouse"
    os.makedirs(warehouse_dir, exist_ok=True)
    
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        
        # Storage
        .config("spark.sql.warehouse.dir", get_spark_path(warehouse_dir))
        .config("spark.local.dir", r"D:\spark\spark-temp")
        
        # REDUCED Memory (was 16g driver, 8g executor)
        .config("spark.driver.memory", "8g")  # Reduced from 16g
        .config("spark.executor.memory", "4g")  # Reduced from 8g
        .config("spark.driver.maxResultSize", "2g")  # Reduced from 4g
        
        # Shuffle
        .config("spark.sql.shuffle.partitions", "64")  # Reduced from 128
        .config("spark.default.parallelism", "64")  # Reduced from 128
        
        # Adaptive Query Execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        
        # Compression
        .config("spark.sql.parquet.compression.codec", "snappy")
        
        # Local filesystem
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.hadoop.fs.default.name", "file:///")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        
        # Windows
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "5000")  # Reduced from 10000
        
        # Timeouts
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.sql.broadcastTimeout", "600")
        
        # ADD THESE TO PREVENT MEMORY ISSUES
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .config("spark.sql.autoBroadcastJoinThreshold", "50m")  # Reduced from default
        
        .getOrCreate()
    )
    
    return spark