# create_validation_customers.py
"""
Create validation dataset with:
- 1000 actual defaults
- 1000 non-defaults
- 500 random cases
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
import sys
import random

# Add project root
sys.path.insert(0, str(Path(__file__).parent))
from config.config import config


class CustomerValidationCreator:
    """
    Create validation dataset from cleaned data.
    """
    
    def __init__(self):
        self.features_dir = Path("D:/Projects/credit_risk_scoring/data/features")
        self.results_dir = Path("D:/Projects/credit_risk_scoring/results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load the full dataset with target
        print("Loading feature dataset with target...")
        self.df = dd.read_parquet(self.features_dir / "dataset_with_target.parquet")
        print(f"Total records: {len(self.df):,}")
        
        # Get loan-level information
        print("Aggregating loan-level data...")
        self.loan_df = self._aggregate_loans()
        print(f"Total loans: {len(self.loan_df):,}")
    
    def _aggregate_loans(self):
        """Aggregate to loan-level for sampling."""
        # Get loan-level aggregates
        loan_agg = self.df.groupby("LOAN_SEQUENCE_NUMBER").agg({
            "target": "max",  # Ever defaulted
            "vintage_year": "first",
            "ORIGINAL_UPB": "first",
            "CREDIT_SCORE": "first",
            "ORIGINAL_LTV": "first",
            "ORIGINAL_DTI": "first",
            "PROPERTY_STATE": "first"
        }).compute()
        
        # Reset index
        loan_agg = loan_agg.reset_index()
        loan_agg.columns = ['LOAN_SEQUENCE_NUMBER', 'ever_defaulted', 'vintage_year', 
                           'ORIGINAL_UPB', 'CREDIT_SCORE', 'ORIGINAL_LTV', 
                           'ORIGINAL_DTI', 'PROPERTY_STATE']
        
        return loan_agg
    
    def create_validation_set(self):
        """
        Create validation dataset with:
        - 1000 actual defaulters
        - 1000 non-defaulters
        - 500 random cases
        
        Uses sampled data to avoid memory issues.
        """
        print("\n" + "=" * 60)
        print("CREATING VALIDATION DATASET")
        print("=" * 60)
        
        # 1. Get defaulted loans from loan_df
        defaulted = self.loan_df[self.loan_df['ever_defaulted'] == 1]
        non_defaulted = self.loan_df[self.loan_df['ever_defaulted'] == 0]
        
        print(f"\nTotal Loans:")
        print(f"  Defaulted: {len(defaulted):,}")
        print(f"  Non-Defaulted: {len(non_defaulted):,}")
        
        # 2. Sample 1000 defaulted loans
        defaulted_sample = defaulted.sample(n=min(1000, len(defaulted)), random_state=42)
        print(f"  Sampled defaulted: {len(defaulted_sample):,}")
        
        # 3. Sample 1000 non-defaulted loans
        non_defaulted_sample = non_defaulted.sample(n=min(1000, len(non_defaulted)), random_state=42)
        print(f"  Sampled non-defaulted: {len(non_defaulted_sample):,}")
        
        # 4. Get 500 random cases (mix of both)
        random_sample = self.loan_df.sample(n=min(500, len(self.loan_df)), random_state=42)
        print(f"  Sampled random: {len(random_sample):,}")
        
        # 5. Combine
        validation_loans = pd.concat([
            defaulted_sample.assign(sample_type='defaulted'),
            non_defaulted_sample.assign(sample_type='non_defaulted'),
            random_sample.assign(sample_type='random')
        ]).drop_duplicates('LOAN_SEQUENCE_NUMBER')
        
        print(f"  Total unique loans: {len(validation_loans):,}")
        
        # 6. Get full records for these loans
        print("\nFetching full records for selected loans...")
        loan_ids = validation_loans['LOAN_SEQUENCE_NUMBER'].tolist()
        
        # OPTIMIZATION: Use sampled data if available
        # Check if we have sampled data
        sampled_path = Path(self.features_dir) / "sampled_dataset_with_target.parquet"
        
        if sampled_path.exists():
            print("  Using pre-sampled dataset...")
            df_sampled = dd.read_parquet(sampled_path)
        else:
            print("  Using full dataset (this may take time)...")
            df_sampled = self.df
        
        # Filter to only selected loans
        validation_data = df_sampled[df_sampled['LOAN_SEQUENCE_NUMBER'].isin(loan_ids)].compute()
        
        if len(validation_data) == 0:
            print("  WARNING: No records found! Trying full dataset...")
            validation_data = self.df[self.df['LOAN_SEQUENCE_NUMBER'].isin(loan_ids)].compute()
        
        # For each loan, keep the latest record
        validation_data = validation_data.sort_values('MONTHLY_REPORTING_PERIOD')
        validation_data = validation_data.groupby('LOAN_SEQUENCE_NUMBER').last().reset_index()
        
        print(f"Final validation dataset: {len(validation_data):,} records")
        
        # Save
        output_path = self.results_dir / "validation_customers_raw.csv"
        validation_data.to_csv(output_path, index=False)
        print(f"\nSaved raw validation data to: {output_path}")
        
        # Also save loan-level summary
        loan_summary = validation_data[['LOAN_SEQUENCE_NUMBER', 'target', 'vintage_year', 
                                        'ORIGINAL_UPB', 'CREDIT_SCORE', 'ORIGINAL_LTV']]
        loan_summary_path = self.results_dir / "validation_loan_summary.csv"
        loan_summary.to_csv(loan_summary_path, index=False)
        print(f"Saved loan summary to: {loan_summary_path}")
        
        return validation_data, validation_loans

def format_for_model(validation_data):
    """
    Format validation data for model input.
    Removes identifier columns and ensures all features are numeric.
    """
    print("\n" + "=" * 60)
    print("FORMATTING FOR MODEL INPUT")
    print("=" * 60)
    
    # Columns to drop (identifiers and target + dropped features)
    drop_cols = [
        'LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'target',
        'ingestion_timestamp', 'vintage_year', 'origination_year',
        'reporting_year', 'delinquency_numeric', 'is_delinquent',
        'is_terminated', 'future_termination', 'future_delinquency_max',
        'delinquency_days', 'row_num', 'cumulative_delinquency',
        'last_delinquent_month', 'last_modification_month',
        'delinquency_streak_id', 'num_modifications',
        'payment_deferral_count', 'rate_reduction_after_mod',
        'CURRENT_NON_INTEREST_BEARING_UPB', 'prev_interest_rate',
        'rolling_std_balance_6m', 'rolling_std_rate_6m',
        'rolling_avg_eltv_6m', 'rolling_std_eltv_6m',
        'rolling_min_eltv_6m', 'rolling_max_eltv_6m'
    ]
    
    # Get feature columns
    feature_cols = [c for c in validation_data.columns if c not in drop_cols]
    
    # Create input DataFrame
    model_input = validation_data[feature_cols].copy()
    
    # Ensure all columns are numeric
    for col in model_input.columns:
        if model_input[col].dtype == 'object':
            model_input[col] = pd.to_numeric(model_input[col], errors='coerce')
    
    # Fill NaN with 0
    model_input = model_input.fillna(0)
    
    # Add back loan identifier for tracking
    model_input['LOAN_SEQUENCE_NUMBER'] = validation_data['LOAN_SEQUENCE_NUMBER'].values
    
    return model_input


if __name__ == "__main__":
    print("=" * 60)
    print("CUSTOMER VALIDATION DATASET CREATOR")
    print("=" * 60)
    
    creator = CustomerValidationCreator()
    validation_data, loan_summary = creator.create_validation_set()
    
    # Format for model
    model_input = format_for_model(validation_data)
    
    # Save model-ready input
    input_path = Path("D:/Projects/credit_risk_scoring/results/validation_customers_input.csv")
    model_input.to_csv(input_path, index=False)
    print(f"\nModel-ready input saved to: {input_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total customers: {len(model_input):,}")
    print(f"Features: {len(model_input.columns) - 1}")  # -1 for LOAN_SEQUENCE_NUMBER
    print(f"\nSample distribution:")
    print(f"  Defaulted: {sum(validation_data['target'] == 1):,}")
    print(f"  Non-Defaulted: {sum(validation_data['target'] == 0):,}")