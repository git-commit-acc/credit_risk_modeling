# setup_sampling.py
"""Quick script to sample data for training."""

import dask.dataframe as dd
import os

# Configuration
SAMPLE_SIZE = 500000  # Change this value as needed
INPUT_PATH = "D:/Projects/credit_risk_scoring/data/features"
OUTPUT_PATH = "D:/Projects/credit_risk_scoring/data/features_sampled"

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load and sample
print(f"Sampling {SAMPLE_SIZE:,} records...")

train = dd.read_parquet(f"{INPUT_PATH}/train_data.parquet")
val = dd.read_parquet(f"{INPUT_PATH}/val_data.parquet")
test = dd.read_parquet(f"{INPUT_PATH}/test_data.parquet")

# Sample
train_sample = train.sample(frac=min(1.0, SAMPLE_SIZE/len(train)), random_state=42)
val_sample = val.sample(frac=min(1.0, SAMPLE_SIZE*0.2/len(val)), random_state=42)
test_sample = test.sample(frac=min(1.0, SAMPLE_SIZE*0.4/len(test)), random_state=42)

# Save
print("Saving sampled data...")
train_sample.to_parquet(f"{OUTPUT_PATH}/train_data.parquet", compression='snappy')
val_sample.to_parquet(f"{OUTPUT_PATH}/val_data.parquet", compression='snappy')
test_sample.to_parquet(f"{OUTPUT_PATH}/test_data.parquet", compression='snappy')

print(f"Done! Sampled data saved to {OUTPUT_PATH}")
print(f"Train: {len(train_sample):,}")
print(f"Val: {len(val_sample):,}")
print(f"Test: {len(test_sample):,}")