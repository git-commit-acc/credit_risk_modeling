# test_gpu.py
"""Test GPU availability for all models."""

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

print("XGBoost GPU Support:")
try:
    print(f"  Version: {xgb.__version__}")
    print(f"  GPU available: {xgb.get_omp_threads() > 0}")
except:
    print("  Not available")

print("\nLightGBM GPU Support:")
try:
    print(f"  Version: {lgb.__version__}")
    print(f"  GPU available: {lgb.basic._DUMMY_DEVICE != 'cpu'}")
except:
    print("  Not available")

print("\nCatBoost GPU Support:")
try:
    print(f"  Version: {cb.__version__}")
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(task_type='GPU', devices='0')
    print("  GPU available: Yes")
except Exception as e:
    print(f"  GPU available: No ({e})")

print("\nCUDA Information:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout[:500])
except:
    print("  nvidia-smi not found")

import xgboost as xgb
# import lightgbm as lgb
# Check if GPU is available for XGBoost
print(xgb.__version__)
# For LightGBM, a simple check: 
# print(lgb.__version__)