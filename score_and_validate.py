# score_and_validate.py
"""
Score validation customers and validate predictions.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from models.base import BaseCreditRiskModel

logger = logging.getLogger(__name__)


class CustomerValidator:
    """Score validation customers and validate against actual outcomes."""
    
    def __init__(self, models_dir="D:/Projects/credit_risk_scoring/models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        print("Loading models...")
        for model_file in self.models_dir.glob("model_*.pkl"):
            model_name = model_file.stem.replace("model_", "")
            with open(model_file, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            print(f"  Loaded: {model_name}")
    
    def prepare_features(self, df, model_name='ensemble'):
        """Prepare features for model prediction."""
        model = self.models[model_name]
        
        # Get feature names from model
        if hasattr(model, 'feature_names') and model.feature_names:
            feature_cols = model.feature_names
        elif hasattr(model, 'feature_importances_') and hasattr(model, 'n_features_in_'):
            feature_cols = df.columns.tolist()
        else:
            feature_cols = [c for c in df.columns if c != 'LOAN_SEQUENCE_NUMBER']
        
        # FIX: Add missing columns that the model expects
        # Check which required columns are missing
        available_cols = df.columns.tolist()
        
        # Common missing columns from config.drop_features
        default_columns = {
            'origination_year': 0,
            'vintage_year': 0,
            'reporting_year': 0,
            'delinquency_numeric': 0,
            'is_delinquent': 0,
            'is_terminated': 0,
            'future_termination': 0,
            'future_delinquency_max': 0,
            'delinquency_days': 0,
            'row_num': 0,
            'cumulative_delinquency': 0,
            'last_delinquent_month': 'MISSING',
            'last_modification_month': 'MISSING',
            'delinquency_streak_id': 0,
            'num_modifications': 0,
            'payment_deferral_count': 0,
            'rate_reduction_after_mod': 0,
            'CURRENT_NON_INTEREST_BEARING_UPB': 0,
            'prev_interest_rate': 0,
            'rolling_std_balance_6m': 0,
            'rolling_std_rate_6m': 0,
            'rolling_avg_eltv_6m': 0,
            'rolling_std_eltv_6m': 0,
            'rolling_min_eltv_6m': 0,
            'rolling_max_eltv_6m': 0,
        }
        
        # Add missing columns with default values
        for col, default_val in default_columns.items():
            if col in feature_cols and col not in available_cols:
                df[col] = default_val
                print(f"  Added missing column: {col} (default={default_val})")
        
        # Filter to only columns that exist and are in feature_cols
        available_cols = df.columns.tolist()
        X_cols = [c for c in feature_cols if c in available_cols]
        
        print(f"  Using {len(X_cols)} features for prediction")
        
        X = df[X_cols].copy()
        X = X.fillna(0)
        
        return X, X_cols
    
    def score_customers(self, df, model_name='ensemble'):
        """Score all customers in the DataFrame."""
        print(f"\nScoring {len(df)} customers using {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Prepare features
        X, feature_cols = self.prepare_features(df, model_name)
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict(X)
        
        # Get class predictions
        pred = (proba >= 0.5).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'LOAN_SEQUENCE_NUMBER': df['LOAN_SEQUENCE_NUMBER'].values,
            'predicted_probability': proba,
            'predicted_default': pred,
            'model_used': model_name
        })
        
        # Add true labels if available
        if 'target' in df.columns:
            results['true_default'] = df['target'].values
            results['correct'] = results['predicted_default'] == results['true_default']
        
        print(f"Scoring complete!")
        return results
    
    def validate_predictions(self, results):
        """Validate predictions against true labels."""
        if 'true_default' not in results.columns:
            print("No true labels available. Skipping validation.")
            return {}
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        y_true = results['true_default'].values
        y_pred = results['predicted_default'].values
        y_proba = results['predicted_probability'].values
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  True Negatives: {cm[0,0]:,}")
        print(f"  False Positives: {cm[0,1]:,}")
        print(f"  False Negatives: {cm[1,0]:,}")
        print(f"  True Positives: {cm[1,1]:,}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return metrics
    
    def generate_report(self, results, output_dir="D:/Projects/credit_risk_scoring/results"):
        """Generate comprehensive validation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save predictions
        results.to_csv(output_dir / "customer_predictions.csv", index=False)
        print(f"\nPredictions saved to: {output_dir}/customer_predictions.csv")
        
        # Metrics
        metrics = self.validate_predictions(results)
        
        return results, metrics


if __name__ == "__main__":
    print("=" * 60)
    print("CUSTOMER SCORING & VALIDATION")
    print("=" * 60)
    
    # Load validation customers
    input_path = Path("D:/Projects/credit_risk_scoring/results/validation_customers_raw.csv")
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run create_validation_customers.py first.")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df):,} validation customers")
    
    # Initialize validator
    validator = CustomerValidator()
    
    # Score using ensemble
    results = validator.score_customers(df, model_name='ensemble')
    
    # Generate report
    results, metrics = validator.generate_report(results)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE!")
    print("=" * 60)