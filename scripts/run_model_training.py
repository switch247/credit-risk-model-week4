import sys
import os
from pathlib import Path
import pandas as pd
import mlflow

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.features.xente_features import build_feature_pipeline
from src.pipeline.tabular_modeling import build_classification_models, split_features_target
from src.pipeline.experiment_tracking import run_experiment, register_best_model

def main():
    # 1. Load Data
    data_path = project_root / "data" / "processed" / "xente_processed.csv"
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Split Data
    print("Splitting data...")
    # Ensure target is int
    df["FraudResult"] = df["FraudResult"].astype(int)
    
    X_train, X_test, y_train, y_test = split_features_target(
        df, target="FraudResult", test_size=0.2, random_state=42
    )
    
    # 3. Build Feature Pipeline
    print("Building feature pipeline...")
    # Define categorical columns excluding the target 'FraudResult'
    categorical_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId',
        'PricingStrategy'
    ]
    
    # We need to identify categorical and numerical columns if we want to override defaults
    feat_pipe, _ = build_feature_pipeline(categorical_cols=categorical_cols)
    
    # 4. Build Models
    print("Building models...")
    # Note: build_classification_models expects a preprocessor (ColumnTransformer)
    # We are passing the whole feature pipeline as the preprocessor.
    # This works because the feature pipeline is a Transformer.
    models = build_classification_models(preprocessor=feat_pipe)
    
    # 5. Run Experiments
    experiment_name = "Credit_Risk_Fraud_Detection"
    # Set tracking URI to a local folder
    mlflow.set_tracking_uri("file:./mlruns")
    
    for model_name, model_pipeline in models.items():
        print(f"Running experiment for {model_name}...")
        
        # Define hyperparameter grid based on model type
        # Using smaller grids for demonstration/speed
        param_grid = None
        if "log_reg" in model_name:
            param_grid = {
                "model__C": [0.1, 1.0],
                # "model__solver": ["liblinear", "lbfgs"] # lbfgs is default and good
            }
        elif "decision_tree" in model_name:
            param_grid = {
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 5]
            }
        elif "random_forest" in model_name:
            param_grid = {
                "model__n_estimators": [50, 100],
                "model__max_depth": [5, 10]
            }
        elif "gradient_boosting" in model_name:
             param_grid = {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1]
            }
        elif "xgb" in model_name:
             param_grid = {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1]
            }
            
        # Run experiment
        # We use 'random' search here to be faster if grid is large, but grid is small so 'grid' is fine.
        # Let's use 'grid' as requested in instructions (or random)
        run_experiment(
            experiment_name=experiment_name,
            model_name=model_name,
            model=model_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            param_grid=param_grid,
            search_type='grid'
        )
        
    # 6. Register Best Model
    print("Registering best model...")
    register_best_model(experiment_name, metric="f1_score")

if __name__ == "__main__":
    main()
