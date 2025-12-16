import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = None):
    """Set up MLflow experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_model_metrics(y_true, y_pred, y_proba=None, prefix="") -> Dict[str, float]:
    """Log evaluation metrics to MLflow."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    mlflow.log_metric(f"{prefix}accuracy", accuracy)
    mlflow.log_metric(f"{prefix}precision", precision)
    mlflow.log_metric(f"{prefix}recall", recall)
    mlflow.log_metric(f"{prefix}f1_score", f1)
    
    metrics = {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    }

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            mlflow.log_metric(f"{prefix}roc_auc", roc_auc)
            metrics["roc_auc"] = roc_auc
        except ValueError:
            pass # Handle cases where ROC AUC cannot be calculated (e.g. only one class)
            
    return metrics

def tune_hyperparameters(model, param_grid, X_train, y_train, search_type='grid', cv=3, scoring='f1'):
    """Tune hyperparameters using Grid or Random search."""
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, n_iter=10)
    else:
        raise ValueError("search_type must be 'grid' or 'random'")
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def run_experiment(experiment_name: str, model_name: str, model, X_train, y_train, X_test, y_test, param_grid: Optional[Dict] = None, search_type: str = 'grid'):
    """Run a full experiment: setup, tune (optional), train, evaluate, log."""
    setup_mlflow_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name):
        if param_grid:
            mlflow.log_param("tuning_method", search_type)
            best_model, best_params = tune_hyperparameters(model, param_grid, X_train, y_train, search_type)
            mlflow.log_params(best_params)
            model = best_model
        else:
            model.fit(X_train, y_train)
            
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            
        # Log metrics
        metrics = log_model_metrics(y_test, y_pred, y_proba)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run {model_name} completed. Metrics: {metrics}")
        return model, metrics

def register_best_model(experiment_name: str, metric: str = "f1_score", higher_is_better: bool = True):
    """Find the best run in the experiment and register it."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment {experiment_name} not found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC" if higher_is_better else f"metrics.{metric} ASC"]
    )
    
    if not runs:
        print("No runs found.")
        return
        
    best_run = runs[0]
    print(f"Best run: {best_run.info.run_id} with {metric}: {best_run.data.metrics.get(metric)}")
    
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri, f"{experiment_name}_best_model")
    print(f"Registered model from run {best_run.info.run_id} as {experiment_name}_best_model")
