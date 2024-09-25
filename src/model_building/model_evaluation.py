import logging
from EDA.logging_config import setup_logging
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Set up logging
setup_logging(log_directory="c:/Users/User/Downloads/film/Rossmann-Pharmaceuticals-Store-Sales-Prediction/logs", log_file="model_evaluation.log")

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the model performance."""
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logging.info(f"Model evaluation completed. MSE: {mse}, MAE: {mae}")
    return mse, mae

def feature_importance_analysis(pipeline, features: list):
    """Analyze feature importance from the RandomForest model."""
    importances = pipeline.named_steps['model'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    logging.info("Feature importance analysis completed.")
    return feature_importance_df.sort_values(by='Importance', ascending=False)

def calculate_confidence_intervals(y_pred: np.array, y_test: np.array, confidence: float = 0.95):
    """Estimate the confidence interval for the predictions."""
    errors = y_pred - y_test
    mean_error = np.mean(errors)
    error_margin = np.std(errors) * confidence
    logging.info("Confidence interval calculated.")
    return mean_error, error_margin