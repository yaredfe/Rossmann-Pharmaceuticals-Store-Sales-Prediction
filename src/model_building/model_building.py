import logging
from EDA.logging_config import setup_logging
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Set up logging
setup_logging(log_directory="c:/Users/User/Downloads/film/Rossmann-Pharmaceuticals-Store-Sales-Prediction/logs", log_file="model_building.log")

def build_pipeline():
    """Build a pipeline for preprocessing and RandomForest modeling."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    logging.info("Pipeline built with StandardScaler and RandomForestRegressor.")
    return pipeline

def train_model(df: pd.DataFrame, target: str):
    """Train the Random Forest model using the pipeline."""
    X = df.drop(columns=[target,"Date"])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    logging.info(f"Model trained. Target: {target}")
    return pipeline, X_test, y_test