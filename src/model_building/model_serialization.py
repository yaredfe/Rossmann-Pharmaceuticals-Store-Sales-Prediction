import logging
from EDA.logging_config import setup_logging
import joblib
import time

# Set up logging
setup_logging(log_directory="c:/Users/User/Downloads/film/Rossmann-Pharmaceuticals-Store-Sales-Prediction/logs", log_file="model_serialization.log")

def serialize_model(pipeline, model_name: str):
    """Save the model with a timestamp."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model_name}-{timestamp}.pkl"
    joblib.dump(pipeline, filename)
    logging.info(f"Model saved as {filename}")
    return filename