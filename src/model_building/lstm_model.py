import logging
from EDA.logging_config import setup_logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore

# Set up logging
setup_logging(log_directory="c:/Users/User/Downloads/film/Rossmann-Pharmaceuticals-Store-Sales-Prediction/logs", log_file="lstm.log")

def build_lstm_model(input_shape):
    """Build an LSTM model for time series prediction."""
    model = Sequential() # type: ignore
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    logging.info("LSTM model built.")
    return model

def scale_time_series_data(data):
    """Scale time series data in the range (-1, 1)."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    logging.info("Time series data scaled.")
    return scaled_data, scaler

def prepare_lstm_data(data, time_step):
    """Transform time series data into supervised learning format."""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    logging.info("Time series data transformed for LSTM.")
    return np.array(X), np.array(y)