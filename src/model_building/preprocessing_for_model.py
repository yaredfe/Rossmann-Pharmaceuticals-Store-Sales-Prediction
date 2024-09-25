import logging
from EDA.logging_config import setup_logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging using the logging_config module
setup_logging(log_directory="c:/Users/User/Downloads/film/Rossmann-Pharmaceuticals-Store-Sales-Prediction/logs", log_file="preprocessing.log")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    logging.info("Handling missing values with forward fill method.")
    return df.ffill()

def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Extract date-related features from a datetime column."""
    df[date_col] = pd.to_datetime(df[date_col])
    df['Weekday'] = df[date_col].dt.weekday
    df['Weekend'] = df['Weekday'] >= 5
    # df['DaysToHoliday'] = (df[date_col] - df['HolidayDate']).dt.days  # Requires 'HolidayDate'
    # df['DaysAfterHoliday'] = (df[date_col] - df['HolidayDate']).dt.days
    df['IsMonthStart'] = df[date_col].dt.is_month_start
    df['IsMonthEnd'] = df[date_col].dt.is_month_end
    logging.info(f"Extracted date features from {date_col}")
    return df

def scale_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Scale selected features using StandardScaler."""
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    logging.info(f"Scaled features: {features}")
    return df

def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Encode categorical columns using LabelEncoder."""
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
        logging.info(f"Encoded categorical column: {column}")
    return df