import pandas as pd
import numpy as np
from scipy import stats
import logging

def handle_missing_values(df):
    """Handle missing values by filling with median."""
    numeric_columns=df.select_dtypes(include="number")
    categorical_columns=df.select_dtypes(include="object")
    numeric_columns.fillna(numeric_columns.mean(), inplace=True)
    categorical_columns = categorical_columns.apply(lambda col: col.fillna(col.mode()[0]))
    df=pd.concat([numeric_columns, categorical_columns], axis=1)
    logging.info('Missing values handled.')
    return df

def detect_and_remove_outliers(df):
    """Detect and remove outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df_cleaned = df[(z_scores < 3).all(axis=1)]
    logging.info('Outliers removed using Z-score method.')
    return df_cleaned