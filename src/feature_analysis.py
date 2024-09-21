import pandas as pd
import logging

def compare_promotion_distribution(train_df, test_df):
    """Check if promotions are distributed similarly in training and test data."""
    train_promo_mean = train_df['Promo2'].mean()
    test_promo_mean = test_df['Promo2'].mean()
    
    logging.info(f'Training promotion mean: {train_promo_mean}, Test promotion mean: {test_promo_mean}')
    return train_promo_mean, test_promo_mean

def analyze_seasonal_sales(df):
    """Check for seasonal behaviors like Christmas, Easter."""
    seasonal_sales = df.groupby('Season')['Sales'].mean()
    logging.info(f'Seasonal sales: {seasonal_sales}')
    return seasonal_sales

def analyze_promo_effect(df):
    """Analyze promo effects on sales and customers."""
    promo_sales = df.groupby('Promo')['Sales'].mean()
    promo_customers = df.groupby('Promo')['Number_of_Customers'].mean()

    logging.info(f'Promo sales: {promo_sales}, Promo customers: {promo_customers}')
    return promo_sales, promo_customers

def analyze_store_promo_recommendations(df):
    """Analyze which stores should deploy promos for more effectiveness."""
    promo_efficiency = df.groupby('Store')['Promo'].mean()
    high_sales_stores = promo_efficiency.sort_values(ascending=False).head(10)
    
    logging.info(f'Stores recommended for promo: {high_sales_stores}')
    return high_sales_stores

def analyze_distance_effect_on_sales(df):
    """Analyze how distance to competitors affects sales."""
    distance_effect = df.groupby('Competitor_Distance')['Sales'].mean()
    logging.info(f'Competitor distance effect: {distance_effect}')
    return distance_effect

def analyze_new_competitor_effect(df):
    """Analyze how new competitors affect stores."""
    new_competitor_effect = df[df['New_Competitor_Flag'] == 1].groupby('Store')['Sales'].mean()
    logging.info(f'Effect of new competitors: {new_competitor_effect}')
    return new_competitor_effect