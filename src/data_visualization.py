import matplotlib.pyplot as plt
import seaborn as sns
import logging

def visualize_promotion_distribution(train_df, test_df):
    """Visualize the distribution of promotions in train and test sets."""
    plt.figure(figsize=(10, 5))
    sns.histplot(train_df['Promo2'], label='Train', color='blue', kde=True)
    sns.histplot(test_df['Promo2'], label='Test', color='orange', kde=True)
    plt.legend()
    plt.title('Distribution of Promotions in Train vs Test Set')
    plt.show()
    logging.info('Promotion distribution visualized.')

def visualize_sales_behavior_during_holidays(df):
    """Visualize sales behavior before, during, and after holidays."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='StateHoliday', y='Sales', data=df)
    plt.title('Sales Behavior During Holidays')
    plt.show()
    logging.info('Sales behavior during holidays visualized.')

def visualize_correlation_sales_customers(df):
    """Visualize the correlation between sales and number of customers."""
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='Customers', y='Sales', data=df)
    plt.title('Correlation Between Sales and Number of Customers')
    plt.show()
    logging.info('Correlation between sales and number of customers visualized.')

def visualize_promo_effect_on_sales(df):
    """Visualize promo effect on sales."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Effect of Promos on Sales')
    plt.show()
    logging.info('Promo effect on sales visualized.')

def visualize_assortment_effect_on_sales(df):
    """Visualize the effect of assortment type on sales."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Assortment', y='Sales', data=df)
    plt.title('Effect of Assortment on Sales')
    plt.show()
    logging.info('Assortment effect on sales visualized.')

def visualize_store_opening_trends(df):
    """Visualize customer behavior during store opening/closing times."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='Open', y='Sales', data=df)
    plt.title('Customer Behavior During Store Opening and Closing Times')
    plt.show()
    logging.info('Customer behavior during store opening and closing times visualized.')