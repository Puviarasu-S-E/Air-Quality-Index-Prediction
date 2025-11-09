import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess_aqi_data():
    """Preprocess AQI dataset for model training"""
    
    # Load dataset
    df = pd.read_csv('data/processed/clean_data.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Remove outliers using IQR method
    def remove_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Remove outliers from key columns
    outlier_cols = ['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
    original_size = len(df)
    
    for col in outlier_cols:
        if col in df.columns:
            df = remove_outliers(df, col)
    
    print(f"Outliers removed: {original_size - len(df)} rows")
    
    # Feature engineering
    df['PM_Ratio'] = df['PM2_5'] / (df['PM10'] + 1)
    df['Pollutant_Sum'] = df['PM2_5'] + df['PM10'] + df['NO2'] + df['SO2'] + df['CO'] + df['O3']
    df['Is_Winter'] = ((df['Month'] >= 11) | (df['Month'] <= 2)).astype(int)
    df['Is_Rush_Hour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9) | 
                         (df['Hour'] >= 17) & (df['Hour'] <= 19)).astype(int)
    
    # Save processed data
    df.to_csv('data/processed/preprocessed_data.csv', index=False)
    
    print(f"Preprocessing completed: {df.shape}")
    return df

if __name__ == "__main__":
    preprocess_aqi_data()