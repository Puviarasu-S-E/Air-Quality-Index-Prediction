import pandas as pd
import numpy as np
import os

def create_aqi_dataset(n_samples=1000):
    """Create realistic AQI dataset"""
    np.random.seed(42)
    
    data = {}
    
    # Pollutant concentrations
    data['PM2_5'] = np.random.lognormal(mean=3.0, sigma=0.8, size=n_samples)
    data['PM10'] = data['PM2_5'] * np.random.uniform(1.5, 2.5, n_samples)
    data['NO2'] = np.random.lognormal(mean=3.2, sigma=0.6, size=n_samples)
    data['SO2'] = np.random.lognormal(mean=2.5, sigma=0.7, size=n_samples)
    data['CO'] = np.random.lognormal(mean=1.8, sigma=0.5, size=n_samples)
    data['O3'] = np.random.lognormal(mean=3.5, sigma=0.4, size=n_samples)
    
    # Meteorological data
    data['Temperature'] = np.random.normal(20, 10, n_samples)
    data['Humidity'] = np.random.beta(2, 2, n_samples) * 100
    data['Wind_Speed'] = np.random.gamma(2, 2, n_samples)
    data['Pressure'] = np.random.normal(1013, 20, n_samples)
    
    # Location and time data
    data['Month'] = np.random.randint(1, 13, n_samples)
    data['Hour'] = np.random.randint(0, 24, n_samples)
    data['City'] = np.random.choice(range(5), n_samples)
    data['Industrial_Area'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['Traffic_Density'] = np.random.poisson(3, n_samples)
    
    df = pd.DataFrame(data)
    
    # Ensure positive values
    pollutant_cols = ['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    for col in pollutant_cols:
        df[col] = np.abs(df[col])
    
    # Calculate AQI
    def calculate_aqi(row):
        pm25_norm = min(row['PM2_5'] / 150, 1.0)
        pm10_norm = min(row['PM10'] / 250, 1.0)
        no2_norm = min(row['NO2'] / 100, 1.0)
        so2_norm = min(row['SO2'] / 80, 1.0)
        co_norm = min(row['CO'] / 20, 1.0)
        o3_norm = min(row['O3'] / 150, 1.0)
        
        aqi_base = (pm25_norm * 0.3 + pm10_norm * 0.2 + no2_norm * 0.2 + 
                   so2_norm * 0.1 + co_norm * 0.1 + o3_norm * 0.1) * 300
        
        temp_effect = 1 + (abs(row['Temperature'] - 25) / 100)
        humidity_effect = 1 + (row['Humidity'] / 500)
        wind_effect = max(0.5, 1 - (row['Wind_Speed'] / 50))
        industrial_effect = 1 + (row['Industrial_Area'] * 0.3)
        traffic_effect = 1 + (row['Traffic_Density'] / 50)
        
        aqi = aqi_base * temp_effect * humidity_effect * wind_effect * industrial_effect * traffic_effect
        aqi += np.random.normal(0, 10)
        
        return max(0, min(500, aqi))
    
    df['AQI'] = df.apply(calculate_aqi, axis=1).round().astype(int)
    
    # Save dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/clean_data.csv', index=False)
    
    print(f"Dataset created: {df.shape}")
    print(f"AQI range: {df['AQI'].min()} - {df['AQI'].max()}")
    
    return df

if __name__ == "__main__":
    create_aqi_dataset()