import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def train_aqi_models():
    """Train and evaluate machine learning models for AQI prediction"""
    
    # Load preprocessed data
    df = pd.read_csv('data/processed/preprocessed_data.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Prepare features and target
    X = df.drop('AQI', axis=1)
    y = df['AQI']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Polynomial Regression': None
    }
    
    results = {}
    trained_models = {}
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'Polynomial Regression':
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train_scaled)
            X_test_poly = poly_features.transform(X_test_scaled)
            
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)
            
            trained_models[f'{name}_poly'] = poly_features
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R²': r2}
        trained_models[name] = model
        
        print(f"  R² Score: {r2:.3f}")
        print(f"  MAE: {mae:.2f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best R² Score: {results[best_model_name]['R²']:.3f}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature importance for Random Forest
    if best_model_name == 'Random Forest':
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models/model_results.csv')
    
    print(f"Models saved to 'models/' directory")
    return trained_models, results

if __name__ == "__main__":
    train_aqi_models()