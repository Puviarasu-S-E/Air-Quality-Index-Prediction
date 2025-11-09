import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def create_evaluation_curves():
    """Create comprehensive evaluation curves for AQI regression model"""
    
    # Load data
    df = pd.read_csv('data/processed/clean_data.csv')
    
    # Select original 15 features (matching the trained model)
    feature_cols = ['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 
                   'Humidity', 'Wind_Speed', 'Pressure', 'Month', 'Hour', 
                   'City', 'Industrial_Area', 'Traffic_Density']
    
    X = df[feature_cols]
    y = df['AQI']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[0, 0].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    axes[0, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    axes[0, 0].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    axes[0, 0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    axes[0, 0].set_xlabel('Training Set Size')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Learning Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Validation Curve (n_estimators)
    param_range = [10, 25, 50, 75, 100, 125, 150, 200]
    train_scores, val_scores = validation_curve(
        RandomForestRegressor(random_state=42), X_train_scaled, y_train,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    
    axes[0, 1].plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    axes[0, 1].plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    axes[0, 1].set_xlabel('Number of Estimators')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('Validation Curve (n_estimators)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction Error Plot
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    axes[0, 2].scatter(y_test, y_pred, alpha=0.6, color='purple')
    axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 2].set_xlabel('Actual AQI')
    axes[0, 2].set_ylabel('Predicted AQI')
    r2 = r2_score(y_test, y_pred)
    axes[0, 2].set_title(f'Prediction Accuracy (R² = {r2:.3f})')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Residuals vs Fitted
    residuals = y_test - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals vs Fitted Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. AQI Category Classification Accuracy
    def get_aqi_category(aqi):
        if aqi <= 50: return 'Good'
        elif aqi <= 100: return 'Moderate'
        elif aqi <= 150: return 'Unhealthy for Sensitive'
        elif aqi <= 200: return 'Unhealthy'
        elif aqi <= 300: return 'Very Unhealthy'
        else: return 'Hazardous'
    
    actual_categories = [get_aqi_category(aqi) for aqi in y_test]
    pred_categories = [get_aqi_category(aqi) for aqi in y_pred]
    
    categories = ['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    category_accuracy = {}
    
    for cat in categories:
        actual_count = actual_categories.count(cat)
        if actual_count > 0:
            correct_count = sum(1 for a, p in zip(actual_categories, pred_categories) if a == cat and p == cat)
            category_accuracy[cat] = correct_count / actual_count
        else:
            category_accuracy[cat] = 0
    
    cats = list(category_accuracy.keys())
    accs = list(category_accuracy.values())
    
    axes[1, 1].bar(range(len(cats)), accs, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('AQI Categories')
    axes[1, 1].set_ylabel('Classification Accuracy')
    axes[1, 1].set_title('AQI Category Classification Accuracy')
    axes[1, 1].set_xticks(range(len(cats)))
    axes[1, 1].set_xticklabels(cats, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Error Distribution
    axes[1, 2].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 2].axvline(x=0, color='r', linestyle='--')
    axes[1, 2].set_xlabel('Prediction Error')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Distribution of Prediction Errors')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/evaluation_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[feature_cols + ['AQI']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel Evaluation Metrics:")
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Mean Category Accuracy: {np.mean(list(category_accuracy.values())):.3f}")
    
    return model, scaler

if __name__ == "__main__":
    create_evaluation_curves()