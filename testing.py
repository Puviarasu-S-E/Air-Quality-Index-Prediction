import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def test_aqi_models():
    """Load saved models and evaluate performance on test data"""
    
    # Load preprocessed data
    df = pd.read_csv('data/processed/preprocessed_data.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Prepare features and target
    X = df.drop('AQI', axis=1)
    y = df['AQI']
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load saved models
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("Models loaded successfully")
    except FileNotFoundError:
        print("Error: Model files not found. Please run training.py first.")
        return
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance on Test Data:")
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual AQI')
    axes[0, 0].set_ylabel('Predicted AQI')
    axes[0, 0].set_title(f'Actual vs Predicted AQI (R² = {r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted AQI')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature importance
    if os.path.exists('models/feature_importance.csv'):
        importance_df = pd.read_csv('models/feature_importance.csv')
        top_features = importance_df.head(10)
        axes[1, 1].barh(top_features['feature'], top_features['importance'])
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top 10 Feature Importance')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig('results/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save test results
    test_results = pd.DataFrame({
        'Actual_AQI': y_test.values,
        'Predicted_AQI': y_pred,
        'Residuals': residuals.values
    })
    
    test_results.to_csv('results/test_predictions.csv', index=False)
    
    # Save metrics
    metrics = {'R2_Score': r2, 'MAE': mae, 'RMSE': rmse, 'MSE': mse}
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('results/test_metrics.csv', index=False)
    
    print(f"\nResults saved to 'results/' directory")
    
    return test_results, metrics

if __name__ == "__main__":
    test_aqi_models()