from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
import json

app = Flask(__name__)

# Load models at startup
model = None
scaler = None

def load_models():
    global model, scaler
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return True
    except:
        return False

def predict_aqi(features):
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return max(0, min(500, int(prediction)))
    except:
        return None

def get_aqi_info(aqi):
    if aqi <= 50:
        return "Good", "#00E400", "Air quality is good. Enjoy outdoor activities!"
    elif aqi <= 100:
        return "Moderate", "#FFFF00", "Air quality is moderate. Sensitive individuals should limit outdoor activities."
    elif aqi <= 150:
        return "Unhealthy for Sensitive", "#FF7E00", "Unhealthy for sensitive groups. Reduce outdoor activities if you're sensitive."
    elif aqi <= 200:
        return "Unhealthy", "#FF0000", "Unhealthy air quality. Everyone should limit outdoor activities."
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97", "Very unhealthy air quality. Avoid outdoor activities."
    else:
        return "Hazardous", "#7E0023", "Hazardous air quality. Stay indoors!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features in the exact order the model expects
        features = np.array([[
            float(data['pm25']),      # PM2_5
            float(data['pm10']),      # PM10
            float(data['no2']),       # NO2
            float(data['so2']),       # SO2
            float(data['co']),        # CO
            float(data['o3']),        # O3
            float(data['temp']),      # Temperature
            float(data['humidity']),  # Humidity
            float(data['wind_speed']), # Wind_Speed
            float(data['pressure']),  # Pressure
            int(data['month']),       # Month
            int(data['hour']),        # Hour
            int(data['city']),        # City
            int(data['industrial']),  # Industrial_Area
            int(data['traffic'])      # Traffic_Density
        ]])
        
        aqi = predict_aqi(features)
        
        if aqi is not None:
            category, color, message = get_aqi_info(aqi)
            return jsonify({
                'success': True,
                'aqi': aqi,
                'category': category,
                'color': color,
                'message': message
            })
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Select only the features the model was trained on
        feature_cols = ['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 
                       'Humidity', 'Wind_Speed', 'Pressure', 'Month', 'Hour', 
                       'City', 'Industrial_Area', 'Traffic_Density']
        
        features_scaled = scaler.transform(df[feature_cols])
        predictions = model.predict(features_scaled)
        
        df['Predicted_AQI'] = [max(0, min(500, int(p))) for p in predictions]
        df['AQI_Category'] = [get_aqi_info(aqi)[0] for aqi in df['Predicted_AQI']]
        
        result = df.to_dict('records')
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(df)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analysis')
def analysis():
    try:
        if os.path.exists('data/processed/clean_data.csv'):
            df = pd.read_csv('data/processed/clean_data.csv')
            
            stats = {
                'total_samples': len(df),
                'features': len(df.columns) - 1,
                'avg_aqi': round(df['AQI'].mean(), 1),
                'max_aqi': int(df['AQI'].max())
            }
            
            aqi_dist = df['AQI'].value_counts().sort_index().to_dict()
            
            feature_importance = []
            if os.path.exists('models/feature_importance.csv'):
                importance_df = pd.read_csv('models/feature_importance.csv')
                feature_importance = importance_df.head(10).to_dict('records')
            
            return jsonify({
                'success': True,
                'stats': stats,
                'aqi_distribution': aqi_dist,
                'feature_importance': feature_importance
            })
        else:
            return jsonify({'success': False, 'error': 'No data found'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    if load_models():
        print("Models loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Error: Models not found. Please run training.py first.")