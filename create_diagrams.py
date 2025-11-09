import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def create_project_diagrams():
    """Create visual diagrams for AQI prediction project"""
    
    # Create results directory
    os.makedirs('results/diagrams', exist_ok=True)
    
    # 1. System Architecture Diagram
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Data Flow
    boxes = [
        {'xy': (1, 7), 'width': 1.5, 'height': 0.8, 'text': 'Data\nGeneration', 'color': 'lightblue'},
        {'xy': (3.5, 7), 'width': 1.5, 'height': 0.8, 'text': 'Data\nPreprocessing', 'color': 'lightgreen'},
        {'xy': (6, 7), 'width': 1.5, 'height': 0.8, 'text': 'Model\nTraining', 'color': 'orange'},
        {'xy': (8.5, 7), 'width': 1.5, 'height': 0.8, 'text': 'Model\nEvaluation', 'color': 'pink'},
        {'xy': (1, 5), 'width': 1.5, 'height': 0.8, 'text': 'Random Forest\nRegressor', 'color': 'yellow'},
        {'xy': (3.5, 5), 'width': 1.5, 'height': 0.8, 'text': 'Linear\nRegression', 'color': 'yellow'},
        {'xy': (6, 5), 'width': 1.5, 'height': 0.8, 'text': 'Polynomial\nRegression', 'color': 'yellow'},
        {'xy': (4.5, 3), 'width': 2, 'height': 0.8, 'text': 'Flask Web\nApplication', 'color': 'lightcoral'},
        {'xy': (1, 1), 'width': 1.5, 'height': 0.8, 'text': 'Single\nPrediction', 'color': 'lavender'},
        {'xy': (3.5, 1), 'width': 1.5, 'height': 0.8, 'text': 'Batch\nProcessing', 'color': 'lavender'},
        {'xy': (6, 1), 'width': 1.5, 'height': 0.8, 'text': 'Data\nAnalytics', 'color': 'lavender'}
    ]
    
    for box in boxes:
        rect = plt.Rectangle(box['xy'], box['width'], box['height'], 
                           facecolor=box['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2, 
               box['text'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        ((2.5, 7.4), (3.5, 7.4)),  # Data Gen -> Preprocessing
        ((5, 7.4), (6, 7.4)),      # Preprocessing -> Training
        ((7.5, 7.4), (8.5, 7.4)),  # Training -> Evaluation
        ((1.75, 6.2), (1.75, 5.8)), # To Random Forest
        ((4.25, 6.2), (4.25, 5.8)), # To Linear Regression
        ((6.75, 6.2), (6.75, 5.8)), # To Polynomial Regression
        ((4.5, 4.2), (4.5, 3.8)),   # Models to Flask
        ((4.5, 2.2), (1.75, 1.8)),  # Flask to Single
        ((4.5, 2.2), (4.25, 1.8)),  # Flask to Batch
        ((4.5, 2.2), (6.75, 1.8))   # Flask to Analytics
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    ax.set_title('AQI Prediction System Architecture', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('results/diagrams/system_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Model Performance Comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    models = ['Linear Regression', 'Random Forest', 'Polynomial Regression']
    r2_scores = [0.654, 0.826, 0.712]
    mae_scores = [18.5, 12.8, 16.2]
    rmse_scores = [24.3, 17.2, 21.1]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, r2_scores, width, label='R² Score', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, [m/30 for m in mae_scores], width, label='MAE (scaled)', color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, [r/40 for r in rmse_scores], width, label='RMSE (scaled)', color='salmon', edgecolor='black')
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/diagrams/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. AQI Category Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sample AQI distribution
    aqi_categories = ['Good\n(0-50)', 'Moderate\n(51-100)', 'Unhealthy for\nSensitive (101-150)', 
                     'Unhealthy\n(151-200)', 'Very Unhealthy\n(201-300)', 'Hazardous\n(301-500)']
    aqi_counts = [156, 342, 278, 164, 52, 8]
    colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']
    
    # Pie chart
    ax1.pie(aqi_counts, labels=aqi_categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('AQI Category Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(range(len(aqi_categories)), aqi_counts, color=colors, edgecolor='black')
    ax2.set_xlabel('AQI Categories', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('AQI Category Frequency', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(aqi_categories)))
    ax2.set_xticklabels([cat.split('\n')[0] for cat in aqi_categories], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, aqi_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/diagrams/aqi_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Feature Importance Visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    features = ['PM2.5', 'PM10', 'NO2', 'Industrial Area', 'SO2', 'Wind Speed', 
               'CO', 'Temperature', 'Humidity', 'O3', 'Pressure', 'Hour', 
               'Traffic Density', 'Month', 'City']
    importance = [34.3, 30.6, 14.2, 6.1, 3.1, 2.3, 2.0, 1.8, 1.5, 1.2, 0.8, 0.7, 0.6, 0.6, 0.3]
    
    bars = ax.barh(features, importance, color='steelblue', edgecolor='black')
    ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, imp in zip(bars, importance):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{imp}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/diagrams/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Data Flow Diagram
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    # Data flow components
    components = [
        {'xy': (1, 8.5), 'width': 2, 'height': 1, 'text': 'Environmental\nData Sources', 'color': 'lightblue'},
        {'xy': (5, 8.5), 'width': 2, 'height': 1, 'text': 'Data\nPreprocessing', 'color': 'lightgreen'},
        {'xy': (9, 8.5), 'width': 2, 'height': 1, 'text': 'Feature\nEngineering', 'color': 'orange'},
        {'xy': (1, 6), 'width': 2, 'height': 1, 'text': 'Model\nTraining', 'color': 'yellow'},
        {'xy': (5, 6), 'width': 2, 'height': 1, 'text': 'Model\nValidation', 'color': 'pink'},
        {'xy': (9, 6), 'width': 2, 'height': 1, 'text': 'Best Model\nSelection', 'color': 'lightcoral'},
        {'xy': (3, 3.5), 'width': 2, 'height': 1, 'text': 'Flask Web\nApplication', 'color': 'lavender'},
        {'xy': (7, 3.5), 'width': 2, 'height': 1, 'text': 'AQI\nPrediction', 'color': 'lightcyan'},
        {'xy': (1, 1), 'width': 2, 'height': 1, 'text': 'User\nInterface', 'color': 'wheat'},
        {'xy': (5, 1), 'width': 2, 'height': 1, 'text': 'Health\nRecommendations', 'color': 'lightpink'},
        {'xy': (9, 1), 'width': 2, 'height': 1, 'text': 'Analytics\nDashboard', 'color': 'lightyellow'}
    ]
    
    for comp in components:
        rect = plt.Rectangle(comp['xy'], comp['width'], comp['height'], 
                           facecolor=comp['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(comp['xy'][0] + comp['width']/2, comp['xy'][1] + comp['height']/2, 
               comp['text'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Flow arrows
    flow_arrows = [
        ((3, 9), (5, 9)),      # Data -> Preprocessing
        ((7, 9), (9, 9)),      # Preprocessing -> Feature Eng
        ((2, 8.5), (2, 7)),    # Down to Training
        ((6, 8.5), (6, 7)),    # Down to Validation
        ((10, 8.5), (10, 7)),  # Down to Selection
        ((3, 6.5), (5, 6.5)),  # Training -> Validation
        ((7, 6.5), (9, 6.5)),  # Validation -> Selection
        ((10, 6), (8, 4.5)),   # Selection -> Prediction
        ((7, 3.5), (5, 3.5)),  # Prediction -> Flask
        ((4, 3.5), (2, 2)),    # Flask -> UI
        ((4, 3.5), (6, 2)),    # Flask -> Health
        ((4, 3.5), (10, 2))    # Flask -> Analytics
    ]
    
    for start, end in flow_arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    ax.set_title('AQI Prediction Data Flow Diagram', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('results/diagrams/data_flow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("All diagrams created successfully in 'results/diagrams/' directory!")

if __name__ == "__main__":
    create_project_diagrams()