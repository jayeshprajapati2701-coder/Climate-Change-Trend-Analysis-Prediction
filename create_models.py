"""
Create trained ML models for the Climate Dashboard
Run this script once to generate all required model files
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def run_model_creation():
    # Create models directory using Pathlib
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    print("Creating ML models for Climate Dashboard...")
    print("-" * 50)

    # ============================================================
    # 1. Random Forest Model for Air Quality (AQI Prediction)
    # ============================================================
    print("1. Training Random Forest model for Air Quality...")

    from sklearn.ensemble import RandomForestRegressor

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 500

    # Features: SO2, NO2, RSPM, SPM
    X_aqi = np.random.uniform(
        low=[0, 10, 50, 100],
        high=[50, 100, 200, 300],
        size=(n_samples, 4)
    )

    # Target: PM2.5 (based on features with some noise)
    y_aqi = (0.3 * X_aqi[:, 0] + 0.4 * X_aqi[:, 1] + 
             0.2 * X_aqi[:, 2] + 0.1 * X_aqi[:, 3] + 
             np.random.normal(0, 10, n_samples))

    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    rf_model.fit(X_aqi, y_aqi)

    # Save model
    joblib.dump(rf_model, models_dir / 'rf_aqi_model.pkl')
    print("✅ Random Forest model saved to models/rf_aqi_model.pkl")
    print(f"   Model score: {rf_model.score(X_aqi, y_aqi):.4f}")

    # ============================================================
    # 2. Linear Regression Model for Temperature
    # ============================================================
    print("\n2. Training Linear Regression model for Temperature...")

    from sklearn.linear_model import LinearRegression

    # Create synthetic historical temperature data (1980-2024)
    years = np.arange(1980, 2025).reshape(-1, 1)

    # Temperature trend with some noise
    base_temp = 24.0
    temp_trend = 0.03  # 0.03 degree rise per year
    y_temp = base_temp + (years - 1980) * temp_trend + np.random.normal(0, 0.5, len(years))

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(years, y_temp)

    # Save model
    joblib.dump(lr_model, models_dir / 'linear_reg_temp.pkl')
    print("✅ Linear Regression model saved to models/linear_reg_temp.pkl")
    print(f"   Temperature slope: {str(lr_model.coef_[0])[:6]} °C/year")
    print(f"   Intercept: {str(lr_model.intercept_)[:6]} °C")

    # ============================================================
    # 3. LSTM Model for Rainfall Prediction
    # ============================================================
    print("\n3. Training LSTM model for Rainfall Prediction...")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.optimizers import Adam
        
        # Create synthetic rainfall data (humidity -> rainfall probability)
        np.random.seed(42)
        n_sequences = 1000
        
        # Generate sequences of 3 days of humidity data
        X_lstm = np.random.uniform(20, 100, size=(n_sequences, 3, 1))  # (samples, timesteps, features)
        
        # Target: rainfall probability (higher humidity → higher rain probability)
        y_lstm = np.mean(X_lstm, axis=(1, 2)) / 100.0  # Normalize to 0-1
        y_lstm = y_lstm + np.random.normal(0, 0.1, n_sequences)
        y_lstm = np.clip(y_lstm, 0, 1)  # Clip to 0-1 range
        
        # Build LSTM model
        lstm_model = Sequential([
            LSTM(32, input_shape=(3, 1), return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Output probability 0-1
        ])
        
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train with minimal epochs for quick setup
        lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32, verbose=0)
        
        # Save model using SavedModel format (compatible with TensorFlow 2.21+)
        lstm_model.save(models_dir / 'lstm_rainfall')  # This creates a directory with SavedModel format
        print("✅ LSTM model saved to models/lstm_rainfall (SavedModel format)")
        
        # Test prediction
        test_input = np.array([[[60], [65], [70]]])
        test_pred = lstm_model.predict(test_input, verbose=0)[0][0]
        print(f"   Test prediction (humidity 60-70%): {test_pred*100:.1f}% rain probability")
        
    except Exception as e:
        print(f"⚠️  TensorFlow/LSTM not available: {str(e)}")
        print("   LSTM model will not be created.")
        print("   Install TensorFlow with: pip install tensorflow")

    print("\n" + "=" * 50)
    print("✅ Model creation completed!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  • models/rf_aqi_model.pkl       (Random Forest for Air Quality)")
    print("  • models/linear_reg_temp.pkl    (Linear Regression for Temperature)")
    print("  • models/lstm_rainfall/         (LSTM for Rainfall - SavedModel format)")

if __name__ == "__main__":
    run_model_creation()
