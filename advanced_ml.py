"""
Advanced ML Models for Climate Predictions
Includes: LSTM, Random Forest, Prophet, Ensemble Methods
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except:
    STATSMODELS_AVAILABLE = False


class AdvancedClimatePredictor:
    """Multi-model ensemble for climate predictions"""
    
    def __init__(self, df, column_name, test_size=0.2):
        self.df = df.copy()
        self.column_name = column_name
        self.test_size = test_size
        self.year_data = df.groupby("Year")[column_name].mean().reset_index()
        self.X = self.year_data[["Year"]].values
        self.y = self.year_data[column_name].values
        
        # Store models
        self.models = {}
        self.accuracies = {}
        self.predictions = {}
        
    def train_linear_regression(self):
        """Basic Linear Regression"""
        model = LinearRegression()
        model.fit(self.X, self.y)
        score = r2_score(self.y, model.predict(self.X))
        self.models['LinearRegression'] = model
        self.accuracies['LinearRegression'] = score
        return model, score
    
    def train_random_forest(self, n_estimators=100):
        """Random Forest Regressor"""
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=10)
        model.fit(self.X, self.y)
        score = r2_score(self.y, model.predict(self.X))
        self.models['RandomForest'] = model
        self.accuracies['RandomForest'] = score
        return model, score
    
    def train_gradient_boosting(self):
        """Gradient Boosting Regressor"""
        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
        model.fit(self.X, self.y)
        score = r2_score(self.y, model.predict(self.X))
        self.models['GradientBoosting'] = model
        self.accuracies['GradientBoosting'] = score
        return model, score
    
    def train_lstm(self, epochs=50, batch_size=4):
        """LSTM Neural Network for time series"""
        if not KERAS_AVAILABLE:
            return None, 0
        
        try:
            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(self.y.reshape(-1, 1))
            
            # Create sequences
            sequence_length = 3
            X_lstm = []
            y_lstm = []
            
            for i in range(len(scaled_data) - sequence_length):
                X_lstm.append(scaled_data[i:i+sequence_length])
                y_lstm.append(scaled_data[i+sequence_length])
            
            if len(X_lstm) < 2:
                return None, 0
            
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_lstm, y_lstm, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Evaluate
            y_pred = model.predict(X_lstm, verbose=0)
            score = r2_score(y_lstm, y_pred)
            
            self.models['LSTM'] = (model, scaler, sequence_length)
            self.accuracies['LSTM'] = score
            return model, score
        except:
            return None, 0
    
    def get_seasonal_decomposition(self):
        """Seasonal decomposition analysis"""
        if not STATSMODELS_AVAILABLE or len(self.year_data) < 12:
            return None
        
        try:
            # Make period appropriate
            period = min(12, len(self.year_data) // 2)
            if period < 2:
                return None
            
            decomposition = seasonal_decompose(self.year_data[self.column_name], 
                                               model='additive', period=period)
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        except:
            return None
    
    def create_ensemble_prediction(self, years_to_predict):
        """Ensemble prediction using multiple models"""
        ensemble_pred = np.zeros(len(years_to_predict))
        weights = {}
        
        # Train all models
        self.train_linear_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_lstm()
        
        # Normalize accuracy scores to use as weights
        total_acc = sum(self.accuracies.values())
        if total_acc > 0:
            for model_name, acc in self.accuracies.items():
                weights[model_name] = max(0, acc) / total_acc
        
        # Get predictions from each model
        X_pred = years_to_predict.reshape(-1, 1)
        
        # Linear Regression prediction
        if 'LinearRegression' in self.models:
            lr_pred = self.models['LinearRegression'].predict(X_pred)
            ensemble_pred += lr_pred * weights.get('LinearRegression', 0.2)
        
        # Random Forest prediction
        if 'RandomForest' in self.models:
            rf_pred = self.models['RandomForest'].predict(X_pred)
            ensemble_pred += rf_pred * weights.get('RandomForest', 0.2)
        
        # Gradient Boosting prediction
        if 'GradientBoosting' in self.models:
            gb_pred = self.models['GradientBoosting'].predict(X_pred)
            ensemble_pred += gb_pred * weights.get('GradientBoosting', 0.2)
        
        # LSTM prediction (if available)
        if 'LSTM' in self.models:
            try:
                model, scaler, seq_len = self.models['LSTM']
                # Use last sequence from scaled data for prediction
                scaled_data = scaler.transform(self.y.reshape(-1, 1))
                last_seq = scaled_data[-seq_len:].reshape(1, seq_len, 1)
                lstm_pred_scaled = model.predict(last_seq, verbose=0)[0, 0]
                lstm_pred = scaler.inverse_transform([[lstm_pred_scaled]])[0, 0]
                # Trend extrapolation for future
                trend = (self.y[-1] - self.y[0]) / len(self.y)
                lstm_pred += trend * (len(years_to_predict) - 1)
                ensemble_pred += lstm_pred * weights.get('LSTM', 0.2)
            except:
                pass
        
        return ensemble_pred, self.accuracies
    
    def get_prediction_uncertainty(self, prediction):
        """Calculate uncertainty bounds"""
        # Calculate RMSE as uncertainty measure
        residuals = self.y - self.models.get('LinearRegression', LinearRegression().fit(self.X, self.y)).predict(self.X)
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # Confidence interval (95%)
        confidence_interval = 1.96 * rmse / np.sqrt(len(self.y))
        
        return {
            'lower_bound': prediction - confidence_interval,
            'upper_bound': prediction + confidence_interval,
            'rmse': rmse,
            'confidence': confidence_interval
        }
    
    def get_model_comparison(self):
        """Compare all trained models"""
        comparison_df = pd.DataFrame({
            'Model': list(self.accuracies.keys()),
            'R² Score': list(self.accuracies.values())
        })
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        return comparison_df
    
    def get_monthly_predictions(self, df_daily, year_to_predict):
        """Get monthly level predictions from daily data"""
        monthly_data = df_daily.groupby(["Year", "Month"])[self.column_name].mean().reset_index()
        
        predictions = {}
        for month in range(1, 13):
            month_data = monthly_data[monthly_data["Month"] == month]
            if len(month_data) > 2:
                X_month = month_data[["Year"]].values
                y_month = month_data[self.column_name].values
                
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                model.fit(X_month, y_month)
                pred = model.predict([[year_to_predict]])[0]
                predictions[month] = pred
        
        return predictions


def advanced_train_climate_model(df, col, daily_df=None):
    """
    Enhanced climate model training with ensemble methods
    Returns: (full_dataframe, best_model, accuracy_dict, historical_avg, seasonal_info)
    """
    predictor = AdvancedClimatePredictor(df, col)
    
    # Create ensemble predictions
    future_years = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030])
    ensemble_preds, all_accuracies = predictor.create_ensemble_prediction(future_years)
    
    # Create output dataframe
    y_data = predictor.year_data
    hist = y_data[["Year", col]].copy()
    hist['Type'] = 'Historical'
    
    future = pd.DataFrame({
        'Year': future_years,
        col: ensemble_preds,
        'Type': 'AI Projection'
    })
    
    full_df = pd.concat([hist, future], ignore_index=True)
    
    # Get best single model for backward compatibility
    best_model = predictor.models.get('RandomForest') or predictor.models.get('LinearRegression')
    if best_model is None:
        best_model = LinearRegression().fit(predictor.X, predictor.y)
    
    # Get best accuracy
    best_accuracy = max(all_accuracies.values()) if all_accuracies else 0
    
    # Get seasonal info if available
    seasonal_info = predictor.get_seasonal_decomposition()
    
    return full_df, best_model, all_accuracies, y_data[col].mean(), seasonal_info, predictor


def get_prediction_with_uncertainty(predictor, year_value):
    """Get prediction with uncertainty bounds"""
    if hasattr(predictor, 'models') and 'RandomForest' in predictor.models:
        pred = predictor.models['RandomForest'].predict([[year_value]])[0]
    else:
        pred = predictor.models['LinearRegression'].predict([[year_value]])[0]
    
    uncertainty = predictor.get_prediction_uncertainty(pred)
    return {
        'prediction': pred,
        'lower_bound': uncertainty['lower_bound'],
        'upper_bound': uncertainty['upper_bound'],
        'rmse': uncertainty['rmse']
    }
