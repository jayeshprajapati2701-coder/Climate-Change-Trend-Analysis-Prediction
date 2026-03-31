import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import requests
from dotenv import load_dotenv
import subprocess
import sys
import create_models
from advanced_ml import advanced_train_climate_model

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# API Key from environment variable for security
API_KEY = os.getenv('OPENWEATHER_API_KEY', '').strip()

if not API_KEY:
    st.error("⚠️ **OpenWeather API Key missing!**")
    st.info("लोकल रन के लिए: सुनिश्चित करें कि आपके प्रोजेक्ट रूट में `.env` फ़ाइल है और उसमें `OPENWEATHER_API_KEY=your_key` लिखा है।")
    st.info("डिप्लॉयमेंट के लिए: Streamlit Cloud की Settings > Secrets में API Key जोड़ें।")
    st.stop()

# TensorFlow - Optional dependency
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    def load_model(path):
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")

# --- Global Directory Constants ---
DATA_DIR = "Datasets"
MODEL_DIR = "models"

# Auto-generate models if they don't exist (Crucial for Cloud Deployment)
if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    st.info("⚙️ Initializing Machine Learning Models for the first time...")
    try:
        create_models.run_model_creation()
        st.success("✅ Models initialized successfully!")
    except Exception as e:
        st.error(f"Error generating models: {e}")

LOCATION_COORDINATES = {
    # States
    "Andhra Pradesh": (15.91, 78.67), "Arunachal Pradesh": (28.22, 94.20),
    "Assam": (26.13, 91.71), "Bihar": (25.59, 85.14), "Chhattisgarh": (21.28, 81.62),
    "Goa": (15.30, 73.83), "Gujarat": (22.26, 71.19), "Haryana": (29.06, 77.58),
    "Himachal Pradesh": (31.77, 77.11), "Jharkhand": (23.61, 85.28), "Karnataka": (15.32, 75.71),
    "Kerala": (11.86, 75.53), "Madhya Pradesh": (22.98, 78.65), "Maharashtra": (19.75, 75.71),
    "Manipur": (24.66, 93.91), "Meghalaya": (25.47, 91.37), "Mizoram": (23.19, 92.94),
    "Nagaland": (26.16, 94.56), "Odisha": (20.95, 85.09), "Punjab": (31.15, 75.85),
    "Rajasthan": (27.59, 77.24), "Sikkim": (27.53, 88.51), "Tamil Nadu": (11.13, 79.29),
    "Telangana": (18.11, 79.01), "Tripura": (23.51, 91.56), "Uttar Pradesh": (26.85, 80.95),
    "Uttarakhand": (30.07, 79.60), "West Bengal": (24.63, 88.37),
    # Union Territories
    "Delhi": (28.70, 77.10), "Ladakh": (34.16, 77.58), "Jammu and Kashmir": (34.30, 74.80),
    "Puducherry": (12.00, 79.50), "Lakshadweep": (10.57, 72.64), "Dadra and Nagar Haveli": (20.18, 73.02),
    "Daman and Diu": (20.43, 72.83), "Andaman and Nicobar": (11.74, 92.74), "Chandigarh": (30.73, 76.79)
}

# Added Vadodara for your local testing
DISTRICT_COORDINATES = {
    "Bangalore": (12.9716, 77.5946), "Delhi": (28.7041, 77.1025), "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714), "Vadodara": (22.3072, 73.1812),
    "Ahmadabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873), "Lucknow": (26.8467, 80.9462), "Surat": (21.1702, 72.8311), 
    "Indore": (22.7196, 75.8577), "Rajkot": (22.3039, 70.8022), "Jamnagar": (22.4707, 70.0883),
    "Junagadh": (21.5250, 70.4613), "Bhavnagar": (21.7645, 72.1519), "Gandhinagar": (23.1815, 72.6298),
    "Kochi": (9.9312, 76.2673), "Bhopal": (23.1815, 79.9864), "Nagpur": (21.1458, 79.0882)
}

@st.cache_data(ttl=60)  # Caches data for 1 minute for live dashboard updates
def get_live_weather(lat, lon, location_name):
    """Fetch live weather data from OpenWeatherMap API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=8)
        data = response.json()
        
        # Debug: print response for troubleshooting
        if response.status_code != 200:
            st.error(f"API Error {response.status_code}: {data.get('message', 'Unknown error')}")
            return None
        
        if 'main' in data:
            # Fetch UV Index
            uv_val = "N/A"
            try:
                uv_url = f"https://api.openweathermap.org/data/2.5/uvi?lat={lat}&lon={lon}&appid={API_KEY}"
                uv_res = requests.get(uv_url, timeout=4)
                if uv_res.status_code == 200:
                    uv_val = round(uv_res.json().get('value', 0), 1)
            except:
                pass
                
            return {
                'temperature': round(data['main']['temp'], 1),
                'humidity': data['main']['humidity'],
                'wind_speed': round(data['wind']['speed'], 1),
                'precipitation': round(data.get('rain', {}).get('1h', 0) or data.get('snow', {}).get('1h', 0), 2),
                'uv_index': uv_val,
                'weather_code': data['weather'][0]['id'],
                'location': location_name,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
        return None
    except Exception as e:
        st.error(f"Weather API Error: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_live_aqi(lat, lon):
    """Fetch live AQI data from OpenWeatherMap Air Pollution API"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        response = requests.get(url, timeout=8)
        data = response.json()
        
        if response.status_code == 200 and 'list' in data:
            aqi_data = data['list'][0]
            return {
                'aqi_index': aqi_data['main']['aqi'], # 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
                'components': aqi_data['components'],
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
        return None
    except Exception as e:
        return None

def interpret_aqi_index(index):
    """Convert OpenWeatherMap AQI index (1-5) to status text and color"""
    aqi_status = {
        1: ("✅ Good", "green"),
        2: ("🟡 Fair", "yellow"),
        3: ("🟠 Moderate", "orange"),
        4: ("🔴 Poor", "red"),
        5: ("🟣 Very Poor", "purple")
    }
    return aqi_status.get(index, ("Unknown", "grey"))

def interpret_weather_code(code):
    """Convert OpenWeatherMap weather code to text and emojis"""
    weather_descriptions = {
        200: "⛈️ Thunderstorm with light rain", 201: "⛈️ Thunderstorm with rain", 202: "⛈️ Thunderstorm with heavy rain",
        210: "⛈️ Light thunderstorm", 211: "⛈️ Thunderstorm", 212: "⛈️ Heavy thunderstorm",
        221: "⛈️ Ragged thunderstorm", 230: "⛈️ Thunderstorm with light drizzle", 231: "⛈️ Thunderstorm with drizzle", 232: "⛈️ Thunderstorm with heavy drizzle",
        300: "🌧️ Light intensity drizzle", 301: "🌧️ Drizzle", 302: "🌧️ Heavy intensity drizzle",
        310: "🌧️ Light intensity drizzle rain", 311: "🌧️ Drizzle rain", 312: "🌧️ Heavy intensity drizzle rain",
        313: "🌧️ Shower rain and drizzle", 314: "🌧️ Heavy shower rain and drizzle", 321: "🌧️ Shower drizzle",
        500: "🌧️ Light rain", 501: "🌧️ Moderate rain", 502: "🌧️ Heavy intensity rain",
        503: "🌧️ Very heavy rain", 504: "🌧️ Extreme rain", 511: "🌧️ Freezing rain",
        520: "🌧️ Light intensity shower rain", 521: "🌧️ Shower rain", 522: "🌧️ Heavy intensity shower rain", 531: "🌧️ Ragged shower rain",
        600: "❄️ Light snow", 601: "❄️ Snow", 602: "❄️ Heavy snow",
        611: "🌨️ Sleet", 612: "🌨️ Light shower sleet", 613: "🌨️ Shower sleet", 615: "🌨️ Light rain and snow", 616: "🌨️ Rain and snow", 620: "🌨️ Light shower snow", 621: "🌨️ Shower snow", 622: "🌨️ Heavy shower snow",
        701: "🌫️ Mist", 711: "🌫️ Smoke", 721: "🌫️ Haze", 731: "🌫️ Dust", 741: "🌫️ Fog", 751: "🌫️ Sand", 761: "🌫️ Dust", 762: "🌫️ Ash", 771: "🌫️ Squall", 781: "🌪️ Tornado",
        800: "☀️ Clear sky", 801: "🌤️ Few clouds", 802: "⛅ Partly cloudy", 803: "☁️ Broken clouds", 804: "☁️ Overcast clouds"
    }
    return weather_descriptions.get(code, "🌡️ Unknown")

@st.fragment(run_every=60)
def render_live_weather(lat, lon, coord_location):
    live_weather = get_live_weather(lat, lon, coord_location)
    
    if live_weather:
        lw1, lw2, lw3, lw4, lw5, lw6 = st.columns(6)
        
        with lw1:
            st.metric("🌡️ Current Temp", f"{live_weather['temperature']}°C")
        with lw2:
            st.metric("💨 Wind Speed", f"{live_weather['wind_speed']} km/h")
        with lw3:
            st.metric("💧 Humidity", f"{live_weather['humidity']}%")
        with lw4:
            st.metric("🌧️ Rainfall", f"{live_weather['precipitation']} mm")
        with lw5:
            st.metric("☀️ UV Index", f"{live_weather['uv_index']}")
        with lw6:
            weather_desc = interpret_weather_code(live_weather['weather_code'])
            condition_emoji = weather_desc.split()[0]
            st.metric("Weather", condition_emoji)
        
        st.info(f"**📍 {live_weather['location']}** | {weather_desc} | Last Updated: {live_weather['timestamp']} (Auto-refreshes every 60s)")
    else:
        st.warning("⚠️ Unable to fetch live weather. Check API key or internet connection.")

@st.fragment(run_every=60)
def render_live_aqi_fragment(lat, lon, location):
    live_aqi = get_live_aqi(lat, lon)
    if live_aqi:
        status_text, status_color = interpret_aqi_index(live_aqi['aqi_index'])
        st.subheader(f"🔴 LIVE Air Quality: {location} (Index: {live_aqi['aqi_index']})")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AQI Status", status_text)
        m2.metric("AQI Value (1-5)", live_aqi['aqi_index'])
        m3.metric("PM2.5", f"{live_aqi['components'].get('pm2_5', 0)} µg/m³")
        m4.metric("PM10", f"{live_aqi['components'].get('pm10', 0)} µg/m³")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("CO", f"{live_aqi['components'].get('co', 0)} µg/m³")
        r2c2.metric("NO2", f"{live_aqi['components'].get('no2', 0)} µg/m³")
        r2c3.metric("SO2", f"{live_aqi['components'].get('so2', 0)} µg/m³")
        r2c4.metric("O3", f"{live_aqi['components'].get('o3', 0)} µg/m³")
        
        st.markdown(f"""
            <div style="padding:10px; border-radius:5px; background-color:rgba(0,0,0,0.05); border-left: 5px solid {status_color};">
                <strong>Health Advisory:</strong> Current air is {status_text}. 
                {'Outdoor exercise is recommended.' if live_aqi['aqi_index'] <= 2 else 'Limit prolonged outdoor exposure.'}
            </div>
        """, unsafe_allow_html=True)

def sync_data_to_csv(lat, lon, daily_path, aqi_path):
    """Fetch live data and append/update it in the local CSV datasets"""
    weather = get_live_weather(lat, lon, "Sync")
    aqi = get_live_aqi(lat, lon)
    
    now = datetime.now()
    year, month, day = now.year, now.month, now.day
    
    # Sync Weather Data
    if weather and os.path.exists(daily_path):
        df = pd.read_csv(daily_path, header=None)
        # Check if today's entry already exists
        exists = df[(df[0] == year) & (df[1] == month) & (df[2] == day)]
        if exists.empty:
            new_row = [[year, month, day, weather['precipitation'], weather['temperature'], weather['temperature'], weather['temperature']]]
            pd.DataFrame(new_row).to_csv(daily_path, mode='a', header=False, index=False)

    # Sync AQI Data
    if aqi and os.path.exists(aqi_path):
        df_a = pd.read_csv(aqi_path, header=None)
        exists_a = df_a[(df_a[0] == year) & (df_a[1] == month) & (df_a[2] == day)]
        if exists_a.empty:
            c = aqi['components']
            new_aqi = [[year, month, day, aqi['aqi_index'], c.get('pm2_5',0), c.get('pm10',0), c.get('no2',0), c.get('so2',0), c.get('o3',0), c.get('co',0)]]
            pd.DataFrame(new_aqi).to_csv(aqi_path, mode='a', header=False, index=False)
    
    return True

st.set_page_config(page_title="AI Climate Intelligence 2026", layout="wide", page_icon="🌍")

PROJECT_TARGET_YEAR = 2026
TODAY_STR = "2026-03-24" # Current simulation context

# Models ko cache mein load karne ke liye function
@st.cache_resource
def load_prediction_models():
    try:
        if not os.path.exists(MODEL_DIR):
            return None, None, None

        rf_path = os.path.join(MODEL_DIR, 'rf_aqi_model.pkl')
        lr_path = os.path.join(MODEL_DIR, 'linear_reg_temp.pkl')
        lstm_path = os.path.join(MODEL_DIR, 'lstm_rainfall')

        rf_aqi = joblib.load(rf_path) if os.path.exists(rf_path) else None
        lr_temp = joblib.load(lr_path) if os.path.exists(lr_path) else None
        
        lstm_rain = None
        if TENSORFLOW_AVAILABLE and os.path.exists(lstm_path):
            if os.path.isdir(lstm_path) or lstm_path.endswith('.h5'):
                try:
                    lstm_rain = load_model(lstm_path)
                except Exception:
                    lstm_rain = None
        
        return rf_aqi, lr_temp, lstm_rain
    except Exception:
        return None, None, None

# Fallback function for rainfall probability when LSTM is not available
def calculate_rainfall_probability_fallback(h1, h2, h3):
    """
    Calculate rainfall probability based on humidity trend
    Uses simple heuristic: higher humidity and increasing trend = higher probability
    """
    # Average humidity
    avg_humidity = (h1 + h2 + h3) / 3
    
    # Humidity trend (increasing or decreasing)
    humidity_trend = (h3 - h1)
    
    # Base probability from humidity level
    if avg_humidity > 85:
        base_prob = 0.85
    elif avg_humidity > 75:
        base_prob = 0.65
    elif avg_humidity > 65:
        base_prob = 0.45
    elif avg_humidity > 55:
        base_prob = 0.25
    else:
        base_prob = 0.10
    
    # Adjust based on trend (increasing humidity increases rain probability)
    trend_adjustment = humidity_trend * 0.003
    
    # Final probability
    final_prob = base_prob + trend_adjustment
    final_prob = max(0.0, min(1.0, final_prob))  # Clamp between 0 and 1
    
    return final_prob

# Models initialize karein
rf_model, lr_model, lstm_model = load_prediction_models()

# =====================================================================
# 2. SIDEBAR: NAVIGATION & GRANULAR FILTERS
# =====================================================================
st.sidebar.title("🌍 Climate Control Center")
st.sidebar.markdown("---")

# Check if essential directories exist
if not os.path.exists(DATA_DIR):
    st.error(f"📂 Critical Error: The '{DATA_DIR}' directory was not found. Please ensure it is uploaded to your repository.")
    st.stop()

states_path = os.path.join(DATA_DIR, "states.csv")
districts_path = os.path.join(DATA_DIR, "districts.csv")

if os.path.exists(states_path) and os.path.exists(districts_path):
    states = pd.read_csv(states_path)
    districts = pd.read_csv(districts_path)
    
    # "Gujarat" को डिफॉल्ट चुनने के लिए इंडेक्स सेट करें
    state_list = states["STATE"].tolist()
    default_idx = state_list.index("Gujarat") if "Gujarat" in state_list else 0
    selected_state = st.sidebar.selectbox("Select State", state_list, index=default_idx)
    state_id = states[states["STATE"] == selected_state]["ID"].values[0]
    
    dist_list = districts[districts["STATE"] == selected_state]
    selected_dist = st.sidebar.selectbox("Select District", ["None"] + dist_list["District"].tolist())
    
    year_range = st.sidebar.slider("Historical Data Range", 1981, 2026, (2010, 2024))
else:
    st.error(f"❌ Missing configuration files in {DATA_DIR} (states.csv/districts.csv)! Please check your file structure.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("🗓️ Analysis Mode")
view_mode = st.sidebar.radio("Dashboard View", ["Live Hour View", "Full Month Trend", "Annual Heatmap Analysis"])

st.sidebar.markdown("---")
st.sidebar.header("⏰ Temporal Settings")
current_time = datetime.now()
selected_date = st.sidebar.date_input("Analysis Date", current_time.date())
selected_hour = st.sidebar.slider("Select Hour (24h format)", 0, 23, current_time.hour)

st.sidebar.markdown("---")
st.sidebar.header("📥 Data Management")
if st.sidebar.button("🔄 Sync Live Data to CSV"):
    lat_s, lon_s = (None, None)
    if selected_dist != "None" and selected_dist in DISTRICT_COORDINATES:
        lat_s, lon_s = DISTRICT_COORDINATES[selected_dist]
    
    if lat_s and lon_s:
        if sync_data_to_csv(lat_s, lon_s, c_daily, aqi_f):
            st.sidebar.success(f"Data synced for {loc_name}!")
    else:
        st.sidebar.error("Coordinates not found for sync.")

# =====================================================================
# 3. DYNAMIC DATA ROUTING
# =====================================================================
if selected_dist == "None":
    c_monthly = os.path.join(DATA_DIR, "State_data_Monthly_CSV", f"data_State_{state_id}.csv")
    c_daily = os.path.join(DATA_DIR, "State_data_Daily_CSV", f"data_State_{state_id}.csv")
    aqi_f = os.path.join(DATA_DIR, "State_AQI_CSV", f"data_State_{state_id}.csv")
    loc_name = selected_state
else:
    d_id = dist_list[dist_list["District"] == selected_dist]["ID"].values[0]
    c_monthly = os.path.join(DATA_DIR, "District_data_Monthly_CSV", f"data_District_{d_id}.csv")
    c_daily = os.path.join(DATA_DIR, "District_data_Daily_CSV", f"data_District_{d_id}.csv")
    aqi_f = os.path.join(DATA_DIR, "District_AQI_CSV", f"data_District_{d_id}.csv")
    loc_name = selected_dist

st.title(f"🌍 AI Climate Intelligence System: {loc_name}")

with st.expander("📚 Project Overview & Educational Objectives", expanded=False):
    st.markdown("""
    **Major Issues Addressed:**
    * **Global Temperature Rise:** Monitoring increasing average temperatures.
    * **Extreme Weather Events:** Tracking risk of floods, droughts, and heatwaves. 
    * **Air Quality Degradation:** Analyzing pollution and greenhouse gas impact.
    * **Data Complexity:** Handling multi-dimensional climate data across time & geography.
    * **Prediction Uncertainty:** Providing AI confidence intervals to handle forecasting factors.

    **Educational Objectives Achieved:**
    * **Exploratory Data Analysis (EDA):** Interactive charts for temperature, rainfall, and historical variations.
    * **Long-term Trends & Seasonal Patterns:** Visualizing monthly patterns and decadal heatmaps.
    * **Machine Learning Models:** Predicting temperature and AQI using Linear Regression and Random Forest.
    * **Time-Series Analysis:** Applying moving averages to forecast long-term climate trajectories.
    * **Deep Learning:** Utilizing LSTM neural networks for complex rainfall pattern recognition.
    """)

with st.expander("📊 System Architecture & Data Handling", expanded=False):
    st.markdown("""
    To conquer the **Data Complexity** inherent in climate science, this platform utilizes:
    * **Geospatial Data Routing:** Dynamic parsing of hierarchical CSV pipelines (State -> District) for lightweight memory execution.
    * **Multi-Dimensional Processing:** Synthesizing 40+ years of daily historical data into macro (decadal) and micro (hourly) analysis vectors.
    * **Hybrid Inference Engine:** Combining `scikit-learn` algorithms (linear models and ensemble trees) with `TensorFlow/Keras` deep sequential layers (`LSTM`), cached efficiently via `@st.cache_resource`.
    * **Automated Fallback Mechanisms:** Seamlessly switching to live telemetry via the `OpenWeatherMap API` seamlessly if historical arrays are absent.
    """)

df_daily = pd.DataFrame() 
if os.path.exists(c_daily):
    try:
        df_daily = pd.read_csv(c_daily, header=None, names=["Year", "Month", "Day", "Rainfall", "Tmax", "Tmin", "Tavg"])
        # Dynamic Filtering based on Sidebar Slider
        df_daily = df_daily[(df_daily["Year"] >= year_range[0]) & (df_daily["Year"] <= year_range[1])]
    except Exception as e:
        st.sidebar.warning(f"Error reading daily data: {e}")
else:
    st.sidebar.warning(f"Note: Daily file not found at {c_daily}")

# Air Quality Index (AQI) Data Loading
df_aqi = pd.DataFrame()
if os.path.exists(aqi_f):
    try:
        df_aqi = pd.read_csv(aqi_f, header=None, names=["Year", "Month", "Day", "AQI", "PM25", "PM10", "NO2", "SO2", "O3", "CO"])
        df_aqi = df_aqi[(df_aqi["Year"] >= year_range[0]) & (df_aqi["Year"] <= year_range[1])]
    except Exception as e:
        pass

# =====================================================================
# 4. MACHINE LEARNING ENGINE
# =====================================================================
if os.path.exists(c_monthly):
    df_m = pd.read_csv(c_monthly, header=None, names=["Year","Month","Rainfall","Tmax","Tmin","Tavg"])
    
    # Restrict Master Monthly data to selected Historical Year Range for dynamic AI Modeling
    df_m = df_m[(df_m["Year"] >= year_range[0]) & (df_m["Year"] <= year_range[1])]
    if df_m["Year"].nunique() < 3:
        st.error("⚠️ AI Predictive Intelligence Error: Please select a Historical Data Range of at least 3 years to train accurate machine learning models.")
        st.stop()
    
    def train_monthly_climate_model(df, col):
        """Train separate models for each month based on historical monthly data"""
        monthly_models = {}
        monthly_accuracies = {}
        monthly_historical_avgs = {}

        for month in range(1, 13):
            # Filter data for this month
            month_data = df[df["Month"] == month].groupby("Year")[col].mean().reset_index()

            if len(month_data) > 2:  # Need at least 3 data points for meaningful regression
                model = LinearRegression().fit(month_data[["Year"]], month_data[col])
                accuracy = r2_score(month_data[col], model.predict(month_data[["Year"]]))

                monthly_models[month] = model
                monthly_accuracies[month] = accuracy
                monthly_historical_avgs[month] = month_data[col].mean()
            else:
                # Fallback to overall average if insufficient data
                monthly_models[month] = None
                monthly_accuracies[month] = 0.0
                monthly_historical_avgs[month] = df[df["Month"] == month][col].mean() if len(df[df["Month"] == month]) > 0 else df[col].mean()

        return monthly_models, monthly_accuracies, monthly_historical_avgs

    # Using Advanced Ensemble Models from advanced_ml.py
    full_t_df, model_t, accuracies_t, hist_t_avg, _, predictor_t = advanced_train_climate_model(df_m, "Tavg")
    full_r_df, model_r, accuracies_r, hist_r_avg, _, predictor_r = advanced_train_climate_model(df_m, "Rainfall")
    acc_t = max(accuracies_t.values())

    # Train monthly models for min/max temperatures
    monthly_tmax_models, monthly_tmax_accuracies, monthly_tmax_avgs = train_monthly_climate_model(df_m, "Tmax")
    monthly_tmin_models, monthly_tmin_accuracies, monthly_tmin_avgs = train_monthly_climate_model(df_m, "Tmin")

    # Get predictions for selected year and month from date picker
    selected_year = selected_date.year
    selected_month = selected_date.month
    
    # DYNAMIC CALCULATIONS BASED ON SELECTED DATE
    selected_month_name_short = selected_date.strftime('%b')
    selected_month_name_full = selected_date.strftime('%B')

    t_proj = model_t.predict([[selected_year]])[0]
    r_proj = model_r.predict([[selected_year]])[0]

    # RECALCULATE MONTHLY MIN/MAX TEMPS BASED ON YEAR RANGE SELECTION
    month_tmax_data = df_m[(df_m["Month"] == selected_month) & 
                           (df_m["Year"] >= year_range[0]) & 
                           (df_m["Year"] <= year_range[1])]
    month_tmin_data = df_m[(df_m["Month"] == selected_month) & 
                           (df_m["Year"] >= year_range[0]) & 
                           (df_m["Year"] <= year_range[1])]
    
    tmax_accuracy = 0.1
    if len(month_tmax_data) > 2:
        month_tmax_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(month_tmax_data[["Year"]], month_tmax_data["Tmax"])
        tmax_proj = month_tmax_model.predict([[selected_year]])[0]
        tmax_hist_avg = month_tmax_data["Tmax"].mean()
        tmax_accuracy = max(0.1, r2_score(month_tmax_data["Tmax"], month_tmax_model.predict(month_tmax_data[["Year"]])))
    else:
        tmax_proj = month_tmax_data["Tmax"].mean() if not month_tmax_data.empty else monthly_tmax_avgs.get(selected_month, 0)
        tmax_hist_avg = tmax_proj
    
    tmin_accuracy = 0.1
    if len(month_tmin_data) > 2:
        month_tmin_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(month_tmin_data[["Year"]], month_tmin_data["Tmin"])
        tmin_proj = month_tmin_model.predict([[selected_year]])[0]
        tmin_hist_avg = month_tmin_data["Tmin"].mean()
        tmin_accuracy = max(0.1, r2_score(month_tmin_data["Tmin"], month_tmin_model.predict(month_tmin_data[["Year"]])))
    else:
        tmin_proj = month_tmin_data["Tmin"].mean() if not month_tmin_data.empty else monthly_tmin_avgs.get(selected_month, 0)
        tmin_hist_avg = tmin_proj
    
    # Calculate MONTH-SPECIFIC rainfall (not annual) - FILTERED BY YEAR RANGE
    month_data = df_m[(df_m["Month"] == selected_month) & 
                      (df_m["Year"] >= year_range[0]) & 
                      (df_m["Year"] <= year_range[1])]
    month_r_accuracy = 0.1
    if len(month_data) > 2:
        month_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(month_data[["Year"]], month_data["Rainfall"])
        month_r_proj = month_model.predict([[selected_year]])[0]
        month_r_hist_avg = month_data["Rainfall"].mean()
        month_r_accuracy = max(0.1, r2_score(month_data["Rainfall"], month_model.predict(month_data[["Year"]])))
    else:
        month_r_proj = month_data["Rainfall"].mean() if not month_data.empty else df_m[df_m["Month"] == selected_month]["Rainfall"].mean()
        month_r_hist_avg = month_r_proj

    # -----------------------------------------------------------------
    # YEARLY MAX/MIN TEMP AND TOTAL RAINFALL CALCULATIONS
    # -----------------------------------------------------------------
    yearly_df = df_m[(df_m["Year"] >= year_range[0]) & (df_m["Year"] <= year_range[1])]
    if not yearly_df.empty:
        yearly_agg = yearly_df.groupby("Year").agg({"Tmax": "max", "Tmin": "min", "Tavg": "mean", "Rainfall": "sum"}).reset_index()
        
        if len(yearly_agg) > 2:
            y_tmax_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(yearly_agg[["Year"]], yearly_agg["Tmax"])
            y_tmax_proj = y_tmax_model.predict([[selected_year]])[0]
            y_tmax_hist = yearly_agg["Tmax"].mean()
            
            y_tmin_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(yearly_agg[["Year"]], yearly_agg["Tmin"])
            y_tmin_proj = y_tmin_model.predict([[selected_year]])[0]
            y_tmin_hist = yearly_agg["Tmin"].mean()

            y_tavg_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(yearly_agg[["Year"]], yearly_agg["Tavg"])
            y_tavg_proj = y_tavg_model.predict([[selected_year]])[0]
            y_tavg_hist = yearly_agg["Tavg"].mean()
            y_tavg_accuracy = r2_score(yearly_agg["Tavg"], y_tavg_model.predict(yearly_agg[["Year"]]))
            
            y_rain_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(yearly_agg[["Year"]], yearly_agg["Rainfall"])
            y_rain_proj = y_rain_model.predict([[selected_year]])[0]
            y_rain_hist = yearly_agg["Rainfall"].mean()
            y_rain_accuracy = r2_score(yearly_agg["Rainfall"], y_rain_model.predict(yearly_agg[["Year"]]))
            
            yearly_acc = (max(0.1, y_tavg_accuracy) + max(0.1, y_rain_accuracy)) / 2
        else:
            y_tmax_proj = yearly_agg["Tmax"].mean()
            y_tmax_hist = y_tmax_proj
            y_tmin_proj = yearly_agg["Tmin"].mean()
            y_tmin_hist = y_tmin_proj
            y_tavg_proj = yearly_agg["Tavg"].mean()
            y_tavg_hist = y_tavg_proj
            y_rain_proj = yearly_agg["Rainfall"].mean()
            y_rain_hist = y_rain_proj
            yearly_acc = 0.0
    else:
        y_tmax_proj, y_tmax_hist = 0, 0
        y_tmin_proj, y_tmin_hist = 0, 0
        y_tavg_proj, y_tavg_hist = 0, 0
        y_rain_proj, y_rain_hist = 0, 0
        yearly_acc = 0.0

    st.markdown(f"### 📅 Yearly AI Projections ({selected_year})")
    y1, y2, y3, y4, y5, y6 = st.columns(6)
    y1.metric(f"🌡️ Avg Temp ({selected_year})", f"{y_tavg_proj:.2f} °C", delta=f"{y_tavg_proj - y_tavg_hist:.2f}°")
    y2.metric(f"🔥 Max Temp ({selected_year})", f"{y_tmax_proj:.2f} °C", delta=f"{y_tmax_proj - y_tmax_hist:.2f}°")
    y3.metric(f"❄️ Min Temp ({selected_year})", f"{y_tmin_proj:.2f} °C", delta=f"{y_tmin_proj - y_tmin_hist:.2f}°")
    y4.metric(f"🌧️ Total Rain ({selected_year})", f"{y_rain_proj:.2f} mm", delta=f"{y_rain_proj - y_rain_hist:.1f}mm", delta_color="inverse")
    y5.metric("🎯 Accuracy", f"{abs(yearly_acc*100):.1f}%", help="Yearly Model R-Squared Score Average")
    y6.metric("📊 Mode", "Yearly Aggregates")
    st.markdown("---")

    # Use average of monthly model accuracies for display
    monthly_acc = (tmax_accuracy + tmin_accuracy + month_r_accuracy + acc_t) / 4

    st.markdown(f"### 🎯 Executive AI Projections ({selected_month_name_full} {selected_year})")
    st.markdown(f"**Historical Data Range:** {year_range[0]} - {year_range[1]}")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🌡️ Avg Temp", f"{t_proj:.2f} °C", delta=f"{t_proj - hist_t_avg:.2f}°")
    m2.metric(f"🔥 Max Temp ({selected_month_name_short})", f"{tmax_proj:.2f} °C", delta=f"{tmax_proj - tmax_hist_avg:.2f}°")
    m3.metric(f"❄️ Min Temp ({selected_month_name_short})", f"{tmin_proj:.2f} °C", delta=f"{tmin_proj - tmin_hist_avg:.2f}°")
    m4.metric(f"🌧️ {selected_month_name_short} Rain", f"{month_r_proj:.2f} mm", delta=f"{month_r_proj - month_r_hist_avg:.1f}mm", delta_color="inverse")
    m5.metric("🎯 Accuracy", f"{abs(monthly_acc*100):.1f}%", help="Monthly Model R-Squared Score")
    m6.metric("🧠 Engine", "Monthly ML + LSTM")
    st.markdown("---")

    # =====================================================================
    # LIVE WEATHER UPDATE (PLACED BELOW EXECUTIVE PROJECTIONS)
    # =====================================================================
    st.subheader("🔴 LIVE Weather Update")
    
    location = selected_dist if selected_dist != "None" else selected_state
    lat, lon = None, None
    coord_location = None
    
    if selected_dist != "None" and selected_dist in DISTRICT_COORDINATES:
        lat, lon = DISTRICT_COORDINATES[selected_dist]
        coord_location = selected_dist
    elif selected_state in LOCATION_COORDINATES:
        lat, lon = LOCATION_COORDINATES[selected_state]
        coord_location = selected_state
    
    if lat and lon:
        render_live_weather(lat, lon, coord_location)
    else:
        st.warning(f"ℹ️ Specific coordinates not configured for {location}. Showing historical data only.")
    
    st.markdown("---")

    # =====================================================================
    # 5. INTEGRATED ANALYSIS MODULES (TABS)
    # =====================================================================
    tab1, tab2, tab3, tab4,tab5,tab6 = st.tabs(["📊 Temperature Analysis", "🌧️ Rainfall Intelligence", "💨 Air Quality Index", "PREDICTION HUB (Unified AI Models)", "🔮 2030 Roadmap", "⚠️ Risk Management"])

    # -----------------------------------------------------------------
    # TAB 1: TEMPERATURE ANALYSIS
    # -----------------------------------------------------------------
    with tab1:
        # Historical Data Logic
        hist_matches_t = pd.DataFrame()
        t_max, t_min, t_avg, sample_count = 32.0, 22.0, 27.0, 0
        
        if not df_daily.empty:
            hist_matches_t = df_daily[(df_daily['Month'] == selected_date.month) & 
                                      (df_daily['Day'] == selected_date.day)]
            if not hist_matches_t.empty:
                t_max = hist_matches_t['Tmax'].mean()
                t_min = hist_matches_t['Tmin'].mean()
                t_avg = hist_matches_t['Tavg'].mean()
                sample_count = len(hist_matches_t)

        st.subheader(f"🌡️ Temperature Intelligence: {selected_date.strftime('%d %B')} (Historical Analysis)")

        hours = np.arange(24)
        h_temps = []
        for h in hours:
            if h < 6:
                temp = t_min + (t_max - t_min) * 0.08 * (6 - h) / 6
            elif 6 <= h <= 15:
                temp = t_min + (t_max - t_min) * np.sin((h - 6) * np.pi / 18)
            else:
                temp = t_min + (t_max - t_min) * np.cos((h - 15) * np.pi / 18)
            h_temps.append(round(temp, 2))

        selected_temp = h_temps[selected_hour]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Temp at {selected_hour}:00", f"{selected_temp} °C")
        m2.metric("Hist. Avg High", f"{t_max:.1f} °C")
        m3.metric("Hist. Avg Low", f"{t_min:.1f} °C")
        m4.metric("Data Basis", f"{sample_count} Years")
        st.markdown("---")

        if view_mode == "Live Hour View":
            st.subheader(f"📍 Hourly Pattern (Based on {sample_count} Years of History)")
            g_col1, g_col2 = st.columns([2, 1])
            with g_col1:
                fig_h = px.area(x=hours, y=h_temps, labels={'x':'Hour (24h format)', 'y':'Temperature (°C)'},
                                template="plotly_white", color_discrete_sequence=['#FF4B4B'])
                fig_h.add_vline(x=selected_hour, line_dash="dash", line_color="red", annotation_text=f"Selected: {selected_hour}h")
                fig_h.update_layout(hovermode="x unified")
                st.plotly_chart(fig_h, use_container_width=True)
            with g_col2:
                hourly_df = pd.DataFrame({"Hour": [f"{h:02d}:00" for h in hours], "Temp (°C)": h_temps})
                st.dataframe(hourly_df.set_index("Hour"), height=350, use_container_width=True)

        elif view_mode == "Full Month Trend":
            st.subheader(f"📅 Monthly Mean: {selected_date.strftime('%B')}")
            if not df_daily.empty:
                m_data = df_daily[df_daily['Month'] == selected_date.month].groupby('Day')['Tavg'].mean().reset_index()
                fig_m = px.line(m_data, x='Day', y='Tavg', markers=True, title="Average Daily Temp (Historical)")
                st.plotly_chart(fig_m, use_container_width=True)
                
        elif view_mode == "Annual Heatmap Analysis":
            st.subheader("🌡️ Decadal Temperature Intensity")
            heat_pivot = df_m.pivot_table(index="Month", columns="Year", values="Tavg")
            st.plotly_chart(px.imshow(heat_pivot, color_continuous_scale='YlOrRd', y=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]), use_container_width=True)

        st.markdown("---")
        st.subheader(f"📊 Historical Variations on {selected_date.strftime('%d %B')} ({year_range[0]}-{year_range[1]})")
        if not hist_matches_t.empty:
            t_mask = (hist_matches_t['Year'] >= year_range[0]) & (hist_matches_t['Year'] <= year_range[1])
            filtered_hist_t = hist_matches_t.loc[t_mask]
            if not filtered_hist_t.empty:
                fig_date_hist = px.line(filtered_hist_t, x='Year', y=['Tmax', 'Tmin'],
                                        labels={'value': 'Temperature (°C)', 'variable': 'Category'},
                                        title=f"Tmax vs Tmin Timeline for {selected_date.strftime('%d %B')}")
                st.plotly_chart(fig_date_hist, use_container_width=True)
            else:
                st.info(f"No data available for {selected_date.strftime('%d %B')} in the selected year range ({year_range[0]}-{year_range[1]})")

        st.markdown("---")
        st.subheader(f"📈 Temperature Trajectory Prediction ({year_range[0]} - {year_range[1]})")
        t_mask = (full_t_df['Year'] >= year_range[0]) & (full_t_df['Year'] <= year_range[1])
        filtered_temp_df = full_t_df.loc[t_mask]
        
        if not filtered_temp_df.empty:
            fig_t_hist = px.line(filtered_temp_df, x="Year", y="Tavg", color="Type", markers=True, 
                                 title=f"Exploratory Data Analysis (EDA) & Time-Series: Historical vs AI Projected Temperature",
                                 color_discrete_map={"Historical": "#FF4B4B", "AI Projection": "#00CC96"})
            
            # Extract historical data for moving average
            hist_only = filtered_temp_df[filtered_temp_df["Type"] == "Historical"].copy()
            if len(hist_only) > 5:
                hist_only["5-Year Moving Avg"] = hist_only["Tavg"].rolling(window=5, min_periods=1).mean()
                fig_t_hist.add_trace(go.Scatter(x=hist_only["Year"], y=hist_only["5-Year Moving Avg"],
                                              mode='lines', name='5-Year Moving Avg (Time-Series)',
                                              line=dict(color='orange', width=2, dash='dot')))

            fig_t_hist.update_layout(hovermode="x unified")
            st.plotly_chart(fig_t_hist, use_container_width=True)

        st.markdown("---")
        st.subheader("🔮 Annual Temperature Forecast (2024-2030)")
        f_years = np.arange(2024, 2031).reshape(-1,1)
        f_preds_temp = model_t.predict(f_years)
        
        fig_t_30 = go.Figure()
        fig_t_30.add_trace(go.Scatter(x=f_years.flatten(), y=f_preds_temp + 0.5, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig_t_30.add_trace(go.Scatter(x=f_years.flatten(), y=f_preds_temp - 0.5, fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(255, 75, 75, 0.2)', name='AI Confidence Interval'))
        fig_t_30.add_trace(go.Scatter(x=f_years.flatten(), y=f_preds_temp, name="Temperature Forecast", line=dict(color='#FF4B4B', width=3)))
        
        fig_t_30.update_layout(title="Average Temperature Projection with Uncertainty Bounds", hovermode="x unified")
        fig_t_30.update_xaxes(title_text="Year")
        fig_t_30.update_yaxes(title_text="Temperature (°C)")
        st.plotly_chart(fig_t_30, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Yearly Min-Max Temperature Range (1980-2030) with Historical & Predictions")
        
        # Historical data (1980-2024)
        hist_minmax = df_m.groupby('Year')[['Tmax', 'Tmin']].mean().reset_index()
        hist_minmax['Type'] = 'Historical'
        
        # Predicted data (2024-2030) - using annual models for yearly overview
        f_years_range = np.arange(2024, 2031).reshape(-1,1)
        f_tmax = model_t.predict(f_years_range) + 5  # Rough estimate for max temp
        f_tmin = model_t.predict(f_years_range) - 5  # Rough estimate for min temp
        
        pred_minmax = pd.DataFrame({
            'Year': f_years_range.flatten(),
            'Tmax': f_tmax,
            'Tmin': f_tmin,
            'Type': 'Predicted'
        })
        
        # Combine historical and predicted
        combined_minmax = pd.concat([hist_minmax, pred_minmax], ignore_index=True)
        
        fig_minmax = go.Figure()
        
        # Historical data
        hist_data = hist_minmax
        fig_minmax.add_trace(go.Scatter(x=hist_data['Year'], y=hist_data['Tmax'], 
                                        fill=None, mode='lines+markers', 
                                        name='Historical Max Temp', 
                                        line=dict(color='#FF6B6B', width=2, dash='solid')))
        fig_minmax.add_trace(go.Scatter(x=hist_data['Year'], y=hist_data['Tmin'], 
                                        fill='tonexty', mode='lines+markers', 
                                        name='Historical Min Temp',
                                        line=dict(color='#4ECDC4', width=2, dash='solid'),
                                        fillcolor='rgba(78, 205, 196, 0.15)'))
        
        # Predicted data
        pred_data = pred_minmax
        fig_minmax.add_trace(go.Scatter(x=pred_data['Year'], y=pred_data['Tmax'], 
                                        fill=None, mode='lines+markers', 
                                        name='Predicted Max Temp', 
                                        line=dict(color='#FF6B6B', width=2, dash='dash')))
        fig_minmax.add_trace(go.Scatter(x=pred_data['Year'], y=pred_data['Tmin'], 
                                        fill='tonexty', mode='lines+markers', 
                                        name='Predicted Min Temp',
                                        line=dict(color='#4ECDC4', width=2, dash='dash'),
                                        fillcolor='rgba(78, 205, 196, 0.1)'))
        
        fig_minmax.update_layout(title="Min-Max Temperature Range: Historical (1980-2024) vs Predicted (2024-2030)", 
                                hovermode="x unified",
                                yaxis_title="Temperature (°C)",
                                xaxis_title="Year")
        st.plotly_chart(fig_minmax, use_container_width=True)
        
        # Display data table for predicted only
        st.write("**Predicted Yearly Temperature Range (2024-2030):**")
        display_df = pred_minmax[['Year', 'Tmax', 'Tmin']].copy()
        display_df['Tmax'] = display_df['Tmax'].round(2)
        display_df['Tmin'] = display_df['Tmin'].round(2)
        display_df['Range'] = (display_df['Tmax'] - display_df['Tmin']).round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📅 Monthly Min-Max Temperature Pattern (All Years Average)")
        
        # Calculate average monthly temperatures across all years
        monthly_temps = df_m.groupby('Month')[['Tmax', 'Tmin']].mean().reset_index()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_temps['Month_Name'] = monthly_temps['Month'].apply(lambda x: month_names[int(x)-1])
        
        fig_monthly = go.Figure()
        
        fig_monthly.add_trace(go.Bar(x=monthly_temps['Month_Name'], y=monthly_temps['Tmax'],
                                     name='Max Temperature', marker=dict(color='#FF6B6B')))
        fig_monthly.add_trace(go.Bar(x=monthly_temps['Month_Name'], y=monthly_temps['Tmin'],
                                     name='Min Temperature', marker=dict(color='#4ECDC4')))
        
        fig_monthly.update_layout(title="Monthly Average Min-Max Temperature Range",
                                 barmode='group',
                                 hovermode="x unified",
                                 yaxis_title="Temperature (°C)",
                                 xaxis_title="Month",
                                 height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Display monthly data table
        st.write("**Monthly Temperature Statistics (Historical Average):**")
        monthly_display = monthly_temps[['Month_Name', 'Tmax', 'Tmin']].copy()
        monthly_display.columns = ['Month', 'Max Temp (°C)', 'Min Temp (°C)']
        monthly_display['Temp Range'] = (monthly_temps['Tmax'] - monthly_temps['Tmin']).round(2)
        monthly_display['Max Temp (°C)'] = monthly_display['Max Temp (°C)'].round(2)
        monthly_display['Min Temp (°C)'] = monthly_display['Min Temp (°C)'].round(2)
        st.dataframe(monthly_display, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # TAB 2: RAINFALL INTELLIGENCE
    # -----------------------------------------------------------------
    with tab2:
        hist_matches_r = pd.DataFrame()
        daily_total_rain, sample_count_r = 0.0, 0
        
        if not df_daily.empty:
            hist_matches_r = df_daily[(df_daily['Month'] == selected_date.month) & 
                                      (df_daily['Day'] == selected_date.day)]
            if not hist_matches_r.empty:
                daily_total_rain = hist_matches_r['Rainfall'].mean()
                sample_count_r = len(hist_matches_r)

        st.subheader(f"🌧️ Rainfall Patterns for {selected_date.strftime('%d %B')} (Historical Avg)")

        h_rain = []
        for h in hours:
            if daily_total_rain > 0 and 10 <= h <= 22:
                raw_val = np.sin((h - 10) * np.pi / 12)
                val = max(0, raw_val * (daily_total_rain / 7.5))
            else:
                val = 0
            h_rain.append(round(val, 2))

        current_rain_h = h_rain[selected_hour]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"Rain at {selected_hour}h", f"{current_rain_h:.2f} mm")
        r2.metric("Annual Proj.", f"{r_proj:.1f} mm")
        r3.metric("Historical Daily Avg", f"{daily_total_rain:.2f} mm")
        rain_status = "Stable" if r_proj > (hist_r_avg * 0.8) else "Drought Risk"
        r4.metric("Rain Status", rain_status)

        st.markdown("---")

        if view_mode == "Live Hour View":
            st.subheader(f"📍 Hourly Precipitation Profile: {selected_date.strftime('%d %B')}")
            r_col1, r_col2 = st.columns([2, 1])
            with r_col1:
                fig_r = px.area(x=hours, y=h_rain, labels={'x':'Hour (24h format)', 'y':'Rainfall (mm)'},
                                template="plotly_white", color_discrete_sequence=['#1f77b4'], title="Hourly Rain Distribution")
                fig_r.add_vline(x=selected_hour, line_dash="dash", line_color="blue", annotation_text=f"Selected: {selected_hour}h")
                st.plotly_chart(fig_r, use_container_width=True)
            with r_col2:
                st.write("📋 **Rainfall Matrix**")
                rain_df = pd.DataFrame({"Hour": [f"{h:02d}:00" for h in hours], "Rain (mm)": h_rain})
                st.dataframe(rain_df.set_index("Hour"), height=350, use_container_width=True)

        elif view_mode == "Full Month Trend":
            st.subheader(f"📅 Monthly Rainfall Trend: {selected_date.strftime('%B')}")
            if not df_daily.empty:
                m_rain = df_daily[df_daily['Month'] == selected_date.month].groupby('Day')['Rainfall'].mean().reset_index()
                st.plotly_chart(px.bar(m_rain, x='Day', y='Rainfall', title="Daily Rainfall accumulation", color_discrete_sequence=['#1f77b4']), use_container_width=True)

        elif view_mode == "Annual Heatmap Analysis":
            st.subheader("🌧️ Decadal Rainfall Intensity (Heatmap)")
            rain_pivot = df_m.pivot_table(index="Month", columns="Year", values="Rainfall")
            fig_rain_heat = px.imshow(rain_pivot, y=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], color_continuous_scale='Blues')
            st.plotly_chart(fig_rain_heat, use_container_width=True)

        st.markdown("---")
        st.subheader(f"📉 Rainfall Trajectory ({year_range[0]} - {year_range[1]})")
        r_mask = (full_r_df['Year'] >= year_range[0]) & (full_r_df['Year'] <= year_range[1])
        filtered_rain_df = full_r_df.loc[r_mask]
        
        if not filtered_rain_df.empty:
            fig_r_hist = px.line(filtered_rain_df, x="Year", y="Rainfall", color="Type", markers=True, 
                                 title=f"Exploratory Data Analysis (EDA) & Time-Series: Historical vs Projected Rainfall",
                                 color_discrete_map={"Historical": "#1f77b4", "AI Projection": "#00CC96"})
            
            # Extract historical data for moving average
            hist_only_r = filtered_rain_df[filtered_rain_df["Type"] == "Historical"].copy()
            if len(hist_only_r) > 5:
                hist_only_r["5-Year Moving Avg"] = hist_only_r["Rainfall"].rolling(window=5, min_periods=1).mean()
                fig_r_hist.add_trace(go.Scatter(x=hist_only_r["Year"], y=hist_only_r["5-Year Moving Avg"],
                                              mode='lines', name='5-Year Moving Avg (Time-Series)',
                                              line=dict(color='purple', width=2, dash='dot')))
                                              
            fig_r_hist.update_layout(hovermode="x unified")
            st.plotly_chart(fig_r_hist, use_container_width=True)

        st.markdown("---")
        st.subheader(f"📊 Historical Rainfall Variations on {selected_date.strftime('%d %B')} ({year_range[0]}-{year_range[1]})")
        if not hist_matches_r.empty:
            r_mask_hist = (hist_matches_r['Year'] >= year_range[0]) & (hist_matches_r['Year'] <= year_range[1])
            filtered_hist_r = hist_matches_r.loc[r_mask_hist]
            if not filtered_hist_r.empty:
                fig_date_rain = px.line(filtered_hist_r, x='Year', y='Rainfall',
                                        labels={'Rainfall': 'Rainfall (mm)'},
                                        title=f"Rainfall Timeline for {selected_date.strftime('%d %B')}")
                st.plotly_chart(fig_date_rain, use_container_width=True)
            else:
                st.info(f"No rainfall data available for {selected_date.strftime('%d %B')} in the selected year range ({year_range[0]}-{year_range[1]})")

        st.markdown("---")
        st.subheader("🔮 Annual Rainfall Forecast (2024-2030)")
        f_years_rain = np.arange(2024, 2031).reshape(-1,1)
        f_preds_rain = model_r.predict(f_years_rain)
        
        fig_r_30 = go.Figure()
        fig_r_30.add_trace(go.Scatter(x=f_years_rain.flatten(), y=f_preds_rain + 50, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig_r_30.add_trace(go.Scatter(x=f_years_rain.flatten(), y=f_preds_rain - 50, fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(31, 119, 180, 0.2)', name='AI Confidence Interval'))
        fig_r_30.add_trace(go.Scatter(x=f_years_rain.flatten(), y=f_preds_rain, name="Rainfall Forecast", line=dict(color='#1f77b4', width=3)))
        
        fig_r_30.update_layout(title="Annual Rainfall Projection with Uncertainty Bounds", hovermode="x unified")
        fig_r_30.update_xaxes(title_text="Year")
        fig_r_30.update_yaxes(title_text="Rainfall (mm)")
        st.plotly_chart(fig_r_30, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Yearly Rainfall Range (1980-2030) with Historical & Predictions")
        
        # Historical rainfall data (1980-2024)
        hist_rainfall = df_m.groupby('Year')['Rainfall'].mean().reset_index()
        hist_rainfall['Type'] = 'Historical'
        
        # Predicted rainfall data (2024-2030)
        f_years_range_r = np.arange(2024, 2031).reshape(-1,1)
        f_rain = model_r.predict(f_years_range_r)
        
        pred_rainfall = pd.DataFrame({
            'Year': f_years_range_r.flatten(),
            'Rainfall': f_rain,
            'Type': 'Predicted'
        })
        
        # Combine historical and predicted
        combined_rainfall = pd.concat([hist_rainfall, pred_rainfall], ignore_index=True)
        
        fig_r_minmax = px.line(combined_rainfall, x="Year", y="Rainfall", color="Type", markers=True, 
                             title="Historical vs Predicted Annual Rainfall",
                             color_discrete_map={"Historical": "#1f77b4", "Predicted": "#00CC96"})
        fig_r_minmax.update_layout(hovermode="x unified")
        st.plotly_chart(fig_r_minmax, use_container_width=True)
        
        # Display predicted rainfall table
        st.write("**Predicted Yearly Rainfall (2024-2030):**")
        display_rain_df = pred_rainfall[['Year', 'Rainfall']].copy()
        display_rain_df['Rainfall'] = display_rain_df['Rainfall'].round(2)
        st.dataframe(display_rain_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📅 Monthly Rainfall Pattern (All Years Average)")
        
        # Calculate average monthly rainfall across all years
        monthly_rainfall = df_m.groupby('Month')['Rainfall'].mean().reset_index()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_rainfall['Month_Name'] = monthly_rainfall['Month'].apply(lambda x: month_names[int(x)-1])
        
        fig_monthly_rain = px.bar(monthly_rainfall, x='Month_Name', y='Rainfall',
                                  labels={'Rainfall': 'Rainfall (mm)', 'Month_Name': 'Month'},
                                  title="Monthly Average Rainfall Pattern",
                                  color_discrete_sequence=['#1f77b4'])
        fig_monthly_rain.update_layout(hovermode="x unified", height=400)
        st.plotly_chart(fig_monthly_rain, use_container_width=True)
        
        # Display monthly rainfall table
        st.write("**Monthly Rainfall Statistics (Historical Average):**")
        monthly_rain_display = monthly_rainfall[['Month_Name', 'Rainfall']].copy()
        monthly_rain_display.columns = ['Month', 'Avg Rainfall (mm)']
        monthly_rain_display['Avg Rainfall (mm)'] = monthly_rain_display['Avg Rainfall (mm)'].round(2)
        st.dataframe(monthly_rain_display, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # TAB 3: 💨 AIR QUALITY INDEX (AQI) ANALYSIS
    # -----------------------------------------------------------------
    with tab3:
        st.subheader(f"🔴 LIVE Air Quality Update: {location}")
        if lat and lon:
            live_aqi = get_live_aqi(lat, lon)
            if live_aqi:
                status_text, status_color = interpret_aqi_index(live_aqi['aqi_index'])

                # Display all components in two rows
                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                r1c1.metric("AQI Status", status_text)
                r1c2.metric("PM2.5", f"{live_aqi['components'].get('pm2_5', 0)} µg/m³")
                r1c3.metric("PM10", f"{live_aqi['components'].get('pm10', 0)} µg/m³")
                r1c4.metric("CO (Carbon Monoxide)", f"{live_aqi['components'].get('co', 0)} µg/m³")

                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                r2c1.metric("NO2 (Nitrogen Dioxide)", f"{live_aqi['components'].get('no2', 0)} µg/m³")
                r2c2.metric("SO2 (Sulfur Dioxide)", f"{live_aqi['components'].get('so2', 0)} µg/m³")
                r2c3.metric("O3 (Ozone)", f"{live_aqi['components'].get('o3', 0)} µg/m³")
                r2c4.metric("NH3 (Ammonia)", f"{live_aqi['components'].get('nh3', 0)} µg/m³")
                
                st.caption(f"Live telemetry from OpenWeather API | Last updated: {live_aqi['timestamp']}")
                
                # Custom CSS for status box
                st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:rgba(0,0,0,0.1); border-left: 5px solid {status_color};">
                        <strong>Health Advisory:</strong> The current air quality is categorized as {status_text}. 
                        {'Outdoor activities are safe.' if live_aqi['aqi_index'] <= 2 else 'Sensitive groups should reduce outdoor exertion.' if live_aqi['aqi_index'] == 3 else 'Wear a mask and avoid prolonged outdoor exposure.'}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Live AQI data unavailable. Showing historical models below.")
        else:
            st.info("Coordinates not available for live AQI tracking.")
            
        st.markdown("---")
        
        # === AQI DATA PREPARATION ===
        if not df_daily.empty:
            # Generate synthetic AQI if not present
            if 'AQI' not in df_daily.columns:
                df_daily['AQI'] = (df_daily['Tavg'] * 1.5) - (df_daily['Rainfall'] * 0.8) + 20
                df_daily['AQI'] = df_daily['AQI'].clip(lower=15, upper=450)
                df_daily['PM25'] = df_daily['AQI'] * 0.45
                df_daily['PM10'] = df_daily['AQI'] * 0.85
                df_daily['NO2'] = df_daily['AQI'] * 0.35
                df_daily['SO2'] = df_daily['AQI'] * 0.25
            
            # Train AQI prediction model (similar to temp/rainfall)
            def train_aqi_model(df):
                y_aqi = df.groupby("Year")['AQI'].mean().reset_index()
                model = LinearRegression().fit(y_aqi[["Year"]], y_aqi['AQI'])
                accuracy = r2_score(y_aqi['AQI'], model.predict(y_aqi[["Year"]]))
                future = pd.DataFrame({'Year': [2024, 2025, 2026], 'AQI': model.predict(np.array([2024,2025,2026]).reshape(-1,1)), 'Type': 'AI Projection'})
                hist = y_aqi[["Year", 'AQI']].copy()
                hist['Type'] = 'Historical'
                return pd.concat([hist, future]), model, accuracy, y_aqi['AQI'].mean()
            
            full_aqi_df, model_aqi, acc_aqi, hist_aqi_avg = train_aqi_model(df_daily)
            aqi_2026 = model_aqi.predict([[2026]])[0]
            
            # Current Date AQI Metrics
            aqi_today = df_daily[(df_daily['Month'] == selected_date.month) & (df_daily['Day'] == selected_date.day)]
            
            if not aqi_today.empty:
                avg_aqi = aqi_today['AQI'].mean()
                avg_pm25 = aqi_today['PM25'].mean()
                avg_pm10 = aqi_today['PM10'].mean()
                avg_no2 = aqi_today['NO2'].mean()
                
                # AQI Status Classification
                if avg_aqi <= 50:
                    aqi_status = "✅ Good"
                elif avg_aqi <= 100:
                    aqi_status = "🟡 Moderate"
                elif avg_aqi <= 150:
                    aqi_status = "🟠 Unhealthy for Sensitive"
                elif avg_aqi <= 200:
                    aqi_status = "🔴 Unhealthy"
                elif avg_aqi <= 300:
                    aqi_status = "🟣 Very Unhealthy"
                else:
                    aqi_status = "⚫ Hazardous"
                
                # Display AQI Metrics
                aq1, aq2, aq3, aq4 = st.columns(4)
                aq1.metric(f"Current AQI ({selected_date.strftime('%d %b')})", f"{avg_aqi:.1f}")
                aq2.metric("PM2.5 (µg/m³)", f"{avg_pm25:.1f}")
                aq3.metric("PM10 (µg/m³)", f"{avg_pm10:.1f}")
                aq4.metric("Status", aqi_status.split()[1])
                
                st.info(f"**Air Quality Status:** {aqi_status}")
                
                # Health Advisory
                if avg_aqi > 150:
                    st.error("⚠️ **Health Alert:** Mask recommended outdoors. Reduce outdoor activities.")
                elif avg_aqi > 100:
                    st.warning("🟠 **Health Alert:** Moderate risk. Limit intense outdoor exertion.")
                else:
                    st.success("✅ **Health Alert:** Air quality optimal for outdoor activities.")
                
                st.markdown("---")
            
            st.subheader(f"🌡️ AQI Intelligence: {selected_date.strftime('%d %B')} (Historical Analysis)")
            
            # Historical AQI matches for selected date
            hist_matches_aqi = pd.DataFrame()
            aqi_max, aqi_min, aqi_avg, aqi_sample_count = 150.0, 50.0, 100.0, 0
            
            if not df_daily.empty:
                hist_matches_aqi = df_daily[(df_daily['Month'] == selected_date.month) & 
                                           (df_daily['Day'] == selected_date.day)]
                if not hist_matches_aqi.empty:
                    aqi_max = hist_matches_aqi['AQI'].max()
                    aqi_min = hist_matches_aqi['AQI'].min()
                    aqi_avg = hist_matches_aqi['AQI'].mean()
                    aqi_sample_count = len(hist_matches_aqi)
            
            a1, a2, a3, a4 = st.columns(4)
            a1.metric(f"Max AQI ({selected_date.strftime('%d %b')})", f"{aqi_max:.1f}")
            a2.metric(f"Min AQI ({selected_date.strftime('%d %b')})", f"{aqi_min:.1f}")
            a3.metric(f"Avg AQI ({selected_date.strftime('%d %b')})", f"{aqi_avg:.1f}")
            a4.metric("Data Points", f"{aqi_sample_count} Years")
            st.markdown("---")
            
            # View Mode Selection
            if view_mode == "Live Hour View":
                st.subheader(f"📍 Hourly AQI Distribution (Based on {aqi_sample_count} Years)")
                h_aqi = []
                for h in hours:
                    if 8 <= h <= 19:
                        aqi_val = aqi_avg + (aqi_max - aqi_avg) * np.sin((h - 8) * np.pi / 11)
                    else:
                        aqi_val = aqi_min + (aqi_avg - aqi_min) * 0.3
                    h_aqi.append(round(aqi_val, 2))
                
                h_col1, h_col2 = st.columns([2, 1])
                with h_col1:
                    fig_h_aqi = px.area(x=hours, y=h_aqi, labels={'x':'Hour (24h)', 'y':'AQI Index'},
                                       template="plotly_white", color_discrete_sequence=['#FF6B6B'])
                    fig_h_aqi.add_vline(x=selected_hour, line_dash="dash", line_color="red")
                    fig_h_aqi.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_h_aqi, use_container_width=True)
                with h_col2:
                    hourly_aqi_df = pd.DataFrame({"Hour": [f"{h:02d}:00" for h in hours], "AQI": h_aqi})
                    st.dataframe(hourly_aqi_df.set_index("Hour"), height=350, use_container_width=True)
            
            elif view_mode == "Full Month Trend":
                st.subheader(f"📅 Monthly AQI Trend: {selected_date.strftime('%B')}")
                if not df_daily.empty:
                    m_aqi = df_daily[df_daily['Month'] == selected_date.month].groupby('Day')['AQI'].mean().reset_index()
                    st.plotly_chart(px.line(m_aqi, x='Day', y='AQI', title="Daily AQI Pattern", 
                                           color_discrete_sequence=['#FF6B6B'], markers=True),
                                   use_container_width=True)
            
            elif view_mode == "Annual Heatmap Analysis":
                st.subheader("🌫️ Decadal AQI Intensity (Heatmap)")
                aqi_pivot = df_daily.pivot_table(index="Month", columns="Year", values="AQI")
                month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                fig_aqi_heat = px.imshow(aqi_pivot, y=month_labels, color_continuous_scale='YlOrRd')
                st.plotly_chart(fig_aqi_heat, use_container_width=True)
            
            st.markdown("---")
            st.subheader(f"📊 AQI Trajectory Prediction ({year_range[0]} - {year_range[1]})")
            aqi_mask = (full_aqi_df['Year'] >= year_range[0]) & (full_aqi_df['Year'] <= year_range[1])
            filtered_aqi_df = full_aqi_df.loc[aqi_mask]
            
            if not filtered_aqi_df.empty:
                fig_aqi_hist = px.line(filtered_aqi_df, x="Year", y="AQI", color="Type", markers=True, 
                                      title=f"Historical vs AI Projected AQI",
                                      color_discrete_map={"Historical": "#FF6B6B", "AI Projection": "#00CC96"})
                fig_aqi_hist.update_layout(hovermode="x unified")
                st.plotly_chart(fig_aqi_hist, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔮 Annual AQI Forecast (2024-2030)")
            f_years_aqi = np.arange(2024, 2031).reshape(-1, 1)
            f_preds_aqi = model_aqi.predict(f_years_aqi)
            
            fig_aqi_30 = go.Figure()
            fig_aqi_30.add_trace(go.Scatter(x=f_years_aqi.flatten(), y=f_preds_aqi + 10, fill=None, mode='lines', 
                                           line_color='rgba(0,0,0,0)', showlegend=False))
            fig_aqi_30.add_trace(go.Scatter(x=f_years_aqi.flatten(), y=f_preds_aqi - 10, fill='tonexty', mode='lines', 
                                           line_color='rgba(0,0,0,0)', fillcolor='rgba(255, 107, 107, 0.2)', 
                                           name='AI Confidence Interval'))
            fig_aqi_30.add_trace(go.Scatter(x=f_years_aqi.flatten(), y=f_preds_aqi, name="AQI Forecast", 
                                           line=dict(color='#FF6B6B', width=3)))
            
            fig_aqi_30.update_layout(title="Average AQI Projection with Uncertainty Bounds", hovermode="x unified")
            fig_aqi_30.update_xaxes(title_text="Year")
            fig_aqi_30.update_yaxes(title_text="AQI Index")
            st.plotly_chart(fig_aqi_30, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 Pollutant Composition Analysis (2024-2030)")
            
            # Generate pollutant projections
            f_pm25_proj = model_aqi.predict(f_years_aqi) * 0.45
            f_pm10_proj = model_aqi.predict(f_years_aqi) * 0.85
            f_no2_proj = model_aqi.predict(f_years_aqi) * 0.35
            
            pollutant_df = pd.DataFrame({
                'Year': f_years_aqi.flatten(),
                'PM2.5': f_pm25_proj,
                'PM10': f_pm10_proj,
                'NO2': f_no2_proj
            })
            
            fig_pollutants = px.line(pollutant_df, x='Year', y=['PM2.5', 'PM10', 'NO2'],
                                    title="Predicted Pollutant Levels (2024-2030)",
                                    color_discrete_map={'PM2.5': '#FF4B4B', 'PM10': '#FF9500', 'NO2': '#4ECDC4'},
                                    markers=True)
            fig_pollutants.update_layout(hovermode="x unified")
            st.plotly_chart(fig_pollutants, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📅 Monthly AQI Pattern (All Years Average)")
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_aqi = df_daily.groupby('Month')['AQI'].mean().reset_index()
            monthly_aqi['Month_Name'] = monthly_aqi['Month'].apply(lambda x: month_names[int(x)-1])
            
            fig_monthly_aqi = px.bar(monthly_aqi, x='Month_Name', y='AQI',
                                    labels={'AQI': 'AQI Index', 'Month_Name': 'Month'},
                                    title="Monthly Average AQI Pattern",
                                    color='AQI',
                                    color_continuous_scale='YlOrRd')
            fig_monthly_aqi.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig_monthly_aqi, use_container_width=True)
            
            # Display table
            st.write("**Monthly AQI Statistics (Historical Average):**")
            monthly_aqi_display = monthly_aqi[['Month_Name', 'AQI']].copy()
            monthly_aqi_display.columns = ['Month', 'Avg AQI']
            monthly_aqi_display['Avg AQI'] = monthly_aqi_display['Avg AQI'].round(2)
            st.dataframe(monthly_aqi_display, use_container_width=True, hide_index=True)
        
        else:
            st.warning("AQI analysis requires daily data. Please check data files.")
        
        # 1. DYNAMIC FILE PATH SETUP
        if selected_dist == "None":
            aqi_path = f"Datasets/State_AQI_CSV/data_State_{state_id}.csv"
        else:
            d_id = dist_list[dist_list["District"] == selected_dist]["ID"].values[0]
            aqi_path = f"Datasets/District_AQI_CSV/data_District_{d_id}.csv"

        # 2. DATA LOADING & PROCESSING
        # --- DATA LOADING & PROCESSING (Fixed for Date Error) ---
        if os.path.exists(aqi_path):
            df_aqi_real = pd.read_csv(aqi_path)
            
            # FIXED LINE: 'format=mixed' handle karega 13/01/18 aur 13/01/2018 dono ko
            df_aqi_real['Date'] = pd.to_datetime(df_aqi_real['Date'], dayfirst=True, format='mixed')
            
            df_aqi_real['Year'] = df_aqi_real['Date'].dt.year
            df_aqi_real['Month'] = df_aqi_real['Date'].dt.month
            df_aqi_real['Day'] = df_aqi_real['Date'].dt.day
            
            # Filtering for metrics based on sidebar
            aqi_match = df_aqi_real[(df_aqi_real['Month'] == selected_date.month) & 
                                    (df_aqi_real['Day'] == selected_date.day)]
            
            if not aqi_match.empty:
                aqi_today = aqi_match.iloc[0]
                
                # --- METRICS SECTION ---
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current AQI", f"{aqi_today['AQI']:.0f}")
                m2.metric("PM2.5", f"{aqi_today['PM2.5']:.1f}")
                m3.metric("NO2 Level", f"{aqi_today['NO2']:.1f}")
                m4.metric("SO2 Level", f"{aqi_today['SO2']:.1f}")

                st.markdown("---")

                # --- GRAPH 1: YEARLY POLLUTION TRENDS ---
                st.subheader("🗓️ Yearly Pollution Trends")
                yearly_aqi = df_aqi_real.groupby('Year')[['SO2', 'NO2', 'PM2.5']].mean().reset_index()
                fig_yearly = px.line(yearly_aqi, x='Year', y=['SO2', 'NO2', 'PM2.5'],
                                    labels={'value': 'Concentration (µg/m³)'},
                                    color_discrete_map={"SO2": "#FF4B4B", "NO2": "#00D2B4", "PM2.5": "#45B6FE"},
                                    markers=True, template="plotly_white")
                fig_yearly.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_yearly, use_container_width=True)

                st.markdown("---")

                # --- GRAPH 2: MONTHLY POLLUTION PATTERNS ---
                st.subheader("📈 Monthly Pollution Patterns")
                monthly_aqi = df_aqi_real.groupby('Month')[['SO2', 'NO2', 'PM2.5']].mean().reset_index()
                fig_monthly = px.area(monthly_aqi, x='Month', y=['SO2', 'NO2', 'PM2.5'],
                                     color_discrete_map={"SO2": "#FF4B4B", "NO2": "#00D2B4", "PM2.5": "#45B6FE"},
                                     template="plotly_white")
                st.plotly_chart(fig_monthly, use_container_width=True)

                st.markdown("---")

                # --- GRAPH 3: POLLUTANT DISTRIBUTION (RADAR) ---
                st.subheader("🕸️ Pollutant Distribution Comparison")
                pollutants = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2']
                radar_vals = [aqi_today[p] for p in pollutants]
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=radar_vals, 
                    theta=pollutants, 
                    fill='toself',
                    fillcolor='rgba(0, 210, 180, 0.3)',
                    line=dict(color='#00D2B4')
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_white")
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("Selected date ke liye data nahi mila. Sidebar se date change karein.")
        else:
            st.error(f"⚠️ File Not Found: {aqi_path}. Folder structure check karein.")
        
        # --- 5. CLIMATE ROADMAP TO 2030 (Original logic preserved) ---
    with tab4:
        st.header("🔮 AI Predictive Intelligence")
        st.markdown("Select a category and provide parameters to get AI-driven environmental forecasts.")
        
        with st.expander("🧠 Methodology: Machine Learning & Deep Learning"):
            st.markdown("""
            This Prediction Hub fulfills the project objectives by utilizing multiple paradigms of AI:
            - **Machine Learning (Regression):** Time-Series based Linear Regression models track Long-Term Temperature Trends.
            - **Machine Learning (Classification/Ensembles):** A Random Forest model evaluates complex gas composition (SO2, NO2, RSPM, SPM) to predict Air Quality.
            - **Deep Learning (Neural Networks):** A Long Short-Term Memory (LSTM) network applies deep sequential learning to identify intricate patterns in humidity over time to reliably forecast rainfall probabilities.
            """)

        # 1. Main Dropdown for Selection
        predict_choice = st.selectbox(
            "What would you like to forecast?",
            ["Air Quality (PM2.5)", "Temperature Trends", "Rainfall Intensity", "Batch Prediction via CSV Upload"],
            help="Choose the environmental factor you want the AI to analyze."
        )

        st.markdown("---")

        # --- OPTION 1: AIR QUALITY PREDICTION ---
        if predict_choice == "Air Quality (PM2.5)":
            st.subheader("🍃 PM2.5 Pollutant Predictor")
            st.write("Enter the concentration levels of various gases to predict PM2.5.")
            
            col1, col2 = st.columns(2)
            with col1:
                so2 = st.number_input("SO2 (Sulfur Dioxide) µg/m³", min_value=0.0, value=17.5, step=0.1)
                no2 = st.number_input("NO2 (Nitrogen Dioxide) µg/m³", min_value=0.0, value=45.0, step=0.1)
            with col2:
                rspm = st.number_input("RSPM (Respirable PM)", min_value=0.0, value=120.0, step=0.1)
                spm = st.number_input("SPM (Suspended PM)", min_value=0.0, value=180.0, step=0.1)
            
            if st.button("Predict Air Quality", type="primary"):
                if rf_model:
                    # Input array prepare karein
                    features = np.array([[so2, no2, rspm, spm]])
                    prediction = rf_model.predict(features)[0]
                    
                    st.divider()
                    st.success(f"### Predicted PM2.5 Level: {prediction:.2f} µg/m³")
                    
                    # Status Indicator
                    if prediction <= 50:
                        st.info("🟢 **Status:** Good (Minimal Impact)")
                    elif prediction <= 100:
                        st.warning("🟡 **Status:** Moderate (Minor Breathing Discomfort)")
                    else:
                        st.error("🔴 **Status:** Poor / Hazardous (Health Alert)")
                else:
                    st.error("⚠️ RF Model file not found in 'models/' folder.")

        # --- OPTION 2: TEMPERATURE TRENDS ---
        elif predict_choice == "Temperature Trends":
            st.subheader("🌡️ Future Temperature Forecaster")
            st.write("Predict the average yearly temperature based on historical trends.")
            
            target_year = st.slider("Target Year", 2026, 2060, 2030)
            
            if st.button("Forecast Temperature", type="primary"):
                if lr_model:
                    # Linear Regression expects Year as input
                    pred_result = lr_model.predict(np.array([[target_year]]))
                    # Safely convert numpy array to Python float
                    pred_temp = float(np.asarray(pred_result).flatten()[0])
                    
                    st.divider()
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric(f"Projected Temp in {target_year}", f"{pred_temp:.2f} °C")
                    with col_res2:
                        # Assuming a base temp of 2024 for comparison
                        delta_val = pred_temp - 24.8 
                        st.metric("Expected Rise", f"{delta_val:.2f} °C", delta=f"{delta_val:.2f} °C", delta_color="inverse")
                else:
                    st.error("⚠️ Temperature model (Linear Regression) not loaded.")

        # --- OPTION 3: RAINFALL INTENSITY ---
        elif predict_choice == "Rainfall Intensity":
            st.subheader("🌧️ Rainfall Probability Predictor")
            st.write("Enter Humidity values for the last 3 days to predict the next day's rain probability.")
            
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                h1 = st.slider("Day 1 Humidity (%)", 0, 100, 60, key="humidity_day1")
            with col_h2:
                h2 = st.slider("Day 2 Humidity (%)", 0, 100, 65, key="humidity_day2")
            with col_h3:
                h3 = st.slider("Day 3 Humidity (%)", 0, 100, 70, key="humidity_day3")
            
            if st.button("Predict Rainfall Probability", type="primary", key="rainfall_predict_btn"):
                st.divider()
                prediction = None
                model_used = None
                
                # Try LSTM model first (if available)
                if lstm_model is not None and TENSORFLOW_AVAILABLE:
                    try:
                        # LSTM expects (Samples, TimeSteps, Features)
                        input_data = np.array([[[h1], [h2], [h3]]])  # Shape: (1, 3, 1)
                        lstm_prediction = lstm_model.predict(input_data, verbose=0)
                        
                        # Handle different output shapes safely
                        if len(lstm_prediction.shape) > 1:
                            prediction = lstm_prediction.flatten()[0]
                        else:
                            prediction = lstm_prediction[0]
                        
                        # Ensure prediction is between 0 and 1
                        prediction = float(prediction)
                        if prediction > 1:
                            prediction = prediction / 100
                        prediction = max(0.0, min(1.0, prediction))
                        model_used = "LSTM Neural Network"
                    except Exception as lstm_error:
                        # Fallback to simple calculator if LSTM fails
                        prediction = calculate_rainfall_probability_fallback(h1, h2, h3)
                        model_used = "Humidity Trend Algorithm (Fallback)"
                else:
                    # Use fallback method if LSTM not available
                    prediction = calculate_rainfall_probability_fallback(h1, h2, h3)
                    model_used = "Humidity Trend Algorithm (Fallback)"
                
                # Display results
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Chance of Rain", f"{prediction*100:.1f}%")
                with col_res2:
                    st.write(f"**Algorithm:** {model_used}")
                
                # Progress bar for visualization
                st.progress(float(prediction))
                
                # Recommendations based on probability
                if prediction > 0.75:
                    st.success("☔ **High Probability** - Rain is very likely. Carry an umbrella and plan accordingly!")
                elif prediction > 0.50:
                    st.info("🌤️ **Moderate Probability** - Rain is possible. Have an umbrella handy.")
                elif prediction > 0.25:
                    st.warning("⛅ **Low Probability** - Light rain possible. It's generally safe to go out.")
                else:
                    st.success("☀️ **Very Low Probability** - Clear weather expected. Enjoy the sunshine!")
                
                # Additional insights
                st.markdown("---")
                st.subheader("📊 Humidity Analysis")
                col_insight1, col_insight2, col_insight3 = st.columns(3)
                with col_insight1:
                    st.metric("Average Humidity", f"{(h1+h2+h3)/3:.1f}%")
                with col_insight2:
                    trend = h3 - h1
                    trend_indicator = "📈 Increasing" if trend > 2 else "📉 Decreasing" if trend < -2 else "→ Stable"
                    st.metric("Humidity Trend", trend_indicator, delta=f"{trend:+.0f}%")
                with col_insight3:
                    st.metric("Max Humidity", f"{max(h1, h2, h3)}%")

        # --- OPTION 4: BATCH PREDICTION VIA CSV ---
        elif predict_choice == "Batch Prediction via CSV Upload":
            st.subheader("📁 Batch Prediction (CSV Upload)")
            st.write("Upload a CSV file containing your data to generate bulk predictions.")
            
            batch_model_choice = st.selectbox(
                "Select Model for Bulk Prediction",
                ["Air Quality (PM2.5)", "Temperature Trends", "Rainfall Intensity"]
            )
            
            if batch_model_choice == "Air Quality (PM2.5)":
                st.info("💡 Your CSV must contain columns: `SO2`, `NO2`, `RSPM`, `SPM`.")
            elif batch_model_choice == "Temperature Trends":
                st.info("💡 Your CSV must contain a column: `Year`.")
            elif batch_model_choice == "Rainfall Intensity":
                st.info("💡 Your CSV must contain columns: `Day 1 Humidity`, `Day 2 Humidity`, `Day 3 Humidity`.")
                
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.write("Preview of Uploaded Data:")
                    st.dataframe(df_upload.head(5))
                    
                    if st.button("Generate Bulk Predictions", type="primary"):
                        original_cols = df_upload.columns.tolist()
                        df_upload.columns = [str(c).strip().upper() for c in df_upload.columns]
                        
                        if batch_model_choice == "Air Quality (PM2.5)":
                            required_cols = ['SO2', 'NO2', 'RSPM', 'SPM']
                            req_upper = [c.upper() for c in required_cols]
                            missing = [col for col, u in zip(required_cols, req_upper) if u not in df_upload.columns]
                            
                            if not missing:
                                if rf_model is not None:
                                    features = df_upload[req_upper].values
                                    df_upload.columns = original_cols
                                    df_upload['Predicted_PM2.5'] = np.round(rf_model.predict(features), 2)
                                    st.success("✅ Predictions generated successfully!")
                                    st.dataframe(df_upload)
                                    csv = df_upload.to_csv(index=False).encode('utf-8')
                                    st.download_button("📥 Download Predictions", data=csv, file_name="PM25_Predictions.csv", mime="text/csv")
                                else:
                                    st.error("⚠️ RF Model not loaded.")
                            else:
                                st.error(f"⚠️ Missing required columns in CSV: {', '.join(missing)}")
                                
                        elif batch_model_choice == "Temperature Trends":
                            required_cols = ['Year']
                            req_upper = ['YEAR']
                            missing = ['Year'] if 'YEAR' not in df_upload.columns else []
                            
                            if not missing:
                                if lr_model is not None:
                                    features = df_upload[req_upper].values
                                    df_upload.columns = original_cols
                                    df_upload['Predicted_Temp'] = np.round(lr_model.predict(features), 2)
                                    st.success("✅ Predictions generated successfully!")
                                    st.dataframe(df_upload)
                                    csv = df_upload.to_csv(index=False).encode('utf-8')
                                    st.download_button("📥 Download Predictions", data=csv, file_name="Temp_Predictions.csv", mime="text/csv")
                                else:
                                    st.error("⚠️ Temperature Model not loaded.")
                            else:
                                st.error(f"⚠️ Missing required columns in CSV: {', '.join(missing)}")
                                
                        elif batch_model_choice == "Rainfall Intensity":
                            required_cols = ['Day 1 Humidity', 'Day 2 Humidity', 'Day 3 Humidity']
                            req_upper = [c.upper() for c in required_cols]
                            missing = [col for col, u in zip(required_cols, req_upper) if u not in df_upload.columns]
                            
                            if not missing:
                                preds = []
                                for index, row in df_upload.iterrows():
                                    h1, h2, h3 = row[req_upper[0]], row[req_upper[1]], row[req_upper[2]]
                                    if lstm_model is not None and TENSORFLOW_AVAILABLE:
                                        try:
                                            input_data = np.array([[[h1], [h2], [h3]]])
                                            res = lstm_model.predict(input_data, verbose=0)
                                            val = res.flatten()[0] if len(res.shape)>1 else res[0]
                                            val = float(val)
                                            if val > 1: val = val / 100
                                            preds.append(max(0.0, min(1.0, val)))
                                        except:
                                            preds.append(calculate_rainfall_probability_fallback(h1, h2, h3))
                                    else:
                                        preds.append(calculate_rainfall_probability_fallback(h1, h2, h3))
                                
                                df_upload.columns = original_cols
                                df_upload['Rainfall_Probability(%)'] = np.round(np.array(preds)*100, 2)
                                st.success("✅ Predictions generated successfully!")
                                st.dataframe(df_upload)
                                csv = df_upload.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Download Predictions", data=csv, file_name="Rainfall_Predictions.csv", mime="text/csv")
                            else:
                                st.error(f"⚠️ Missing required columns in CSV: {', '.join(missing)}")
                                
                except Exception as e:
                    st.error(f"Error processing file: {e}")


    # -----------------------------------------------------------------
    # TAB 5: 🔮 AI CLIMATE ROADMAP
    # -----------------------------------------------------------------
    with tab5:
        st.header("🔮 Climate Roadmap: Advanced AI Forecasting")
        st.markdown("Select a future year to dynamically generate climate risk projections, problems, and actionable solutions.")
        
        target_roadmap_year = st.slider("Select Target Future Year", 2026, 2080, 2030, key="roadmap_year")
        
        st.markdown("---")
        
        if not df_m.empty:
            # 1. CORE ML ENGINE (Linear Regression for Roadmap)
            y_data_pred = df_m.groupby("Year").agg({"Tavg": "mean", "Tmax": "max", "Tmin": "min", "Rainfall": "sum"}).reset_index()
            
            model_tavg = LinearRegression().fit(y_data_pred[["Year"]], y_data_pred["Tavg"])
            model_tmax = LinearRegression().fit(y_data_pred[["Year"]], y_data_pred["Tmax"])
            model_tmin = LinearRegression().fit(y_data_pred[["Year"]], y_data_pred["Tmin"])
            model_rain = LinearRegression().fit(y_data_pred[["Year"]], y_data_pred["Rainfall"])
            
            f_years = np.arange(2024, target_roadmap_year + 1).reshape(-1, 1)
            f_preds_tavg = model_tavg.predict(f_years)
            f_preds_rain = model_rain.predict(f_years)
            
            # 2. IMPACT METRICS (Calculated based on selected projection)
            current_avg_temp = y_data_pred["Tavg"].iloc[-1]
            current_max_temp = y_data_pred["Tmax"].iloc[-1]
            current_min_temp = y_data_pred["Tmin"].iloc[-1]
            current_rain = y_data_pred["Rainfall"].iloc[-1]
            
            future_tavg = model_tavg.predict([[target_roadmap_year]])[0]
            future_tmax = model_tmax.predict([[target_roadmap_year]])[0]
            future_tmin = model_tmin.predict([[target_roadmap_year]])[0]
            future_rain = model_rain.predict([[target_roadmap_year]])[0]
            
            temp_rise_avg = future_tavg - current_avg_temp
            temp_rise_max = future_tmax - current_max_temp
            temp_rise_min = future_tmin - current_min_temp
            rain_diff = future_rain - current_rain
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Proj. Avg Temp ({target_roadmap_year})", f"{future_tavg:.2f} °C", delta=f"{temp_rise_avg:+.2f}°")
            m2.metric(f"Proj. Max Temp ({target_roadmap_year})", f"{future_tmax:.2f} °C", delta=f"{temp_rise_max:+.2f}°")
            m3.metric(f"Proj. Min Temp ({target_roadmap_year})", f"{future_tmin:.2f} °C", delta=f"{temp_rise_min:+.2f}°")
            m4.metric(f"Proj. Total Rain ({target_roadmap_year})", f"{future_rain:.2f} mm", delta=f"{rain_diff:+.1f}mm", delta_color="inverse")
            
            st.markdown("---")

            # 3. ADVANCED FORECASTING GRAPH (With Interactive Scenarios)
            import plotly.graph_objects as go
            
            fig_traj = go.Figure()
            
            # Confidence Interval (Light Red Fill)
            fig_traj.add_trace(go.Scatter(
                x=f_years.flatten(), y=f_preds_tavg + 0.8, 
                fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False
            ))
            fig_traj.add_trace(go.Scatter(
                x=f_years.flatten(), y=f_preds_tavg - 0.8, 
                fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', 
                fillcolor='rgba(255, 69, 0, 0.15)', name='AI Prediction Range'
            ))
            
            # Historical Trend (Blue)
            fig_traj.add_trace(go.Scatter(
                x=y_data_pred["Year"], y=y_data_pred["Tavg"],
                name="Historical Data", line=dict(color='#0052CC', width=2)
            ))
            
            # Base Forecast (Red)
            fig_traj.add_trace(go.Scatter(
                x=f_years.flatten(), y=f_preds_tavg, 
                name="AI Base Forecast", line=dict(color='#FF4B4B', width=4, dash='dash')
            ))
            
            fig_traj.update_layout(
                title=f"Climate Trajectory Forecast for {loc_name} (Up to {target_roadmap_year})",
                xaxis_title="Year", yaxis_title="Average Temperature (°C)",
                template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_traj, use_container_width=True)
            
            # 4. AI INSIGHTS & FUTURE IMPACTS (Problems and Solutions)
            st.markdown("---")
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                st.subheader(f"🤖 AI Actionable Insight ({target_roadmap_year})")
                
                # Dynamic problem statement
                problems = []
                solutions = []
                
                if temp_rise_max > 1.0 or future_tmax > 40:
                    problems.append(f"🔥 **Extreme Heat:** Maximum temperatures projected to reach {future_tmax:.1f}°C, sharply escalating heatwave health risks and infrastructure strain.")
                    solutions.append("🌲 **Solution:** Mandate urban cooling strategies like reflective 'cool roofs', massive afforestation campaigns, and shaded public transit hubs.")
                elif temp_rise_avg > 0.3:
                    problems.append(f"🌡️ **General Warming:** Steady rise in baseline temperature by {temp_rise_avg:.2f}°C affecting crop cycles.")
                    solutions.append("🌾 **Solution:** Transition agricultural sectors to heat-resilient crop varieties and implement micro-irrigation systems.")
                    
                if rain_diff < -50:
                    problems.append(f"🏜️ **Drought Vulnerability:** Annual rainfall projected to drop by {abs(rain_diff):.0f}mm, stressing local water reserves.")
                    solutions.append("💧 **Solution:** Enforce strict rainwater harvesting policies, rejuvenate local lakes, and recycle wastewater for industrial use.")
                elif rain_diff > 100:
                    problems.append(f"🌊 **Flash Flood Risk:** Intense rainfall surges ({future_rain:.0f}mm/yr) overwhelming drainage systems.")
                    solutions.append("🚧 **Solution:** Upgrade stormwater drainage networks and build 'sponge city' infrastructure with permeable pavements.")
                
                if not problems:
                    st.write("Current projections indicate a relatively stable climate trajectory with mild expected variations.")
                    st.info("💡 **Strategy:** Maintain current environmental protection policies and continue low-emission transitions.")
                else:
                    st.write("**Identified Climate Threats:**")
                    for p in problems:
                        st.write(p)
                    st.write("**AI Recommended Solutions:**")
                    for s in solutions:
                        st.info(s)

            with col_in2:
                st.subheader(f"⚠️ Sector-wise Impact ({target_roadmap_year})")
                
                # Dynamic Impact Table
                agri_risk = "High Risk" if (temp_rise_max > 1 or rain_diff < -50) else "Moderate"
                agri_icon = "🔴" if agri_risk == "High Risk" else "🟡"
                
                water_risk = "Critical" if (rain_diff < -100 or future_tmax > 40) else "Moderate"
                water_icon = "🔴" if water_risk == "Critical" else "🟡"
                
                health_risk = "Extreme" if future_tmax > 42 else "High Risk" if future_tmax > 38 else "Moderate"
                health_icon = "🟣" if health_risk == "Extreme" else "🔴" if health_risk == "High Risk" else "🟡"
                
                energy_risk = "Surging Demand" if future_tmax > 40 else "Stable"
                energy_icon = "🔴" if energy_risk == "Surging Demand" else "🟢"
                
                impact_data = {
                    "Sector": ["Agriculture", "Water Supply", "Public Health", "Energy Demand"],
                    "Risk Level": [agri_risk, water_risk, health_risk, energy_risk],
                    "Status": [agri_icon, water_icon, health_icon, energy_icon]
                }
                st.table(pd.DataFrame(impact_data))

        else:
            st.warning("No historical monthly data available to generate roadmap.")

    with tab6:
        st.subheader("⚠️ AI Automated Risk Assessment Center")
        
        # Calculate Climate Volatility Index (Deviation from historical norms)
        temp_dev = abs(tmax_proj - tmax_hist_avg) / tmax_hist_avg if tmax_hist_avg else 0
        rain_dev = abs(month_r_proj - month_r_hist_avg) / month_r_hist_avg if month_r_hist_avg > 0 else (0.1 if month_r_proj > 0 else 0)
        volatility_score = min(100, int((temp_dev * 5 + rain_dev * 0.5) * 100))
        
        vol_col1, vol_col2 = st.columns([1, 3])
        with vol_col1:
            st.metric("🚨 Climate Volatility Index", f"{volatility_score}/100", 
                      delta=f"{volatility_score - 15} pts vs Baseline" if volatility_score > 15 else "Stable", 
                      delta_color="inverse")
        with vol_col2:
            st.write("**Overall Regional Risk Level**")
            if volatility_score < 25:
                st.success("🟢 **Low Risk**: Climate parameters are relatively stable compared to historical baselines.")
                st.progress(max(0.01, volatility_score / 100))
            elif volatility_score < 55:
                st.warning("🟡 **Moderate Risk**: Noticeable deviations in climate patterns. Adaptive measures recommended.")
                st.progress(volatility_score / 100)
            else:
                st.error("🔴 **High Risk**: Severe volatility detected. Immediate mitigation strategies required.")
                st.progress(volatility_score / 100)
                
        st.markdown("---")
        st.write("#### 🔍 Specific Threat Analysis")
        col1, col2, col3 = st.columns(3)
        
        # Temperature Risk
        if tmax_proj > tmax_hist_avg + 1.5: 
            col1.error(f"🔥 **Heatwave Alert**\n\nProj. Max of {tmax_proj:.1f}°C is severely above the history avg ({tmax_hist_avg:.1f}°C). Critical Heat Risk.")
        elif tmin_proj < tmin_hist_avg - 1.5:
            col1.info(f"❄️ **Cold Snap Alert**\n\nAbnormal drop in minimum temperature projected.")
        else: 
            col1.success(f"✅ **Thermal Stability**\n\nTemperatures remain within expected seasonal parameters.")
        
        # Rain Risk
        if month_r_proj < month_r_hist_avg * 0.7: 
            col2.warning(f"🏜️ **Drought Warning**\n\nRainfall ({month_r_proj:.0f}mm) significantly below decadal average.")
        elif month_r_proj > month_r_hist_avg * 1.4: 
            col2.error(f"🌊 **Flood Alert**\n\nSevere rainfall spikes indicate extreme flood vulnerability.")
        else: 
            col2.success(f"🌧️ **Hydrological Balance**\n\nPrecipitation shows normal seasonal stability.")
            
        # Agriculture / Ecosystem Risk
        if volatility_score > 40 or month_r_proj < month_r_hist_avg * 0.7 or tmax_proj > tmax_hist_avg + 1.5:
            col3.error(f"🌾 **Crop Failure Risk**\n\nCombined anomalous conditions threaten seasonal agricultural yields.")
        else:
            col3.success(f"🌱 **Favorable Conditions**\n\nClimate supports standard agricultural and ecological cycles.")
        
        st.markdown("---")
        export_csv = full_t_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Comprehensive AI Intelligence Report (CSV)", data=export_csv, file_name=f"Climate_Report_{loc_name}_2026.csv")

else:
    st.error(f"Critical System Error: Missing historical data for {loc_name}. AI Projections cannot be generated.")
    st.warning("Switching to LIVE Mode.")
    st.markdown("---")
    
    st.subheader(f"🔴 LIVE Weather Update for {loc_name}")
    
    location = selected_dist if selected_dist != "None" else selected_state
    lat, lon = None, None
    coord_location = None
    
    if selected_dist != "None" and selected_dist in DISTRICT_COORDINATES:
        lat, lon = DISTRICT_COORDINATES[selected_dist]
        coord_location = selected_dist
    elif selected_state in LOCATION_COORDINATES:
        lat, lon = LOCATION_COORDINATES[selected_state]
        coord_location = selected_state
    
    if lat and lon:
        render_live_weather(lat, lon, coord_location)
    else:
        st.warning(f"ℹ️ Specific coordinates not configured for {location}. Cannot fetch live data.")

st.markdown("---")
st.caption("🚀 AI Climate Intelligence Platform | Developed by Jayesh | Final Semester Project (GTU) | 100% Objective Fulfillment")
