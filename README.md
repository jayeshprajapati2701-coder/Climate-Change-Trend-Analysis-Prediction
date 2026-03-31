# AI Climate Intelligence System 🌍
### Climate-Change-Trend-Analysis-Prediction

An advanced climate monitoring and forecasting dashboard built with **Streamlit**, **Machine Learning (Scikit-learn)**, and **Deep Learning (TensorFlow/LSTM)**. This system provides real-time weather updates, live Air Quality Index (AQI) tracking, and AI-driven projections for temperature and rainfall up to 2030 for Indian regions.

## 🚀 Features
- **Live Telemetry:** Real-time weather and AQI data using OpenWeatherMap API.
- **AI Projections:** Ensemble models (Random Forest, Linear Regression, LSTM) for climate forecasting.
- **Interactive Analysis:** Temporal patterns, decadal heatmaps, and historical trend analysis.
- **Risk Management:** Automated assessment of heatwaves, droughts, and flood risks.
- **Roadmap 2030:** Strategic AI-driven climate scenarios and actionable solutions.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Data Analysis:** Pandas, NumPy, Plotly
- **Machine Learning:** Scikit-Learn
- **Deep Learning:** TensorFlow (LSTM)
- **API:** OpenWeatherMap API

## 📦 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/jayeshprajapati2701-coder/Climate-Change-Trend-Analysis-Prediction.git
   cd Climate-Change-Trend-Analysis-Prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Setup Environment Variables:**
   Create a `.env` file in the root directory and add your key:
   ```env
   OPENWEATHER_API_KEY=your_api_key_here
   ```
4. **Generate ML Models (If missing):**
   ```bash
   python create_models.py
   ```
5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```