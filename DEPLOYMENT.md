# Climate Dashboard Deployment Requirements

## 1. Project Overview
A Streamlit-based climate intelligence dashboard for Indian states/districts.

- UI: `app.py`
- Model files: `models/rf_aqi_model.pkl`, `models/linear_reg_temp.pkl`, `models/lstm_rainfall/`
- Data sources: `states.csv`, `districts.csv`, plus per-state/per-district CSV folders
- Optional deep learning path: TensorFlow LSTM

---

## 2. Prerequisites

1. Python 3.11+ (3.10 also acceptable; avoid 3.7/3.8 for dependency compatibility).
2. Git (for cloning repo) or zip download.
3. 8+ GB RAM recommended for training / TensorFlow model; 2+ GB for simple runtime.

---

## 3. Required Python packages
Install from `requirements.txt`:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Unix/Mac
pip install -r requirements.txt
```

Packages needed:
- streamlit
- joblib
- pandas
- numpy
- plotly
- scikit-learn
- requests
- tensorflow (optional; required for LSTM model file load/training)

---

## 4. Data requirements

Ensure these files/folders exist at root:

- `states.csv`
- `districts.csv`

Folder structure expected by app:

- `State_AQI_CSV/`
- `State_data_Daily_CSV/`
- `State_data_Monthly_CSV/`
- `District_AQI_CSV/`
- `District_data_Daily_CSV/`
- `District_data_Monthly_CSV/`

Each file name pattern:

- `data_State_{ID}.csv`
- `data_District_{ID}.csv`

If missing, app warns and may not render full function.

---

## 5. Model files

`app.py` loads models using `load_prediction_models()`:

- `models/rf_aqi_model.pkl`
- `models/linear_reg_temp.pkl`
- `models/lstm_rainfall` (TensorFlow format, optional)

If missing, app falls back to non-ML behavior but still runs.

To generate models:

```bash
python create_models.py
```

---

## 6. Local run

```bash
streamlit run app.py
```

Access at `http://localhost:8501`.

---

## 7. Production deployment (Docker + cloud)

### Dockerfile (example)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build + run

```bash
docker build -t climate-dashboard:latest .
docker run -p 8501:8501 --name climate-dashboard -d climate-dashboard:latest
```

### Notes for cloud
- Persist `models/`, `states.csv`, `districts.csv`, and data folders via mounted volume or remote blob store.
- Provide environment variables if needed for custom config.

---

## 8. Optional Model Training (offline)

- `create_models.py` writes ML models into `models/`.
- TensorFlow model creation requires `tensorflow` and can be heavy.

---

## 9. Validation checklist before deployment

- [ ] `python --version` is 3.11+
- [ ] `pip install -r requirements.txt` successful
- [ ] `states.csv` + `districts.csv` present
- [ ] CSV folder structure exists with expected files
- [ ] `models/rf_aqi_model.pkl` and `models/linear_reg_temp.pkl` exist
- [ ] Optional: `models/lstm_rainfall/` exists
- [ ] `streamlit run app.py` starts without fatal errors
- [ ] API usage (Open-Meteo) allowed; verify network egress

---

## 10. Troubleshooting

- `Missing configuration files` = verify CSV path and file names.
- `TensorFlow not installed` = either install or rely on fallback (`calculate_rainfall_probability_fallback`).
- API timeout = increase requests timeout or disable live request block.
- permission errors = run in project directory with read/write access.
