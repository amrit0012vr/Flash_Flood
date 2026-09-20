# Flash Flood Prediction System — End-to-End Prototype

This is a hackathon-ready software prototype for Problem Statement 26192.

## Architecture
- Python + scikit-learn: ML prediction
- FastAPI: backend REST API
- Streamlit: dashboard
- Folium: interactive risk map
- Simulated IoT sensors: real-time soil-moisture/rainfall input
- CSV/JSON: lightweight local storage

## 1. Create environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

## 2. Generate training data and train model
```bash
python scripts/train_model.py
```

This creates:
- data/training_data.csv
- models/flood_model.joblib
- models/model_meta.json

## 3. Start backend
Open Terminal 1:
```bash
uvicorn backend.main:app --reload --port 8000
```

API:
- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

## 4. Start dashboard
Open Terminal 2:
```bash
streamlit run frontend/app.py
```

Open the URL shown by Streamlit, normally:
http://localhost:8501

## 5. Simulate live sensors
Open Terminal 3:
```bash
python scripts/sensor_simulator.py
```

The simulator posts changing sensor values to FastAPI. Refresh the dashboard to see updated values.

## Demo flow
1. Open dashboard.
2. Select a village.
3. Use the sidebar to simulate rainfall and soil moisture.
4. Click "Send simulated sensor reading".
5. Click "Refresh live data".
6. Watch the risk score and map change.
7. Demonstrate LOW -> MEDIUM -> HIGH conditions.

## Important
The training data in this prototype is synthetic for demonstration. Do NOT claim that it is operationally validated or suitable for real evacuation decisions. For a real deployment, replace it with validated IMD/satellite/DEM/IoT/disaster datasets and conduct rigorous calibration and field validation.
