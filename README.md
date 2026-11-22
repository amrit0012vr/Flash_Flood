<div align="center">

# Flash Flood Prediction & Early Warning System

**A machine learning system for predicting localized flash-flood risk levels using environmental and simulated IoT hydrological sensor data.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-9ece6a?style=flat-square)

[⚙️ Installation](#️-installation) • [▶️ Usage](#️-usage) • [📁 Project Structure](#-project-structure)

</div>

## 🎯 Problem Statement
Flash floods develop rapidly within hours of heavy localized rainfall, catching vulnerable settlements unprepared. Traditional regional weather forecasts often lack the spatial granularity and real-time hydrological telemetry necessary to provide actionable, localized early warnings before inundation strikes.

## 💡 Solution
This project provides an end-to-end flash flood prediction and early warning prototype. It pairs an environmental machine learning classification model with a FastAPI REST inference service, a real-time Streamlit GIS dashboard with interactive Folium maps, and an IoT sensor simulator feeding dynamic rainfall and soil moisture data.

## 🧠 Methodology
```mermaid
flowchart LR
    A[Meteorological & IoT Sensor Data] --> B[Data Preprocessing & Scaling] --> C[Feature Engineering: Rain, Slope, Soil] --> D[Random Forest Classifier] --> E[Evaluation: Accuracy & Risk Score] --> F[FastAPI REST API & Streamlit Dashboard]
```
- **Data Ingestion:** Collects hydrological indicators (1h, 3h, 6h, and 24h rainfall, soil moisture percentage, slope degree, elevation, distance to river, and historical event count).
- **Feature Pipeline:** Normalized numerical features processed through a Scikit-learn classification pipeline.
- **Inference & UI:** Exposes predictions via FastAPI endpoints (`/predict`) and visualizes village-level risk categories (`LOW`, `MEDIUM`, `HIGH`) on an interactive map.

## 🛠️ Tech Stack
- **Machine Learning & Data:** Python, Scikit-learn, Pandas, NumPy, Joblib, SciPy
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Frontend Dashboard:** Streamlit, Folium, Streamlit-Folium, Plotly

## 📊 Results
| Metric / Attribute | Value |
|--------------------|-------|
| Classification Accuracy | 85.67% |
| Model Algorithm | Random Forest Classifier (`joblib`) |
| Ingestion Telemetry | 9 Hydrological and Topographical Features |
| Risk Tiers | 3 Levels (Low, Medium, High Alert) |

## ⚙️ Installation
```bash
git clone https://github.com/amrit0012vr/Flash_Flood.git
cd Flash_Flood

# Virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

### 1. Train the Model
```bash
python scripts/train_model.py
```
*Generates `models/flood_model.joblib` and `models/model_meta.json`.*

### 2. Start Backend API
```bash
uvicorn backend.main:app --reload --port 8000
```
*Swagger docs accessible at `http://127.0.0.1:8000/docs`.*

### 3. Launch Dashboard
```bash
streamlit run frontend/app.py
```
*Open `http://localhost:8501` in your browser.*

### 4. Run Simulated Live Telemetry
```bash
python scripts/sensor_simulator.py
```

## 📁 Project Structure
```text
Flash_Flood/
├── backend/                # FastAPI application and route handlers
├── data/                   # Environmental datasets and training records
├── frontend/               # Streamlit application and Folium map views
├── models/                 # Saved model weights (.joblib) and metadata (.json)
├── scripts/                # Training pipelines and IoT sensor simulators
├── requirements.txt        # Python dependency manifest
└── README.md
```

## 🔮 Future Improvements
- Integrate live satellite precipitation data (e.g. IMD / NASA GPM API feeds).
- Implement automated SMS/WhatsApp alerts for local panchayat emergency coordinators.
- Add LSTM/GRU sequence modeling for multi-hour hydrograph discharge forecasting.

## 👤 Author
**Amrit Preetam Nayak** — [GitHub](https://github.com/amrit0012vr) • [LinkedIn](https://www.linkedin.com/in/amrit-preetam-nayak-667a7232b/)
