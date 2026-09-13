from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "flood_model.joblib"
VILLAGE_PATH = ROOT / "data" / "villages.csv"
STATE_PATH = ROOT / "data" / "live_state.json"

app = FastAPI(
    title="Flash Flood Prediction API",
    version="1.0.0",
    description="Hackathon prototype for hyper-local flash flood risk prediction."
)

model_bundle = None
villages = []
live = {}

def load_state():
    global model_bundle, villages, live
    if MODEL_PATH.exists():
        model_bundle = joblib.load(MODEL_PATH)
    else:
        model_bundle = None

    import pandas as pd
    if VILLAGE_PATH.exists():
        villages = pd.read_csv(VILLAGE_PATH).to_dict("records")

    if STATE_PATH.exists():
        live = json.loads(STATE_PATH.read_text())
    else:
        for v in villages:
            live[v["village"]] = {
                "rain_1h_mm": 15.0,
                "rain_3h_mm": 30.0,
                "rain_6h_mm": 45.0,
                "rain_24h_mm": 70.0,
                "soil_moisture_pct": 45.0,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        save_state()

def save_state():
    STATE_PATH.write_text(json.dumps(live, indent=2))

load_state()

class SensorReading(BaseModel):
    village: str
    rain_1h_mm: float = Field(ge=0, le=1000)
    rain_3h_mm: float = Field(ge=0, le=2000)
    rain_6h_mm: float = Field(ge=0, le=3000)
    rain_24h_mm: float = Field(ge=0, le=5000)
    soil_moisture_pct: float = Field(ge=0, le=100)

def village_info(name):
    for v in villages:
        if v["village"] == name:
            return v
    raise HTTPException(status_code=404, detail="Village not found")

def predict(name):
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model not trained. Run scripts/train_model.py first.")
    v = village_info(name)
    s = live[name]
    features = model_bundle["features"]
    row = pd.DataFrame([{
        "rain_1h_mm": s["rain_1h_mm"],
        "rain_3h_mm": s["rain_3h_mm"],
        "rain_6h_mm": s["rain_6h_mm"],
        "rain_24h_mm": s["rain_24h_mm"],
        "soil_moisture_pct": s["soil_moisture_pct"],
        "slope_deg": v["slope_deg"],
        "elevation_m": v["elevation_m"],
        "distance_to_river_m": v["distance_to_river_m"],
        "historical_events": v["historical_events"]
    }], columns=features)
    model = model_bundle["model"]
    inv_label_order = model_bundle.get("inv_label_order", {0: "LOW", 1: "MEDIUM", 2: "HIGH"})
    
    probs = model.predict_proba(row)[0]
    classes = list(model.classes_)
    probability_map = {inv_label_order.get(c, str(c)): float(p) for c, p in zip(classes, probs)}
    
    risk_encoded = model.predict(row)[0]
    risk = inv_label_order.get(risk_encoded, str(risk_encoded))
    p_high = probability_map.get("HIGH", 0.0)
    p_medium = probability_map.get("MEDIUM", 0.0)

    # Demo lead-time heuristic. It is deliberately described as an estimate.
    intensity = (
        0.45 * min(s["rain_1h_mm"] / 100, 1)
        + 0.25 * min(s["rain_3h_mm"] / 180, 1)
        + 0.20 * s["soil_moisture_pct"] / 100
        + 0.10 * min(v["slope_deg"] / 45, 1)
    )
    lead_hours = max(0.5, round(4.0 - 3.2 * intensity, 1))

    if risk == "HIGH":
        action = "Prepare evacuation and move away from rivers, streams and low-lying areas. Follow local authority instructions."
    elif risk == "MEDIUM":
        action = "Remain alert, prepare emergency resources and monitor official warnings."
    else:
        action = "Continue monitoring. No immediate evacuation action is suggested by this prototype."

    return {
        "village": name,
        "risk": risk,
        "probabilities": probability_map,
        "high_risk_probability_pct": round(p_high * 100, 1),
        "estimated_warning_window_hours": lead_hours,
        "action": action,
        "sensor_data": s,
        "terrain": v,
        "updated_at": s["updated_at"]
    }

@app.get("/")
def root():
    return {"message": "Flash Flood Prediction API", "docs": "/docs"}

@app.get("/villages")
def get_villages():
    return villages

@app.get("/sensor-data/{village}")
def get_sensor_data(village: str):
    village_info(village)
    return live[village]

@app.post("/sensor-data")
def ingest_sensor(reading: SensorReading):
    village_info(reading.village)
    live[reading.village] = {
        "rain_1h_mm": reading.rain_1h_mm,
        "rain_3h_mm": reading.rain_3h_mm,
        "rain_6h_mm": reading.rain_6h_mm,
        "rain_24h_mm": reading.rain_24h_mm,
        "soil_moisture_pct": reading.soil_moisture_pct,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    save_state()
    return {"status": "accepted", "prediction": predict(reading.village)}

@app.get("/risk/{village}")
def get_risk(village: str):
    village_info(village)
    return predict(village)

@app.get("/risk")
def get_all_risks():
    return [predict(v["village"]) for v in villages]
