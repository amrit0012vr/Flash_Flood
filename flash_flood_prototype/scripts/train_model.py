from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n = 3000 

rain_1h = rng.gamma(2.2, 12, n).clip(0, 160)
rain_3h = (rain_1h * rng.uniform(1.5, 3.2, n) + rng.normal(0, 8, n)).clip(0, 300)
rain_6h = (rain_3h * rng.uniform(1.4, 2.4, n) + rng.normal(0, 12, n)).clip(0, 500)
rain_24h = (rain_6h * rng.uniform(2.0, 5.0, n) + rng.normal(0, 25, n)).clip(0, 1000)
soil = rng.beta(4, 2, n) * 100
slope = rng.uniform(5, 55, n)
elevation = rng.uniform(300, 3000, n)
river_dist = rng.uniform(50, 3000, n)
historical = rng.integers(0, 10, n)

# Synthetic demonstration label: intentionally nonlinear and noisy.
risk_score = (
    0.32 * np.clip(rain_1h / 100, 0, 1)
    + 0.20 * np.clip(rain_3h / 180, 0, 1)
    + 0.15 * np.clip(rain_24h / 600, 0, 1)
    + 0.14 * (soil / 100)
    + 0.08 * np.clip(slope / 45, 0, 1)
    + 0.06 * np.clip((1200 - river_dist) / 1200, 0, 1)
    + 0.05 * (historical / 10)
    + rng.normal(0, 0.035, n)
)
y = pd.cut(risk_score, bins=[-np.inf, 0.30, 0.56, np.inf],
           labels=["LOW", "MEDIUM", "HIGH"]).astype(str)

df = pd.DataFrame({
    "rain_1h_mm": rain_1h,
    "rain_3h_mm": rain_3h,
    "rain_6h_mm": rain_6h,
    "rain_24h_mm": rain_24h,
    "soil_moisture_pct": soil,
    "slope_deg": slope,
    "elevation_m": elevation,
    "distance_to_river_m": river_dist,
    "historical_events": historical,
    "risk": y
})
df.to_csv(DATA / "training_data.csv", index=False)

features = [c for c in df.columns if c != "risk"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["risk"], test_size=0.2, random_state=42, stratify=df["risk"]
)

model = RandomForestClassifier(
    n_estimators=250, max_depth=12, min_samples_leaf=3,
    random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

joblib.dump({"model": model, "features": features}, MODELS / "flood_model.joblib")
meta = {
    "accuracy": round(float(acc), 4),
    "features": features,
    "warning": "Synthetic demonstration dataset. Not for operational use."
}
(MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))

print(f"Training complete. Test accuracy: {acc:.3f}")
print(classification_report(y_test, pred))
print(f"Saved: {MODELS / 'flood_model.joblib'}")
