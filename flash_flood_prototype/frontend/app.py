from pathlib import Path
import requests
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

API = ("https://flash-flood.onrender.com") 

st.set_page_config(
    page_title="Flash Flood Early Warning",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Flash Flood Prediction & Early Warning System")
st.caption("Problem Statement 26192 • Hyper-local hilly-region risk prototype")

def api_get(path):
    try:
        r = requests.get(API + path, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error("Backend is not running. Start: uvicorn backend.main:app --reload --port 8000")
        st.stop()

villages = api_get("/villages")
names = [v["village"] for v in villages]
 
with st.sidebar:
    st.link_button("⚡ Open Flagship Command Center", "http://127.0.0.1:8000", use_container_width=True)
    st.divider()
    st.header("🎛️ Controls")
    selected = st.selectbox("Select village", names)
    st.divider()
    st.write("Simulate a live sensor reading")
    rain1 = st.slider("Rainfall — last 1 hour (mm)", 0.0, 200.0, 30.0, 1.0)
    rain3 = st.slider("Rainfall — last 3 hours (mm)", 0.0, 400.0, 65.0, 1.0)
    rain6 = st.slider("Rainfall — last 6 hours (mm)", 0.0, 700.0, 100.0, 1.0)
    rain24 = st.slider("Rainfall — last 24 hours (mm)", 0.0, 1200.0, 180.0, 1.0)
    soil = st.slider("Soil moisture (%)", 0.0, 100.0, 55.0, 1.0)

    if st.button("📡 Send simulated sensor reading", use_container_width=True):
        payload = {
            "village": selected,
            "rain_1h_mm": rain1,
            "rain_3h_mm": rain3,
            "rain_6h_mm": rain6,
            "rain_24h_mm": rain24,
            "soil_moisture_pct": soil
        }
        try:
            rr = requests.post(API + "/sensor-data", json=payload, timeout=5)
            rr.raise_for_status()
            st.success("Sensor data accepted.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not send data: {e}")

    if st.button("🔄 Refresh live data", use_container_width=True):
        st.rerun()

data = api_get(f"/risk/{selected}")
risk = data["risk"]
prob = data["high_risk_probability_pct"]
lead = data["estimated_warning_window_hours"]
terrain = data["terrain"]
sensor = data["sensor_data"]

if risk == "HIGH":
    icon = "🔴"
elif risk == "MEDIUM":
    icon = "🟠"
else:
    icon = "🟢"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current risk", f"{icon} {risk}")
c2.metric("High-risk probability", f"{prob}%")
c3.metric("Estimated warning window", f"{lead} h")
c4.metric("Soil moisture", f'{sensor["soil_moisture_pct"]:.0f}%')

if risk == "HIGH":
    st.error("🚨 HIGH FLASH-FLOOD RISK — " + data["action"])
elif risk == "MEDIUM":
    st.warning("⚠️ MEDIUM RISK — " + data["action"])
else:
    st.success("✅ LOW RISK — " + data["action"])

left, right = st.columns([1.2, 1])

with left:
    st.subheader("🗺️ Hyper-local risk map")
    m = folium.Map(
        location=[30.35, 78.08],
        zoom_start=11,
        tiles="OpenStreetMap"
    )
    colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
    all_risks = api_get("/risk")
    for item in all_risks:
        t = item["terrain"]
        r = item["risk"]
        popup = (
            f"<b>{item['village']}</b><br>"
            f"Risk: {r}<br>"
            f"High-risk probability: {item['high_risk_probability_pct']}%<br>"
            f"Slope: {t['slope_deg']}°<br>"
            f"Elevation: {t['elevation_m']} m"
        )
        folium.CircleMarker(
            location=[t["latitude"], t["longitude"]],
            radius=10 if item["village"] == selected else 7,
            popup=popup,
            tooltip=f"{item['village']} — {r}",
            color=colors[r],
            fill=True,
            fill_opacity=0.75
        ).add_to(m)
    st_folium(m, width=None, height=520)

with right:
    st.subheader("📊 Current conditions")
    metrics = pd.DataFrame({
        "Feature": ["Rainfall 1h", "Rainfall 3h", "Rainfall 6h", "Rainfall 24h",
                    "Soil moisture", "Slope"],
        "Value": [
            sensor["rain_1h_mm"], sensor["rain_3h_mm"], sensor["rain_6h_mm"],
            sensor["rain_24h_mm"], sensor["soil_moisture_pct"], terrain["slope_deg"]
        ],
        "Unit": ["mm", "mm", "mm", "mm", "%", "°"]
    })
    st.dataframe(metrics, hide_index=True, use_container_width=True)

    st.subheader("🤖 Model probability")
    probs = data["probabilities"]
    fig = go.Figure(go.Bar(
        x=list(probs.keys()),
        y=[probs[k] * 100 for k in probs.keys()],
        text=[f"{probs[k]*100:.1f}%" for k in probs.keys()],
        textposition="auto"
    ))
    fig.update_yaxes(range=[0, 100], title="Probability (%)")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🧠 Why this location is at risk")
reason_rows = [
    ("Rainfall — 1h", sensor["rain_1h_mm"], "Higher intensity increases runoff"),
    ("Soil moisture", sensor["soil_moisture_pct"], "Saturated soil reduces infiltration"),
    ("Slope", terrain["slope_deg"], "Steeper terrain can accelerate runoff"),
    ("Distance to river", terrain["distance_to_river_m"], "Closer drainage channels can increase exposure"),
    ("Historical events", terrain["historical_events"], "Past events indicate local vulnerability"),
]
st.table(pd.DataFrame(reason_rows, columns=["Factor", "Value", "Interpretation"]))

st.caption(
    "Prototype disclaimer: predictions use synthetic training data and simulated sensors. "
    "This dashboard is for demonstration and must not be used for real evacuation decisions."
)
