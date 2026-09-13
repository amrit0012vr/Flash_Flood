from pathlib import Path
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Flash Flood AI Early Warning Command Center",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Hackathon Command Center UI
st.markdown("""
<style>
    /* Metric Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Risk Badge glow indicators */
    .badge-low {
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.4);
        background: rgba(34, 197, 94, 0.12);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        background: rgba(245, 158, 11, 0.12);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        background: rgba(239, 68, 68, 0.15);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        animation: pulse-border 1.5s infinite;
    }

    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Alert Banner */
    .alert-banner-high {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.25), rgba(185, 28, 28, 0.15));
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .alert-banner-medium {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.22), rgba(180, 83, 9, 0.12));
        border-left: 6px solid #f59e0b;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .alert-banner-low {
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.2), rgba(21, 128, 61, 0.1));
        border-left: 6px solid #22c55e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    /* Live status dot */
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #22c55e;
    }
</style>
""", unsafe_allow_html=True)

# Helper API functions
def api_get(path):
    try:
        r = requests.get(API + path, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def api_post(path, payload):
    try:
        r = requests.post(API + path, json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

# Check backend health
villages = api_get("/villages")

if villages is None:
    st.error("### 🔌 Backend Disconnected")
    st.markdown("""
    The FastAPI prediction service is currently offline. Start it in a terminal:
    ```bash
    cd flash_flood_prototype
    uvicorn backend.main:app --reload --port 8000
    ```
    """)
    if st.button("🔄 Retry Connection", use_container_width=False):
        st.rerun()
    st.stop()

# Header Section
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("🌊 Flash Flood Early Warning Command Center")
    st.caption("AI-Powered Real-time Risk Prediction & Telemetry • Problem Statement 26192")

with col_status:
    st.markdown("""
        <div style="text-align: right; padding-top: 15px;">
            <span class="live-indicator"></span><strong style="color: #22c55e;">TELEMETRY ACTIVE</strong><br>
            <span style="font-size: 0.8rem; color: #94a3b8;">Local AI Inference Node: Online</span>
        </div>
    """, unsafe_allow_html=True)

names = [v["village"] for v in villages]

# Initialize session state for sliders if not present
default_village = names[0] if names else "Village A"
if "village" not in st.session_state:
    st.session_state.village = default_village
if "rain1" not in st.session_state:
    st.session_state.rain1 = 30.0
if "rain3" not in st.session_state:
    st.session_state.rain3 = 65.0
if "rain6" not in st.session_state:
    st.session_state.rain6 = 100.0
if "rain24" not in st.session_state:
    st.session_state.rain24 = 180.0
if "soil" not in st.session_state:
    st.session_state.soil = 55.0

# Sidebar with 1-Click Simulation Presets
with st.sidebar:
    st.header("🎛️ Disaster Simulator")
    selected = st.selectbox("🎯 Target Village", names, index=names.index(st.session_state.village) if st.session_state.village in names else 0)
    st.session_state.village = selected

    st.markdown("---")
    st.subheader("⚡ 1-Click Demo Scenarios")
    st.caption("Instantly test how the AI responds to extreme weather events:")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("☀️ Normal Weather", use_container_width=True):
            st.session_state.rain1 = 4.0
            st.session_state.rain3 = 8.0
            st.session_state.rain6 = 15.0
            st.session_state.rain24 = 25.0
            st.session_state.soil = 32.0
            payload = {
                "village": selected,
                "rain_1h_mm": 4.0, "rain_3h_mm": 8.0,
                "rain_6h_mm": 15.0, "rain_24h_mm": 25.0,
                "soil_moisture_pct": 32.0
            }
            api_post("/sensor-data", payload)
            st.rerun()

        if st.button("🌧️ Heavy Monsoon", use_container_width=True):
            st.session_state.rain1 = 45.0
            st.session_state.rain3 = 85.0
            st.session_state.rain6 = 130.0
            st.session_state.rain24 = 210.0
            st.session_state.soil = 68.0
            payload = {
                "village": selected,
                "rain_1h_mm": 45.0, "rain_3h_mm": 85.0,
                "rain_6h_mm": 130.0, "rain_24h_mm": 210.0,
                "soil_moisture_pct": 68.0
            }
            api_post("/sensor-data", payload)
            st.rerun()

    with p_col2:
        if st.button("⛈️ Cloudburst Emergency", use_container_width=True):
            st.session_state.rain1 = 135.0
            st.session_state.rain3 = 240.0
            st.session_state.rain6 = 320.0
            st.session_state.rain24 = 480.0
            st.session_state.soil = 92.0
            payload = {
                "village": selected,
                "rain_1h_mm": 135.0, "rain_3h_mm": 240.0,
                "rain_6h_mm": 320.0, "rain_24h_mm": 480.0,
                "soil_moisture_pct": 92.0
            }
            api_post("/sensor-data", payload)
            st.rerun()

        if st.button("⛰️ Slope Saturated", use_container_width=True):
            st.session_state.rain1 = 75.0
            st.session_state.rain3 = 140.0
            st.session_state.rain6 = 190.0
            st.session_state.rain24 = 280.0
            st.session_state.soil = 85.0
            payload = {
                "village": selected,
                "rain_1h_mm": 75.0, "rain_3h_mm": 140.0,
                "rain_6h_mm": 190.0, "rain_24h_mm": 280.0,
                "soil_moisture_pct": 85.0
            }
            api_post("/sensor-data", payload)
            st.rerun()

    st.markdown("---")
    st.subheader("Manual Sensor Sliders")
    rain1 = st.slider("Rainfall — last 1 hour (mm)", 0.0, 200.0, float(st.session_state.rain1), 1.0)
    rain3 = st.slider("Rainfall — last 3 hours (mm)", 0.0, 400.0, float(st.session_state.rain3), 1.0)
    rain6 = st.slider("Rainfall — last 6 hours (mm)", 0.0, 700.0, float(st.session_state.rain6), 1.0)
    rain24 = st.slider("Rainfall — last 24 hours (mm)", 0.0, 1200.0, float(st.session_state.rain24), 1.0)
    soil = st.slider("Soil moisture saturation (%)", 0.0, 100.0, float(st.session_state.soil), 1.0)

    st.session_state.rain1 = rain1
    st.session_state.rain3 = rain3
    st.session_state.rain6 = rain6
    st.session_state.rain24 = rain24
    st.session_state.soil = soil

    if st.button("📡 Push Live Sensor Readings", use_container_width=True, type="primary"):
        payload = {
            "village": selected,
            "rain_1h_mm": rain1,
            "rain_3h_mm": rain3,
            "rain_6h_mm": rain6,
            "rain_24h_mm": rain24,
            "soil_moisture_pct": soil
        }
        res = api_post("/sensor-data", payload)
        if res:
            st.success("Telemetry synchronized with inference engine!")
            st.rerun()
        else:
            st.error("Failed to push telemetry.")

    if st.button("🔄 Refresh Data Feed", use_container_width=True):
        st.rerun()

# Fetch active risk prediction for the selected village
data = api_get(f"/risk/{selected}")
if not data:
    st.warning("No prediction data received.")
    st.stop()

risk = data["risk"]
prob = data["high_risk_probability_pct"]
lead = data["estimated_warning_window_hours"]
terrain = data["terrain"]
sensor = data["sensor_data"]
probs = data["probabilities"]

# Banner Alert Section
if risk == "HIGH":
    st.markdown(f"""
        <div class="alert-banner-high">
            <h3 style="margin: 0; color: #ef4444; display: flex; align-items: center; gap: 8px;">
                🚨 CRITICAL WARNING: HIGH FLASH-FLOOD HAZARD IMMINENT
            </h3>
            <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: #fecaca;">
                <strong>Emergency Action Directive:</strong> {data["action"]}
            </p>
        </div>
    """, unsafe_allow_html=True)
elif risk == "MEDIUM":
    st.markdown(f"""
        <div class="alert-banner-medium">
            <h3 style="margin: 0; color: #f59e0b; display: flex; align-items: center; gap: 8px;">
                ⚠️ WATCH ADVISORY: ELEVATED RISK OF RUNOFF & INUNDATION
            </h3>
            <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: #fef3c7;">
                <strong>Precautionary Measure:</strong> {data["action"]}
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="alert-banner-low">
            <h3 style="margin: 0; color: #22c55e; display: flex; align-items: center; gap: 8px;">
                ✅ NORMAL MONITORING: CONDITIONS STABLE
            </h3>
            <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: #dcfce7;">
                <strong>Standard Routine:</strong> {data["action"]}
            </p>
        </div>
    """, unsafe_allow_html=True)

# 4 Key Hero Metric Cards
badge_class = f"badge-{risk.lower()}"
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Threat Assessment</div>
            <div class="metric-value"><span class="{badge_class}">{risk}</span></div>
            <div class="metric-sub">Location: {selected}</div>
        </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Flood Probability (High)</div>
            <div class="metric-value" style="color: {'#ef4444' if prob >= 60 else '#f59e0b' if prob >= 30 else '#22c55e'};">{prob}%</div>
            <div class="metric-sub">Confidence: 85.7% (Macro F1: 0.81)</div>
        </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Warning Lead-Time</div>
            <div class="metric-value" style="color: {'#ef4444' if lead <= 1.5 else '#38bdf8'};">{lead} hrs</div>
            <div class="metric-sub">Window before crest formation</div>
        </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Soil Saturation</div>
            <div class="metric-value" style="color: {'#ef4444' if sensor['soil_moisture_pct'] > 80 else '#38bdf8'};">{sensor["soil_moisture_pct"]:.0f}%</div>
            <div class="metric-sub">Threshold limit: 75%</div>
        </div>
    """, unsafe_allow_html=True)

# Main Visual Workspace (Interactive Map & Plotly Gauges)
left_view, right_view = st.columns([1.3, 1])

with left_view:
    st.subheader("🗺️ Regional Terrain & Hazard Heatmap")

    # Folium Map with multiple base tile layers
    map_lat = terrain["latitude"]
    map_lon = terrain["longitude"]
    m = folium.Map(
        location=[map_lat, map_lon],
        zoom_start=12,
        tiles="CartoDB dark_matter"
    )

    colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
    all_risks = api_get("/risk") or []

    for item in all_risks:
        t = item["terrain"]
        r = item["risk"]
        is_curr = (item["village"] == selected)

        # Pulse circle radius
        radius = 16 if is_curr else 9

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; width: 170px;">
            <b style="font-size: 14px;">{item['village']}</b><br>
            <span style="color: {colors[r]}; font-weight: bold;">Risk Level: {r}</span><br>
            <b>Probability:</b> {item['high_risk_probability_pct']}%<br>
            <b>Elevation:</b> {t['elevation_m']} m<br>
            <b>Slope:</b> {t['slope_deg']}°<br>
            <b>River Distance:</b> {t['distance_to_river_m']} m
        </div>
        """

        folium.CircleMarker(
            location=[t["latitude"], t["longitude"]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{item['village']} — {r} ({item['high_risk_probability_pct']}%)",
            color=colors[r],
            fill=True,
            fill_color=colors[r],
            fill_opacity=0.85 if is_curr else 0.65,
            weight=3 if is_curr else 1
        ).add_to(m)

        # Highlight radius ripple for high risk
        if r == "HIGH":
            folium.Circle(
                location=[t["latitude"], t["longitude"]],
                radius=1800,
                color="#ef4444",
                fill=True,
                fill_color="#ef4444",
                fill_opacity=0.15,
                weight=1
            ).add_to(m)

    st_folium(m, width=None, height=520)

with right_view:
    st.subheader("📊 AI Risk Gauge & Multi-Class Confidence")

    # Plotly Gauge Chart for High Risk Probability
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Flash Flood Risk Index (%)", 'font': {'size': 16, 'color': '#cbd5e1'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': '#f8fafc'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#ffffff", 'thickness': 0.22},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#475569",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(34, 197, 94, 0.45)'},
                {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.45)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.55)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 3},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    gauge_fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc")
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Class Probability Distribution Bar Chart
    class_order = ["LOW", "MEDIUM", "HIGH"]
    class_colors = ["#22c55e", "#f59e0b", "#ef4444"]
    y_vals = [probs.get(c, 0.0) * 100 for c in class_order]

    bar_fig = go.Figure(go.Bar(
        x=class_order,
        y=y_vals,
        marker_color=class_colors,
        text=[f"{val:.1f}%" for val in y_vals],
        textposition="auto",
        opacity=0.85
    ))
    bar_fig.update_layout(
        title="Model Prediction Distribution",
        title_font=dict(size=14, color="#cbd5e1"),
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", tickfont=dict(color="#cbd5e1")),
        yaxis=dict(range=[0, 100], title="Confidence (%)", gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#cbd5e1")),
        font=dict(color="#f8fafc")
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("---")

# Deep-Dive Tabs: Telemetry, Regional Surveillance, and Emergency SOP
tab_telemetry, tab_surveillance, tab_sop = st.tabs([
    "🔬 Live Telemetry & Vulnerability Factors",
    "🏘️ Multi-Village Regional Grid",
    "📢 Disaster Response SOP & SMS Dispatch"
])

with tab_telemetry:
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.write("#### 📡 Real-time Sensor Metrics vs Safe Thresholds")
        sensor_df = pd.DataFrame([
            {"Metric": "Rainfall (1 Hour)", "Current": f"{sensor['rain_1h_mm']:.1f} mm", "Safe Baseline": "< 25.0 mm", "Status": "CRITICAL" if sensor['rain_1h_mm'] > 60 else "ELEVATED" if sensor['rain_1h_mm'] > 25 else "NORMAL"},
            {"Metric": "Rainfall (3 Hours)", "Current": f"{sensor['rain_3h_mm']:.1f} mm", "Safe Baseline": "< 50.0 mm", "Status": "CRITICAL" if sensor['rain_3h_mm'] > 120 else "ELEVATED" if sensor['rain_3h_mm'] > 50 else "NORMAL"},
            {"Metric": "Rainfall (6 Hours)", "Current": f"{sensor['rain_6h_mm']:.1f} mm", "Safe Baseline": "< 80.0 mm", "Status": "CRITICAL" if sensor['rain_6h_mm'] > 180 else "ELEVATED" if sensor['rain_6h_mm'] > 80 else "NORMAL"},
            {"Metric": "Rainfall (24 Hours)", "Current": f"{sensor['rain_24h_mm']:.1f} mm", "Safe Baseline": "< 120.0 mm", "Status": "CRITICAL" if sensor['rain_24h_mm'] > 250 else "ELEVATED" if sensor['rain_24h_mm'] > 120 else "NORMAL"},
            {"Metric": "Soil Saturation", "Current": f"{sensor['soil_moisture_pct']:.1f} %", "Safe Baseline": "< 65.0 %", "Status": "CRITICAL" if sensor['soil_moisture_pct'] > 80 else "ELEVATED" if sensor['soil_moisture_pct'] > 65 else "NORMAL"},
        ])
        st.dataframe(sensor_df, hide_index=True, use_container_width=True)

    with col_t2:
        st.write("#### ⛰️ Geo-Spatial Vulnerability Profile")
        terrain_df = pd.DataFrame([
            {"Parameter": "Terrain Slope", "Value": f"{terrain['slope_deg']}°", "Hazard Weight": "High (> 25° accelerates flash runoff)"},
            {"Parameter": "Elevation", "Value": f"{terrain['elevation_m']} m", "Hazard Weight": "Moderate (Mountain catchment zone)"},
            {"Parameter": "River Distance", "Value": f"{terrain['distance_to_river_m']} m", "Hazard Weight": "High (< 400m increases inundation exposure)"},
            {"Parameter": "Historical Floods", "Value": f"{terrain['historical_events']} events", "Hazard Weight": "Empirical risk multiplier"},
        ])
        st.dataframe(terrain_df, hide_index=True, use_container_width=True)

with tab_surveillance:
    st.write("#### 🌐 Active Surveillance Across All Settlement Sectors")
    if all_risks:
        surv_rows = []
        for v_item in all_risks:
            t = v_item["terrain"]
            surv_rows.append({
                "Village / Sector": v_item["village"],
                "Threat Level": v_item["risk"],
                "High-Risk Prob": f"{v_item['high_risk_probability_pct']}%",
                "Lead Window": f"{v_item['estimated_warning_window_hours']} h",
                "Slope": f"{t['slope_deg']}°",
                "River Distance": f"{t['distance_to_river_m']} m",
                "Last Updated": v_item.get("updated_at", "Just now")[:19].replace("T", " ")
            })
        surv_df = pd.DataFrame(surv_rows)
        st.dataframe(surv_df, hide_index=True, use_container_width=True)
    else:
        st.info("No multi-village telemetry available.")

with tab_sop:
    st.write(f"#### 🚨 Disaster Management Protocol for **{selected}** ({risk} RISK)")
    
    sop_col1, sop_col2 = st.columns([1.2, 1])
    with sop_col1:
        if risk == "HIGH":
            st.error("""
            **Priority 1 Actions (Immediate Execution):**
            - 🚨 Sound local sirens and activate automated community loudspeaker alert.
            - 🚶 Initiate mandatory evacuation for riverbank settlements within 500m of drainage channels.
            - 🚑 Mobilize SDRF / NDRF quick-response units to pre-designated assembly points.
            - 🚧 Close low-lying bridges, culverts, and vulnerable hill road sections.
            """)
        elif risk == "MEDIUM":
            st.warning("""
            **Priority 2 Actions (Standby & Heightened Readiness):**
            - ⚠️ Put local disaster response coordinators and Gram Panchayat volunteers on high alert.
            - 📦 Pre-position emergency food, drinking water, and first-aid kits at school relief camps.
            - 📡 Increase sensor monitoring frequency to 5-minute sampling intervals.
            - 📢 Issue caution bulletin advising citizens to refrain from riverbed crossings.
            """)
        else:
            st.success("""
            **Routine Monitoring Protocol:**
            - ✅ Standard automated sensor logging active every 10 seconds.
            - 💧 Regular drainage channels clear. No emergency mobilization required.
            - 📊 Routine weather bureau radar sync active.
            """)

    with sop_col2:
        st.write("##### 📲 Automated Resident Warning Broadcast Simulation")
        recipient_count = 1420 if selected == "Village A" else 890 if selected == "Village B" else 1250
        st.caption(f"Broadcast group: Registered residents in **{selected}** catchment ({recipient_count} verified numbers).")
        
        if st.button("📢 Simulate Emergency SMS Broadcast", type="secondary", use_container_width=True):
            st.toast(f"✅ Emergency SMS queued for {recipient_count} residents in {selected}!", icon="📲")
            st.code(
                f"[DISASTER ALERT - UTTARAKHAND SDMA]\n"
                f"Flash Flood Warning for {selected}: Threat Level {risk}.\n"
                f"Estimated lead time: {lead} hrs.\n"
                f"{data['action']}\n"
                f"Emergency helpline: 1070 / 112.",
                language="markdown"
            )

st.caption(
    "⚠️ **Prototype Notice:** Problem Statement 26192 demonstration project. "
    "Predictions are calculated via machine learning trained on synthetic terrain-weather correlations. "
    "Not intended for live civil protection decision making without certified field calibration."
)
