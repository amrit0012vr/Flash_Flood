import time
import random
import requests

API = "http://127.0.0.1:8000"  
VILLAGE = "Village C"

print("Starting simulated IoT sensor. Press Ctrl+C to stop.") 

while True:
    # Slowly vary conditions so the dashboard feels live.
    rain1 = random.uniform(20, 130)
    rain3 = rain1 * random.uniform(1.8, 2.8)
    rain6 = rain3 * random.uniform(1.5, 2.1)
    rain24 = rain6 * random.uniform(2.0, 4.0)
    soil = random.uniform(55, 92)

    payload = { 
        "village": VILLAGE,
        "rain_1h_mm": round(rain1, 1),
        "rain_3h_mm": round(rain3, 1),
        "rain_6h_mm": round(rain6, 1),
        "rain_24h_mm": round(rain24, 1), 
        "soil_moisture_pct": round(soil, 1),
    }

    try:
        r = requests.post(API + "/sensor-data", json=payload, timeout=5)
        print(r.json()["prediction"]["risk"], payload)
    except Exception as e:
        print("Backend unavailable:", e)

    time.sleep(10)
