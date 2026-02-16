from fastapi import FastAPI
import requests
import math
import os
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from fastapi.responses import JSONResponse

from .predict import predict_leak
from .prescribe import get_prescription

app = FastAPI()

# =====================================================
# ENVIRONMENT VARIABLES
# =====================================================
CHANNEL_ID = os.getenv("CHANNEL_ID")
READ_API_KEY = os.getenv("READ_API_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING"
)

RAW_CONTAINER = "digital-twin-raw"
PROCESSED_CONTAINER = "digital-twin-processed"

if not CHANNEL_ID or not READ_API_KEY:
    raise ValueError("Missing ThingSpeak environment variables")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("Missing Azure Storage connection string")

# =====================================================
# AZURE BLOB SETUP
# =====================================================
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)

raw_container_client = blob_service_client.get_container_client(
    RAW_CONTAINER
)

processed_container_client = blob_service_client.get_container_client(
    PROCESSED_CONTAINER
)

for container in [raw_container_client, processed_container_client]:
    try:
        container.create_container()
    except:
        pass

# =====================================================
# THINGSPEAK CONFIG
# =====================================================
THINGSPEAK_URL = (
    f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    f"?api_key={READ_API_KEY}&results=1"
)

DEFAULT_PRESSURE = 45.0
DEFAULT_FLOW = 100.0

SENSOR_CONFIG = [
    {"sensor_id": 1, "pressure_field": "field1", "flow_field": "field2"},
    {"sensor_id": 2, "pressure_field": "field3", "flow_field": "field4"},
    {"sensor_id": 3, "pressure_field": "field5", "flow_field": "field6"},
]

# =====================================================
# UTILITIES
# =====================================================
def safe_float(x, default):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except:
        return default


def clean_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                out[k] = 0.0
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def save_to_blob(container_client, filename, data: dict):
    blob_client = container_client.get_blob_client(filename)
    blob_client.upload_blob(
        json.dumps(data, indent=2),
        overwrite=True
    )

# =====================================================
# CORE DIGITAL TWIN FUNCTION
# =====================================================
def run_digital_twin():
    try:
        r = requests.get(THINGSPEAK_URL, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "feeds" not in data or len(data["feeds"]) == 0:
            return {"error": "No feeds available from ThingSpeak"}

        feed = data["feeds"][-1]

    except Exception as e:
        return {"error": f"ThingSpeak error: {str(e)}"}

    timestamp = datetime.utcnow().isoformat()
    filename_time = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    raw_output = {
        "timestamp": timestamp,
        "channel_id": CHANNEL_ID,
        "sensors": {}
    }

    processed_output = {
        "timestamp": timestamp,
        "channel_id": CHANNEL_ID,
        "sensors": []
    }

    try:
        for sensor in SENSOR_CONFIG:

            sensor_id = sensor["sensor_id"]

            pressure = safe_float(
                feed.get(sensor["pressure_field"]),
                DEFAULT_PRESSURE
            )

            flow = safe_float(
                feed.get(sensor["flow_field"]),
                DEFAULT_FLOW
            )

            raw_output["sensors"][f"sensor_{sensor_id}"] = {
                "pressure": pressure,
                "flow": flow
            }

            result = predict_leak(pressure, flow)

            leak_lpm = safe_float(result.get("leak_lpm"), 0)
            leak_mm = safe_float(result.get("leak_mm"), 0)
            prob = safe_float(result.get("prob"), 0)

            prescription = {
                "severity": "Normal",
                "action_type": "No action required",
                "priority": 0
            }

            if result.get("leak") == 1:
                size_ratio = leak_mm / 1000
                mag_ratio = leak_lpm / 10800
                pres = get_prescription(size_ratio, mag_ratio)
                prescription = clean_dict(pres)

            processed_output["sensors"].append(clean_dict({
                "sensor_id": sensor_id,
                "pressure": round(pressure, 2),
                "flow": round(flow, 2),
                "leak": int(result.get("leak", 0)),
                "probability": round(prob, 4),
                "leak_lpm": leak_lpm,
                "leak_mm": leak_mm,
                "prescription": prescription
            }))

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

    try:
        save_to_blob(
            raw_container_client,
            f"{filename_time}_raw.json",
            raw_output
        )

        save_to_blob(
            processed_container_client,
            f"{filename_time}_processed.json",
            processed_output
        )

    except Exception as e:
        return {"error": f"Azure Blob error: {str(e)}"}

    return processed_output # 👈 IMPORTANT

# =====================================================
# API ROUTES
# =====================================================
@app.get("/")
def home():
    return {"status": "Digital Twin API Running"}

@app.get("/live")
def live_trigger():
    result = run_digital_twin()
    return JSONResponse(content=result)

    
