from fastapi import FastAPI
import requests
import math
import os
import json
from datetime import datetime

from azure.storage.blob import BlobServiceClient

from .predict import predict_leak
from .prescribe import get_prescription

app = FastAPI()


# =============================
# Load env variables
# =============================
CHANNEL_ID = os.getenv("CHANNEL_ID")
READ_API_KEY = os.getenv("READ_API_KEY")

AZURE_STORAGE_CONNECTION_STRING = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING"
)

CONTAINER_NAME = "digital-twin-data"


if not CHANNEL_ID or not READ_API_KEY:
    raise ValueError("Missing ThingSpeak environment variables")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("Missing Azure Storage connection string")


# =============================
# Azure Blob Setup
# =============================
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)

container_client = blob_service_client.get_container_client(
    CONTAINER_NAME
)

# Create container if not exists
try:
    container_client.create_container()
except:
    pass


# =============================
# ThingSpeak URL
# =============================
THINGSPEAK_URL = (
    f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    f"?api_key={READ_API_KEY}&results=1"
)


DEFAULT_PRESSURE = 45.0
DEFAULT_FLOW = 100.0


# =============================
# Helpers
# =============================
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


# =============================
# Save to Azure Blob
# =============================
def save_to_blob(data: dict):

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{timestamp}.json"

    blob_client = container_client.get_blob_client(filename)

    blob_client.upload_blob(
        json.dumps(data, indent=2),
        overwrite=True
    )


# =============================
# Routes
# =============================
@app.get("/")
def home():
    return {"status": "Digital Twin Running"}


@app.get("/predict")
def manual_predict(pressure: float, flow: float):
    return process(pressure, flow)


@app.get("/live")
def live_predict():

    try:

        r = requests.get(THINGSPEAK_URL, timeout=10)
        data = r.json()

        feed = data["feeds"][-1]

        pressure = safe_float(
            feed.get("field1"),
            DEFAULT_PRESSURE
        )

        flow = safe_float(
            feed.get("field2"),
            DEFAULT_FLOW
        )

    except:

        pressure = DEFAULT_PRESSURE
        flow = DEFAULT_FLOW

    return process(pressure, flow)


# =============================
# Core Logic
# =============================
def process(pressure, flow):

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

    response = clean_dict({

        "timestamp": datetime.utcnow().isoformat(),

        "pressure": round(pressure, 2),
        "flow": round(flow, 2),

        "leak": int(result.get("leak", 0)),
        "probability": round(prob, 4),

        "leak_lpm": leak_lpm,
        "leak_mm": leak_mm,

        "prescription": prescription
    })

    # Save in Azure Blob
    save_to_blob(response)

    return response
