from fastapi import FastAPI
import requests
import math

from predict import predict_leak
from prescribe import get_prescription

app = FastAPI()


CHANNEL_ID = "3149051"
READ_API_KEY = "NWKBNFL3252PISBY"

THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"


DEFAULT_PRESSURE = 45.0
DEFAULT_FLOW = 100.0


# -----------------------
# Safe float
# -----------------------
def safe_float(x, default):

    try:
        v = float(x)

        if math.isnan(v) or math.isinf(v):
            return default

        return v

    except:
        return default


# -----------------------
# Clean dict
# -----------------------
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


@app.get("/")
def home():
    return {"status": "Digital Twin Running"}


# Manual
@app.get("/predict")
def manual_predict(pressure: float, flow: float):

    return process(pressure, flow)


# Live
@app.get("/live")
def live_predict():

    try:

        r = requests.get(THINGSPEAK_URL, timeout=10)
        data = r.json()

        feed = data["feeds"][-1]

        pressure = safe_float(feed.get("field1"), DEFAULT_PRESSURE)
        flow = safe_float(feed.get("field2"), DEFAULT_FLOW)

    except:

        pressure = DEFAULT_PRESSURE
        flow = DEFAULT_FLOW

    return process(pressure, flow)


# Core
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

    return clean_dict({

        "pressure": round(pressure, 2),
        "flow": round(flow, 2),

        "leak": int(result.get("leak", 0)),
        "probability": round(prob, 4),

        "leak_lpm": leak_lpm,
        "leak_mm": leak_mm,

        "prescription": prescription
    })
