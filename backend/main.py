from fastapi import FastAPI
import requests
import math
import os
import json
import uuid
from datetime import datetime
from azure.storage.blob import BlobServiceClient

from .scalers import NamedRobustScaler  # noqa: F401
from .predict import predict_leak
from .prescribe import get_prescription

app = FastAPI()

# =========================
# ENVIRONMENT VARIABLES
# =========================
CHANNEL_ID                    = os.getenv("CHANNEL_ID")
READ_API_KEY                  = os.getenv("READ_API_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

RAW_CONTAINER       = "digital-twin-raw"
PROCESSED_CONTAINER = "digital-twin-processed"

if not CHANNEL_ID or not READ_API_KEY:
    raise ValueError("Missing ThingSpeak environment variables: CHANNEL_ID, READ_API_KEY")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("Missing Azure Storage connection string: AZURE_STORAGE_CONNECTION_STRING")


# =========================
# AZURE BLOB SETUP
# =========================
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)
raw_container_client       = blob_service_client.get_container_client(RAW_CONTAINER)
processed_container_client = blob_service_client.get_container_client(PROCESSED_CONTAINER)

for container in [raw_container_client, processed_container_client]:
    try:
        container.create_container()
    except Exception:
        pass   # container already exists — safe to ignore


# =========================
# THINGSPEAK CONFIG
# =========================
THINGSPEAK_URL = (
    f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    f"?api_key={READ_API_KEY}&results=1"
)

# =========================
# SENSOR & PIPE CONFIG
# =========================
# Pipe cross-section: DN25 (25 mm nominal diameter)
# Area = π/4 × 25² = 490.87 mm²
PIPE_AREA = 490.87   # mm²

# Field normal operating point — must match predict.py FIELD_FLOW_LPS / FIELD_PRESSURE_BAR
FIELD_FLOW_LPM     = 25.0    # lpm  (normal operating flow in field)
FIELD_PRESSURE_BAR = 0.48    # bar  (normal operating pressure in field)

# Sensor-to-physical-unit correction factors
# Calibrated from field meter comparison
FLOW_FACTOR     = 7.177033493    # raw ADC → lpm
PRESSURE_FACTOR = 0.390450136    # raw ADC → bar

# Safety clamps (reject obvious sensor faults)
FLOW_MAX_LPM     = 200.0   # lpm  — adjust if pipe can carry more
PRESSURE_MAX_BAR = 10.0    # bar  — burst pressure safety limit

# ThingSpeak field mapping per sensor
SENSOR_CONFIG = [
    {"sensor_id": 1, "flow_field": "field1", "pressure_field": "field2"},
    {"sensor_id": 2, "flow_field": "field3", "pressure_field": "field4"},
    {"sensor_id": 3, "flow_field": "field5", "pressure_field": "field6"},
]

# CesiumJS / pipe network metadata
SENSOR_METADATA = {
    1: {"pressure_sensor_id": "PP-001", "flow_sensor_id": "F-001", "pipe_id": "P-002"},
    2: {"pressure_sensor_id": "PP-002", "flow_sensor_id": "F-002", "pipe_id": "P-006"},
    3: {"pressure_sensor_id": "PP-003", "flow_sensor_id": "F-003", "pipe_id": "P-009"},
}


# =========================
# UTILITIES
# =========================
def safe_float(x, default: float) -> float:
    """Parse a value to float; return default on any error or non-finite result."""
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def clean_dict(d: dict) -> dict:
    """Replace NaN/Inf floats with 0.0 so JSON serialization never fails."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = 0.0
        else:
            out[k] = v
    return out


def save_to_blob(container_client, filename: str, data: dict):
    blob = container_client.get_blob_client(filename)
    blob.upload_blob(json.dumps(data, indent=2), overwrite=True)


# =========================
# CORE DIGITAL TWIN LOOP
# =========================
def run_digital_twin() -> dict:

    # ── 1. Fetch latest feed from ThingSpeak ────────────────
    try:
        r = requests.get(THINGSPEAK_URL, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "feeds" not in data or len(data["feeds"]) == 0:
            return {"error": "No feeds available from ThingSpeak"}

        feed = data["feeds"][-1]

    except Exception as e:
        return {"error": f"ThingSpeak fetch failed: {str(e)}"}

    # ── 2. Initialise output envelopes ──────────────────────
    timestamp     = datetime.utcnow().isoformat()
    filename_time = f"{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex}"

    raw_output = {
        "timestamp":  timestamp,
        "channel_id": CHANNEL_ID,
        "sensors":    {}
    }

    processed_output = {
        "timestamp":  timestamp,
        "channel_id": CHANNEL_ID,
        "sensors":    []
    }

    # ── 3. Per-sensor processing ─────────────────────────────
    try:
        for sensor in SENSOR_CONFIG:
            sid = sensor["sensor_id"]

            # ── 3a. Read raw ADC values from ThingSpeak ──────
            # Use field normal as fallback if sensor read fails
            pressure_raw = max(safe_float(feed.get(sensor["pressure_field"]), FIELD_PRESSURE_BAR / PRESSURE_FACTOR), 0.0)
            flow_raw     = max(safe_float(feed.get(sensor["flow_field"]),     FIELD_FLOW_LPM    / FLOW_FACTOR    ), 0.0)

            # ── 3b. Apply calibration correction factors ──────
            flow_lpm  = min(flow_raw     * FLOW_FACTOR,     FLOW_MAX_LPM)
            pressure  = min(pressure_raw * PRESSURE_FACTOR, PRESSURE_MAX_BAR)

            # ── 3c. Convert flow to lps for model input ───────
            # predict.py expects lps; FIELD_FLOW_LPS = 25/60 = 0.4167 lps
            flow_lps = flow_lpm / 60.0

            # ── 3d. Store human-readable raw values ───────────
            raw_output["sensors"][f"sensor_{sid}"] = {
                "pressure_bar": round(pressure, 3),
                "flow_lpm":     round(flow_lpm, 2),
            }

            # ── 3e. ML prediction ─────────────────────────────
            result = predict_leak(pressure, flow_lps)

            # ── 3f. Extract quantification results ───────────
            leak_lps  = safe_float(result.get("Leak_Magnitude_LPS"), 0.0)
            leak_lpm  = safe_float(result.get("Leak_Magnitude_LPM"), leak_lps * 60.0)
            leak_area = safe_float(result.get("Leak_Area_mm2"),       0.0)
            leak_dia  = safe_float(result.get("Leak_Diameter_mm"),    0.0)
            prob      = safe_float(result.get("prob"),                 0.0)

            # ── 3g. Prescription engine ───────────────────────
            # Default: no leak → normal status
            prescription = {
                "severity":       "Normal",
                "action_type":    "No action required",
                "failure_type":   "",
                "repair_strategy": "",
            }
            size_ratio = 0.0

            if result.get("leak") == 1:
                # Leak size as fraction of pipe cross-section area
                # Ref: IWA Real Losses framework, AWWA M36
                size_ratio = (leak_area / PIPE_AREA) if PIPE_AREA > 0 else 0.0

                pres_raw = get_prescription(size_ratio, leak_lps)
                pres     = clean_dict(pres_raw)

                prescription = {
                    "severity":        pres.get("severity",        "N/A"),
                    "action_type":     pres.get("action_type",     "N/A"),
                    "failure_type":    pres.get("failure_type",    "N/A"),
                    "repair_strategy": pres.get("repair_strategy", "N/A"),
                }

            # ── 3h. Sensor metadata ───────────────────────────
            meta = SENSOR_METADATA.get(sid, {})

            # ── 3i. Append to processed output ───────────────
            processed_output["sensors"].append(clean_dict({
                "sensor_numeric_id":   sid,
                "pressure_sensor_id":  meta.get("pressure_sensor_id"),
                "flow_sensor_id":      meta.get("flow_sensor_id"),
                "pipe_id":             meta.get("pipe_id"),

                "pressure":            round(pressure, 3),   # bar
                "flow_lpm":            round(flow_lpm, 2),   # lpm (for display)

                "leak":                int(result.get("leak", 0)),
                "probability":         round(prob, 4),

                "leak_lpm":            round(leak_lpm, 2),   # lpm
                "leak_area_mm2":       leak_area,
                "leak_diameter_mm":    leak_dia,
                "leak_size_ratio":     round(size_ratio, 5),

                "prescription":        prescription,
            }))

    except Exception as e:
        return {"error": f"Prediction processing error: {str(e)}"}

    # ── 4. Persist to Azure Blob Storage ────────────────────
    try:
        # Time-stamped historical files (never overwritten)
        save_to_blob(raw_container_client,       f"{filename_time}_raw.json",       raw_output)
        save_to_blob(processed_container_client, f"{filename_time}_processed.json", processed_output)

        # latest.json — always overwritten; used by CesiumJS frontend
        save_to_blob(processed_container_client, "latest.json", processed_output)

    except Exception as e:
        return {"error": f"Azure Blob Storage write failed: {str(e)}"}

    return processed_output


# =========================
# API ROUTES
# =========================
@app.get("/")
def home():
    return {
        "status": "Digital Twin API running",
        "pipe_diameter_mm": 25,
        "field_normal_flow_lpm": FIELD_FLOW_LPM,
        "field_normal_pressure_bar": FIELD_PRESSURE_BAR,
    }


@app.get("/live")
def live_trigger():
    """Fetch latest ThingSpeak reading, run ML prediction, persist to blob, return result."""
    return run_digital_twin()
