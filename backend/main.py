"""
main.py  —  FastAPI Digital Twin Backend
─────────────────────────────────────────────────────────────────────
Architecture:
  IoT sensors → ThingSpeak → this API → Azure Blob Storage
                                       → CesiumJS 3D frontend

Routes:
  GET /       status + config
  GET /live   fetch → predict → prescribe → persist → return
"""

from fastapi import FastAPI
import requests, math, os, json, uuid
from datetime import datetime
from azure.storage.blob import BlobServiceClient

from .predict  import predict_leak
from .prescribe import get_prescription

app = FastAPI(title="Predictive Digital Twin — Leak Detection API")

# ══════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES (set in Azure App Service → Configuration)
# ══════════════════════════════════════════════════════════════════
CHANNEL_ID   = os.getenv("CHANNEL_ID")
READ_API_KEY = os.getenv("READ_API_KEY")
AZURE_CONN   = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

RAW_CONTAINER       = "digital-twin-raw"
PROCESSED_CONTAINER = "digital-twin-processed"

if not CHANNEL_ID or not READ_API_KEY:
    raise ValueError("Missing env vars: CHANNEL_ID, READ_API_KEY")
if not AZURE_CONN:
    raise ValueError("Missing env var: AZURE_STORAGE_CONNECTION_STRING")

# ── Azure Blob ─────────────────────────────────────────────────────
_blob_svc = BlobServiceClient.from_connection_string(AZURE_CONN)
_raw_ctr  = _blob_svc.get_container_client(RAW_CONTAINER)
_proc_ctr = _blob_svc.get_container_client(PROCESSED_CONTAINER)

for ctr in [_raw_ctr, _proc_ctr]:
    try: ctr.create_container()
    except Exception: pass   # already exists

# ── ThingSpeak ─────────────────────────────────────────────────────
THINGSPEAK_URL = (
    f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    f"?api_key={READ_API_KEY}&results=1"
)

# ══════════════════════════════════════════════════════════════════
# PIPE & SENSOR CONFIGURATION
# ══════════════════════════════════════════════════════════════════
PIPE_AREA = 490.87          # mm²  (DN25 pipe: π/4 × 25²)

# Field normal operating point (must match predict.py values)
FIELD_FLOW_LPM     = 25.0   # lpm
FIELD_PRESSURE_BAR = 0.48   # bar

# ADC → physical unit correction factors (calibrated from field meter)
FLOW_FACTOR     = 7.177033493    # raw → lpm
PRESSURE_FACTOR = 0.390450136    # raw → bar

# Safety clamps (reject clearly faulty sensor values)
FLOW_MAX_LPM     = 200.0    # lpm
PRESSURE_MAX_BAR = 10.0     # bar

# ThingSpeak field assignments
SENSOR_CONFIG = [
    {"sensor_id": 1, "flow_field": "field1", "pressure_field": "field2"},
    {"sensor_id": 2, "flow_field": "field3", "pressure_field": "field4"},
    {"sensor_id": 3, "flow_field": "field5", "pressure_field": "field6"},
]

# Pipe network metadata (used by CesiumJS 3D visualisation)
SENSOR_META = {
    1: {"pressure_sensor_id": "PP-001", "flow_sensor_id": "F-001", "pipe_id": "P-002"},
    2: {"pressure_sensor_id": "PP-002", "flow_sensor_id": "F-002", "pipe_id": "P-006"},
    3: {"pressure_sensor_id": "PP-003", "flow_sensor_id": "F-003", "pipe_id": "P-009"},
}

# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════
def _float(x, default: float) -> float:
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default

def _clean(d: dict) -> dict:
    """Replace NaN/Inf so JSON serialisation never fails."""
    return {k: (0.0 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            for k, v in d.items()}

def _save_blob(ctr, name: str, data: dict):
    ctr.get_blob_client(name).upload_blob(json.dumps(data, indent=2), overwrite=True)

# ══════════════════════════════════════════════════════════════════
# CORE DIGITAL TWIN LOOP
# ══════════════════════════════════════════════════════════════════
def run_digital_twin() -> dict:

    # 1. Fetch from ThingSpeak ────────────────────────────────────
    try:
        resp = requests.get(THINGSPEAK_URL, timeout=10)
        resp.raise_for_status()
        feeds = resp.json().get("feeds", [])
        if not feeds:
            return {"error": "No feeds from ThingSpeak"}
        feed = feeds[-1]
    except Exception as e:
        return {"error": f"ThingSpeak: {e}"}

    ts   = datetime.utcnow().isoformat()
    hour = datetime.utcnow().hour
    fid  = f"{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex}"

    raw_out  = {"timestamp": ts, "channel_id": CHANNEL_ID, "sensors": {}}
    proc_out = {"timestamp": ts, "channel_id": CHANNEL_ID, "sensors": []}

    # 2. Per-sensor processing ────────────────────────────────────
    try:
        for s in SENSOR_CONFIG:
            sid = s["sensor_id"]

            # Read & calibrate
            p_raw = max(_float(feed.get(s["pressure_field"]),
                                FIELD_PRESSURE_BAR / PRESSURE_FACTOR), 0.0)
            f_raw = max(_float(feed.get(s["flow_field"]),
                                FIELD_FLOW_LPM / FLOW_FACTOR), 0.0)

            flow_lpm = min(f_raw * FLOW_FACTOR,     FLOW_MAX_LPM)
            pressure = min(p_raw * PRESSURE_FACTOR, PRESSURE_MAX_BAR)
            flow_lps = flow_lpm / 60.0

            # Store raw
            raw_out["sensors"][f"sensor_{sid}"] = {
                "pressure_bar": round(pressure, 3),
                "flow_lpm":     round(flow_lpm, 2),
            }

            # Predict (rolling buffer updated inside predict_leak)
            result = predict_leak(pressure, flow_lps,
                                  sensor_id=sid, hour=hour)

            leak_lps  = _float(result.get("Leak_Magnitude_LPS"), 0.0)
            leak_lpm  = _float(result.get("Leak_Magnitude_LPM"), leak_lps*60)
            leak_area = _float(result.get("Leak_Area_mm2"),       0.0)
            leak_dia  = _float(result.get("Leak_Diameter_mm"),    0.0)
            prob      = _float(result.get("prob"),                 0.0)

            # Prescription
            prescription = {
                "severity": "Normal", "action_type": "No action required",
                "failure_type": "", "repair_strategy": "",
            }
            size_ratio = 0.0

            if result.get("leak") == 1:
                size_ratio = (leak_area / PIPE_AREA) if PIPE_AREA > 0 else 0.0
                pres = _clean(get_prescription(size_ratio, leak_lps))
                prescription = {
                    "severity":        pres.get("severity",        "N/A"),
                    "action_type":     pres.get("action_type",     "N/A"),
                    "failure_type":    pres.get("failure_type",    "N/A"),
                    "repair_strategy": pres.get("repair_strategy", "N/A"),
                }

            meta = SENSOR_META.get(sid, {})
            proc_out["sensors"].append(_clean({
                "sensor_numeric_id":  sid,
                "pressure_sensor_id": meta.get("pressure_sensor_id"),
                "flow_sensor_id":     meta.get("flow_sensor_id"),
                "pipe_id":            meta.get("pipe_id"),

                "pressure":           round(pressure, 3),   # bar
                "flow_lpm":           round(flow_lpm, 2),   # lpm

                "leak":               int(result.get("leak", 0)),
                "probability":        round(prob, 4),
                "physics_fired":      bool(result.get("physics_fired", False)),
                "p_drop_pct":         round(_float(result.get("p_drop_pct"), 0.0), 4),

                "leak_lpm":           round(leak_lpm, 2),
                "leak_area_mm2":      leak_area,
                "leak_diameter_mm":   leak_dia,
                "leak_size_ratio":    round(size_ratio, 5),
                "alert_state": result.get("alert_state", "normal"),

                "prescription":       prescription,
            }))

    except Exception as e:
        return {"error": f"Prediction error: {e}"}

    # 3. Persist to Azure Blob ─────────────────────────────────────
    try:
        _save_blob(_raw_ctr,  f"{fid}_raw.json",       raw_out)
        _save_blob(_proc_ctr, f"{fid}_processed.json", proc_out)
        _save_blob(_proc_ctr, "latest.json",            proc_out)   # live feed
    except Exception as e:
        return {"error": f"Blob Storage: {e}"}

    return proc_out


# ══════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════
@app.get("/")
def home():
    return {
        "status":                    "Digital Twin API running",
        "model":                     "Extra Trees (CalibratedClassifierCV)",
        "model_version":             "7.0-clean",
        "pipe_diameter_mm":          25,
        "field_normal_pressure_bar": FIELD_PRESSURE_BAR,
        "field_normal_flow_lpm":     FIELD_FLOW_LPM,
        "custom_classes_in_pkl":     "NONE",
    }


@app.get("/live")
def live():
    """Fetch ThingSpeak → predict → prescribe → persist → return."""
    return run_digital_twin()
