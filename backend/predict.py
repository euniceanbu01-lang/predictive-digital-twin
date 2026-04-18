import joblib
import pandas as pd
import math
import os

# ── CRITICAL: import before joblib.load ──────────────────────────────────────
# NamedRobustScaler was defined in __main__ during training (notebook).
# Importing it here puts it in sys.modules so pickle can find it at startup.
from .scalers import NamedRobustScaler  # noqa: F401

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "leak_model_systemA.pkl")

bundle = joblib.load(MODEL_PATH)

model     = bundle["model"]
threshold = bundle.get("threshold", 0.5)

# =========================
# FIELD OPERATING NORMALS
# ─────────────────────────
# These are the FIELD-SCALE baseline values used to compute
# ratio features. They MUST reflect actual normal operation
# at the deployment site — NOT the training data medians.
#
# Why: The model was trained on ratio features (pressure_ratio,
# flow_ratio, etc.) where 1.0 = normal. If we use training-data
# medians (116 lps, 2.92 bar) against field readings (0.4 lps,
# 0.48 bar), every ratio becomes ~0.003 — completely outside the
# training distribution and the model breaks.
#
# Using field normals keeps ratio=1.0 during normal operation
# and ratio<1.0 during a leak, which matches the training pattern.
# =========================
FIELD_FLOW_LPS      = 25.0 / 60.0   # 25 lpm → 0.4167 lps  (normal field flow)
FIELD_PRESSURE_BAR  = 0.48          # bar                   (normal field pressure)

# =========================
# PHYSICAL CONSTANTS
# =========================
Cd  = 0.80   # discharge coefficient (AWWA, orifice in pipe)
g   = 9.81   # m/s²


# =========================
# LEAK QUANTIFICATION
# Uses orifice flow equation:  Q = Cd · A · √(2gh)
# Ref: AWWA M36, Torricelli's theorem for pipe orifices
# =========================
def _calculate_leak(flow_lps: float, pressure_bar: float):
    """
    Calculate leak magnitude, orifice area, and equivalent diameter.

    Args:
        flow_lps:     measured flow (lps) during leak event
        pressure_bar: measured pressure (bar) during leak event

    Returns:
        (leak_lps, area_mm2, diameter_mm)
    """
    # Deviation from expected normal flow = leak magnitude
    leak_lps = abs(flow_lps - FIELD_FLOW_LPS)

    if leak_lps <= 0 or pressure_bar <= 0:
        return 0.0, 0.0, 0.0

    leak_m3s = leak_lps / 1000.0           # lps → m³/s
    head_m   = pressure_bar * 10.2         # bar → metres head

    if head_m <= 0:
        return 0.0, 0.0, 0.0

    # Orifice area (m²) → convert to mm²
    A_m2      = leak_m3s / (Cd * math.sqrt(2.0 * g * head_m))
    area_mm2  = A_m2 * 1.0e6
    dia_mm    = math.sqrt((4.0 * A_m2) / math.pi) * 1000.0

    return (
        round(leak_lps,  4),
        round(area_mm2,  4),
        round(dia_mm,    4)
    )


# =========================
# MAIN PREDICTION FUNCTION
# =========================
def predict_leak(pressure_bar: float, flow_lps: float) -> dict:
    """
    Run the Random Forest leak detection model on a single reading.

    Features are computed as ratios relative to FIELD_FLOW_LPS and
    FIELD_PRESSURE_BAR so the model receives inputs in the same
    scale-normalised form it was trained on.

    Args:
        pressure_bar: corrected pressure in bar
        flow_lps:     corrected flow in litres per second

    Returns:
        dict with keys:
            leak                  – 0 or 1
            prob                  – model confidence (0.0–1.0)
            Leak_Magnitude_LPS    – estimated leak flow (lps)
            Leak_Magnitude_LPM    – estimated leak flow (lpm)
            Leak_Area_mm2         – orifice area (mm²)
            Leak_Diameter_mm      – equivalent orifice diameter (mm)
    """

    # ── Safety guard ──────────────────────────────────────────
    if pressure_bar <= 0 or flow_lps <= 0:
        return _no_leak(0.0)

    # ── Feature engineering ───────────────────────────────────
    # Ratios relative to FIELD normals (1.0 = normal operation)
    pressure_ratio = pressure_bar  / FIELD_PRESSURE_BAR
    flow_ratio     = flow_lps      / FIELD_FLOW_LPS

    pressure_drop  = 1.0 - pressure_ratio   # +ve when pressure falls
    flow_drop      = 1.0 - flow_ratio       # +ve when flow falls

    interaction    = flow_ratio * pressure_ratio
    leak_index     = pressure_drop * flow_ratio   # rises with P-drop & F anomaly
    pressure_gradient = 0.0                       # single-reading API; no history

    X = pd.DataFrame([{
        "Pressure_ratio":             pressure_ratio,
        "Pressure_drop":              pressure_drop,
        "Flow_ratio":                 flow_ratio,
        "Flow_drop":                  flow_drop,
        "Flow_Pressure_interaction":  interaction,
        "Leak_index":                 leak_index,
        "Pressure_gradient":          pressure_gradient,
    }])

    # ── Model inference ──────────────────────────────────────
    prob       = float(model.predict_proba(X)[0][1])
    leak_flag  = int(prob >= threshold)

    # ── Physics override (optional safety net) ────────────────
    # Uncomment if you want a hard rule for large pressure drops:
    # if pressure_ratio < 0.74:   # >26% drop → force leak
    #     leak_flag = 1

    # ── No leak ──────────────────────────────────────────────
    if leak_flag == 0:
        return _no_leak(prob)

    # ── Leak quantification ──────────────────────────────────
    lps, area_mm2, dia_mm = _calculate_leak(flow_lps, pressure_bar)

    return {
        "leak":               1,
        "prob":               round(prob, 4),
        "Leak_Magnitude_LPS": lps,
        "Leak_Magnitude_LPM": round(lps * 60.0, 2),
        "Leak_Area_mm2":      area_mm2,
        "Leak_Diameter_mm":   dia_mm,
    }


def _no_leak(prob: float) -> dict:
    return {
        "leak":               0,
        "prob":               round(prob, 4),
        "Leak_Magnitude_LPS": 0.0,
        "Leak_Magnitude_LPM": 0.0,
        "Leak_Area_mm2":      0.0,
        "Leak_Diameter_mm":   0.0,
    }
