import joblib
import pandas as pd
import math
import os

# LOAD MODEL (once only)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_leak_model.pkl")

model = joblib.load(MODEL_PATH)["model"]

# CONSTANTS

EXPECTED_FLOW = 5.0          # LPS
EXPECTED_PRESSURE = 1.8      # bar

Cd = 0.90
g = 9.81

# LEAK CALCULATION

def _calculate_leak(flow_lps, pressure_bar):
    leak_lps = flow_lps - EXPECTED_FLOW

    # Fast exit (no leak physically possible)
    if leak_lps <= 0 or pressure_bar <= 0:
        return 0.0, 0.0, 0.0

    leak_m3s = leak_lps / 1000
    head = pressure_bar * 10.2

    if head <= 0:
        return 0.0, 0.0, 0.0

    # Orifice equation
    A = leak_m3s / (Cd * math.sqrt(2 * g * head))

    # Derived values
    area_mm2 = A * 1e6
    diameter_mm = math.sqrt((4 * A) / math.pi) * 1000

    return (
        round(leak_lps, 4),
        round(area_mm2, 4),
        round(diameter_mm, 4)
    )


# =========================
# MAIN FUNCTION
# =========================
def predict_leak(pressure_bar, flow_lps):

    # -------- Safety --------
    if pressure_bar <= 0 or flow_lps <= 0:
        return {
            "leak": 0,
            "prob": 0.0,
            "Leak_Magnitude_LPS": 0.0,
            "Leak_Area_mm2": 0.0,
            "Leak_Diameter_mm": 0.0
        }

    # -------- Feature Engineering (inline, fast) --------
    fr = flow_lps / EXPECTED_FLOW
    pr = pressure_bar / EXPECTED_PRESSURE

    X = pd.DataFrame([[
        fr,
        pr,
        fr * pr,
        1 - fr,
        1 - pr
    ]], columns=[
        "Flow_ratio",
        "Pressure_ratio",
        "Flow_Pressure_interaction",
        "Flow_drop",
        "Pressure_drop"
    ])

    # -------- Prediction --------
    prob = float(model.predict_proba(X)[0][1])

    if prob <= 0.2:
        return {
            "leak": 0,
            "prob": prob,
            "Leak_Magnitude_LPS": 0.0,
            "Leak_Area_mm2": 0.0,
            "Leak_Diameter_mm": 0.0
        }

    # -------- Leak Calculation --------
    lps, area, dia = _calculate_leak(flow_lps, pressure_bar)

    return {
        "leak": 1,
        "prob": prob,
        "Leak_Magnitude_LPS": lps,
        "Leak_Area_mm2": area,
        "Leak_Diameter_mm": dia
    }
