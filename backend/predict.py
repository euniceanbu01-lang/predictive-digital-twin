import joblib
import numpy as np
import math
import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

bundle = joblib.load(MODEL_PATH)


model = bundle["model"]
threshold = bundle["threshold"]

Cd = 0.62
g = 9.81


def predict_leak(pressure_bar, flow_lpm):

    pressure_psi = pressure_bar * 14.5038
    flow_gpm = flow_lpm / 3.78541

    X = pd.DataFrame([[pressure_psi, flow_gpm]],
                     columns=["Pressure", "Flow_Rate"])

    prob = model.predict_proba(X)[0][1]
    leak = int(prob >= threshold)

    # If NO leak → return early
    if leak == 0:
        return {
            "leak": 0,
            "prob": float(prob),
            "leak_lpm": 0,
            "leak_mm": 0
        }

    # -------------------------
    # Leak calculations
    # -------------------------

    expected = 0.70 * flow_gpm
    leak_gpm = max(0, flow_gpm - expected)

    leak_lpm = leak_gpm * 3.78541
    leak_m3s = leak_lpm / 1000 / 60

    head_m = pressure_bar * 10.2

    if leak_m3s <= 0 or head_m <= 0:
        leak_mm = 0
    else:
        A = leak_m3s / (Cd * math.sqrt(2 * g * head_m))
        d = math.sqrt((4 * A) / math.pi)
        leak_mm = d * 1000

    return {
        "leak": 1,
        "prob": float(prob),
        "leak_lpm": round(leak_lpm, 3),
        "leak_mm": round(leak_mm, 3)
    }

