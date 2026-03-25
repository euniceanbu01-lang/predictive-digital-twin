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

    if leak == 0:
        return {
            "leak": 0,
            "prob": float(prob),
            "leak_lpm": 0,
            "leak_area": 0
        }

    expected = 0.70 * flow_gpm
    leak_gpm = max(0, flow_gpm - expected)

    leak_lpm = leak_gpm * 3.78541
    leak_m3s = leak_lpm / 1000 / 60

    head_m = pressure_bar * 10.2

    A = 0

    if leak_m3s > 0 and head_m > 0:
        A = leak_m3s / (Cd * math.sqrt(2 * g * head_m))

    # ✅ convert m² → mm²
    A_mm2 = A * 1_000_000

    return {
        "leak": 1,
        "prob": float(prob),
        "leak_lpm": round(leak_lpm, 3),
        "leak_area": round(A_mm2, 3)
    }
