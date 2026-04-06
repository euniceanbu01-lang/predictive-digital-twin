import joblib
import pandas as pd
import math
import os


# LOAD MODEL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_leak_model_rf.pkl")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
threshold = bundle.get("threshold", 0.5)
flow_median = bundle["flow_median"]
pressure_median = bundle["pressure_median"]


# CONSTANTS (KEEP YOURS)

EXPECTED_FLOW = 5.0
EXPECTED_PRESSURE = 1.8

Cd = 0.90
g = 9.81


# LEAK CALCULATION 

def _calculate_leak(flow_lps, pressure_bar):

    leak_lps = flow_lps - EXPECTED_FLOW

    if leak_lps <= 0 or pressure_bar <= 0:
        return 0.0, 0.0, 0.0

    leak_m3s = leak_lps / 1000
    head = pressure_bar * 10.2

    if head <= 0:
        return 0.0, 0.0, 0.0

    A = leak_m3s / (Cd * math.sqrt(2 * g * head))

    area_mm2 = A * 1e6
    diameter_mm = math.sqrt((4 * A) / math.pi) * 1000

    return (
        round(leak_lps, 4),
        round(area_mm2, 4),
        round(diameter_mm, 4)
    )


# MAIN PREDICTION

def predict_leak(pressure_bar, flow_lps):

    
    if pressure_bar <= 0 or flow_lps <= 0:
        return {
            "leak": 0,
            "prob": 0.0,
            "Leak_Magnitude_LPS": 0.0,
            "Leak_Area_mm2": 0.0,
            "Leak_Diameter_mm": 0.0
        }

    
    #  CORRECT FEATURE ENGINEERING
  
    flow_ratio = flow_lps / flow_median
    pressure_ratio = pressure_bar / pressure_median

    flow_drop = 1 - flow_ratio
    pressure_drop = 1 - pressure_ratio
    interaction = flow_ratio * pressure_ratio
    leak_index = (1 - pressure_ratio) * flow_ratio
    pressure_gradient = 0  # no time data

    X = pd.DataFrame([{
        "Pressure_ratio": pressure_ratio,
        "Pressure_drop": pressure_drop,
        "Flow_ratio": flow_ratio,
        "Flow_drop": flow_drop,
        "Flow_Pressure_interaction": interaction,
        "Leak_index": leak_index,
        "Pressure_gradient": pressure_gradient
    }])

   
    # MODEL PREDICTION
    
    prob = float(model.predict_proba(X)[0][1])

    # HYBRID DECISION 
 
    if pressure_ratio < 0.75:
        leak_flag = 1
    else:
        leak_flag = int(prob > threshold)

    
    # NO LEAK
    
    if leak_flag == 0:
        return {
            "leak": 0,
            "prob": round(prob, 4),
            "Leak_Magnitude_LPS": 0.0,
            "Leak_Area_mm2": 0.0,
            "Leak_Diameter_mm": 0.0
        }

    
    # LEAK CALCULATION 

    lps, area, dia = _calculate_leak(flow_lps, pressure_bar)

    return {
        "leak": 1,
        "prob": round(prob, 4),
        "Leak_Magnitude_LPS": lps,
        "Leak_Area_mm2": area,
        "Leak_Diameter_mm": dia
    }
