import pandas as pd
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "prescription.csv")

df = pd.read_csv(CSV_PATH).fillna("")
df.columns = df.columns.str.strip().str.lower()

def clean_value(x):
    if isinstance(x, float) and math.isnan(x):
        return ""
    return x

def to_float(x):
    try:
        return float(x)
    except:
        return None


def get_prescription(leak_size, magnitude):

    selected_row = None


    # STEP 1: SIZE-BASED SEVERITY
    # =========================
    for _, row in df.iterrows():

        smin = to_float(row.get("leak_size_min"))
        smax = to_float(row.get("leak_size_max"))

        size_ok = (
            (smin is None or leak_size >= smin) and
            (smax is None or leak_size < smax)
        )

        if size_ok:
            selected_row = row
            break

    # =========================
    # STEP 2: MAGNITUDE REFINEMENT
    # =========================
    if selected_row is not None:

        result = {k: clean_value(v) for k, v in selected_row.to_dict().items()}

        mmin = to_float(selected_row.get("magnitude_min"))
        mmax = to_float(selected_row.get("magnitude_max"))

        mag_ok = (
            (mmin is None or magnitude >= mmin) and
            (mmax is None or magnitude < mmax)
        )

        # 🔥 If magnitude is extreme → upgrade action (NOT severity)
        if not mag_ok:
            result["action_type"] = result.get("action_type", "") + " (High flow impact)"

        return result

    # =========================
    # STEP 3: SAFETY FALLBACK
    # =========================
    # (should rarely happen)
    row = df.iloc[-1]  # Catastrophic
    return {k: clean_value(v) for k, v in row.to_dict().items()}
