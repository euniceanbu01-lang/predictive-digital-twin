import pandas as pd
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "prescription.csv")

df = pd.read_csv(CSV_PATH).fillna("")


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

    for _, row in df.iterrows():

        smin = to_float(row.get("leak_size_min"))
        smax = to_float(row.get("leak_size_max"))
        mmin = to_float(row.get("magnitude_min"))
        mmax = to_float(row.get("magnitude_max"))

        size_ok = True
        mag_ok = True

        # ✅ SAME LOGIC STYLE (no breaking change)
        if smin is not None and leak_size < smin:
            size_ok = False
        if smax is not None and leak_size > smax:
            size_ok = False

        if mmin is not None and magnitude < mmin:
            mag_ok = False
        if mmax is not None and magnitude > mmax:
            mag_ok = False

        if size_ok and mag_ok:
            return {k: clean_value(v) for k, v in row.to_dict().items()}

    # ✅ IMPORTANT: keep your old fallback behavior (Azure safe)
    moderate = df[df["severity"] == "Moderate"]

    if len(moderate) > 0:
        row = moderate.iloc[0]
        return {k: clean_value(v) for k, v in row.to_dict().items()}

    return {"message": "No prescription found"}
