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

    for _, row in df.iterrows():

        smin = to_float(row.get("leak_size_min"))
        smax = to_float(row.get("leak_size_max"))
        mmin = to_float(row.get("magnitude_min"))
        mmax = to_float(row.get("magnitude_max"))

        size_ok = (
            (smin is None or leak_size >= smin) and
            (smax is None or leak_size < smax)
        )

        mag_ok = (
            (mmin is None or magnitude >= mmin) and
            (mmax is None or magnitude < mmax)
        )

        if size_ok and mag_ok:
            return {k: clean_value(v) for k, v in row.to_dict().items()}

    # ===== SAFE FALLBACK (NO UNKNOWN) =====
    if leak_size < 0.0001:
        fallback = "Minor"
    elif leak_size < 0.0025:
        fallback = "Moderate"
    elif leak_size < 0.01:
        fallback = "Major"
    else:
        fallback = "Catastrophic"

    row = df[df["severity"].str.lower() == fallback.lower()].iloc[0]

    return {k: clean_value(v) for k, v in row.to_dict().items()}
