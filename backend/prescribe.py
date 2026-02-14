import pandas as pd
import math
import os


# Get current file directory (backend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to prescription.csv
CSV_PATH = os.path.join(BASE_DIR, "prescription.csv")


# Load CSV safely
df = pd.read_csv(CSV_PATH)

# Replace NaN
df = df.fillna("")


def clean_value(x):
    if isinstance(x, float):
        if math.isnan(x):
            return ""
    return x


def get_prescription(leak_size, magnitude):

    for _, row in df.iterrows():

        if row.get("severity") == "Catastrophic":
            continue

        smin = row.get("leak_size_min", "")
        smax = row.get("leak_size_max", "")
        mmin = row.get("magnitude_min", "")
        mmax = row.get("magnitude_max", "")

        size_ok = True
        mag_ok = True

        try:
            if smin != "" and leak_size < float(smin):
                size_ok = False
            if smax != "" and leak_size > float(smax):
                size_ok = False

            if mmin != "" and magnitude < float(mmin):
                mag_ok = False
            if mmax != "" and magnitude > float(mmax):
                mag_ok = False

        except:
            pass

        if size_ok and mag_ok:

            result = {}

            for k, v in row.to_dict().items():
                result[k] = clean_value(v)

            return result


    # Default fallback (Moderate)
    moderate = df[df["severity"] == "Moderate"]

    if len(moderate) > 0:
        row = moderate.iloc[0]

        result = {}

        for k, v in row.to_dict().items():
            result[k] = clean_value(v)

        return result

    return {"message": "No prescription found"}
