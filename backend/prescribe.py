import pandas as pd
import math

df = pd.read_csv("prescription.csv")

# Replace NaN in dataframe
df = df.fillna("")


def clean_value(x):

    if isinstance(x, float):
        if math.isnan(x):
            return ""

    return x


def get_prescription(leak_size, magnitude):

    for _, row in df.iterrows():

        if row["severity"] == "Catastrophic":
            continue

        smin = row["leak_size_min"]
        smax = row["leak_size_max"]
        mmin = row["magnitude_min"]
        mmax = row["magnitude_max"]

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

    row = df[df["severity"] == "Moderate"].iloc[0]

    result = {}

    for k, v in row.to_dict().items():
        result[k] = clean_value(v)

    return result
