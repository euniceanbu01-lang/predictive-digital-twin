"""
predict.py  —  Leak Detection Inference
────────────────────────────────────────────────────────────────────
DEPLOYMENT NOTES
  - leak_detector.pkl uses ONLY standard sklearn classes.
    joblib.load() works on any Python 3.11+ environment without
    NamedRobustScaler injection or __main__ patching.
  - All 12 model features are dimensionless ratios / percentages.
  - A per-sensor SensorBuffer maintains the rolling window.
  - Field normals are read from the bundle (set at training time).
"""

import math, os, joblib, numpy as np, pandas as pd
from collections import deque

# ── 1. Load bundle (standard sklearn only — no custom classes) ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_bundle    = joblib.load(os.path.join(BASE_DIR, "leak_detector.pkl"))
_model     = _bundle["model"]
_threshold = _bundle["threshold"]
_fn        = _bundle["field_normals"]
_phys      = _bundle["physics_rule"]

# ── 2. Field normals (read from bundle) ──────────────────────────
FIELD_PRESSURE_BAR = _fn["pressure_bar"]   # 0.48 bar
FIELD_FLOW_LPS     = _fn["flow_lps"]       # 0.4167 lps

# ── 3. Thresholds ────────────────────────────────────────────────
PHYS_DROP_PCT = _phys["threshold"]         # 0.26
ML_THRESHOLD  = _threshold                 # 0.5178

# ── 4. Physical constants ─────────────────────────────────────────
Cd = 0.80; g = 9.81

# ── 5. Per-sensor rolling buffer (pre-seeded with field normals) ──
WINDOW = 12

class SensorBuffer:
    def __init__(self):
        self._p = deque([FIELD_PRESSURE_BAR] * WINDOW, maxlen=WINDOW)
        self._f = deque([FIELD_FLOW_LPS]     * WINDOW, maxlen=WINDOW)

    def push(self, p, f):
        self._p.append(p); self._f.append(f)

    def features(self, hour):
        p = list(self._p); f = list(self._f)
        cp, cf = p[-1], f[-1]
        pm3=np.mean(p[-3:]); pm6=np.mean(p[-6:]); pm12=np.mean(p)
        ps6=max(np.std(p[-6:]),1e-6); ps12=max(np.std(p),1e-6)
        fm3=np.mean(f[-3:]); fm6=np.mean(f[-6:]); fs6=max(np.std(f[-6:]),1e-6)
        p1=(cp-p[-2])/(pm3+1e-6); p2=p1-(p[-2]-p[-3])/(pm3+1e-6)
        f1=(cf-f[-2])/(fm3+1e-6)
        pn=cp/(pm6+1e-6); fn=cf/(fm6+1e-6); pfr=pn/(fn+1e-6)
        pfm=np.mean([pp/(ff+1e-6) for pp,ff in zip(p[-6:],f[-6:])])
        return pd.DataFrame([{
            "P_drop_pct6":  (pm6-cp)/(pm6+1e-6),
            "P_drop_pct12": (pm12-cp)/(pm12+1e-6),
            "P_cv6":        ps6/(pm6+1e-6),
            "F_cv6":        fs6/(fm6+1e-6),
            "P_roc1": p1, "P_roc2": p2, "F_roc1": f1,
            "P_zscore6":    (cp-pm6)/(ps6+1e-6),
            "P_zscore12":   (cp-pm12)/(ps12+1e-6),
            "PF_norm_ratio": pfr,
            "PF_ratio_dev":  pfr-pfm,
            "hour": float(hour),
        }])

    @property
    def p6_mean(self):
        return float(np.mean(list(self._p)[-6:]))


_buffers = {}
def _buf(sid): 
    if sid not in _buffers: _buffers[sid] = SensorBuffer()
    return _buffers[sid]


# ── 6. Leak quantification (orifice equation) ─────────────────────
def _quantify(flow_lps, pressure_bar):
    lps = abs(flow_lps - FIELD_FLOW_LPS)
    if lps <= 0 or pressure_bar <= 0: return 0.0, 0.0, 0.0
    head = pressure_bar * 10.2
    if head <= 0: return 0.0, 0.0, 0.0
    A = (lps/1000.0)/(Cd*math.sqrt(2.0*g*head))
    return round(lps,4), round(A*1e6,4), round(math.sqrt(4*A/math.pi)*1000,4)

def _no_leak(prob, pdrop=0.0):
    return {"leak":0,"prob":round(prob,4),"physics_fired":False,
            "p_drop_pct":round(pdrop,4),"Leak_Magnitude_LPS":0.0,
            "Leak_Magnitude_LPM":0.0,"Leak_Area_mm2":0.0,"Leak_Diameter_mm":0.0}


# ── 7. Main prediction ────────────────────────────────────────────
def predict_leak(pressure_bar: float,
                 flow_lps: float,
                 sensor_id: int = 1,
                 hour: int = 12) -> dict:
    """
    Args:
        pressure_bar : corrected pressure in bar
        flow_lps     : corrected flow in litres per second
        sensor_id    : sensor index (each keeps its own rolling buffer)
        hour         : UTC hour of reading (0-23)
    Returns dict with: leak, prob, physics_fired, p_drop_pct,
        Leak_Magnitude_LPS/LPM, Leak_Area_mm2, Leak_Diameter_mm
    """
    if pressure_bar <= 0 or flow_lps <= 0:
        return _no_leak(0.0)

    buf = _buf(sensor_id)
    buf.push(pressure_bar, flow_lps)

    # Physics rule (primary — scale-invariant, ROC-AUC=0.951)
    pdrop = (buf.p6_mean - pressure_bar) / (buf.p6_mean + 1e-6)
    phys  = pdrop >= PHYS_DROP_PCT

    # ML model (secondary — Extra Trees, MCC=0.666)
    try:
        prob = float(_model.predict_proba(buf.features(hour))[0][1])
    except Exception:
        prob = 0.0

    if not (phys or prob >= ML_THRESHOLD):
        return _no_leak(prob, pdrop)

    lps, area, dia = _quantify(flow_lps, pressure_bar)
    return {
        "leak":               1,
        "prob":               round(prob, 4),
        "physics_fired":      bool(phys),
        "p_drop_pct":         round(pdrop, 4),
        "Leak_Magnitude_LPS": lps,
        "Leak_Magnitude_LPM": round(lps * 60.0, 2),
        "Leak_Area_mm2":      area,
        "Leak_Diameter_mm":   dia,
    }
