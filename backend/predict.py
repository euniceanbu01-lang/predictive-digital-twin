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

# ── Load bundle ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_bundle    = joblib.load(os.path.join(BASE_DIR, "leak_detector.pkl"))
_model     = _bundle["model"]
_threshold = _bundle["threshold"]
_fn        = _bundle["field_normals"]
_phys      = _bundle["physics_rule"]

FIELD_PRESSURE_BAR = _fn["pressure_bar"]   # 0.48
FIELD_FLOW_LPS     = _fn["flow_lps"]       # 0.4167

PHYS_DROP_PCT = _phys["threshold"]         # 0.26
ML_THRESHOLD  = _threshold                 # 0.5178

# ── Recovery threshold ───────────────────────────────────────────
# Once a leak is latched, pressure must recover to at least this
# fraction of rolling mean before the alert clears.
# 0.85 = pressure must return to within 15% of normal.
RECOVERY_THRESHOLD = 0.85

# Minimum consecutive stable readings before latch clears.
# Prevents a single fluke reading from resetting the alert.
RECOVERY_READINGS  = 3

Cd = 0.80; g = 9.81
WINDOW = 12


class SensorBuffer:
    """
    Rolling window of pressure/flow readings for one sensor.
    Pre-seeded with field normals — accurate from reading #1.
    
    Also tracks sticky alert state:
      state = 'normal'     → no leak detected
      state = 'latched'    → leak detected, alert stays on
      state = 'recovering' → pressure recovered but waiting for
                             RECOVERY_READINGS consecutive stable readings
    """

    def __init__(self):
        self._p     = deque([FIELD_PRESSURE_BAR] * WINDOW, maxlen=WINDOW)
        self._f     = deque([FIELD_FLOW_LPS]     * WINDOW, maxlen=WINDOW)
        self.state  = "normal"      # 'normal' | 'latched' | 'recovering'
        self._stable_count = 0      # consecutive stable readings counter

    def push(self, p: float, f: float):
        self._p.append(p)
        self._f.append(f)

    def features(self, hour: int) -> pd.DataFrame:
        p = list(self._p); f = list(self._f)
        cp, cf = p[-1], f[-1]
        pm3 = np.mean(p[-3:]); pm6 = np.mean(p[-6:]); pm12 = np.mean(p)
        ps6 = max(np.std(p[-6:]), 1e-6); ps12 = max(np.std(p), 1e-6)
        fm3 = np.mean(f[-3:]); fm6 = np.mean(f[-6:]); fs6 = max(np.std(f[-6:]), 1e-6)
        p1  = (cp - p[-2]) / (pm3 + 1e-6)
        p2  = p1 - (p[-2] - p[-3]) / (pm3 + 1e-6)
        f1  = (cf - f[-2]) / (fm3 + 1e-6)
        pn  = cp / (pm6 + 1e-6); fn = cf / (fm6 + 1e-6); pfr = pn / (fn + 1e-6)
        pfm = np.mean([pp / (ff + 1e-6) for pp, ff in zip(p[-6:], f[-6:])])
        return pd.DataFrame([{
            "P_drop_pct6":   (pm6  - cp) / (pm6  + 1e-6),
            "P_drop_pct12":  (pm12 - cp) / (pm12 + 1e-6),
            "P_cv6":         ps6 / (pm6 + 1e-6),
            "F_cv6":         fs6 / (fm6 + 1e-6),
            "P_roc1": p1, "P_roc2": p2, "F_roc1": f1,
            "P_zscore6":     (cp - pm6)  / (ps6  + 1e-6),
            "P_zscore12":    (cp - pm12) / (ps12 + 1e-6),
            "PF_norm_ratio": pfr,
            "PF_ratio_dev":  pfr - pfm,
            "hour":          float(hour),
        }])

    @property
    def p6_mean(self) -> float:
        return float(np.mean(list(self._p)[-6:]))

    def update_state(self, raw_leak: bool, pressure_bar: float) -> tuple[bool, str]:
        """
        Apply sticky latch logic on top of the raw ML+physics decision.

        Returns (final_leak_bool, state_string)

        State machine:
          normal     → leak detected            → latched
          latched    → pressure recovers        → recovering
          recovering → stable for N readings    → normal
          recovering → pressure drops again     → latched
        """
        p6m = self.p6_mean

        # Is current pressure "recovered" (back to near-normal)?
        # Use FIELD_PRESSURE_BAR as the reference, not rolling mean,
        # because rolling mean drifts down during a sustained leak.
        pressure_ok = (pressure_bar / FIELD_PRESSURE_BAR) >= RECOVERY_THRESHOLD

        if self.state == "normal":
            if raw_leak:
                self.state = "latched"
                self._stable_count = 0
            return raw_leak, self.state

        elif self.state == "latched":
            if pressure_ok and not raw_leak:
                self.state = "recovering"
                self._stable_count = 1
            # stays latched regardless of ML/physics output
            return True, self.state

        elif self.state == "recovering":
            if raw_leak or not pressure_ok:
                # relapsed
                self.state = "latched"
                self._stable_count = 0
                return True, self.state
            else:
                self._stable_count += 1
                if self._stable_count >= RECOVERY_READINGS:
                    self.state = "normal"
                    self._stable_count = 0
                    return False, self.state
                # still recovering — keep alerting
                return True, self.state

        return raw_leak, self.state


# ── Per-sensor buffer registry ───────────────────────────────────
_buffers: dict[int, SensorBuffer] = {}

def _buf(sid: int) -> SensorBuffer:
    if sid not in _buffers:
        _buffers[sid] = SensorBuffer()
    return _buffers[sid]


# ── Orifice quantification ────────────────────────────────────────
def _quantify(flow_lps: float, pressure_bar: float) -> tuple:
    lps = abs(flow_lps - FIELD_FLOW_LPS)
    if lps <= 0 or pressure_bar <= 0:
        return 0.0, 0.0, 0.0
    head = pressure_bar * 10.2
    if head <= 0:
        return 0.0, 0.0, 0.0
    A = (lps / 1000.0) / (Cd * math.sqrt(2.0 * g * head))
    return round(lps, 4), round(A * 1e6, 4), round(math.sqrt(4 * A / math.pi) * 1000, 4)

def _no_leak(prob: float, pdrop: float = 0.0, state: str = "normal") -> dict:
    return {
        "leak": 0, "prob": round(prob, 4),
        "alert_state": state,
        "physics_fired": False, "p_drop_pct": round(pdrop, 4),
        "Leak_Magnitude_LPS": 0.0, "Leak_Magnitude_LPM": 0.0,
        "Leak_Area_mm2": 0.0, "Leak_Diameter_mm": 0.0,
    }


# ── Main prediction ───────────────────────────────────────────────
def predict_leak(pressure_bar: float,
                 flow_lps: float,
                 sensor_id: int = 1,
                 hour: int = 12) -> dict:
    """
    Args:
        pressure_bar : corrected pressure in bar
        flow_lps     : corrected flow in lps
        sensor_id    : each sensor keeps its own buffer and latch state
        hour         : UTC hour (0-23)

    Returns dict with:
        leak          – 0 or 1 (sticky: stays 1 until recovery confirmed)
        alert_state   – 'normal' | 'latched' | 'recovering'
        prob          – ML probability
        physics_fired – True if physics rule triggered this reading
        p_drop_pct    – actual % pressure drop
        Leak_Magnitude_LPS/LPM, Leak_Area_mm2, Leak_Diameter_mm
    """
    if pressure_bar <= 0 or flow_lps <= 0:
        return _no_leak(0.0)

    buf = _buf(sensor_id)
    buf.push(pressure_bar, flow_lps)

    # Raw detection (ML + physics, no memory)
    pdrop = (buf.p6_mean - pressure_bar) / (buf.p6_mean + 1e-6)
    phys  = pdrop >= PHYS_DROP_PCT

    try:
        prob = float(_model.predict_proba(buf.features(hour))[0][1])
    except Exception:
        prob = 0.0

    raw_leak = phys or (prob >= ML_THRESHOLD)

    # Apply sticky latch state machine
    final_leak, state = buf.update_state(raw_leak, pressure_bar)

    if not final_leak:
        return _no_leak(prob, pdrop, state)

    lps, area, dia = _quantify(flow_lps, pressure_bar)
    return {
        "leak":               1,
        "prob":               round(prob, 4),
        "alert_state":        state,          # 'latched' or 'recovering'
        "physics_fired":      bool(phys),
        "p_drop_pct":         round(pdrop, 4),
        "Leak_Magnitude_LPS": lps,
        "Leak_Magnitude_LPM": round(lps * 60.0, 2),
        "Leak_Area_mm2":      area,
        "Leak_Diameter_mm":   dia,
    }
