"""Scenario laboratory: sensitivity analysis of the trained model under explicit hypothetical inputs.
This is NOT a forecast and NOT causal inference: the model learned associations from observational data, and only the
variables you change (plus the documented derived feature) are modified."""
import numpy as np
import pandas as pd

from .weather import fwi_lite

EDITABLE = ("tmax_mean", "rh_mean", "wind_max", "precip_sum", "precip14", "dry_days", "et0_sum", "ndvi")
FWI_INPUTS = ("tmax_mean", "wind_max", "precip14", "rh_mean")


def editable_variables(info):
    return [v for v in EDITABLE if v in info["features"]]


def baseline_row(g, cy, cx, feats):
    row = g[(g["cy"] == cy) & (g["cx"] == cx)].sort_values("week").tail(1)
    if row.empty:
        raise ValueError("Cell not found in dataset")
    return row[feats].reset_index(drop=True)


def _predict(info, X):
    p = float(info["model"].predict_proba(X[info["features"]])[:, 1][0])
    cal = info["calibrator"]
    prob = float(cal.predict([p])[0]) if (cal is not None and info["calibration_supported"]) else None
    return p, prob


def _range_warnings(info, overrides):
    st, out = info["train_stats"], []
    for k, v in overrides.items():
        if k not in st.index:
            continue
        if v < st.loc[k, "min"] or v > st.loc[k, "max"]:
            out.append(f"{k}={v:g} is OUTSIDE the training range [{st.loc[k, 'min']:.3g}, {st.loc[k, 'max']:.3g}]: extrapolation, not reliable")
        elif v < st.loc[k, "q01"] or v > st.loc[k, "q99"]:
            out.append(f"{k}={v:g} is at the extreme edge of the training data (beyond the 1st-99th percentile)")
    return out


def run_scenario(info, row, overrides, update_derived=True):
    bad = [k for k in overrides if k not in editable_variables(info)]
    if bad:
        raise ValueError(f"Not editable for this model: {bad}. Editable: {editable_variables(info)}")
    r = row.copy()
    for k, v in overrides.items():
        r[k] = float(v)
    notes = []
    if update_derived and "fwi" in info["features"] and any(k in overrides for k in FWI_INPUTS):
        g = lambda c: pd.Series([r[c].iloc[0] if c in r else np.nan])
        r["fwi"] = float(fwi_lite(g("tmax_mean"), g("wind_max"), g("precip14"), g("rh_mean")).iloc[0])
        notes.append("derived feature 'fwi' was recomputed from the changed inputs")
    if "precip_sum" in overrides and "precip14" in info["features"] and "precip14" not in overrides:
        notes.append("precip_sum changed but precip14 (14-day history) was left unchanged; change it too if intended")
    s0, p0 = _predict(info, row)
    s1, p1 = _predict(info, r)
    return {"baseline_score": s0, "scenario_score": s1, "delta_score": s1 - s0, "baseline_probability": p0,
            "scenario_probability": p1, "overrides": dict(overrides), "notes": notes,
            "warnings": _range_warnings(info, overrides),
            "disclaimer": "Sensitivity analysis of a statistical model; not a forecast and not evidence of causal effects."}


def sweep(info, row, var, n=25, lo=None, hi=None, update_derived=True):
    st = info["train_stats"]
    lo = float(st.loc[var, "q01"]) if lo is None else lo
    hi = float(st.loc[var, "q99"]) if hi is None else hi
    rows = []
    for v in np.linspace(lo, hi, n):
        out = run_scenario(info, row, {var: v}, update_derived)
        rows.append({var: v, "model_score": out["scenario_score"], "calibrated_probability": out["scenario_probability"],
                     "within_training_range": not any("OUTSIDE" in w for w in out["warnings"])})
    return pd.DataFrame(rows)


def sweep2d(info, row, v1, v2, n=12, update_derived=True):
    st = info["train_stats"]
    a = np.linspace(st.loc[v1, "q01"], st.loc[v1, "q99"], n)
    b = np.linspace(st.loc[v2, "q01"], st.loc[v2, "q99"], n)
    z = np.array([[run_scenario(info, row, {v1: x, v2: y}, update_derived)["scenario_score"] for x in a] for y in b])
    return pd.DataFrame(z, index=np.round(b, 3), columns=np.round(a, 3))
