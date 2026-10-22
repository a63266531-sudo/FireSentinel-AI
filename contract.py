"""Risk output contract. Never shows a bare number as a probability: every estimate carries its model score, a calibrated
probability ONLY if calibration was demonstrated on a held-out period, an empirical reliability interval, data completeness,
data freshness, out-of-distribution flags, scope, model version and a status (ok / unreliable / unavailable) with reasons."""
import numpy as np
import pandas as pd

from .config import CELL, LIMITATIONS, RISK_BINS, RISK_LABELS
from .model import calibration_table

STALE_HOURS = 72.0
MIN_COMPLETENESS = 0.80
OOD_BEYOND_Q = 0.25   # share of inputs beyond the 1st-99th training percentile
OOD_OUT_RANGE = 2     # number of inputs outside the training min-max


def info_from_res(res, version="unsaved", test_period=None):
    return {"model": res["models"][res["best"]], "calibrator": res["calibrators"][res["best"]], "name": res["best"],
            "features": res["feats"], "train_stats": res["train_stats"], "train_ref": res["train_ref"],
            "calib_table": res["calib_table"], "beats_baseline_ci": res["beats_baseline_ci"],
            "calibration_supported": res["calibration_supported"], "ece": res["ece"], "ci": res["ci"], "version": version,
            "test_period": res["info"]["test_weeks"]}


def info_from_bundle(bundle, meta):
    return {"model": bundle["model"], "calibrator": bundle.get("calibrator"), "name": meta.get("model"),
            "features": bundle["features"], "train_stats": bundle["train_stats"], "train_ref": bundle.get("train_ref"),
            "calib_table": pd.DataFrame(meta.get("calib_table", [])), "beats_baseline_ci": meta.get("beats_baseline_ci", False),
            "calibration_supported": meta.get("calibration_supported", False), "ece": meta.get("ece"),
            "ci": pd.DataFrame(meta.get("ci", [])), "version": bundle.get("version"),
            "test_period": (meta.get("split") or {}).get("test_weeks")}


def build_contract(info, g, df, now, cell=CELL, synthetic=False):
    """Returns (per_cell_df, summary_dict) for the week AFTER the latest week in g."""
    feats = info["features"]
    last = g[g["week"] == g["week"].max()].copy()
    X = last[feats]
    st = info["train_stats"].reindex(feats)
    completeness = 1 - X.isna().mean(axis=1)
    p_raw = info["model"].predict_proba(X)[:, 1]
    p_cal = info["calibrator"].predict(p_raw) if info["calibrator"] is not None else None
    out_rng = (X.lt(st["min"], axis=1) | X.gt(st["max"], axis=1)).sum(axis=1)
    beyond = (X.lt(st["q01"], axis=1) | X.gt(st["q99"], axis=1)).mean(axis=1)
    fresh_h = float((pd.Timestamp(now).tz_localize(None) - df["acq_datetime"].max()) / pd.Timedelta(hours=1))
    rank = pd.Series(p_raw).rank(pct=True).to_numpy() * 100
    ct = info["calib_table"]
    rows = []
    for i, r in enumerate(last.itertuples()):
        reasons, status = [], "ok"
        if completeness.iloc[i] < MIN_COMPLETENESS:
            status = "unavailable"
            reasons.append(f"only {completeness.iloc[i]:.0%} of model inputs available (< {MIN_COMPLETENESS:.0%})")
        else:
            if not info["beats_baseline_ci"]:
                status = "unreliable"
                reasons.append("model has not been shown to beat simple baselines (95% block-bootstrap interval includes 0)")
            if fresh_h > STALE_HOURS:
                status = "unreliable"
                reasons.append(f"newest observation is {fresh_h:.0f} h old (> {STALE_HOURS:.0f} h)")
            if out_rng.iloc[i] >= OOD_OUT_RANGE or beyond.iloc[i] > OOD_BEYOND_Q:
                status = "unreliable"
                reasons.append("inputs lie outside the range seen in training")
        prob = float(p_cal[i]) if (p_cal is not None and info["calibration_supported"] and status != "unavailable") else None
        lo = hi = None
        if prob is not None and len(ct):
            b = ct[(ct["lo"] <= prob) & ((prob < ct["hi"]) | (ct["hi"] >= 1.0))].head(1)
            if len(b) and b["n"].iloc[0] >= 30:
                lo, hi = float(b["wilson_lo"].iloc[0]), float(b["wilson_hi"].iloc[0])
        if prob is None and status != "unavailable":
            reasons.append("score is not a calibrated probability (calibration not demonstrated on held-out data)")
        level = (RISK_LABELS[int(np.searchsorted(RISK_BINS, prob, side="right") - 1)] if prob is not None else
                 ("unavailable" if status == "unavailable" else
                  "relatively high (rank)" if rank[i] >= 90 else "relatively elevated (rank)" if rank[i] >= 60 else "relatively lower (rank)"))
        rows.append({"cy": r.cy, "cx": r.cx, "lat": (r.cy + .5) * cell, "lon": (r.cx + .5) * cell,
                     "forecast_week_start": r.week + pd.Timedelta(days=7), "model_score": float(p_raw[i]),
                     "calibrated_probability": prob, "reliability_low": lo, "reliability_high": hi,
                     "level": level, "relative_rank_pct": round(float(rank[i]), 1), "status": status,
                     "data_completeness": round(float(completeness.iloc[i]), 3), "data_freshness_hours": round(fresh_h, 1),
                     "inputs_out_of_training_range": int(out_rng.iloc[i]), "synthetic_data": bool(synthetic),
                     "model_version": info["version"], "reasons": "; ".join(reasons)})
    c = pd.DataFrame(rows)
    ci = info["ci"]
    pr = ci[ci["metric"] == "PR-AUC"].head(1) if len(ci) else ci
    summary = {"model": info["name"], "model_version": info["version"], "forecast_scope": "next calendar week, grid cells in the training selection",
               "grid_cell_degrees": cell, "test_period": info["test_period"],
               "test_PR_AUC": None if pr.empty else {k: float(pr[k].iloc[0]) for k in ("estimate", "lo", "hi")},
               "beats_baselines_95CI": bool(info["beats_baseline_ci"]), "calibration_supported": bool(info["calibration_supported"]),
               "ECE_test": None if info.get("ece") is None else round(float(info["ece"]), 3),
               "data_freshness_hours": round(fresh_h, 1), "synthetic_data": bool(synthetic),
               "status_counts": c["status"].value_counts().to_dict(), "limitations": LIMITATIONS}
    return c, summary
