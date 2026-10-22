"""Input / prediction distribution monitoring (population stability index, out-of-range checks)."""
import numpy as np
import pandas as pd

PSI_STABLE, PSI_MAJOR = 0.10, 0.25  # conventional rules of thumb, not statistical tests


def psi(expected, actual, bins=10):
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) < 30 or len(a) < 10:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")  # (near-)constant feature
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, edges)[0] / len(e), 1e-4, None)
    pa = np.clip(np.histogram(a, edges)[0] / len(a), 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def _status(v):
    if np.isnan(v):
        return "n/a"
    return "stable" if v < PSI_STABLE else ("moderate shift" if v < PSI_MAJOR else "major shift")


def drift_report(train_ref: pd.DataFrame, recent: pd.DataFrame, feats) -> pd.DataFrame:
    rows = []
    for f in feats:
        lo, hi = train_ref[f].min(), train_ref[f].max()
        r = recent[f]
        v = psi(train_ref[f], r)
        rows.append({"feature": f, "PSI": round(v, 3) if not np.isnan(v) else np.nan, "status": _status(v),
                     "train_median": train_ref[f].median(), "recent_median": r.median(),
                     "share_outside_train_range": float(((r < lo) | (r > hi)).mean()), "share_missing": float(r.isna().mean())})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False, na_position="last").reset_index(drop=True).round(3)


def prediction_shift(test_scores, recent_scores) -> dict:
    v = psi(test_scores, recent_scores)
    return {"PSI": None if np.isnan(v) else round(v, 3), "status": _status(v),
            "test_mean": round(float(np.mean(test_scores)), 3), "recent_mean": round(float(np.mean(recent_scores)), 3)}
