"""Geospatial analytics: clustering, hotspots, period comparison, anomalies."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .config import CELL


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cell_ids(df, cell=CELL):
    d = df.copy()
    d["cy"] = np.floor(d["latitude"] / cell).astype(int)
    d["cx"] = np.floor(d["longitude"] / cell).astype(int)
    return d


def cluster(df, eps_km=3.0, min_samples=3):
    """DBSCAN on haversine distance. A cluster is a group of detections,
    NOT a confirmed single fire. -1 = isolated detection."""
    if len(df) == 0:
        return df.assign(cluster=pd.Series(dtype=int))
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    labels = DBSCAN(eps=eps_km / 6371.0088, min_samples=min_samples,
                    metric="haversine", algorithm="ball_tree").fit_predict(coords)
    return df.assign(cluster=labels)


def cluster_summary(dfc):
    cols = ["cluster", "detections", "frp_sum", "lat", "lon", "radius_km", "first_seen", "last_seen", "active_days"]
    c = dfc[dfc["cluster"] >= 0]
    if c.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for cid, g in c.groupby("cluster"):
        la, lo = g["latitude"].mean(), g["longitude"].mean()
        rows.append({
            "cluster": int(cid), "detections": len(g),
            "frp_sum": float(g["frp"].sum()) if "frp" in g else 0.0,
            "lat": round(la, 4), "lon": round(lo, 4),
            "radius_km": round(float(haversine_km(la, lo, g["latitude"], g["longitude"]).max()), 2),
            "first_seen": g["acq_datetime"].min(), "last_seen": g["acq_datetime"].max(),
            "active_days": int(g["acq_date"].nunique()),
        })
    return pd.DataFrame(rows, columns=cols).sort_values("detections", ascending=False).reset_index(drop=True)


def hotspot_grid(df, cell=CELL):
    if df.empty:
        return pd.DataFrame(columns=["cy", "cx", "lat", "lon", "detections", "frp_sum", "share", "area_km2", "per_1000km2"])
    d = cell_ids(df, cell)
    g = d.groupby(["cy", "cx"]).agg(detections=("latitude", "size"), frp_sum=("frp", "sum")).reset_index()
    g["lat"], g["lon"] = (g["cy"] + 0.5) * cell, (g["cx"] + 0.5) * cell
    g["share"] = g["detections"] / g["detections"].sum()
    from .geodesy import cell_area_km2  # cells shrink towards the poles: normalise by true spherical area
    g["area_km2"] = cell_area_km2(g["cy"].to_numpy(), cell)
    g["per_1000km2"] = g["detections"] / g["area_km2"] * 1000
    return g.sort_values("detections", ascending=False).reset_index(drop=True)


def compare_periods(df, a, b, cell=CELL):
    """Daily detection rate per grid cell in period A vs period B (inclusive ranges)."""
    a0, a1, b0, b1 = (pd.Timestamp(x).normalize() for x in (*a, *b))
    da, db = (a1 - a0).days + 1, (b1 - b0).days + 1
    pa = cell_ids(df[(df["acq_date"] >= a0) & (df["acq_date"] <= a1)], cell)
    pb = cell_ids(df[(df["acq_date"] >= b0) & (df["acq_date"] <= b1)], cell)
    ca = pa.groupby(["cy", "cx"]).size().rename("a")
    cb = pb.groupby(["cy", "cx"]).size().rename("b")
    t = pd.concat([ca, cb], axis=1).fillna(0).reset_index()
    if t.empty:
        return t.assign(rate_a=[], rate_b=[], change=[], change_pct=[], lat=[], lon=[])
    t["rate_a"], t["rate_b"] = t["a"] / da, t["b"] / db
    t["change"] = t["rate_b"] - t["rate_a"]
    t["change_pct"] = np.where(t["rate_a"] > 0, (t["rate_b"] / t["rate_a"].replace(0, np.nan) - 1) * 100, np.nan)
    t["lat"], t["lon"] = (t["cy"] + 0.5) * cell, (t["cx"] + 0.5) * cell
    return t.sort_values("change", ascending=False).reset_index(drop=True)


def daily_counts(df):
    if df.empty:
        return pd.Series(dtype=float)
    s = df.groupby("acq_date").size()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0)


def activity_anomalies(df, window=14, z=2.5):
    """Flag days whose count is far above the trailing-window baseline."""
    s = daily_counts(df)
    out = pd.DataFrame({"date": s.index, "count": s.to_numpy()})
    prev = s.shift(1).rolling(window, min_periods=5)
    out["baseline"] = prev.mean().to_numpy()
    out["std"] = prev.std().clip(lower=1.0).to_numpy()
    out["zscore"] = (out["count"] - out["baseline"]) / out["std"]
    out["anomaly"] = out["zscore"] >= z
    return out


def diurnal(df):
    h = df["acq_datetime"].dt.hour
    return h.value_counts().reindex(range(24), fill_value=0).rename_axis("hour_utc").reset_index(name="detections")
