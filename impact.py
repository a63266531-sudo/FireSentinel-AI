"""Exposure estimate: which villages / roads / protected areas lie near detection clusters.
Sources: OpenStreetMap (Overpass API, no key), a user CSV (name, lat, lon, type), or synthetic demo assets."""
import numpy as np
import pandas as pd
import requests

from .geo import haversine_km

OVERPASS = "https://overpass-api.de/api/interpreter"
COLS = ["name", "type", "lat", "lon", "population"]


def validate_assets(df):
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    miss = {"name", "lat", "lon"} - set(d.columns)
    if miss:
        raise ValueError(f"Assets CSV needs columns name, lat, lon (missing {sorted(miss)})")
    d["lat"], d["lon"] = pd.to_numeric(d["lat"], errors="coerce"), pd.to_numeric(d["lon"], errors="coerce")
    d = d.dropna(subset=["lat", "lon"])
    d = d[d["lat"].between(-90, 90) & d["lon"].between(-180, 180)]
    for c, default in (("type", "asset"), ("population", np.nan)):
        if c not in d.columns:
            d[c] = default
    return d[COLS].reset_index(drop=True)


def overpass_assets(lat, lon, radius_km=15, timeout=40):
    r = int(radius_km * 1000)
    q = (f'[out:json][timeout:{timeout}];('
         f'node["place"~"^(city|town|village|hamlet)$"](around:{r},{lat},{lon});'
         f'way["boundary"="protected_area"](around:{r},{lat},{lon});'
         f'way["leisure"="nature_reserve"](around:{r},{lat},{lon});'
         f'way["highway"~"^(motorway|trunk|primary)$"](around:{r},{lat},{lon}););out center tags 80;')
    resp = requests.post(OVERPASS, data={"data": q}, timeout=timeout + 20)
    resp.raise_for_status()
    rows = []
    for e in resp.json().get("elements", []):
        tags, c = e.get("tags", {}), (e.get("center") or {"lat": e.get("lat"), "lon": e.get("lon")})
        if c.get("lat") is None:
            continue
        typ = tags.get("place") or ("protected area" if tags.get("boundary") == "protected_area"
                                    or tags.get("leisure") == "nature_reserve" else "major road")
        rows.append({"name": tags.get("name", typ), "type": typ, "lat": c["lat"], "lon": c["lon"],
                     "population": pd.to_numeric(tags.get("population"), errors="coerce")})
    return pd.DataFrame(rows, columns=COLS)


def exposure(summ, assets, radius_km=10.0, top=10):
    """Returns (detail, per_cluster). Distance is measured from the cluster centre minus its radius."""
    detail, per = [], []
    for c in summ.head(top).itertuples():
        if assets.empty:
            continue
        d = haversine_km(c.lat, c.lon, assets["lat"].to_numpy(), assets["lon"].to_numpy())
        edge = np.clip(d - c.radius_km, 0, None)
        near = assets[edge <= radius_km].assign(cluster=c.cluster, distance_km=np.round(edge[edge <= radius_km], 1))
        detail.append(near)
        per.append({"cluster": c.cluster, "detections": c.detections, "assets_within_km": len(near),
                    "nearest": near.sort_values("distance_km")["name"].iloc[0] if len(near) else "-",
                    "nearest_km": float(near["distance_km"].min()) if len(near) else np.nan,
                    "people_in_range": float(near["population"].sum()) if len(near) else 0.0})
    detail_df = (pd.concat(detail, ignore_index=True).sort_values(["cluster", "distance_km"])
                 if detail else pd.DataFrame(columns=COLS + ["cluster", "distance_km"]))
    return detail_df, pd.DataFrame(per, columns=["cluster", "detections", "assets_within_km", "nearest", "nearest_km", "people_in_range"])
