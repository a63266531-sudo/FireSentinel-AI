"""Vegetation context: NDVI (MODIS MOD13Q1) and land cover (MCD12Q1) through the free ORNL DAAC
MODIS web service (no login), or synthetic data for offline demos."""
import json
import time

import numpy as np
import pandas as pd
import requests

from .config import CACHE, get_logger
from .synthetic import synth_landcover, synth_ndvi

log = get_logger()
ORNL = "https://modis.ornl.gov/rst/api/v1"
LC_GROUP = {1: "forest", 2: "forest", 3: "forest", 4: "forest", 5: "forest", 6: "shrub", 7: "shrub", 8: "shrub",
            9: "shrub", 10: "grass", 11: "other", 12: "crop", 13: "other", 14: "crop", 15: "other", 16: "other", 17: "other"}
LC_FEATS = ["lc_forest", "lc_shrub", "lc_grass", "lc_crop"]
NDVI_AVAILABILITY_LAG_DAYS = 16  # a 16-day composite is only known after its window ends


def _get(path, params, retries=3):
    last = None
    for i in range(retries):
        try:
            r = requests.get(f"{ORNL}/{path}", params=params, headers={"Accept": "application/json"}, timeout=90)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:150]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 ** i)
    raise RuntimeError(f"ORNL request failed: {last}")


def fetch_ndvi_ornl(lat, lon, start, end):
    dates = _get("MOD13Q1/dates", {"latitude": lat, "longitude": lon})["dates"]
    sel = [d["modis_date"] for d in dates
           if start - pd.Timedelta(days=32) <= pd.to_datetime(d["calendar_date"]) <= end]
    rows = []
    for i in range(0, len(sel), 10):
        ch = sel[i:i + 10]
        j = _get("MOD13Q1/subset", {"latitude": lat, "longitude": lon, "band": "250m_16_days_NDVI",
                                    "startDate": ch[0], "endDate": ch[-1], "kmAboveBelow": 0, "kmLeftRight": 0})
        for s in j.get("subset", []):
            v = np.array(s["data"], dtype=float)
            v = v[v > -3000]  # fill value is -3000
            if len(v):
                rows.append({"date": pd.to_datetime(s["calendar_date"]) + pd.Timedelta(days=NDVI_AVAILABILITY_LAG_DAYS),
                             "ndvi": float(v.mean() * 0.0001)})
    if not rows:
        raise RuntimeError("No NDVI values returned")
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def get_ndvi(lat, lon, start, end, mode="synthetic"):
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if mode == "synthetic":
        return synth_ndvi(lat, lon, pd.date_range(start, end, freq="D"))
    path = CACHE / f"ndvi_{lat:.2f}_{lon:.2f}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    d = fetch_ndvi_ornl(lat, lon, start, end)
    d.to_csv(path, index=False)
    return d


def weekly_ndvi(nd, start, end):
    s = nd.set_index("date")["ndvi"].sort_index()
    idx = pd.date_range(min(s.index.min(), pd.Timestamp(start)), pd.Timestamp(end), freq="D")
    daily = s.reindex(idx).ffill()  # forward fill only: never use later composites for earlier days
    wk = daily.index.to_period("W").start_time.astype("datetime64[ns]")
    return daily.groupby(wk).mean().rename("ndvi").rename_axis("week").reset_index()


def get_landcover(lat, lon, year, mode="synthetic"):
    """Returns one of forest / shrub / grass / crop / other."""
    if mode == "synthetic":
        return synth_landcover(lat, lon)
    path = CACHE / f"lc_{lat:.2f}_{lon:.2f}_{year}.json"
    if path.exists():
        return json.loads(path.read_text())["group"]
    dates = _get("MCD12Q1/dates", {"latitude": lat, "longitude": lon})["dates"]
    cand = [d for d in dates if pd.to_datetime(d["calendar_date"]).year <= year] or dates
    d = cand[-1]
    j = _get("MCD12Q1/subset", {"latitude": lat, "longitude": lon, "band": "LC_Type1", "startDate": d["modis_date"],
                                "endDate": d["modis_date"], "kmAboveBelow": 0, "kmLeftRight": 0})
    group = LC_GROUP.get(int(j["subset"][0]["data"][0]), "other")
    path.write_text(json.dumps({"group": group}))
    return group
