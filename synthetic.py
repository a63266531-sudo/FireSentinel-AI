"""Synthetic (offline) data so the full workflow can be demonstrated without
internet or an API key. Weather, NDVI and land cover drive fire probability, so the
ML pipeline has signal to find. NEVER present synthetic results as real findings."""
import numpy as np
import pandas as pd

from .config import CELL

LC_NAMES = ["forest", "shrub", "grass", "crop", "other"]
LC_EFFECT = {"forest": 0.5, "shrub": 0.4, "grass": 0.1, "crop": -0.3, "other": -0.6}


def _seed(lat, lon, salt=0):
    return (abs(int(round(lat * 100))) * 100003 + abs(int(round(lon * 100))) * 7919 + salt) % (2 ** 32)


def synth_weather(lat, lon, dates):
    dates = pd.DatetimeIndex(dates)
    seed = _seed(lat, lon)
    rng = np.random.default_rng(seed)
    n = len(dates)
    season = np.sin((dates.dayofyear.to_numpy() - 110) / 365 * 2 * np.pi)
    smooth = lambda x: pd.Series(x).rolling(4, min_periods=1).mean().to_numpy()
    temp = 24 + 9 * season - 0.15 * abs(lat - 30) + smooth(rng.normal(0, 4, n))
    wind = np.clip(rng.gamma(4, 3.0, n), 0, None)
    rain = rng.random(n) < (0.20 - 0.10 * season)
    precip = np.where(rain, rng.exponential(7, n), 0.0)
    rh = np.clip(55 - 16 * season - 0.5 * (temp - 24) + 25 * (precip > 0) + rng.normal(0, 6, n), 8, 100)
    et0 = np.clip(0.2 * temp - 1.5 + 0.04 * wind + rng.normal(0, 0.4, n), 0.2, None)
    return pd.DataFrame({"date": dates, "temp_max": temp, "precip": precip, "wind_max": wind,
                         "et0": et0, "rh_mean": rh, "elevation": float(seed % 1500)})


def synth_ndvi(lat, lon, dates):
    dates = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(_seed(lat, lon, 11))
    base = rng.uniform(0.30, 0.60)
    season = 0.18 * np.sin((dates.dayofyear.to_numpy() - 80) / 365 * 2 * np.pi)
    noise = pd.Series(rng.normal(0, 0.05, len(dates))).rolling(10, min_periods=1).mean().to_numpy()
    return pd.DataFrame({"date": dates, "ndvi": np.clip(base + season + noise, 0.05, 0.9)})


def synth_landcover(lat, lon):
    return LC_NAMES[_seed(lat, lon, 5) % len(LC_NAMES)]


def generate_fires(bbox, start, end, n_cells=30, seed=7, cell=CELL):
    """FIRMS-like detections. Rate depends on the same weather/NDVI/land cover the context modules serve."""
    rng = np.random.default_rng(seed)
    w, s, e, n = bbox
    cys = np.arange(int(np.floor(s / cell)), int(np.floor(n / cell)))
    cxs = np.arange(int(np.floor(w / cell)), int(np.floor(e / cell)))
    picks = set()
    while len(picks) < min(n_cells, len(cys) * len(cxs)):
        picks.add((int(rng.choice(cys)), int(rng.choice(cxs))))
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for cy, cx in sorted(picks):
        lat0, lon0 = (cy + 0.5) * cell, (cx + 0.5) * cell
        wx = synth_weather(lat0, lon0, dates)
        nd = synth_ndvi(lat0, lon0, dates)["ndvi"]
        zt = ((wx["temp_max"] - wx["temp_max"].mean()) / wx["temp_max"].std()).to_numpy()
        zw = ((wx["wind_max"] - wx["wind_max"].mean()) / wx["wind_max"].std()).to_numpy()
        zn = ((nd - nd.mean()) / (nd.std() + 1e-9)).to_numpy()
        p14 = wx["precip"].rolling(14, min_periods=1).sum().to_numpy()
        lc = LC_EFFECT[synth_landcover(lat0, lon0)]
        base, prev = rng.uniform(-2.9, -1.7), 0
        for i, day in enumerate(dates):
            lam = np.exp(base + lc + 1.2 * zt[i] + 0.6 * zw[i] - 0.05 * p14[i] - 0.6 * zn[i]) + 0.35 * prev
            k = int(min(rng.poisson(min(lam, 40)), 60))
            prev = k
            for _ in range(k):
                t = int(rng.choice([130, 145, 1320, 1335, 1420]))
                rows.append({
                    "latitude": round(cy * cell + rng.uniform(0.001, cell - 0.001), 5),
                    "longitude": round(cx * cell + rng.uniform(0.001, cell - 0.001), 5),
                    "bright_ti4": round(float(rng.normal(340, 15)), 1),
                    "acq_date": day.strftime("%Y-%m-%d"), "acq_time": t,
                    "satellite": "N", "instrument": "VIIRS",
                    "confidence": str(rng.choice(["l", "n", "h"], p=[0.15, 0.65, 0.20])),
                    "frp": round(float(rng.gamma(2, 3)), 2),
                    "daynight": "D" if t >= 600 else "N",
                })
    return pd.DataFrame(rows)


def generate_assets(bbox, n=40, seed=3):
    """Fake villages / roads / reserves for the offline impact demo."""
    rng = np.random.default_rng(seed)
    w, s, e, nn = bbox
    types = rng.choice(["village", "town", "major road", "protected area"], size=n, p=[0.5, 0.2, 0.15, 0.15])
    return pd.DataFrame({"name": [f"Demo {t} {i + 1}" for i, t in enumerate(types)], "type": types,
                         "lat": rng.uniform(s, nn, n).round(4), "lon": rng.uniform(w, e, n).round(4),
                         "population": np.where(np.isin(types, ["village", "town"]), rng.integers(300, 60000, n), np.nan)})
