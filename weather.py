"""Weather / terrain context: Open-Meteo (free, no key) or synthetic for demos."""
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import CACHE, get_logger
from .synthetic import synth_weather

log = get_logger()
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
DAILY = ("temperature_2m_max,precipitation_sum,wind_speed_10m_max,"
         "et0_fao_evapotranspiration,relative_humidity_2m_mean")
RENAME = {"time": "date", "temperature_2m_max": "temp_max", "precipitation_sum": "precip",
          "wind_speed_10m_max": "wind_max", "et0_fao_evapotranspiration": "et0",
          "relative_humidity_2m_mean": "rh_mean"}


def _request(url, params):
    last = None
    for daily in (DAILY, DAILY.rsplit(",", 1)[0]):  # retry without humidity if unsupported
        try:
            r = requests.get(url, params={**params, "daily": daily, "timezone": "UTC"}, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = r.text[:200]
        except requests.RequestException as e:
            last = str(e)
            time.sleep(1)
    raise RuntimeError(f"Open-Meteo request failed: {last}")


def _to_df(j):
    d = pd.DataFrame(j["daily"]).rename(columns=RENAME)
    d["date"] = pd.to_datetime(d["date"])
    d["elevation"] = float(j.get("elevation") or 0.0)
    return d


def fetch_open_meteo(lat, lon, start, end):
    today = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()
    cut = today - timedelta(days=7)  # the archive lags a few days behind
    parts = []
    if start <= cut:
        parts.append(_to_df(_request(ARCHIVE, {"latitude": lat, "longitude": lon,
                                               "start_date": start.strftime("%Y-%m-%d"),
                                               "end_date": min(end, cut).strftime("%Y-%m-%d")})))
    if end > cut:
        past = int(min(92, (today - max(start, cut + timedelta(days=1))).days + 1))
        parts.append(_to_df(_request(FORECAST, {"latitude": lat, "longitude": lon,
                                                "past_days": past, "forecast_days": 16})))
    d = pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="first")
    return d[(d["date"] >= start) & (d["date"] <= end)].sort_values("date").reset_index(drop=True)


def get_weather(lat, lon, start, end, mode="synthetic"):
    """Daily weather for one point. mode: 'synthetic' | 'open-meteo'."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if mode == "synthetic":
        return synth_weather(lat, lon, pd.date_range(start, end, freq="D"))
    path = CACHE / f"wx_{lat:.2f}_{lon:.2f}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    recent = end > pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) - timedelta(days=8)
    if path.exists() and (not recent or time.time() - path.stat().st_mtime < 12 * 3600):
        return pd.read_csv(path, parse_dates=["date"])
    d = fetch_open_meteo(lat, lon, start, end)
    d.to_csv(path, index=False)
    log.info("Weather fetched lat=%.2f lon=%.2f rows=%d", lat, lon, len(d))
    return d


def fwi_lite(tmax, wind, precip14, rh):
    """Transparent heuristic dryness/fire-weather index (NOT the official Canadian FWI)."""
    rh = rh.fillna(50)
    return (tmax.clip(lower=0) / 30) * (1 + wind / 20) * (1 + (100 - rh) / 100) / (1 + precip14 / 10)


def weekly_weather(daily):
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["precip14"] = d["precip"].rolling(14, min_periods=1).sum()
    d["week"] = d["date"].dt.to_period("W").dt.start_time.astype("datetime64[ns]")
    if "rh_mean" not in d:
        d["rh_mean"] = float("nan")
    w = d.groupby("week").agg(
        tmax_mean=("temp_max", "mean"), precip_sum=("precip", "sum"), wind_max=("wind_max", "max"),
        et0_sum=("et0", "sum"), rh_mean=("rh_mean", "mean"),
        dry_days=("precip", lambda s: int((s < 1).sum())),
        precip14=("precip14", "last"), elevation=("elevation", "first")).reset_index()
    w["fwi"] = fwi_lite(w["tmax_mean"], w["wind_max"], w["precip14"], w["rh_mean"])
    return w
