"""NASA FIRMS retrieval, raw storage and validation."""
import io
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from .config import RAW, get_logger

log = get_logger()
API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_chunk(key, source, bbox, days=5, date=None, retries=3):
    """One FIRMS area request (max 5 days). Retries on transient errors."""
    url = f"{API}/{key}/{source}/{','.join(str(x) for x in bbox)}/{days}"
    if date:
        url += f"/{date}"
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            txt = r.text.strip()
            if not txt:
                return pd.DataFrame()
            if "latitude" not in txt.split("\n")[0].lower():
                if txt.lower().startswith(("invalid", "error")):
                    raise PermissionError(txt[:200])  # no point retrying
                raise ValueError(txt[:200])
            return pd.read_csv(io.StringIO(txt))
        except PermissionError:
            raise
        except Exception as e:  # network / server problems; the key is inside the URL -> redact
            last = str(e).replace(str(key), "***")
            log.warning("FIRMS attempt %d failed: %s", i + 1, last)
            time.sleep(2 ** i)
    raise RuntimeError(f"FIRMS fetch failed after {retries} attempts: {last}")


def fetch_range(key, source, bbox, start, end, progress=None):
    """Fetch [start, end] in 5-day chunks. Returns (df, errors)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    parts, errors = [], []
    n_chunks = max(1, -(-((end - start).days + 1) // 5))
    for i in range(n_chunks):
        s = start + timedelta(days=5 * i)
        days = min(5, (end - s).days + 1)
        try:
            d = fetch_chunk(key, source, bbox, days, s.strftime("%Y-%m-%d"))
            if len(d):
                parts.append(d)
        except Exception as e:
            errors.append(f"{s.date()}: {e}")
            log.error("Chunk %s failed: %s", s.date(), e)
            if isinstance(e, PermissionError):
                break
        if progress:
            progress((i + 1) / n_chunks)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df, errors


def save_raw(df, source, bbox, retrieved_at, out_dir=None):
    """Keep an untouched copy of every download plus a metadata file."""
    stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    path = (Path(out_dir) if out_dir else RAW) / f"{source}_{stamp}.csv"
    df.to_csv(path, index=False)
    meta = {"source": source, "bbox": list(bbox), "rows": int(len(df)),
            "retrieved_at_utc": retrieved_at.isoformat()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def _conf_level(s: pd.Series) -> pd.Series:
    out = pd.Series("nominal", index=s.index, dtype="object")
    txt = s.astype(str).str.lower().str[0]
    num = pd.to_numeric(s, errors="coerce")
    out[txt == "l"] = "low"
    out[txt == "h"] = "high"
    out[num.notna() & (num < 30)] = "low"
    out[num.notna() & (num >= 80)] = "high"
    return out


def validate(df: pd.DataFrame, source: str | None = None):
    """Validate and standardise detections. Returns (clean_df, quality_report)."""
    rep = {"rows_raw": int(len(df))}
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    missing = {"latitude", "longitude", "acq_date"} - set(d.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
    d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
    ok = d["latitude"].between(-90, 90) & d["longitude"].between(-180, 180)
    rep["bad_coordinates"] = int((~ok).sum())
    d = d.loc[ok].copy()

    if "acq_time" not in d.columns and "acq_datetime" in d.columns:  # e.g. rows loaded back from the database
        d["acq_datetime"] = pd.to_datetime(d["acq_datetime"], errors="coerce")
    else:
        if "acq_time" in d.columns:
            t = pd.to_numeric(d["acq_time"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
        else:
            t = "0000"
        d["acq_datetime"] = pd.to_datetime(d["acq_date"].astype(str).str[:10] + " " + t,
                                           format="%Y-%m-%d %H%M", errors="coerce")
    rep["bad_timestamps"] = int(d["acq_datetime"].isna().sum())
    d = d.dropna(subset=["acq_datetime"])

    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    fut = d["acq_datetime"] > now + pd.Timedelta(days=1)
    rep["future_timestamps"] = int(fut.sum())
    d = d.loc[~fut].copy()

    if "source" not in d.columns:
        d["source"] = source or "unknown"
    for c in ("bright_ti4", "brightness"):
        if c in d.columns:
            d["brightness"] = pd.to_numeric(d[c], errors="coerce")
            break
    d["frp"] = pd.to_numeric(d["frp"], errors="coerce") if "frp" in d.columns else 0.0
    d["conf_level"] = _conf_level(d["confidence"]) if "confidence" in d.columns else "nominal"
    if "daynight" not in d.columns:
        d["daynight"] = "U"

    n = len(d)
    d = d.drop_duplicates(subset=["latitude", "longitude", "acq_datetime", "source"])
    rep["duplicates_removed"] = n - len(d)
    rep["missing_values"] = int(d[["latitude", "longitude", "acq_datetime", "frp"]].isna().sum().sum())
    d["frp"] = d["frp"].fillna(0.0)
    d["acq_date"] = d["acq_datetime"].dt.normalize()
    d = d.sort_values("acq_datetime").reset_index(drop=True)
    rep["rows_clean"] = int(len(d))
    rep["time_convention"] = "UTC (naive timestamps; FIRMS acq_time is UTC)"
    rep["crs"] = "EPSG:4326 (WGS84 degrees)"
    rep["first_observation"] = str(d["acq_datetime"].min()) if len(d) else None
    rep["last_observation"] = str(d["acq_datetime"].max()) if len(d) else None
    return d, rep


def dedupe_across_sensors(df, deg=0.005, hours=3):
    """Merge near-simultaneous detections of the same spot seen by DIFFERENT products
    (e.g. VIIRS S-NPP + NOAA-20 + MODIS): keep the best record (confidence, then FRP) per
    ~500 m / 3 h window. Returns (df, n_removed). Same-sensor neighbours are never merged."""
    if df.empty or df["source"].nunique() < 2:
        return df, 0
    d = df.copy()
    tb = (d["acq_datetime"] - pd.Timestamp("2000-01-01")) // pd.Timedelta(hours=hours)
    d["_k"] = (np.round(d["latitude"] / deg).astype(int).astype(str) + "_" +
               np.round(d["longitude"] / deg).astype(int).astype(str) + "_" + tb.astype(str))
    d["_rank"] = d["conf_level"].map({"low": 0, "nominal": 1, "high": 2}).fillna(1) if "conf_level" in d else 1
    multi = d.groupby("_k")["source"].transform("nunique") > 1
    best = d[multi].sort_values(["_rank", "frp"], ascending=False).drop_duplicates("_k")
    out = pd.concat([d[~multi], best]).sort_values("acq_datetime").drop(columns=["_k", "_rank"]).reset_index(drop=True)
    return out, len(df) - len(out)


def load_csv_chunked(file, bbox=None, max_rows=2_000_000, chunksize=200_000):
    """Lazy CSV reader for large archives: filters each chunk by bbox and stops at max_rows (memory limit)."""
    parts, total = [], 0
    for chunk in pd.read_csv(file, chunksize=chunksize):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        if bbox is not None and {"latitude", "longitude"} <= set(chunk.columns):
            w, s, e, n = bbox
            lat, lon = pd.to_numeric(chunk["latitude"], errors="coerce"), pd.to_numeric(chunk["longitude"], errors="coerce")
            chunk = chunk[lat.between(s, n) & lon.between(w, e)]
        parts.append(chunk)
        total += len(chunk)
        if total >= max_rows:
            log.warning("CSV truncated at %d rows (memory limit)", max_rows)
            break
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
