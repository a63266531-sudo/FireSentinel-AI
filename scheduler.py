"""Automatic updates: fetch -> store -> alerts -> notify.   python -m firesentinel.cli schedule --key ..."""
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import alerts, data, geo, notify, store
from .config import DATA, REGIONS, get_logger

log = get_logger()


def load_zones():
    p = DATA / "zones.csv"  # optional: name,lat,lon,radius_km
    return pd.read_csv(p) if p.exists() else None


def run_once(key, sources, region, days=2, channels=(), min_severity="warning", dry_run=True,
             fetcher=None, db_path=None):
    fetcher = fetcher or data.fetch_range
    kw = {"path": db_path} if db_path else {}
    bbox, now = REGIONS[region], pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    summary = {"fetched": 0, "inserted": 0, "errors": [], "alerts": 0, "notify": None}
    for s in sources:
        raw, errs = fetcher(key, s, bbox, now - timedelta(days=days - 1), now)
        summary["errors"] += errs
        if len(raw):
            data.save_raw(raw, s, bbox, now, out_dir=(__import__('pathlib').Path(db_path).parent if db_path else None))
            clean, _ = data.validate(raw, s)
            ins = store.upsert(clean, now, **kw)
            store.log_run(now, s, bbox, len(raw), ins, "partial" if errs else "ok", "; ".join(errs), **kw)
            summary["fetched"] += len(raw)
            summary["inserted"] += ins
        else:
            store.log_run(now, s, bbox, 0, 0, "failed", "; ".join(errs), **kw)
    df = store.load(bbox, now - timedelta(days=14), now, **kw)
    if df.empty:
        return summary
    df, _ = data.validate(df)
    if len(sources) > 1:
        df, _ = data.dedupe_across_sensors(df)
    al = alerts.collect(df, geo.cluster(df, 3.0, 3), load_zones(), now)
    summary["alerts"] = len(al)
    if channels:
        summary["notify"] = notify.dispatch(al, list(channels), min_severity, dry_run, **kw)
    log.info("scheduler run: %s", {k: v for k, v in summary.items() if k != "notify"})
    return summary


def loop(every_hours=3.0, **kw):
    while True:
        try:
            print(run_once(**kw))
        except Exception as e:  # keep the service alive
            log.exception("scheduler run failed: %s", e)
            print("run failed:", e)
        time.sleep(every_hours * 3600)
