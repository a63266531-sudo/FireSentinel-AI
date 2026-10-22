"""Service layer: all API/CLI-facing operations as plain functions returning JSON-safe dicts (testable without a web server)."""
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import __version__, contract, data, experiments, geo, geodesy, model, pipeline, providers, store
from .config import DB_PATH, EXPERIMENTS, MODELS, REGIONS
from .security import redact

MAX_DAYS, MAX_WEEKS, MAX_CELLS = 120, 104, 200
WEATHER_MODES = ("open-meteo", "synthetic")


class ServiceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _now():
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


def _json(d: pd.DataFrame):
    return json.loads(d.to_json(orient="records", date_format="iso"))


def _region(region):
    if region not in REGIONS:
        raise ServiceError(404, f"Unknown region '{region}'. Options: {list(REGIONS)}")
    return REGIONS[region]


def _frame(region, days, db_path, max_days=MAX_DAYS):
    if not 1 <= days <= max_days:
        raise ServiceError(422, f"days must be between 1 and {max_days}")
    now = _now()
    df = store.load(_region(region), now - timedelta(days=days), now, path=db_path)
    if df.empty:
        raise ServiceError(404, "No data stored for this region/period. Run the fetch or scheduler first.")
    return data.validate(df)[0]


def health():
    return {"status": "ok", "version": __version__, "time_utc": _now().isoformat()}


def readiness(db_path=DB_PATH, models_dir=MODELS):
    checks = {}
    try:
        con = store.connect(db_path)
        n = con.execute("SELECT COUNT(*), MAX(acq_datetime) FROM detections").fetchone()
        con.close()
        checks["database"] = {"ok": True, "detections": n[0], "latest_observation": n[1]}
    except Exception as e:
        checks["database"] = {"ok": False, "error": redact(e)}
    try:
        b, m = model.load_latest(models_dir)
        checks["model"] = {"ok": True, "present": b is not None, "version": (m or {}).get("version")}
    except Exception as e:
        checks["model"] = {"ok": False, "present": True, "error": redact(e)}
    return {"ready": checks["database"]["ok"] and checks["model"]["ok"], "checks": checks}


def sources_status(db_path=DB_PATH):
    keys = {"FIRMS_MAP_KEY": bool(os.environ.get("FIRMS_MAP_KEY")), "SMTP": bool(os.environ.get("SMTP_HOST")),
            "TELEGRAM": bool(os.environ.get("TELEGRAM_BOT_TOKEN"))}
    runs = store.recent_runs(5, db_path)
    return {"providers": _json(providers.describe_all()[["id", "name", "product", "access", "auth", "license", "verified_live"]]),
            "credentials_configured": keys, "recent_fetch_runs": _json(runs),
            "last_successful_fetch": next((r["retrieved_at"] for r in _json(runs) if r["status"] == "ok"), None),
            "note": "A listed provider is not proof that it is reachable; failed runs appear with status 'failed' or 'partial'."}


def model_metadata(models_dir=MODELS):
    b, m = model.load_latest(models_dir)
    if b is None:
        return {"model": None, "message": "No saved model. Train one and save it (dashboard or CLI)."}
    return {k: m.get(k) for k in ("version", "trained_at_utc", "model", "features", "split", "beats_baseline", "beats_baseline_ci",
                                   "calibration_supported", "ece", "ci", "origin", "weather", "limitations", "sha256")}


def validate_region(region, days=7, db_path=DB_PATH):
    df = _frame(region, days, db_path)
    _, rep = data.validate(df)
    return {"quality": rep, "coordinates": geodesy.check_geographic_degrees(df),
            "coverage": geodesy.coverage_summary(df, _region(region))}


def risk(region, weeks=26, weather="open-meteo", db_path=DB_PATH, models_dir=MODELS, max_cells=100):
    if weather not in WEATHER_MODES:
        raise ServiceError(422, f"weather must be one of {WEATHER_MODES}")
    if not 12 <= weeks <= MAX_WEEKS or not 10 <= max_cells <= MAX_CELLS:
        raise ServiceError(422, f"weeks must be 12..{MAX_WEEKS} and max_cells 10..{MAX_CELLS}")
    try:
        bundle, meta = model.load_latest(models_dir)
    except ValueError as e:
        raise ServiceError(500, redact(e))
    if bundle is None:
        raise ServiceError(404, "No saved model. Train one and save it first.")
    df = _frame(region, weeks * 7, db_path, MAX_WEEKS * 7)
    g, _ = pipeline.build_only(df, weather, max_cells=max_cells, **pipeline.flags_from_features(bundle["features"]))
    info = contract.info_from_bundle(bundle, meta)
    cells, summary = contract.build_contract(info, g, df, _now(), synthetic=(meta.get("origin") == "demo" or weather == "synthetic"))
    cells["forecast_week_start"] = cells["forecast_week_start"].astype(str)
    return {"summary": summary, "cells": _json(cells), "weather_source": weather,
            "note": "Estimates, not observations or official warnings. Probabilities appear only when calibration was demonstrated."}


def geojson(region, days=3, kind="detections", eps_km=3.0, min_samples=3, db_path=DB_PATH):
    df = _frame(region, days, db_path)
    if kind == "detections":
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.longitude, r.latitude]},
                  "properties": {"time_utc": str(r.acq_datetime), "frp": float(r.frp), "confidence": r.conf_level, "source": r.source}}
                 for r in df.tail(20000).itertuples()]
    elif kind == "clusters":
        s = geo.cluster_summary(geo.cluster(df, eps_km, min_samples))
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                  "properties": {"cluster": int(r.cluster), "detections": int(r.detections), "radius_km": r.radius_km, "frp_sum": r.frp_sum}}
                 for r in s.itertuples()]
    else:
        raise ServiceError(422, "kind must be 'detections' or 'clusters'")
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"crs": "EPSG:4326", "note": "Unconfirmed satellite thermal anomalies."}}


def list_experiments(out_dir=None):
    return {"experiments": _json(experiments.list_experiments(out_dir))}


def get_experiment(eid, out_dir=None):
    try:
        return experiments.get_experiment(eid, out_dir)
    except ValueError:
        raise ServiceError(422, "invalid experiment id")
    except OSError:
        raise ServiceError(404, "experiment not found")
