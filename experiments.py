"""Reproducible experiments: every run records data fingerprint, label definition, split boundaries, features, model
parameters, seeds, software versions, metrics (with block-bootstrap CIs), spatial / spatio-temporal / walk-forward validation
and the exact command to repeat it. Re-running with the same inputs, seed and library versions should give the same
metrics; `verify_reproduction` checks this on your machine (it does not claim bit-for-bit equality across machines)."""
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from . import __version__, model, monitoring, pipeline, providers, report
from .config import EXPERIMENTS
from .security import safe_join

LABEL_DEFINITION = ("Positive (1): at least one FIRMS active-fire detection (any confidence) inside the grid cell during the NEXT ISO week "
                    "(Mon-Sun, UTC). Negative (0): no detection. A 0 does NOT prove that no fire occurred (cloud, overpass gaps, sensor "
                    "limits); a 1 is an unconfirmed thermal anomaly (it may be industrial heat or a gas flare).")
FEATURE_DOCS = {
    "count": "detections in the cell this week", "lag1": "detections one week ago", "lag2": "detections two weeks ago",
    "roll4": "mean weekly detections over the last 4 weeks", "hist_mean": "expanding mean of weekly detections up to this week",
    "frp": "mean fire radiative power this week (MW)", "frp_sum": "total FRP this week (MW)", "m_sin": "sin(2*pi*month/12)", "m_cos": "cos(2*pi*month/12)",
    "tmax_mean": "mean daily max temperature (C)", "precip_sum": "weekly precipitation (mm)", "wind_max": "max daily wind speed (km/h)",
    "et0_sum": "weekly reference evapotranspiration (mm)", "rh_mean": "mean relative humidity (%)", "dry_days": "days with < 1 mm rain",
    "precip14": "precipitation in the last 14 days (mm)", "fwi": "heuristic dryness index (NOT the official FWI)", "elevation": "elevation (m)",
    "ndvi": "weekly NDVI (16-day composite, 16-day availability lag)", "ndvi_anom": "NDVI minus the cell's expanding mean",
    "ndvi_chg": "week-to-week NDVI change", "lc_forest": "land cover forest", "lc_shrub": "land cover shrub/savanna",
    "lc_grass": "land cover grassland", "lc_crop": "land cover cropland", "nx_tmax": "NEXT week mean Tmax (perfect-prognosis)",
    "nx_wind": "NEXT week max wind (perfect-prognosis)", "nx_precip": "NEXT week precipitation (perfect-prognosis)",
    "nx_fwi": "NEXT week dryness index (perfect-prognosis)", "nb_count": "detections in the 8 neighbouring cells this week",
    "nb_lag1": "detections in the 8 neighbouring cells last week"}
DEFAULT_CONFIG = dict(name="experiment", wx_mode="synthetic", cell=0.5, max_cells=60, neg_ratio=0.3, use_weather=True, use_veg=True,
                      use_nextwx=False, use_spatial=True, tune=False, seed=42)


def dataset_fingerprint(df) -> str:
    cols = [c for c in ("latitude", "longitude", "acq_datetime", "frp", "source") if c in df.columns]
    d = df[cols].sort_values(cols[:3]).reset_index(drop=True)
    return providers.fingerprint(d.assign(**{c: d[c].round(5) for c in ("latitude", "longitude") if c in d}))


def software_versions() -> dict:
    return {"firesentinel": __version__, "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}


def _recs(d):
    return None if d is None else d.reset_index(drop=True).to_dict("records")


def run_experiment(df, config=None, origin="demo", demo_params=None, progress=None, out_dir=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    seed = int(cfg["seed"])
    g, feats = pipeline.build_only(df, cfg["wx_mode"], cfg["cell"], cfg["max_cells"], cfg["neg_ratio"], cfg["use_weather"],
                                   cfg["use_veg"], cfg["use_nextwx"], cfg["use_spatial"], progress)
    res, msg = model.evaluate(g, feats, seed=seed, tune=cfg["tune"])
    if res is None:
        raise ValueError(msg)
    best = res["best"]
    st = model.evaluate_spatiotemporal(g, feats, best, seed)
    wf, pooled = model.walk_forward(g, feats, best, seed=seed)
    gcv = model.geo_cv(g, feats, best, seed=seed)
    imp = model.importance(res)
    recent = g[g["week"] >= g["week"].max() - pd.Timedelta(days=21)]
    drift = monitoring.drift_report(res["train_ref"], recent, feats)
    est = res["models"][best]
    final = est.steps[-1][1] if hasattr(est, "steps") else est
    fp = dataset_fingerprint(df)
    now = datetime.now(timezone.utc)
    eid = f"exp-{now:%Y%m%dT%H%M%SZ}-{fp[:8]}"
    synthetic = origin == "demo"
    cmd = (f"python -m firesentinel.cli experiment --demo --region {demo_params['region']} --weeks {demo_params['weeks']} "
           f"--demo-seed {demo_params['demo_seed']} --end {demo_params['end']} --seed {seed}" if synthetic and demo_params else
           f"python -m firesentinel.cli experiment --region <REGION> --weather {cfg['wx_mode']} --seed {seed}  # same database content required")
    exp = {
        "id": eid, "created_at_utc": now.isoformat(), "name": cfg["name"], "origin": origin,
        "scientific_evidence": False if synthetic else "only for the stated region/period; see limitations",
        "data": {"fingerprint_sha256": fp, "rows": int(len(df)), "first_observation": str(df["acq_datetime"].min()),
                 "last_observation": str(df["acq_datetime"].max()), "sources": sorted(map(str, df["source"].unique())),
                 "extraction_note": "Raw downloads are kept under data/raw with retrieval timestamps; provenance table in the database.",
                 "demo_params": demo_params, "provenance_weather": g.attrs.get("weather_source")},
        "label_definition": LABEL_DEFINITION, "unit": f"{cfg['cell']} degree cell x ISO week; horizon 1 week",
        "cell_selection": {"cutoff": g.attrs.get("selection_cutoff"), "fallback_used": g.attrs.get("selection_fallback"), "n_cells": g.attrs.get("n_cells"),
                           "rule": "positive cells = most active cells BEFORE the cutoff only (no test-period information)"},
        "split": res["info"], "features": feats, "feature_definitions": {f: FEATURE_DOCS.get(f, "") for f in feats},
        "preprocessing": "median imputation + scaling fitted inside each model pipeline on training rows only; isotonic calibration and decision "
                         "threshold fitted on the validation period only",
        "model": {"selected": best, "selection_rule": res["info"]["selection"], "estimator": type(final).__name__,
                  "parameters": {k: str(v) for k, v in final.get_params().items()}, "tuned": bool(cfg["tune"])},
        "seeds": {"experiment": seed, "demo": (demo_params or {}).get("demo_seed")}, "config": cfg, "software": software_versions(),
        "metrics_test": _recs(res["metrics"]), "bootstrap_ci_test": _recs(res["ci"]),
        "calibration": {"ECE": round(res["ece"], 4), "supported": res["calibration_supported"], "table": _recs(res["calib_table"])},
        "beats_baselines": {"point_estimate": res["beats_baseline"], "ci_excludes_zero": res["beats_baseline_ci"]},
        "spatiotemporal_holdout": {"table": _recs(st), **({} if st is None else {k: v for k, v in st.attrs.items()})} if st is not None else None,
        "walk_forward_hindcast": {"per_origin": _recs(wf), "pooled": pooled},
        "spatial_block_cv": _recs(gcv), "permutation_importance": _recs(imp.head(15)) if imp is not None else None,
        "input_drift_last_3_weeks": _recs(drift.head(10)), "tuning": _recs(res["tuning"]),
        "reproduce": cmd,
        "limitations": ["Labels are satellite detections, not ground truth.", "Absence of a detection is not absence of fire.",
                        "Few test weeks give wide confidence intervals; see n_test_weeks.", "Importance is not causation.",
                        "One region/period does not establish generalisation to other seasons or places.",
                        "Synthetic data prove only that the software runs." if synthetic else "Real-data skill must be re-measured per region and season."],
    }
    out = Path(out_dir or EXPERIMENTS) / eid
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.json").write_text(json.dumps(exp, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (out / "report.md").write_text(report.build_experiment_md(exp))
    res["metrics"].to_csv(out / "metrics_test.csv", index=False)
    return exp, out


def run_demo_experiment(region="Greece", weeks=26, demo_seed=7, end=None, config=None, out_dir=None, progress=None):
    from . import data
    if end is None:
        end = str(pipeline.make_demo(region, 7, demo_seed)[0]["acq_date"].max())  # last complete Sunday
    raw, _ = pipeline.make_demo(region, weeks * 7, demo_seed, end)
    df, _ = data.validate(raw, "SYNTHETIC")
    params = {"region": region, "weeks": weeks, "demo_seed": demo_seed, "end": str(pd.Timestamp(end).date())}
    return run_experiment(df, config, "demo", params, progress, out_dir)


def verify_reproduction(exp_dir, out_dir=None, tol=1e-9):
    """Re-runs a DEMO experiment from its recorded parameters and compares the test metrics."""
    exp = json.loads((Path(exp_dir) / "experiment.json").read_text())
    p = (exp["data"] or {}).get("demo_params")
    if exp["origin"] != "demo" or not p:
        raise ValueError("Automatic verification is implemented for demo experiments (real-data runs need the same database content).")
    import tempfile
    out = Path(out_dir or tempfile.mkdtemp())
    new, _ = run_demo_experiment(p["region"], p["weeks"], p["demo_seed"], p["end"], exp["config"], out)
    a, b = pd.DataFrame(exp["metrics_test"]), pd.DataFrame(new["metrics_test"])
    num = a.select_dtypes("number").columns
    diffs = [] if a.shape == b.shape and (a["Model"] == b["Model"]).all() else ["table shape/model order differs"]
    if not diffs:
        d = (a[num].astype(float) - b[num].astype(float)).abs().max().max()
        diffs = [f"max abs metric difference {d:.3g}"] if not d <= tol else []
    return {"match": not diffs and new["data"]["fingerprint_sha256"] == exp["data"]["fingerprint_sha256"], "differences": diffs,
            "same_data_fingerprint": new["data"]["fingerprint_sha256"] == exp["data"]["fingerprint_sha256"],
            "software_changed": new["software"] != exp["software"]}


def list_experiments(out_dir=None):
    rows = []
    for p in sorted(Path(out_dir or EXPERIMENTS).glob("exp-*/experiment.json"), reverse=True):
        try:
            e = json.loads(p.read_text())
            rows.append({"id": e["id"], "created_at_utc": e["created_at_utc"], "origin": e["origin"], "model": e["model"]["selected"],
                         "beats_baselines_95CI": e["beats_baselines"]["ci_excludes_zero"], "name": e["name"]})
        except (OSError, ValueError, KeyError):
            continue
    return pd.DataFrame(rows)


def get_experiment(eid, out_dir=None):
    if not re.fullmatch(r"exp-\d{8}T\d{6}Z-[0-9a-f]{8}", eid):
        raise ValueError("invalid experiment id")
    return json.loads((safe_join(Path(out_dir or EXPERIMENTS), eid) / "experiment.json").read_text())
