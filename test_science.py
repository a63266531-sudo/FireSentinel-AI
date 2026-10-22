"""Scientific-validity tests: leakage prevention, uncertainty, risk contract, scenario lab, geodesy, drift,
providers, experiments, service layer."""
import ast
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from firesentinel import (alerts, config, contract, data, experiments, features, geodesy, model, monitoring, pipeline,
                          providers, scenario, service, store)
from tests.test_extra import _demo

ROOT = Path(__file__).resolve().parent.parent


class Recorder:
    """Stand-in estimator that records which rows it was trained and asked to predict on."""

    def __init__(self):
        self.fit_idx, self.pred_idx = [], []

    def fit(self, X, y):
        self.fit_idx.append(X.index.to_numpy())
        return self

    def predict_proba(self, X):
        self.pred_idx.append(X.index.to_numpy())
        return np.c_[np.full(len(X), 0.7), np.full(len(X), 0.3)]


# ------------------------------------------------------------------ leakage prevention
def test_cell_selection_uses_only_history_before_cutoff():
    rows = [{"latitude": 10.1, "longitude": 10.1, "acq_datetime": pd.Timestamp("2025-01-10"), "frp": 1.0},   # cell A, early
            {"latitude": 20.1, "longitude": 20.1, "acq_datetime": pd.Timestamp("2025-03-10"), "frp": 1.0}]   # cell B, only after cutoff
    cells = features.select_cells(pd.DataFrame(rows), (0, 0, 40, 40), max_cells=10, neg_ratio=0.0, before=pd.Timestamp("2025-02-01"))
    got = set(zip(cells["cy"], cells["cx"]))
    assert (20, 20) in got and (40, 40) not in got


def test_missing_values_stay_nan_until_the_model_pipeline_fits_on_training_rows():
    weeks = pd.date_range("2025-01-06", periods=8, freq="W-MON").astype("datetime64[ns]")
    ts = pd.to_datetime(["2025-01-06 12:00", "2025-01-13 12:00", "2025-03-02 12:00"])
    df = pd.DataFrame({"latitude": 10.1, "longitude": 10.1, "frp": 1.0, "acq_datetime": ts})
    wx = pd.DataFrame({"cy": 20, "cx": 20, "week": weeks, "tmax_mean": [20.0, np.nan] * 4, "wind_max": 5.0, "precip_sum": 0.0, "fwi": 1.0})
    g = features.build_dataset(df, pd.DataFrame({"cy": [20], "cx": [20]}), wx)
    assert g["tmax_mean"].isna().any(), "no global median may be filled before the train/test split"
    X, y = g[["count", "tmax_mean"]].fillna({"count": 0}), (np.arange(len(g)) % 2)
    for name, m in model._models(1).items():
        m.fit(X, y)
        assert m.predict_proba(X).shape == (len(X), 2), name


def test_imputer_statistics_come_from_training_rows_only():
    pipe = model._models(1)["Logistic Regression"]
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan] * 10, "b": np.arange(40.0)})
    pipe.fit(train, (np.arange(40) % 2))
    assert pipe.steps[0][1].statistics_[0] == 2.0  # median of the training column
    pipe.predict_proba(pd.DataFrame({"a": [1000.0, np.nan], "b": [1.0, 2.0]}))
    assert pipe.steps[0][1].statistics_[0] == 2.0  # unchanged by what is predicted


def test_walk_forward_never_trains_on_labels_from_the_future():
    df, rep, g, res, bbox = _demo()
    rec = Recorder()
    with mock.patch.object(model, "_models", lambda seed=0: {"Logistic Regression": rec}):
        table, pooled = model.walk_forward(g, res["feats"], "Logistic Regression", n_origins=5)
    assert table is not None and len(rec.fit_idx) == len(rec.pred_idx) >= 2
    for fit_i, pred_i in zip(rec.fit_idx, rec.pred_idx):
        assert g.loc[fit_i, "week"].max() <= g.loc[pred_i, "week"].min() - pd.Timedelta(days=7)


def test_spatiotemporal_holdout_separates_places_and_time():
    df, rep, g, res, bbox = _demo()
    rec = Recorder()
    with mock.patch.object(model, "_models", lambda seed=0: {"Logistic Regression": rec}):
        out = model.evaluate_spatiotemporal(g, res["feats"], "Logistic Regression")
    assert out is not None and rec.fit_idx and rec.pred_idx
    blk = lambda i: (g.loc[i, "cy"] // 4).astype(str) + "_" + (g.loc[i, "cx"] // 4).astype(str)
    held = set(out.attrs["held_out_blocks"])
    assert not (set(blk(rec.fit_idx[-1])) & held) and set(blk(rec.pred_idx[-1])) <= held
    assert g.loc[rec.pred_idx[-1], "week"].min() >= pd.Timestamp(out.attrs["split_week"]) > g.loc[rec.fit_idx[-1], "week"].max()


# ------------------------------------------------------------------ uncertainty
def test_bootstrap_ci_and_baseline_gate_are_consistent():
    df, rep, g, res, bbox = _demo()
    ci = res["ci"]
    assert {"PR-AUC", "Brier", "PR-AUC minus best baseline"} <= set(ci["metric"])
    row = ci[ci["metric"] == "PR-AUC"].iloc[0]
    assert row["valid_resamples"] >= 20 and row["lo"] <= row["hi"] and row["n_test_weeks"] >= 3
    delta = ci[ci["metric"] == "PR-AUC minus best baseline"].iloc[0]
    assert res["beats_baseline_ci"] == bool(delta["lo"] > 0)


def test_probability_display_requires_demonstrated_calibration():
    df, rep, g, res, bbox = _demo()
    if res["calibration_supported"]:
        assert res["ece"] <= 0.10
    assert abs(model._ece([0, 1, 0, 1], [0, 1, 0, 1])) < 1e-9 and model._ece([0, 0, 0, 0], [1, 1, 1, 1]) == 1.0
    lo, hi = model.wilson_interval(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8 and all(np.isnan(model.wilson_interval(0, 0)))


# ------------------------------------------------------------------ risk contract
def _info(res, **kw):
    return {**contract.info_from_res(res), **kw}


def test_contract_withholds_probability_without_calibration_and_flags_unreliable_model():
    df, rep, g, res, bbox = _demo()
    now = df["acq_datetime"].max() + pd.Timedelta(hours=2)
    c, s = contract.build_contract(_info(res, calibration_supported=False, beats_baseline_ci=False), g, df, now, synthetic=True)
    assert c["calibrated_probability"].isna().all() and (c["status"] == "unreliable").all()
    assert c["reasons"].str.contains("not a calibrated probability").all() and c["reasons"].str.contains("beat simple baselines").all()
    assert c["synthetic_data"].all() and s["calibration_supported"] is False and "limitations" in s
    assert c["relative_rank_pct"].between(0, 100).all() and c["level"].str.contains("rank").all()


def test_contract_status_ok_unavailable_stale_and_out_of_distribution():
    df, rep, g, res, bbox = _demo()
    now = df["acq_datetime"].max() + pd.Timedelta(hours=2)
    good = _info(res, calibration_supported=True, beats_baseline_ci=True)
    # In the raw demo the latest (autumn) week lies outside the summer training range -> correctly flagged, not 'ok'.
    seasonal, _ = contract.build_contract(good, g, df, now)
    assert seasonal["reasons"].str.contains("outside the range").any()
    gi = g.copy()
    lastw = gi["week"] == gi["week"].max()
    for f in res["feats"]:
        gi.loc[lastw, f] = float(res["train_stats"].loc[f, "q50"])
    c, _ = contract.build_contract(good, gi, df, now)
    assert (c["status"] == "ok").any() and c.loc[c["status"] == "ok", "calibrated_probability"].between(0, 1).all()
    stale, _ = contract.build_contract(good, gi, df, now + pd.Timedelta(hours=200))
    assert stale["reasons"].str.contains("old").all() and not (stale["status"] == "ok").any()
    gm = gi.copy()
    last = gm["week"] == gm["week"].max()
    gm.loc[last, res["feats"][: len(res["feats"]) // 2 + 2]] = np.nan
    miss, _ = contract.build_contract(good, gm, df, now)
    assert (miss["status"] == "unavailable").all() and miss["calibrated_probability"].isna().all()
    go = gi.copy()
    go.loc[go["week"] == go["week"].max(), ["count", "lag1", "roll4"]] = 1e6
    ood, _ = contract.build_contract(good, go, df, now)
    assert ood["reasons"].str.contains("outside the range").all() and not (ood["status"] == "ok").any()


def test_prediction_alerts_come_only_from_ok_calibrated_rows():
    base = {"lat": 1.0, "lon": 2.0, "forecast_week_start": pd.Timestamp("2026-01-05"), "model_version": "v", "reliability_low": np.nan, "reliability_high": np.nan}
    c = pd.DataFrame([{**base, "status": "ok", "calibrated_probability": 0.9, "reasons": ""},
                      {**base, "status": "unreliable", "calibrated_probability": 0.95, "reasons": "x"},
                      {**base, "status": "unavailable", "calibrated_probability": np.nan, "reasons": "y"}])
    out = alerts.prediction_alerts(c, None, 0.6)
    assert len(out) == 1 and out[0]["type"] == "Elevated modelled risk"


# ------------------------------------------------------------------ scenario laboratory
def test_scenario_changes_only_requested_inputs_and_documents_derived_ones():
    df, rep, g, res, bbox = _demo()
    info = contract.info_from_res(res)
    rec_inputs = []
    real = info["model"]

    class Spy:
        def predict_proba(self, X):
            rec_inputs.append(X.copy())
            return real.predict_proba(X)

    info = {**info, "model": Spy()}
    c = g.sort_values("count").iloc[-1]
    row = scenario.baseline_row(g, int(c["cy"]), int(c["cx"]), info["features"])
    out = scenario.run_scenario(info, row, {"tmax_mean": float(row["tmax_mean"].iloc[0]) + 5})
    base, scen = rec_inputs[-2], rec_inputs[-1]
    changed = {k for k in base.columns if not np.isclose(base[k].iloc[0], scen[k].iloc[0], equal_nan=True)}
    assert changed <= {"tmax_mean", "fwi"} and "tmax_mean" in changed
    assert any("fwi" in n for n in out["notes"]) and "not a forecast" in out["disclaimer"]
    far = scenario.run_scenario(info, row, {"tmax_mean": 200.0})
    assert any("OUTSIDE the training range" in w for w in far["warnings"])
    try:
        scenario.run_scenario(info, row, {"count": 5})
        assert False
    except ValueError:
        pass
    sw = scenario.sweep(info, row, "tmax_mean", n=7)
    assert len(sw) == 7 and sw["tmax_mean"].is_monotonic_increasing and sw["within_training_range"].all()
    assert scenario.sweep2d(info, row, "tmax_mean", "wind_max", n=4).shape == (4, 4)


def test_spread_is_an_extension_point_only():
    from firesentinel import spread
    assert spread.available_models() == [] and "ignition" in spread.SpreadRequest().missing()


# ------------------------------------------------------------------ geodesy / monitoring
def test_geodesy_equal_area_bbox_projection_and_index():
    total = sum(float(geodesy.cell_area_km2(cy, 1.0)) * 360 for cy in range(-90, 90))
    assert abs(total / (4 * np.pi * geodesy.R_EARTH_KM ** 2) - 1) < 1e-6
    assert geodesy.cell_area_km2(120, 0.5) < 0.6 * geodesy.cell_area_km2(0, 0.5)  # cells shrink poleward
    for bad in [(10, 0, 5, 5), (0, 0, 10, 100), (0, 10, 10, 5), (-200, 0, 10, 5)]:
        try:
            geodesy.validate_bbox(bad)
            assert False, bad
        except ValueError:
            pass
    assert geodesy.check_geographic_degrees(pd.DataFrame({"latitude": [5e6], "longitude": [3e5]}))["looks_projected_metres"]
    assert geodesy.check_geographic_degrees(pd.DataFrame({"latitude": [95.0, 10.0], "longitude": [10.0, 10.0]}))["lat_out_of_range"] == 1
    kl, kn = geodesy.km_per_degree(0.0)
    assert abs(kl - 110.57) < 0.1 and abs(kn - 111.32) < 0.1
    x, y = geodesy.to_local_km(30.3, 31.4, 30.0, 31.0)
    assert abs(np.hypot(x, y) - float(geo_hav(30.0, 31.0, 30.3, 31.4))) / np.hypot(x, y) < 0.01
    idx = geodesy.SpatialIndex([30.0, 30.1, 40.0], [31.0, 31.0, 10.0])
    assert sorted(idx.within(30.0, 31.0, 20)) == [0, 1] and len(idx.within(0, 0, 5)) == 0


def geo_hav(*a):
    from firesentinel.geo import haversine_km
    return haversine_km(*a)


def test_psi_detects_shift_and_ignores_small_or_constant_inputs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    assert monitoring.psi(a, rng.normal(0, 1, 500)) < 0.1
    assert monitoring.psi(a, rng.normal(2, 1, 500)) > 0.25
    assert np.isnan(monitoring.psi(a[:10], a[:100])) and np.isnan(monitoring.psi(np.zeros(100), np.zeros(50)))


def test_hotspot_density_uses_true_area():
    d = pd.DataFrame({"latitude": [0.1, 60.1], "longitude": [0.1, 0.1], "frp": 1.0})
    h = geo_hotspots(d).set_index("cy")
    assert h.loc[120, "area_km2"] < h.loc[0, "area_km2"] and h.loc[120, "per_1000km2"] > h.loc[0, "per_1000km2"]


def geo_hotspots(d):
    from firesentinel import geo
    return geo.hotspot_grid(d)


# ------------------------------------------------------------------ providers / storage / logging
def test_provider_schema_validation_and_provenance_determinism():
    bad = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "ndvi": [5.0]})
    assert any("ndvi" in p for p in providers.validate_schema(bad, "ndvi"))
    assert any("missing column" in p for p in providers.validate_schema(pd.DataFrame({"x": [1]}), "weather"))
    p = providers.REGISTRY["synthetic_demo"]
    d1, v1 = p.fetch(region="Greece", days=42, seed=1, end="2026-10-18")
    d2, v2 = p.fetch(region="Greece", days=42, seed=1, end="2026-10-18")
    assert v1["sha256"] == v2["sha256"] and v1["schema_ok"] and "key" not in v1["request"]
    broken = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3), "temp_max": 20.0, "precip": -5.0, "wind_max": 3.0, "et0": 1.0, "elevation": 0.0})
    with mock.patch("firesentinel.weather.get_weather", return_value=broken):
        try:
            providers.REGISTRY["open_meteo"].fetch(lat=1, lon=1, start="2025-01-01", end="2025-01-03")
            assert False
        except ValueError as e:
            assert "schema validation failed" in str(e)
    md = providers.providers_markdown()
    assert "verified_live=False" in md and all(k in md for k in ("MOD13Q1", "license", "known limitations", "crs"))


def test_provenance_table_and_chunked_csv(tmp_path):
    db = tmp_path / "p.db"
    df, v = providers.REGISTRY["synthetic_demo"].fetch(region="Greece", days=28, seed=2, end="2026-10-18")
    store.log_provenance(v, db)
    assert store.recent_provenance(5, db)["rows"].iloc[0] == len(df)
    p = tmp_path / "a.csv"
    pd.DataFrame({"latitude": np.arange(10.0), "longitude": 0.0, "acq_date": "2025-01-01"}).to_csv(p, index=False)
    assert len(data.load_csv_chunked(p, bbox=(-1, 2, 1, 5))) == 4
    assert len(data.load_csv_chunked(p, max_rows=3, chunksize=2)) == 4  # stops after the chunk that crosses the limit


def test_logs_never_contain_secret_values():
    os.environ["FIRMS_MAP_KEY"] = "SUPERSECRET-KEY-12345"
    try:
        log = config.get_logger()
        log.warning("request to https://x/api/SUPERSECRET-KEY-12345/area failed")
        for h in log.handlers:
            h.flush()
        tail = (config.LOGS / "pipeline.log").read_text(encoding="utf-8").splitlines()[-1]
        assert "SUPERSECRET-KEY-12345" not in tail and "***" in tail
    finally:
        del os.environ["FIRMS_MAP_KEY"]


# ------------------------------------------------------------------ experiments
def test_experiment_is_reproducible_and_ids_are_safe(tmp_path):
    cfg = {"name": "repro", "seed": 5}
    e1, p1 = experiments.run_demo_experiment("Greece", 22, 3, "2026-10-18", cfg, tmp_path)
    e2, p2 = experiments.run_demo_experiment("Greece", 22, 3, "2026-10-18", cfg, tmp_path)
    assert e1["data"]["fingerprint_sha256"] == e2["data"]["fingerprint_sha256"] and pd.DataFrame(e1["metrics_test"]).equals(pd.DataFrame(e2["metrics_test"]))
    assert e1["scientific_evidence"] is False and "--demo" in e1["reproduce"]
    for k in ("label_definition", "split", "features", "seeds", "software", "bootstrap_ci_test", "walk_forward_hindcast", "limitations"):
        assert e1[k]
    assert (p1 / "report.md").read_text().startswith("# Experiment report")
    assert experiments.verify_reproduction(p1)["match"]
    assert len(experiments.list_experiments(tmp_path)) == 2 and experiments.get_experiment(e1["id"], tmp_path)["id"] == e1["id"]
    for bad in ("../etc", "exp-1", "exp-20250101T000000Z-zzzzzzzz/../.."):
        try:
            experiments.get_experiment(bad, tmp_path)
            assert False, bad
        except ValueError:
            pass


# ------------------------------------------------------------------ service layer (what the API serves)
def test_service_layer_end_to_end(tmp_path):
    df, rep, g, res, bbox = _demo()
    db, models = tmp_path / "s.db", tmp_path / "m"
    models.mkdir()
    assert service.health()["status"] == "ok"
    assert service.readiness(db, models)["ready"] is True
    assert service.model_metadata(models)["model"] is None
    for bad_call, status in ((lambda: service.risk("Greece", 26, "synthetic", db, models), 404), (lambda: service.validate_region("Atlantis", 7, db), 404),
                             (lambda: service.risk("Greece", 26, "bogus", db, models), 422), (lambda: service.get_experiment("../x", tmp_path), 422)):
        try:
            bad_call()
            assert False
        except service.ServiceError as e:
            assert e.status == status
    store.upsert(df, pd.Timestamp.now(), db)
    ver, meta = model.save_model(res, models, {"origin": "demo"})
    assert service.model_metadata(models)["version"] == ver
    v = service.validate_region("Greece", 120, db)
    assert v["coordinates"]["crs"] == "EPSG:4326" and v["coverage"]["cells_with_detections"] > 0
    assert service.geojson("Greece", 120, "detections", db_path=db)["features"] and service.geojson("Greece", 120, "clusters", db_path=db)["type"] == "FeatureCollection"
    r = service.risk("Greece", 26, "synthetic", db, models, max_cells=40)
    assert r["cells"] and r["summary"]["model_version"] == ver and r["summary"]["synthetic_data"] is True
    assert {"model_score", "calibrated_probability", "status", "data_completeness", "data_freshness_hours", "model_version"} <= set(r["cells"][0])
    s = service.sources_status(db)
    assert s["credentials_configured"] and any(p["id"] == "nasa_firms" and p["verified_live"] is False for p in s["providers"])
    (models / f"{ver}.joblib").write_bytes(b"tampered")
    rd = service.readiness(db, models)
    assert rd["ready"] is False and "integrity" in str(rd["checks"]["model"])


def test_api_routes_exist_when_fastapi_is_installed():
    try:
        import fastapi  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("fastapi not installed in this environment")
    from firesentinel import api
    paths = {r.path for r in api.app.routes}
    assert {"/health", "/ready", "/sources", "/model", "/validate", "/risk", "/export/geojson", "/experiments"} <= paths


def test_api_source_has_safe_defaults():
    src = (ROOT / "firesentinel" / "api.py").read_text()
    ast.parse(src)
    assert "compare_digest" in src and "FIRESENTINEL_API_KEY" in src and "LOOPBACK" in src and "allow_origins" not in src
    assert ".post(" not in src and ".put(" not in src and ".delete(" not in src  # read-only


# ------------------------------------------------------------------ deployment hardening
def test_docker_ci_and_dependency_constraints_are_hardened():
    docker = (ROOT / "Dockerfile").read_text()
    assert "USER " in docker and "HEALTHCHECK" in docker and "root" not in docker.split("USER ")[1].split("\n")[0]
    comp = (ROOT / "docker-compose.yml").read_text()
    assert comp.count('"127.0.0.1:') >= 2 and "0.0.0.0:" not in comp.replace("--host 0.0.0.0", "")
    assert ".env" in (ROOT / ".dockerignore").read_text()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert all(v in ci for v in ("3.10", "3.11", "3.12")) and "security_check" in ci
    cons = (ROOT / "constraints.txt").read_text()
    for pkg in ("streamlit", "pandas", "numpy", "scikit-learn", "scipy", "plotly", "requests", "joblib", "reportlab", "matplotlib", "markdown"):
        assert any(line.lower().startswith(pkg) and "<" in line for line in cons.splitlines()), pkg
