import ast
import functools
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import requests

from firesentinel import (alerts, config, data, features, geo, i18n, impact, model, notify, pipeline,
                          report, scheduler, security, store, synthetic, vegetation)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


@functools.lru_cache(maxsize=1)
def _demo():
    raw, bbox = pipeline.make_demo("Greece", 168, 11)
    df, rep = data.validate(raw, "SYNTHETIC")
    g, res, msg = pipeline.prepare_and_train(df, "synthetic", max_cells=40, tune=True)
    assert res is not None, msg
    return df, rep, g, res, bbox


def _det(rows):
    d = pd.DataFrame(rows, columns=["latitude", "longitude", "acq_datetime", "source", "conf_level", "frp"])
    d["acq_datetime"] = pd.to_datetime(d["acq_datetime"])
    return d


# ---------------------------------------------------------------- data
def test_dedupe_across_sensors_only_merges_different_products():
    d = _det([[30.0, 31.0, "2025-01-01 10:00", "VIIRS_A", "nominal", 3.0],
              [30.001, 31.001, "2025-01-01 10:30", "MODIS", "high", 2.0],      # same spot, other sensor -> merged
              [40.0, 10.0, "2025-01-01 10:00", "VIIRS_A", "nominal", 1.0],
              [40.001, 10.001, "2025-01-01 10:10", "VIIRS_A", "nominal", 1.0]])  # same sensor -> kept
    out, removed = data.dedupe_across_sensors(d)
    assert removed == 1 and len(out) == 3
    assert (out[(out.latitude < 31)]["source"] == "MODIS").all()  # higher confidence kept


def test_validate_keeps_time_of_day_for_database_rows():
    d = pd.DataFrame({"latitude": [1.0], "longitude": [2.0], "acq_date": [pd.Timestamp("2025-03-01")],
                      "acq_datetime": [pd.Timestamp("2025-03-01 13:45")], "source": ["X"]})
    out, _ = data.validate(d)
    assert out["acq_datetime"].iloc[0] == pd.Timestamp("2025-03-01 13:45")


def test_fetch_chunk_never_leaks_key_and_fetch_range_degrades():
    key = "SECRETKEY123456"
    err = requests.ConnectionError(f"HTTPSConnectionPool url=/api/area/csv/{key}/VIIRS failed")
    with mock.patch("firesentinel.data.requests.get", side_effect=err), mock.patch("firesentinel.data.time.sleep"):
        try:
            data.fetch_chunk(key, "VIIRS_SNPP_NRT", (0, 0, 1, 1), 1, "2025-01-01", retries=2)
            assert False
        except RuntimeError as e:
            assert key not in str(e) and "***" in str(e)
        df, errs = data.fetch_range(key, "VIIRS_SNPP_NRT", (0, 0, 1, 1), "2025-01-01", "2025-01-07")
    assert df.empty and len(errs) == 2 and all(key not in x for x in errs)


# ---------------------------------------------------------------- features / model
def test_vegetation_synthetic_weekly_values():
    nd = vegetation.get_ndvi(30.0, 31.0, "2025-01-01", "2025-03-31", "synthetic")
    w = vegetation.weekly_ndvi(nd, "2025-01-01", "2025-03-31")
    assert {"week", "ndvi"} <= set(w.columns) and w["ndvi"].between(0, 1).all()
    assert vegetation.get_landcover(30.0, 31.0, 2025, "synthetic") in {"forest", "shrub", "grass", "crop", "other"}


def test_next_week_weather_is_shifted_not_current():
    weeks = pd.date_range("2025-01-06", periods=6, freq="W-MON").astype("datetime64[ns]")
    ts = ["2025-01-06 12:00", "2025-01-13 12:00", "2025-02-16 12:00"]
    df = pd.DataFrame({"latitude": 10.1, "longitude": 10.1, "frp": 1.0, "acq_datetime": pd.to_datetime(ts)})
    cells = pd.DataFrame({"cy": [20], "cx": [20]})
    wx = pd.DataFrame({"cy": 20, "cx": 20, "week": weeks.append(pd.DatetimeIndex([weeks[-1] + pd.Timedelta(days=7)])),
                       "tmax_mean": np.arange(7.0) * 10, "wind_max": 1.0, "precip_sum": 0.0, "fwi": 1.0})
    g = features.build_dataset(df, cells, wx, nextwx=True)
    for r in g.itertuples():
        i = list(wx["week"]).index(r.week)
        assert r.nx_tmax == wx["tmax_mean"].iloc[i + 1]


def test_neighbour_features_see_adjacent_cells():
    ts = pd.date_range("2025-01-06 12:00", periods=5, freq="7D")
    rows = [{"latitude": 10.1, "longitude": 10.1, "frp": 1.0, "acq_datetime": t} for t in ts]  # cell (20,20)
    rows += [{"latitude": 10.6, "longitude": 10.1, "frp": 1.0, "acq_datetime": t} for t in ts[:3]]  # neighbour (21,20)
    rows.append({"latitude": 10.1, "longitude": 10.1, "frp": 1.0, "acq_datetime": pd.Timestamp("2025-02-09 12:00")})
    g = features.build_dataset(pd.DataFrame(rows), pd.DataFrame({"cy": [20], "cx": [20]}))
    assert g["nb_count"].max() == 1 and (g["nb_lag1"] <= 1).all()


def test_tuning_intensity_explanation_and_baselines():
    df, rep, g, res, bbox = _demo()
    assert res["tuning"] is not None and {"Model", "params", "CV PR-AUC"} <= set(res["tuning"].columns)
    inten = model.evaluate_intensity(g, res["feats"])
    assert inten is not None and "Baseline: persistence" in set(inten["Model"])
    top = g.sort_values("count").iloc[-1]
    ex = model.explain_cell(res, g, int(top["cy"]), int(top["cx"]))
    assert {"feature", "effect_on_risk"} <= set(ex.columns) and 0 <= ex.attrs["risk"] <= 1
    assert len(model.importance(res)) == len(res["feats"])


def test_time_split_has_no_overlap():
    df, rep, g, res, bbox = _demo()
    i = res["info"]
    assert i["train_weeks"][1] == i["val_weeks"][0] and i["val_weeks"][1] == i["test_weeks"][0]
    assert res["test"]["week"].min() >= pd.Timestamp(i["test_weeks"][0])


def test_model_integrity_check(tmp_path):
    df, rep, g, res, bbox = _demo()
    assert model.load_latest(tmp_path) == (None, None)
    ver, meta = model.save_model(res, tmp_path)
    bundle, meta2 = model.load_latest(tmp_path)
    assert bundle["version"] == ver and meta2["sha256"] == meta["sha256"]
    out = model.score_with_bundle(bundle, g)
    assert out["risk_prob"].between(0, 1).all()
    (tmp_path / f"{ver}.joblib").write_bytes(b"tampered")
    try:
        model.load_latest(tmp_path)
        assert False
    except ValueError as e:
        assert "integrity" in str(e)


def test_score_with_bundle_rejects_missing_features():
    df, rep, g, res, bbox = _demo()
    try:
        model.score_with_bundle({"model": None, "features": ["does_not_exist"]}, g)
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------- notify / impact / scheduler
def _alert_df():
    return pd.DataFrame([{"severity": "warning", "category": "observation", "type": "Detections in watch zone",
                          "time": pd.Timestamp("2025-01-01 10:00"), "lat": 30.0, "lon": 31.0, "source": "T",
                          "explanation": "3 detections"}])


def test_notify_dry_run_then_dedupe(tmp_path):
    db, sent = tmp_path / "n.db", []
    senders = {"telegram": lambda s, b: sent.append(b)}
    out = notify.dispatch(_alert_df(), ["telegram"], "warning", True, db, senders)
    assert out["new"] == 1 and not sent and "not an official emergency" in out["preview"].lower().replace("official emergency warning", "not an official emergency")
    out = notify.dispatch(_alert_df(), ["telegram"], "warning", False, db, senders)
    assert out["channels"] == ["telegram"] and len(sent) == 1
    assert notify.dispatch(_alert_df(), ["telegram"], "warning", False, db, senders)["new"] == 0


def test_notify_failure_is_not_remembered(tmp_path):
    db = tmp_path / "n.db"

    def boom(s, b):
        raise RuntimeError("smtp down")
    out = notify.dispatch(_alert_df(), ["email"], "warning", False, db, {"email": boom})
    assert out["errors"] and notify.dispatch(_alert_df(), ["email"], "warning", True, db, {"email": boom})["new"] == 1


def test_impact_exposure_and_asset_validation():
    summ = pd.DataFrame([{"cluster": 0, "detections": 9, "lat": 30.0, "lon": 31.0, "radius_km": 1.0}])
    assets = pd.DataFrame({"name": ["near", "far"], "type": ["village", "town"], "lat": [30.05, 35.0], "lon": [31.0, 40.0],
                           "population": [1000.0, 5000.0]})
    detail, per = impact.exposure(summ, assets, 10)
    assert list(detail["name"]) == ["near"] and per["people_in_range"].iloc[0] == 1000.0
    try:
        impact.validate_assets(pd.DataFrame({"a": [1]}))
        assert False
    except ValueError:
        pass


def test_scheduler_run_once_offline(tmp_path):
    today = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    fake = pd.DataFrame({"latitude": 30 + rng.normal(0, 0.01, 40), "longitude": 31 + rng.normal(0, 0.01, 40),
                         "acq_date": today, "acq_time": 300, "confidence": "h", "frp": 5.0, "daynight": "N"})
    fetcher = lambda key, src, bbox, a, b, progress=None: (fake, [])
    out = scheduler.run_once("k", ["VIIRS_SNPP_NRT"], "Egypt", 1, [], dry_run=True, fetcher=fetcher, db_path=tmp_path / "s.db")
    assert out["fetched"] == 40 and out["inserted"] == 40 and out["alerts"] >= 1


# ---------------------------------------------------------------- report / i18n / security / config
def test_pdf_html_markdown_reports():
    df, rep, g, res, bbox = _demo()
    from firesentinel import contract
    c, cs = contract.build_contract(contract.info_from_res(res), g, df, pd.Timestamp.now(), synthetic=True)
    al = alerts.collect(df, geo.cluster(df, 3, 3), None, pd.Timestamp.now(), c, cs)
    pdf = report.build_pdf("demo", rep, df, res, None, model.importance(res), "synthetic", al, c, None, None)
    assert pdf[:4] == b"%PDF" and len(pdf) > 10_000
    md = report.build_report("demo", rep, df, res, None, None, "synthetic", al, None, res["tuning"], None)
    html = report.to_html(md)
    assert "SYNTHETIC DEMO DATA" in md and "<table" in html and "<!doctype html>" in html


def test_i18n_arabic_and_fallback():
    try:
        i18n.set_lang("ar")
        assert i18n.t("Region") != "Region" and i18n.t("not in dictionary") == "not in dictionary"
        assert i18n.t("{n} rows downloaded.").format(n=3).count("3") == 1
    finally:
        i18n.set_lang("en")
    assert i18n.t("Region") == "Region"


def test_security_helpers(tmp_path):
    assert security.safe_join(tmp_path, "a/b.csv").name == "b.csv"
    for bad in ("../x", "/etc/passwd"):
        try:
            security.safe_join(tmp_path, bad)
            assert False
        except ValueError:
            pass
    try:
        security.check_upload_size(60 * 1024 * 1024)
        assert False
    except ValueError:
        pass
    os.environ["FIRMS_MAP_KEY"] = "abcd1234efgh"
    try:
        assert "abcd1234efgh" not in security.redact("url/abcd1234efgh/x")
    finally:
        del os.environ["FIRMS_MAP_KEY"]


def test_env_loader_does_not_override(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FS_TEST_A=from_file\nFS_TEST_B='quoted'\n# comment\nbad line\n")
    os.environ["FS_TEST_A"] = "already"
    try:
        config.load_env(p)
        assert os.environ["FS_TEST_A"] == "already" and os.environ["FS_TEST_B"] == "quoted"
    finally:
        os.environ.pop("FS_TEST_A", None)
        os.environ.pop("FS_TEST_B", None)


def test_project_has_no_secrets_or_risky_calls():
    import security_check
    assert security_check.scan(ROOT) == []


def test_api_module_is_valid_python():
    ast.parse((ROOT / "firesentinel" / "api.py").read_text())
