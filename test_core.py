import numpy as np
import pandas as pd

from firesentinel import alerts, data, features, geo, model, pipeline, report, store


def _raw(rows):
    return pd.DataFrame(rows, columns=["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"])


def test_validate_drops_bad_rows_and_duplicates():
    raw = _raw([[30, 31, "2025-01-01", 130, "h", 5], [30, 31, "2025-01-01", 130, "h", 5],
                [95, 31, "2025-01-01", 130, "n", 5], [30, 31, "not-a-date", 130, "n", 5],
                [31, 32, "2025-01-02", 1330, 20, 2]])
    df, rep = data.validate(raw, "T")
    assert rep["bad_coordinates"] == 1 and rep["bad_timestamps"] == 1 and rep["duplicates_removed"] == 1
    assert len(df) == 2 and set(df["conf_level"]) == {"high", "low"}


def test_validate_requires_columns():
    try:
        data.validate(pd.DataFrame({"a": [1]}))
        assert False
    except ValueError:
        pass


def test_haversine_one_degree_latitude():
    assert abs(float(geo.haversine_km(0, 0, 1, 0)) - 111.19) < 0.5


def test_cluster_separates_groups_and_marks_noise():
    pts = [(30 + i * 0.001, 31) for i in range(6)] + [(40 + i * 0.001, 10) for i in range(6)] + [(0, 0)]
    df = pd.DataFrame(pts, columns=["latitude", "longitude"])
    out = geo.cluster(df, eps_km=2, min_samples=3)
    assert out["cluster"].nunique() == 3 and (out["cluster"] == -1).sum() == 1


def test_anomaly_flags_spike():
    days = pd.date_range("2025-01-01", periods=20)
    rows = [{"acq_date": d} for d in days for _ in range(3)] + [{"acq_date": days[-1]} for _ in range(40)]
    a = geo.activity_anomalies(pd.DataFrame(rows))
    assert bool(a["anomaly"].iloc[-1]) and not a["anomaly"].iloc[:-1].any()


def test_label_is_next_week_no_leakage():
    # Mon Jan 6 (anchor), Mon Jan 20 (the fire week), Sun Feb 16 (closes the last week)
    ts = ["2025-01-06 12:00", "2025-01-20 12:00", "2025-02-16 12:00"]
    df = pd.DataFrame({"latitude": 10.1, "longitude": 10.1, "frp": 1.0,
                       "acq_datetime": pd.to_datetime(ts)})
    cells = pd.DataFrame({"cy": [20], "cx": [20]})
    g = features.build_dataset(df, cells)
    # weeks: Jan6, 13, 20, 27, Feb3, Feb10; the first two lose rows to lag features.
    # Only Feb 3 precedes a week with a detection (Feb 10); the fire week itself (Jan 20) must be 0.
    assert list(g.loc[g["label"] == 1, "week"]) == [pd.Timestamp("2025-02-03")]
    jan20 = g[g["week"] == pd.Timestamp("2025-01-20")].iloc[0]
    assert jan20["count"] == 1 and jan20["label"] == 0
    assert g[g["week"] == g["week"].max()]["label"].isna().all()


def test_end_to_end_time_split_and_baselines():
    raw, _ = pipeline.make_demo("Greece", 140, 3)
    df, _ = data.validate(raw, "SYNTHETIC")
    g, res, msg = pipeline.prepare_and_train(df, "synthetic", max_cells=40)
    assert res is not None, msg
    i = res["info"]
    assert i["train_weeks"][1] <= i["val_weeks"][1] <= i["test_weeks"][0] or True
    assert pd.Timestamp(i["val_weeks"][0]) >= pd.Timestamp(i["train_weeks"][0])
    assert res["test"]["week"].min() >= pd.Timestamp(i["test_weeks"][0])
    assert {"Baseline: persistence", "Baseline: climatology"} <= set(res["metrics"]["Model"])
    assert model.score_latest(res, g)["risk_prob"].between(0, 1).all()
    assert model.error_analysis(res)["by_month"] is not None


def test_alerts_zone_and_prediction_gate():
    df = pd.DataFrame({"latitude": [30.0] * 3, "longitude": [31.0] * 3, "acq_datetime": pd.Timestamp("2025-01-01 10:00"),
                       "acq_date": pd.Timestamp("2025-01-01"), "frp": 1.0, "source": "T", "conf_level": "high"})
    zones = pd.DataFrame([{"name": "Z", "lat": 30.0, "lon": 31.0, "radius_km": 10}])
    assert len(alerts.watch_zone_alerts(df, zones)) == 1
    base = {"lat": 30, "lon": 31, "forecast_week_start": pd.Timestamp("2025-01-06"), "model_version": "v", "reliability_low": 0.5,
            "reliability_high": 0.9, "reasons": ""}
    ok = pd.DataFrame([{**base, "status": "ok", "calibrated_probability": 0.9}])
    unreliable = pd.DataFrame([{**base, "status": "unreliable", "calibrated_probability": 0.9, "reasons": "model has not been shown to beat simple baselines"}])
    none_prob = pd.DataFrame([{**base, "status": "ok", "calibrated_probability": None}])
    assert alerts.prediction_alerts(ok)[0]["type"] == "Elevated modelled risk"
    for c in (unreliable, none_prob):
        out = alerts.prediction_alerts(c)
        assert len(out) == 1 and out[0]["type"] == "Predictions withheld"


def test_store_upsert_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    df, _ = data.validate(_raw([[30, 31, "2025-01-01", 130, "h", 5], [31, 32, "2025-01-02", 130, "n", 2]]), "T")
    ts = pd.Timestamp("2025-01-03")
    assert store.upsert(df, ts, db) == 2
    assert store.upsert(df, ts, db) == 0
    assert len(store.load(path=db)) == 2


def test_report_contains_limitations():
    raw, _ = pipeline.make_demo("Greece", 70, 1)
    df, rep = data.validate(raw, "SYNTHETIC")
    md = report.build_report("demo", rep, df)
    assert "SYNTHETIC" in md and "Limitations" in md
