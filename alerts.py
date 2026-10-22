"""Alert engine. Observation alerts, data alerts and prediction alerts are kept separate.
Nothing here claims a detection is a confirmed or newly started fire."""
import pandas as pd

from .geo import activity_anomalies, cluster_summary, haversine_km

COLS = ["severity", "category", "type", "time", "lat", "lon", "source", "explanation"]
ORDER = {"critical": 0, "warning": 1, "info": 2}


def _a(sev, cat, typ, time, lat, lon, source, expl):
    return dict(severity=sev, category=cat, type=typ, time=time, lat=lat, lon=lon, source=source, explanation=expl)


def _src(df):
    return ", ".join(sorted(map(str, df["source"].unique()))) if "source" in df and len(df) else "n/a"


def watch_zone_alerts(df, zones, hours=24):
    out = []
    if df.empty or zones is None or len(zones) == 0:
        return out
    zones = zones.dropna(subset=["lat", "lon", "radius_km"])
    ref = df["acq_datetime"].max()
    rec = df[df["acq_datetime"] >= ref - pd.Timedelta(hours=hours)]
    for z in zones.itertuples(index=False):
        d = haversine_km(z.lat, z.lon, rec["latitude"].to_numpy(), rec["longitude"].to_numpy())
        hit = rec[d <= z.radius_km]
        if len(hit):
            out.append(_a("critical" if len(hit) >= 20 else "warning", "observation", "Detections in watch zone",
                          hit["acq_datetime"].max(), z.lat, z.lon, _src(hit),
                          f"{len(hit)} satellite detection(s) within {z.radius_km:.0f} km of '{z.name}' in the "
                          f"{hours} h up to the latest observation ({ref}). Detections are unconfirmed thermal anomalies."))
    return out


def surge_alerts(df, z=2.5, window=14, last_days=3):
    out = []
    if df.empty:
        return out
    a = activity_anomalies(df, window, z)
    recent = a[a["anomaly"] & (a["date"] >= a["date"].max() - pd.Timedelta(days=last_days - 1))]
    for r in recent.itertuples():
        out.append(_a("warning", "observation", "Activity surge", r.date, df["latitude"].mean(), df["longitude"].mean(),
                      _src(df), f"{int(r.count)} detections on {r.date.date()} vs trailing {window}-day mean "
                                f"{r.baseline:.1f} (z={r.zscore:.1f}). May reflect real activity or more satellite passes."))
    return out


def cluster_alerts(dfc, min_det=10, min_frp=0.0, within_days=2):
    out = []
    s = cluster_summary(dfc)
    if s.empty:
        return out
    ref = dfc["acq_datetime"].max()
    s = s[(s["last_seen"] >= ref - pd.Timedelta(days=within_days)) &
          ((s["detections"] >= min_det) | ((min_frp > 0) & (s["frp_sum"] >= min_frp)))]
    for r in s.itertuples():
        out.append(_a("warning", "observation", "Detection cluster", r.last_seen, r.lat, r.lon, _src(dfc),
                      f"Cluster {r.cluster}: {r.detections} detections, radius {r.radius_km} km, total FRP "
                      f"{r.frp_sum:.0f} MW, active {r.active_days} day(s). A cluster is not proof of one single fire."))
    return out


def data_alerts(df, retrieved_at, max_age_hours=12):
    out = []
    if df.empty:
        return [_a("critical", "data", "No data", retrieved_at, 0, 0, "n/a", "No detections loaded.")]
    age = (pd.Timestamp(retrieved_at).tz_localize(None) - df["acq_datetime"].max()) / pd.Timedelta(hours=1)
    if age > max_age_hours:
        out.append(_a("warning", "data", "Stale data", retrieved_at, df["latitude"].mean(), df["longitude"].mean(),
                      _src(df), f"Newest observation is {age:.0f} h older than the retrieval time (limit {max_age_hours} h). "
                                "Pipeline may be delayed, or this is historical data."))
    s = df.groupby("acq_date").size()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    gaps = [d for d in full if d not in s.index]
    if len(gaps) >= 2:
        out.append(_a("info", "data", "Days without detections", retrieved_at, df["latitude"].mean(), df["longitude"].mean(),
                      _src(df), f"{len(gaps)} day(s) have zero detections (first: {gaps[0].date()}). "
                                "Could be a data gap or genuinely no activity."))
    low = float((df["conf_level"] == "low").mean()) if "conf_level" in df else 0.0
    if low > 0.5:
        out.append(_a("info", "data", "Many low-confidence detections", retrieved_at, df["latitude"].mean(),
                      df["longitude"].mean(), _src(df), f"{low:.0%} of detections are low confidence."))
    return out


def prediction_alerts(contract, summary=None, min_prob=0.6):
    """Predictive alerts come ONLY from contract rows whose status is 'ok' and that carry a CALIBRATED probability.
    Otherwise a single informational alert explains why predictions are withheld."""
    if contract is None or len(contract) == 0:
        return []
    ok = contract[(contract["status"] == "ok") & contract["calibrated_probability"].notna()]
    if ok.empty:
        why = contract["reasons"].replace("", pd.NA).dropna().str.split("; ").explode().value_counts()
        reason = why.index[0] if len(why) else "no estimate met the reliability requirements"
        return [_a("info", "prediction", "Predictions withheld", pd.Timestamp.now(), 0, 0, "model",
                   f"No estimate qualifies for a predictive alert. Most common reason: {reason}.")]
    hi = ok[ok["calibrated_probability"] >= min_prob]
    return [_a("warning", "prediction", "Elevated modelled risk", r.forecast_week_start, r.lat, r.lon, f"model {r.model_version}",
               f"Calibrated probability {r.calibrated_probability:.0%} (empirical interval "
               f"{'n/a' if pd.isna(r.reliability_low) else f'{r.reliability_low:.0%}-{r.reliability_high:.0%}'}) of >=1 detection in this cell "
               f"during the week starting {str(r.forecast_week_start)[:10]}. Statistical estimate, not an observation or an emergency warning.")
            for r in hi.itertuples()]


def collect(df, dfc, zones, retrieved_at, contract=None, summary=None, hours=24, z=2.5,
            min_det=10, min_prob=0.6, max_age_hours=12):
    items = (watch_zone_alerts(df, zones, hours) + surge_alerts(df, z) + cluster_alerts(dfc, min_det)
             + data_alerts(df, retrieved_at, max_age_hours) + prediction_alerts(contract, summary, min_prob))
    out = pd.DataFrame(items, columns=COLS)
    if out.empty:
        return out
    return out.sort_values("severity", key=lambda s: s.map(ORDER), kind="stable").reset_index(drop=True)
