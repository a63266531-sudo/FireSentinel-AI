"""FireSentinel AI - scientific workspace (Streamlit).   Run: streamlit run app.py"""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from firesentinel import (alerts, contract, data, experiments, features, geo, geodesy, i18n, impact, model, monitoring, notify,
                          pipeline, providers, report, scenario, security, spread, store, synthetic)
from firesentinel import __version__
from firesentinel.config import LIMITATIONS, LOGS, MODELS, REGIONS, SOURCES
from firesentinel.i18n import t

st.set_page_config(page_title="FireSentinel AI", page_icon="🔥", layout="wide")
S = st.session_state
# Colour-blind-safe (Okabe-Ito) palette; every colour is also given a text label / numeric legend.
LEVEL_COLORS = {"Low": "#56B4E9", "Moderate": "#F0E442", "High": "#E69F00", "Very high": "#D55E00",
                "relatively lower (rank)": "#56B4E9", "relatively elevated (rank)": "#E69F00", "relatively high (rank)": "#D55E00",
                "unavailable": "#999999"}
STATUS_ICON = {"ok": "OK", "unreliable": "UNRELIABLE", "unavailable": "UNAVAILABLE"}
MODEL_KEYS = ("g", "res", "info", "contract", "csummary", "imp", "geocv", "err", "cal", "meta", "wx_used", "inten", "cell", "exposure",
              "pdf", "stt", "wf", "drift", "conf", "impact", "exp_view", "exp_pdf")
DEFAULT_OPTS = dict(cell=0.5, max_cells=60, neg=0.3, use_wx=True, use_veg=True, use_nx=False, use_sp=True, tune=False)


def now_utc():
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


def zoom_for(df):
    span = max(df["latitude"].max() - df["latitude"].min(), df["longitude"].max() - df["longitude"].min(), 0.5)
    return float(np.clip(np.log2(360 / span), 1, 9))


def set_data(raw, source, retrieved, origin, dedupe=False, demo_params=None):
    df, rep = data.validate(raw, source)
    if dedupe:
        df, n = data.dedupe_across_sensors(df)
        rep["cross_sensor_duplicates_removed"] = n
    for k in MODEL_KEYS:
        S.pop(k, None)
    S.update(df=df, rep=rep, retrieved=retrieved, origin=origin, demo_params=demo_params)
    if len(df):
        S.zones = pd.DataFrame([{"name": "Zone 1", "lat": round(float(df["latitude"].mean()), 2),
                                 "lon": round(float(df["longitude"].mean()), 2), "radius_km": 50.0}])


def train_into_state(df, wx_mode, o, bar=None):
    g, res, msg = pipeline.prepare_and_train(
        df, wx_mode, cell=o["cell"], max_cells=o["max_cells"], neg_ratio=o["neg"], use_weather=o["use_wx"],
        progress=bar.progress if bar else None, use_veg=o["use_veg"], use_nextwx=o["use_nx"], use_spatial=o["use_sp"], tune=o["tune"])
    for k in MODEL_KEYS:
        S.pop(k, None)
    S.g, S.cell = g, o["cell"]
    S.wx_used = wx_mode if (o["use_wx"] or o["use_nx"]) else "none"
    if res is None:
        return msg
    S.res = res
    S.info = contract.info_from_res(res)
    S.contract, S.csummary = contract.build_contract(S.info, g, df, now_utc(), o["cell"], synthetic=(S.get("origin") == "demo"))
    S.imp = model.importance(res)
    S.geocv = model.geo_cv(g, res["feats"], res["best"])
    S.err = model.error_analysis(res)
    S.cal = model.calibration_df(res)
    S.inten = model.evaluate_intensity(g, res["feats"])
    S.stt = model.evaluate_spatiotemporal(g, res["feats"], res["best"])
    S.wf = model.walk_forward(g, res["feats"], res["best"])
    S.conf = model.confusion_and_pr(res)
    S.drift = monitoring.drift_report(res["train_ref"], g[g["week"] >= g["week"].max() - pd.Timedelta(days=21)], res["feats"])
    return ""


@st.cache_data(show_spinner=False)
def cached_summary(_df, key, eps, mins):
    return geo.cluster_summary(geo.cluster(_df, eps, mins))


def map_fig(d, kind, color=None, hover=None, zoom=3):
    kw = dict(lat="latitude", lon="longitude", zoom=zoom, height=520)
    fig = px.density_mapbox(d, z="frp", radius=12, **kw) if kind == "Heatmap" else px.scatter_mapbox(d, color=color, hover_data=hover, **kw)
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    return fig


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    lang = st.radio("Language / اللغة", ["English", "العربية"], horizontal=True)
    i18n.set_lang("ar" if lang == "العربية" else "en")
    if lang == "العربية":
        st.markdown("<style>.stApp{direction:rtl}.stMarkdown,.stMetric,.stAlert{text-align:right}</style>", unsafe_allow_html=True)
    st.title("🔥 FireSentinel AI")
    st.caption(f"v{__version__}")
    modes = ["Demo (synthetic)", "NASA FIRMS API", "Upload CSV", "Local database"]
    mode = st.radio(t("Data source"), modes, format_func=t)
    region = st.selectbox(t("Region"), list(REGIONS))
    bbox = REGIONS[region]
    if mode == "Demo (synthetic)":
        weeks = st.slider(t("Weeks of history"), 14, 52, 26)
        seed = st.number_input(t("Random seed"), value=7, step=1)
        if st.button(t("Generate demo data"), type="primary"):
            end = pipeline.demo_end()
            raw, _ = pipeline.make_demo(region, weeks * 7, int(seed), end)
            set_data(raw, "SYNTHETIC", now_utc(), "demo", demo_params={"region": region, "weeks": int(weeks), "demo_seed": int(seed), "end": str(end.date())})
    elif mode == "NASA FIRMS API":
        key = st.text_input(t("FIRMS MAP_KEY"), value=os.environ.get("FIRMS_MAP_KEY", ""), type="password", help="https://firms.modaps.eosdis.nasa.gov/api/map_key/")
        sources = st.multiselect(t("Products"), list(SOURCES), default=["VIIRS_SNPP_NRT"], format_func=lambda k: f"{k} - {SOURCES[k]}")
        d1, d2 = st.columns(2)
        start = d1.date_input(t("From"), datetime.now(timezone.utc).date() - timedelta(days=4))
        end = d2.date_input(t("To"), datetime.now(timezone.utc).date())
        merge = st.checkbox(t("Merge duplicates across sensors"), value=True)
        st.caption(t("Each fetch is stored in the local database, so history accumulates over time."))
        if st.button(t("Fetch & store"), type="primary"):
            if not key or not sources:
                st.warning(t("Enter a MAP_KEY and choose at least one product."))
            else:
                ts, errs_all, got = now_utc(), [], 0
                bar = st.progress(0.0)
                for si, src in enumerate(sources):
                    raw, errs = data.fetch_range(key, src, bbox, start, end, lambda p, si=si: bar.progress((si + p) / len(sources)))
                    errs_all += errs
                    if len(raw):
                        data.save_raw(raw, src, bbox, ts)
                        clean, _ = data.validate(raw, src)
                        ins = store.upsert(clean, ts)
                        store.log_run(ts, src, bbox, len(raw), ins, "partial" if errs else "ok", "; ".join(errs))
                        prob = providers.validate_schema(clean, "fire")
                        store.log_provenance(providers.make_provenance(providers.REGISTRY["nasa_firms"], {"source": src, "bbox": bbox, "start": start, "end": end}, clean, prob))
                        got += len(raw)
                    else:
                        store.log_run(ts, src, bbox, 0, 0, "failed", "; ".join(errs))
                db = store.load(bbox, start, end)
                if len(db):
                    set_data(db, None, ts, "FIRMS", dedupe=merge and len(sources) > 1)
                    st.success(t("{n} rows downloaded.").format(n=got))
                else:
                    st.error(t("No data returned.") + " " + security.redact("; ".join(errs_all)))
                for e in errs_all:
                    st.warning(security.redact(e))
    elif mode == "Upload CSV":
        up = st.file_uploader(t("FIRMS CSV (archive download gives a long history)"), type="csv")
        if up is not None and S.get("upload_name") != up.name:
            S.upload_name = up.name
            try:
                security.check_upload_size(up.size)
                set_data(data.load_csv_chunked(up, bbox), "upload", now_utc(), "upload")
            except Exception as ex:
                st.error(f"{ex}")
    else:
        if st.button(t("Load from local database"), type="primary"):
            db = store.load(bbox)
            if db.empty:
                st.warning(t("Database is empty for this region."))
            else:
                set_data(db, None, now_utc(), "database")
    st.divider()
    wx_label = st.radio(t("Context data source"), ["Synthetic (demo)", "Real (Open-Meteo + NASA MODIS)"],
                        index=0 if mode == "Demo (synthetic)" else 1, format_func=t)
    wx_mode = "synthetic" if wx_label.startswith("Synthetic") else "open-meteo"
    st.caption(t("Research and decision-support tool. Not an official emergency warning service."))

# ------------------------------------------------------------------ landing
if "df" not in S:
    st.title("🔥 FireSentinel AI")
    st.write(t("Observe the Earth. Understand the Evidence. Anticipate the Risk. Protect the Future."))
    st.info(t("Pick a data source in the sidebar, or run the offline demo with one click."))
    if st.button(t("▶ One-click demo (offline)"), type="primary"):
        with st.spinner(t("Generating data and training models...")):
            end = pipeline.demo_end()
            raw, _ = pipeline.make_demo("Mediterranean", 182, 7, end)
            set_data(raw, "SYNTHETIC", now_utc(), "demo", demo_params={"region": "Mediterranean", "weeks": 26, "demo_seed": 7, "end": str(end.date())})
            train_into_state(S.df, "synthetic", DEFAULT_OPTS)
        st.rerun()
    st.stop()

df, rep = S.df, S.rep
if S.origin == "demo":
    st.error(t("Synthetic demo data: results demonstrate the workflow only and are not real findings."))
if df.empty:
    st.warning(t("No detections in the loaded data."))
    st.stop()

tabs = st.tabs([t("🏠 Overview"), t("🛰️ Observations"), t("🎯 Risk Explorer"), t("📈 Historical Trends"), t("🔬 Model Diagnostics"),
                t("🧪 Scenario Lab"), t("🗂️ Data & Provenance"), t("📄 Experiment Reports"), t("⚙️ System Health")])

# ------------------------------------------------------------------ 1 overview
with tabs[0]:
    st.header(t("Executive overview"))
    summ0 = cached_summary(df, (len(df), str(df["acq_datetime"].max())), 3.0, 3)
    hs = geo.hotspot_grid(df)
    fresh_h = (now_utc() - df["acq_datetime"].max()) / pd.Timedelta(hours=1)
    m = st.columns(5)
    m[0].metric(t("Detections"), f"{len(df):,}")
    m[1].metric(t("Clusters"), len(summ0))
    m[2].metric(t("Active days"), df["acq_date"].nunique())
    m[3].metric(t("Newest observation (UTC)"), str(df["acq_datetime"].max())[:16])
    m[4].metric(t("Data age (hours)"), f"{fresh_h:.0f}")
    cs = S.get("csummary")
    if cs:
        counts = cs["status_counts"]
        st.info(t("Risk estimates: {c}. Model {m} (v {v}); beats baselines with 95% interval: {b}; calibrated probabilities shown: {p}.").format(
            c=", ".join(f"{STATUS_ICON[k]}={v}" for k, v in counts.items()), m=cs["model"], v=cs["model_version"],
            b=cs["beats_baselines_95CI"], p=cs["calibration_supported"]))
    else:
        st.info(t("No risk model yet. Train one in Model Diagnostics, or use the one-click demo."))
    st.subheader(t("Scientific tasks (kept separate)"))
    st.dataframe(pd.DataFrame([
        {"task": "A. Active-fire observation", "status": "implemented", "evidence": "FIRMS detections; unconfirmed thermal anomalies"},
        {"task": "B. Next-week occurrence risk", "status": "implemented; real-data skill not yet demonstrated", "evidence": "see Model Diagnostics (test period, CIs)"},
        {"task": "C. Fire spread / impact scenarios", "status": "NOT implemented (extension point only)", "evidence": "needs fuel, terrain, wind and perimeter validation"},
        {"task": "D. Hindcasting", "status": "implemented (walk-forward)", "evidence": "trains only on labels known at each forecast time"}]),
        use_container_width=True)
    if st.button(t("Run / refresh full pipeline")):
        with st.spinner(t("Generating data and training models...")):
            msg = train_into_state(df, wx_mode, DEFAULT_OPTS)
        st.warning(msg) if msg else st.rerun()
    st.caption(t("2-minute presentation script: docs/DEMO_SCRIPT.md"))

# ------------------------------------------------------------------ 2 observations
with tabs[1]:
    dmin, dmax = df["acq_date"].min().date(), df["acq_date"].max().date()
    f1, f2, f3, f4 = st.columns(4)
    rng = f1.date_input(t("Dates"), (dmin, dmax), min_value=dmin, max_value=dmax)
    conf = f2.multiselect(t("Confidence"), ["low", "nominal", "high"], default=["low", "nominal", "high"])
    dn_opts = sorted(df["daynight"].astype(str).unique())
    dn = f3.multiselect(t("Day/night"), dn_opts, default=dn_opts)
    frp_min = f4.slider(t("Min FRP (MW)"), 0.0, float(max(df["frp"].max(), 1.0)), 0.0)
    v = df[df["conf_level"].isin(conf) & df["daynight"].astype(str).isin(dn) & (df["frp"] >= frp_min)]
    if len(rng) == 2:
        v = v[(v["acq_date"] >= pd.Timestamp(rng[0])) & (v["acq_date"] <= pd.Timestamp(rng[1]))]
    kind = st.radio(t("View"), ["Points", "Clusters", "Heatmap", "Time animation"], horizontal=True, format_func=t)
    c1, c2 = st.columns(2)
    eps = c1.slider(t("Cluster radius (km)"), 1.0, 25.0, 3.0)
    mins = c2.slider(t("Min detections per cluster"), 2, 30, 3)
    dfc = geo.cluster(v, eps, mins)
    summ = geo.cluster_summary(dfc)
    m = st.columns(5)
    m[0].metric(t("Detections"), len(v))
    m[1].metric(t("Clusters"), len(summ))
    m[2].metric(t("Total FRP (MW)"), f"{v['frp'].sum():,.0f}")
    m[3].metric(t("High confidence"), f"{(v['conf_level'] == 'high').mean():.0%}" if len(v) else "-")
    m[4].metric(t("Active days"), v["acq_date"].nunique())
    label = "SYNTHETIC DEMO - " if S.origin == "demo" else ""
    if len(v):
        p = dfc.sample(min(len(dfc), 15000), random_state=0).copy()
        hov = [c for c in ["acq_datetime", "source", "conf_level", "frp", "daynight"] if c in p.columns]
        if kind == "Clusters":
            p["group"] = np.where(p["cluster"] < 0, "isolated", "cluster " + p["cluster"].astype(str))
            fig = map_fig(p, kind, "group", hov, zoom_for(v))
            fig.update_layout(showlegend=False)
        elif kind == "Time animation":
            p["day"] = p["acq_date"].dt.strftime("%Y-%m-%d")
            p = p.sort_values("day")
            fig = px.scatter_mapbox(p, lat="latitude", lon="longitude", animation_frame="day", color="frp", zoom=zoom_for(v), height=520,
                                    range_color=(0, float(p["frp"].quantile(0.95))), color_continuous_scale="Viridis")
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        else:
            fig = map_fig(p, kind, "frp" if kind == "Points" else None, hov, zoom_for(v))
        fig.update_layout(title=label + t("Observed detections (not model output)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t("A detection is a satellite-reported thermal anomaly, not a confirmed wildfire."))
    st.subheader(t("Clusters"))
    st.dataframe(summ.head(100), use_container_width=True)
    with st.expander(t("Spatial and temporal coverage")):
        st.json(geodesy.coverage_summary(v if len(v) else df, bbox if mode != "Upload CSV" else features.data_bbox(df)))
    gj = {"type": "FeatureCollection", "properties": {"crs": "EPSG:4326"}, "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.longitude, r.latitude]},
         "properties": {"time_utc": str(r.acq_datetime), "frp": float(r.frp), "cluster": int(r.cluster)}} for r in dfc.itertuples()]}
    e1, e2, e3 = st.columns(3)
    e1.download_button(t("Detections CSV"), dfc.to_csv(index=False), "detections.csv")
    e2.download_button(t("Detections GeoJSON"), json.dumps(gj), "detections.geojson")
    e3.download_button(t("Clusters CSV"), summ.to_csv(index=False), "clusters.csv")

# ------------------------------------------------------------------ 3 risk explorer
with tabs[2]:
    C, CS = S.get("contract"), S.get("csummary")
    if C is None:
        st.info(t("No risk model yet. Train one in Model Diagnostics, or use the one-click demo."))
    else:
        r1, r2, r3, r4 = st.tabs([t("Risk map"), t("Cell detail"), t("Exposure"), t("Alerts and notifications")])
        with r1:
            st.warning(t("Scores are model estimates for the NEXT calendar week. A probability is shown only when calibration was demonstrated on held-out data; otherwise only a score and a relative rank are given."))
            if S.origin == "demo":
                st.error(t("Synthetic demo data: results demonstrate the workflow only and are not real findings."))
            plot = C.assign(size=C["model_score"].clip(0, 1) + 0.05, probability=C["calibrated_probability"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0%}"))
            fig = px.scatter_mapbox(plot, lat="lat", lon="lon", color="level", size="size", size_max=18, color_discrete_map=LEVEL_COLORS,
                                    hover_data={"status": True, "model_score": ":.3f", "probability": True, "relative_rank_pct": True, "size": False},
                                    symbol="status", zoom=zoom_for(df), height=500)
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0),
                              title=("SYNTHETIC DEMO - " if S.origin == "demo" else "") + t("Modelled risk (colour = level, symbol = status)"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(t("Legend: with calibrated probability - Low <10%, Moderate 10-30%, High 30-60%, Very high >=60%. Without calibration - rank-based labels only. Symbols mark status (ok / unreliable / unavailable)."))
            st.dataframe(C.sort_values("model_score", ascending=False)[["lat", "lon", "level", "status", "model_score", "calibrated_probability", "reliability_low", "reliability_high",
                                                                         "data_completeness", "data_freshness_hours", "inputs_out_of_training_range", "reasons"]].head(60),
                         use_container_width=True)
            b1, b2 = st.columns(2)
            b1.download_button(t("Risk contract CSV"), C.to_csv(index=False), "risk_contract.csv")
            b2.download_button(t("Risk contract JSON"), json.dumps({"summary": CS, "cells": json.loads(C.to_json(orient="records", date_format="iso"))}, default=str), "risk_contract.json")
        with r2:
            cs_ = S.get("cell", 0.5)
            tops = C.sort_values("model_score", ascending=False).head(30)
            pick = st.selectbox(t("Grid cell"), list(zip(tops["cy"], tops["cx"])), key="explain_cell",
                                format_func=lambda c: f"lat {(c[0] + .5) * cs_:.2f}, lon {(c[1] + .5) * cs_:.2f}")
            row = C[(C["cy"] == pick[0]) & (C["cx"] == pick[1])].iloc[0]
            k = st.columns(4)
            k[0].metric(t("Model score"), f"{row['model_score']:.3f}")
            k[1].metric(t("Calibrated probability"), "n/a" if pd.isna(row["calibrated_probability"]) else f"{row['calibrated_probability']:.0%}")
            k[2].metric(t("Status"), STATUS_ICON[row["status"]])
            k[3].metric(t("Data completeness"), f"{row['data_completeness']:.0%}")
            st.write(t("Reasons / caveats:") + " " + (row["reasons"] or "-"))
            ex = model.explain_cell(S.res, S.g, int(pick[0]), int(pick[1]))
            if ex is not None:
                st.plotly_chart(px.bar(ex.head(10).iloc[::-1], x="effect_on_risk", y="feature", orientation="h",
                                       title=t("Effect of each feature vs a typical value (occlusion, not causal)")), use_container_width=True)
            gc = S.g[(S.g["cy"] == pick[0]) & (S.g["cx"] == pick[1])]
            cols = [c for c in ("tmax_mean", "rh_mean", "ndvi") if c in gc]
            if cols:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=gc["week"], y=gc["count"], name="detections"), secondary_y=False)
                for c in cols:
                    fig.add_trace(go.Scatter(x=gc["week"], y=gc[c], name=c), secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
        with r3:
            st.write(t("Which villages, roads and protected areas lie near detection clusters? Exposure is a proximity estimate only."))
            src = st.radio(t("Assets source"), ["Demo assets (synthetic)", "Upload CSV", "OpenStreetMap (internet)"], horizontal=True, format_func=t)
            i1, i2, i3, i4 = st.columns(4)
            days_w = i1.slider(t("Use last N days"), 1, 60, 7)
            rad = i2.slider(t("Exposure radius (km)"), 1, 50, 10)
            topn = i3.slider(t("Top clusters"), 3, 20, 8)
            eps_i = i4.slider(t("Cluster radius (km)"), 1.0, 25.0, 3.0, key="eps_i")
            up_assets = st.file_uploader(t("Assets CSV: name, lat, lon, type, population"), type="csv") if src == "Upload CSV" else None
            if st.button(t("Estimate exposure"), type="primary"):
                recent = df[df["acq_date"] >= df["acq_date"].max() - pd.Timedelta(days=days_w - 1)]
                sm = geo.cluster_summary(geo.cluster(recent, eps_i, 3))
                if sm.empty:
                    st.warning(t("No clusters in this window."))
                else:
                    try:
                        if src == "Demo assets (synthetic)":
                            assets = synthetic.generate_assets(features.data_bbox(df, 0.0), 80)
                        elif src == "Upload CSV":
                            if up_assets is not None:
                                security.check_upload_size(up_assets.size)
                            assets = impact.validate_assets(pd.read_csv(up_assets)) if up_assets else pd.DataFrame(columns=impact.COLS)
                        else:
                            with st.spinner("OpenStreetMap..."):
                                assets = pd.concat([impact.overpass_assets(r.lat, r.lon, rad + r.radius_km) for r in sm.head(min(topn, 5)).itertuples()],
                                                   ignore_index=True).drop_duplicates(["name", "lat", "lon"])
                        detail, per = impact.exposure(sm, assets, rad, topn)
                        S.exposure, S.impact = per, (sm.head(topn), assets, detail)
                    except Exception as ex:
                        st.error(security.redact(ex))
            if S.get("impact"):
                sm, assets, detail = S.impact
                st.dataframe(S.exposure, use_container_width=True)
                fig = go.Figure()
                fig.add_trace(go.Scattermapbox(lat=assets["lat"], lon=assets["lon"], mode="markers", name="assets", text=assets["name"], marker=dict(size=7, color="#0072B2")))
                fig.add_trace(go.Scattermapbox(lat=sm["lat"], lon=sm["lon"], mode="markers", name="clusters", text=sm["cluster"].astype(str), marker=dict(size=14, color="#D55E00")))
                fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=float(sm["lat"].mean()), lon=float(sm["lon"].mean())), zoom=zoom_for(df)),
                                  height=460, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(detail, use_container_width=True)
                if "Demo" in str(src):
                    st.caption(t("Demo assets are random and synthetic."))
        with r4:
            st.write(t("Observation, data and prediction alerts are kept in separate categories."))
            zones = st.data_editor(S.zones, num_rows="dynamic", use_container_width=True, key="zones_ed")
            S.zones = zones
            a1, a2, a3, a4 = st.columns(4)
            hours = a1.slider(t("Zone window (h)"), 6, 72, 24)
            zthr = a2.slider(t("Surge z-score"), 1.5, 5.0, 2.5)
            min_det = a3.slider(t("Cluster min detections"), 3, 50, 10)
            min_p = a4.slider(t("Risk alert probability"), 0.3, 0.95, 0.6)
            al = alerts.collect(df, geo.cluster(df, 3.0, 3), zones, S.retrieved, C, CS, hours, zthr, min_det, min_p)
            S.alerts = al
            if al.empty:
                st.success(t("No alerts with the current settings."))
            else:
                cats = st.multiselect(t("Category"), ["observation", "data", "prediction"], default=["observation", "data", "prediction"])
                shown = al[al["category"].isin(cats)]
                st.dataframe(shown, use_container_width=True)
                st.download_button(t("Alerts CSV"), shown.to_csv(index=False), "alerts.csv")
                st.subheader(t("Send notifications"))
                cfg = notify.configured()
                st.caption(f"Email: {'OK' if cfg['email'] else t('not configured')} | Telegram: {'OK' if cfg['telegram'] else t('not configured')} - " + t("set environment variables, see README"))
                n1, n2, n3 = st.columns(3)
                chans = n1.multiselect(t("Channels"), ["email", "telegram"], default=[c for c in ("email", "telegram") if cfg[c]])
                sev = n2.selectbox(t("Minimum severity"), ["warning", "critical", "info"], format_func=t)
                dry = n3.checkbox(t("Preview only (dry run)"), value=True)
                if st.button(t("Send")):
                    out = notify.dispatch(shown, chans, sev, dry)
                    if out["errors"]:
                        st.error(security.redact("; ".join(out["errors"])))
                    st.write(t("{n} new alert(s); sent via: {c}").format(n=out["new"], c=", ".join(out["channels"]) or "-"))
                    if out["preview"]:
                        st.code(out["preview"])

# ------------------------------------------------------------------ 4 trends
with tabs[3]:
    a = geo.activity_anomalies(df)
    fig = go.Figure(go.Bar(x=a["date"], y=a["count"], name="detections"))
    fig.add_trace(go.Scatter(x=a["date"], y=a["baseline"], name="trailing mean", line=dict(color="gray")))
    an = a[a["anomaly"]]
    fig.add_trace(go.Scatter(x=an["date"], y=an["count"], mode="markers", name="anomaly", marker=dict(color="#D55E00", size=10)))
    fig.update_layout(title=t("Daily detections with anomaly flags"), height=380)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(geo.diurnal(df), x="hour_utc", y="detections", title=t("Hour of day (UTC)")), use_container_width=True)
    c2.plotly_chart(px.histogram(df, x="frp", nbins=40, title=t("Fire radiative power distribution")), use_container_width=True)
    st.subheader(t("Hotspot grid (0.5 degrees)"))
    st.dataframe(geo.hotspot_grid(df).head(20).round(4), use_container_width=True)
    st.caption(t("per_1000km2 divides by the true spherical cell area, so cells at different latitudes are comparable."))
    st.subheader(t("Compare two periods"))
    dmin, dmax = df["acq_date"].min().date(), df["acq_date"].max().date()
    mid = dmin + (dmax - dmin) / 2
    p1, p2 = st.columns(2)
    pa = p1.date_input(t("Period A"), (dmin, mid), key="pa")
    pb = p2.date_input(t("Period B"), (mid + timedelta(days=1), dmax), key="pb")
    if len(pa) == 2 and len(pb) == 2:
        cmp_ = geo.compare_periods(df, pa, pb)
        if len(cmp_):
            st.metric(t("Daily rate change"), f"{(cmp_['rate_b'].sum() / max(cmp_['rate_a'].sum(), 1e-9) - 1) * 100:+.0f}%")
            cmp_["abs_change"] = cmp_["change"].abs()
            fig = px.scatter_mapbox(cmp_, lat="lat", lon="lon", color="change", size="abs_change", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                                    zoom=zoom_for(df), height=420, hover_data=["rate_a", "rate_b", "change_pct"])
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cmp_.head(15).round(3), use_container_width=True)
    st.subheader(t("Environmental context"))
    g = S.get("g")
    if g is None:
        st.info(t("Build the dataset in the Model Diagnostics tab first."))
    else:
        st.caption(t("Context source used:") + f" {S.get('wx_used', '?')}")
        top = g.groupby(["cy", "cx"])["count"].sum().sort_values(ascending=False).head(30).index.tolist()
        cs = S.get("cell", 0.5)
        cell_sel = st.selectbox(t("Grid cell"), top, format_func=lambda c: f"lat {(c[0] + .5) * cs:.2f}, lon {(c[1] + .5) * cs:.2f}", key="env_cell")
        gc = g[(g["cy"] == cell_sel[0]) & (g["cx"] == cell_sel[1])]
        wcols = [c for c in features.WX + features.VEG if c in gc and c != "elevation" and gc[c].notna().any()]
        if wcols:
            var = st.selectbox(t("Variable"), wcols, index=wcols.index("tmax_mean") if "tmax_mean" in wcols else 0)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=gc["week"], y=gc["count"], name="detections"), secondary_y=False)
            fig.add_trace(go.Scatter(x=gc["week"], y=gc[var], name=var, line=dict(color="#E69F00")), secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            lab = g[g["label"].notna()]
            cols = [c for c in features.WX + features.VEG + features.NX + features.SP if c in lab and lab[c].notna().any()] + ["count", "label"]
            corr = lab[cols].corr(method="spearman")["label"].drop("label")
            st.plotly_chart(px.bar(corr.sort_values(), orientation="h", title=t("Spearman correlation with detection next week (association only)")), use_container_width=True)
            lcs = [c.replace("lc_", "") for c in ("lc_forest", "lc_shrub", "lc_grass", "lc_crop") if c in gc and gc[c].iloc[0] > 0]
            k1, k2 = st.columns(2)
            if "elevation" in gc:
                k1.metric(t("Elevation (m)"), f"{gc['elevation'].iloc[0]:.0f}")
            k2.metric(t("Land cover"), lcs[0] if lcs else "other")
        else:
            st.warning(t("No context features in the dataset. Enable them in the Model Diagnostics tab."))

# ------------------------------------------------------------------ 5 diagnostics
with tabs[4]:
    st.write(t("Target: will a grid cell have at least one detection next week? Chronological split: 60% train, 20% validation (calibration and threshold), 20% untouched test, compared with persistence and climatology baselines."))
    s1, s2, s3, s4 = st.columns(4)
    o = dict(DEFAULT_OPTS)
    o["cell"] = s1.selectbox(t("Cell size (degrees)"), [0.25, 0.5, 1.0], index=1)
    o["max_cells"] = s2.slider(t("Max cells"), 20, 200, 60)
    o["neg"] = s3.slider(t("Extra never-fire cells"), 0.0, 1.0, 0.3, help=t("Reduces selection bias but makes the task easier."))
    o["tune"] = s4.checkbox(t("Tune hyper-parameters"), value=False)
    u1, u2, u3, u4 = st.columns(4)
    o["use_wx"] = u1.checkbox(t("Weather"), value=True)
    o["use_veg"] = u2.checkbox(t("NDVI and land cover"), value=True)
    o["use_sp"] = u3.checkbox(t("Neighbour cells"), value=True)
    o["use_nx"] = u4.checkbox(t("Next-week weather (assumes a perfect forecast)"), value=False)
    if o["use_nx"]:
        st.warning(t("With next-week weather the test scores are an upper bound that assumes a perfect forecast."))
    if S.origin == "demo" and wx_mode != "synthetic":
        st.warning(t("Demo detections are synthetic; real context data will not match them."))
    if st.button(t("Build dataset and train"), type="primary"):
        bar = st.progress(0.0, t("Loading context data..."))
        try:
            msg = train_into_state(df, wx_mode, o, bar)
        except Exception as ex:
            st.error(security.redact(ex))
            st.stop()
        bar.empty()
        if msg:
            st.warning(msg)
    res = S.get("res")
    if res:
        i = res["info"]
        st.caption(f"Train {i['train_weeks'][0]} to {i['train_weeks'][1]} | Val to {i['val_weeks'][1]} | Test {i['test_weeks'][0]} to {i['test_weeks'][1]} | "
                   f"rows {i['n_train']}/{i['n_val']}/{i['n_test']} | selection: {i['selection']} | {len(res['feats'])} features | cells chosen from history before {S.g.attrs.get('selection_cutoff', '?')[:10]}")
        ok = res["beats_baseline_ci"]
        msg = (t("Best model: {m}. It beats the simple baselines on the held-out test period.") if ok
               else t("Best model: {m}. It does NOT beat the simple baselines on the held-out test period."))
        (st.success if ok else st.warning)(msg.format(m=res["best"]) + f" (random-guess PR-AUC = {i['test_positive_rate']:.2f}; " + t("criterion: 95% bootstrap interval of the PR-AUC gain excludes 0") + ")")
        st.dataframe(res["metrics"], use_container_width=True)
        st.subheader(t("Uncertainty: block bootstrap over test weeks"))
        st.dataframe(res["ci"], use_container_width=True)
        st.caption(t("With few test weeks the intervals are wide and unreliable; n_test_weeks is shown."))
        st.plotly_chart(px.bar(res["metrics"], x="Model", y="PR-AUC", color="Type", title=t("Test PR-AUC")), use_container_width=True)
        if res.get("tuning") is not None:
            with st.expander(t("Hyper-parameter search")):
                st.dataframe(res["tuning"], use_container_width=True)
        c1, c2 = st.columns(2)
        if S.get("cal") is not None:
            f = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="perfect"),
                           go.Scatter(x=S.cal["mean predicted"], y=S.cal["observed rate"], mode="lines+markers", name="model")])
            f.update_layout(title=t("Calibration (test)") + f" ECE={res['ece']:.3f}", xaxis_title="predicted", yaxis_title="observed")
            c1.plotly_chart(f, use_container_width=True)
        if S.get("imp") is not None:
            c2.plotly_chart(px.bar(S.imp.head(12).iloc[::-1], x="importance", y="feature", orientation="h", title=t("Permutation importance (not causal)")), use_container_width=True)
        st.caption(t("Calibrated probabilities are displayed only if ECE <= 0.10 and the Brier score beats the base-rate forecast on the test period. Supported now:") + f" {res['calibration_supported']}")
        st.dataframe(res["calib_table"], use_container_width=True)
        if S.get("conf"):
            cc1, cc2 = st.columns(2)
            cc1.dataframe(S.conf["matrix"], use_container_width=True)
            cc2.plotly_chart(px.line(S.conf["pr_curve"], x="recall", y="precision", title=t("Precision-recall curve (test)")), use_container_width=True)
        st.subheader(t("Stricter validation"))
        st.markdown("**" + t("Geographic validation (unseen regions)") + "**")
        if S.get("geocv") is not None:
            st.dataframe(S.geocv, use_container_width=True)
            st.caption(f"PR-AUC {S.geocv['PR-AUC'].mean():.3f} vs persistence {S.geocv['persistence PR-AUC'].mean():.3f}. " + t("Tests transfer to new regions, not to future weeks."))
        else:
            st.info(t("Not enough spatial blocks for geographic validation."))
        st.markdown("**" + t("Spatio-temporal hold-out (unseen regions AND later period)") + "**")
        if S.get("stt") is not None:
            st.dataframe(S.stt, use_container_width=True)
        else:
            st.info(t("Not enough blocks or weeks for the spatio-temporal hold-out."))
        st.markdown("**" + t("Walk-forward hindcast (only information available at each forecast time)") + "**")
        wf, pooled = S.get("wf") or (None, None)
        if wf is not None:
            st.dataframe(wf, use_container_width=True)
            st.json(pooled)
        st.subheader(t("Fire-intensity forecast"))
        if S.get("inten") is not None:
            st.dataframe(S.inten, use_container_width=True)
            st.caption(t("Target: log(1 + total FRP) next week per cell. Lower MAE and higher R2 than persistence = real skill."))
        st.subheader(t("Error analysis"))
        t1, t2, t3, t4 = st.tabs([t("By month"), t("By region block"), t("Top false positives"), t("Top false negatives")])
        t1.dataframe(S.err["by_month"], use_container_width=True)
        t2.dataframe(S.err["by_region"], use_container_width=True)
        t3.dataframe(S.err["top_false_positives"].round({"proba": 3}), use_container_width=True)
        t4.dataframe(S.err["top_false_negatives"].round({"proba": 3}), use_container_width=True)
        st.subheader(t("Input drift (last 3 weeks vs training)"))
        st.dataframe(S.drift, use_container_width=True)
        st.caption(t("PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 major (rules of thumb). Seasonal change is expected to show as drift."))
        b1, b2, b3 = st.columns(3)
        if b1.button(t("Save model and model card")):
            ver, meta = model.save_model(res, MODELS, {"origin": S.origin, "weather": S.get("wx_used")})
            S.meta = meta
            st.success(t("Saved {v} in models/").format(v=ver))
        b2.download_button(t("Dataset CSV"), S.g.to_csv(index=False), "cell_week_dataset.csv")
        b3.download_button(t("Test predictions CSV"), res["test"].to_csv(index=False), "test_predictions.csv")

# ------------------------------------------------------------------ 6 scenario lab
with tabs[5]:
    st.warning(t("Scenario Lab is a SENSITIVITY ANALYSIS of the trained model with hypothetical inputs. It is not a forecast, not a fire-spread simulator and not evidence of causal effects."))
    info = S.get("info")
    if info is None:
        st.info(t("No risk model yet. Train one in Model Diagnostics, or use the one-click demo."))
    else:
        cs_ = S.get("cell", 0.5)
        tops = S.contract.sort_values("model_score", ascending=False).head(30)
        pick = st.selectbox(t("Grid cell"), list(zip(tops["cy"], tops["cx"])), key="scen_cell", format_func=lambda c: f"lat {(c[0] + .5) * cs_:.2f}, lon {(c[1] + .5) * cs_:.2f}")
        row = scenario.baseline_row(S.g, int(pick[0]), int(pick[1]), info["features"])
        evars = scenario.editable_variables(info)
        stt = info["train_stats"]
        allow = st.checkbox(t("Allow values beyond the training range (extrapolation)"), value=False)
        overrides = {}
        cols = st.columns(3)
        for n_, v_ in enumerate(evars):
            lo, hi = float(stt.loc[v_, "min"]), float(stt.loc[v_, "max"])
            if hi <= lo:
                continue
            if allow:
                lo, hi = lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo)
            cur = float(row[v_].iloc[0]) if not pd.isna(row[v_].iloc[0]) else float(stt.loc[v_, "q50"])
            if cols[n_ % 3].checkbox(t("Change") + f" {v_}", key=f"chg_{v_}"):
                overrides[v_] = cols[n_ % 3].slider(v_, lo, hi, float(np.clip(cur, lo, hi)), key=f"val_{v_}")
        if overrides:
            out = scenario.run_scenario(info, row, overrides)
            k = st.columns(3)
            k[0].metric(t("Baseline score"), f"{out['baseline_score']:.3f}")
            k[1].metric(t("Scenario score"), f"{out['scenario_score']:.3f}", f"{out['delta_score']:+.3f}")
            k[2].metric(t("Scenario probability"), "n/a" if out["scenario_probability"] is None else f"{out['scenario_probability']:.0%}")
            for w in out["warnings"] + out["notes"]:
                st.warning(w)
            st.caption(out["disclaimer"])
        else:
            st.info(t("Tick a variable to change it; unchanged variables stay at the cell's latest observed values."))
        if evars:
            sv = st.selectbox(t("Sweep variable"), evars, key="sweep_var")
            sw = scenario.sweep(info, row, sv)
            st.plotly_chart(px.line(sw, x=sv, y="model_score", title=t("Sensitivity of the model score (other inputs fixed)")), use_container_width=True)
            if len(evars) >= 2:
                v1, v2 = st.columns(2)
                a_ = v1.selectbox(t("X variable"), evars, index=0, key="x2d")
                b_ = v2.selectbox(t("Y variable"), evars, index=min(1, len(evars) - 1), key="y2d")
                if a_ != b_:
                    st.plotly_chart(px.imshow(scenario.sweep2d(info, row, a_, b_), origin="lower", aspect="auto", labels=dict(x=a_, y=b_, color="score"),
                                              title=t("Two-variable sensitivity (training 1st-99th percentile range)")), use_container_width=True)
        with st.expander(t("Fire spread: extension point (not implemented)")):
            st.write(t("A defensible spread model needs the inputs below and validation against observed perimeters."))
            st.json(spread.REQUIRED_INPUTS)
            st.write(spread.VALIDATION_REQUIREMENTS)

# ------------------------------------------------------------------ 7 data & provenance
with tabs[6]:
    q1, q2 = st.columns(2)
    q1.json({**rep, "retrieved_at_utc": str(S.retrieved), "origin": S.origin, "dataset_fingerprint_sha256": experiments.dataset_fingerprint(df)})
    q2.json(geodesy.check_geographic_degrees(df))
    q2.dataframe(df.groupby(["source", "conf_level"]).size().rename("detections").reset_index(), use_container_width=True)
    st.subheader(t("Data providers"))
    st.dataframe(providers.describe_all()[["id", "name", "product", "spatial_resolution", "temporal_resolution", "license", "verified_live"]], use_container_width=True)
    with st.expander(t("Full provider cards (units, CRS, time convention, quality flags, limitations)")):
        st.markdown(providers.providers_markdown())
    st.subheader(t("Provenance log"))
    st.dataframe(store.recent_provenance(), use_container_width=True)
    if S.get("g") is not None:
        st.caption(f"Model dataset: {S.g.attrs}")
    st.subheader(t("Pipeline runs"))
    st.dataframe(store.recent_runs(), use_container_width=True)

# ------------------------------------------------------------------ 8 experiment reports
with tabs[7]:
    st.subheader(t("Reproducible experiment"))
    x1, x2, x3 = st.columns(3)
    ename = x1.text_input(t("Experiment name"), value="experiment")
    eseed = x2.number_input(t("Experiment seed"), value=42, step=1)
    etune = x3.checkbox(t("Tune hyper-parameters"), value=False, key="exp_tune")
    if st.button(t("Run experiment")):
        with st.spinner(t("Running experiment...")):
            try:
                cfg = dict(name=ename, wx_mode=wx_mode, seed=int(eseed), tune=etune)
                exp, path = experiments.run_experiment(df, cfg, S.origin, S.get("demo_params"))
                S.exp_view = exp
                st.success(f"{path.name}")
            except Exception as ex:
                st.error(security.redact(ex))
    exps = experiments.list_experiments()
    st.dataframe(exps, use_container_width=True)
    if len(exps):
        pick = st.selectbox(t("Open experiment"), list(exps["id"]))
        try:
            ev = experiments.get_experiment(pick)
        except Exception as ex:
            ev = None
            st.error(security.redact(ex))
        if ev:
            md_e = report.build_experiment_md(ev)
            d1, d2 = st.columns(2)
            d1.download_button(t("Experiment JSON"), json.dumps(ev, indent=2), f"{pick}.json")
            d2.download_button(t("Experiment report (Markdown)"), md_e, f"{pick}.md")
            if ev["origin"] == "demo" and st.button(t("Verify reproduction")):
                with st.spinner("..."):
                    st.json(experiments.verify_reproduction(experiments.EXPERIMENTS / pick))
            st.markdown(md_e)
    st.divider()
    st.subheader(t("Analysis report"))
    args = dict(origin=S.origin, rep=rep, df=df, res=S.get("res"), geo_cv=S.get("geocv"), imp=S.get("imp"), wx_mode=S.get("wx_used"), alerts=S.get("alerts"))
    md = report.build_report(**args, intensity=S.get("inten"), tuning=(S.res.get("tuning") if S.get("res") else None), exposure=S.get("exposure"))
    r1, r2, r3 = st.columns(3)
    r1.download_button(t("Markdown"), md, "firesentinel_report.md")
    r2.download_button(t("HTML (print to PDF)"), report.to_html(md), "firesentinel_report.html")
    if r3.button(t("Build PDF with charts")):
        with st.spinner("PDF..."):
            S.pdf = report.build_pdf(S.origin, rep, df, S.get("res"), S.get("geocv"), S.get("imp"), S.get("wx_used"), S.get("alerts"), S.get("contract"), S.get("inten"), S.get("exposure"))
    if S.get("pdf"):
        st.download_button(t("Download PDF"), S.pdf, "firesentinel_report.pdf", mime="application/pdf")

# ------------------------------------------------------------------ 9 system health
with tabs[8]:
    h1, h2 = st.columns(2)
    h1.json({"app_version": __version__, **experiments.software_versions()})
    cfg = notify.configured()
    h2.json({"FIRMS_MAP_KEY configured": bool(os.environ.get("FIRMS_MAP_KEY")), "email configured": cfg["email"], "telegram configured": cfg["telegram"],
             "API key configured": bool(os.environ.get("FIRESENTINEL_API_KEY")), "note": "values are never displayed"})
    st.subheader(t("Saved models (integrity-checked)"))
    try:
        b_, m_ = model.load_latest(MODELS)
        st.json({"latest": None if b_ is None else {k: m_.get(k) for k in ("version", "model", "origin", "sha256", "beats_baseline_ci", "calibration_supported")}})
    except ValueError as ex:
        st.error(security.redact(ex))
    st.subheader(t("Fetch runs"))
    runs = store.recent_runs(10)
    st.dataframe(runs, use_container_width=True)
    if len(runs) and (runs["status"] != "ok").any():
        st.warning(t("Some recent fetch attempts failed or were partial; monitoring is only as current as the last successful run."))
    st.subheader(t("Log (last 30 lines)"))
    lp = LOGS / "pipeline.log"
    st.code("\n".join(lp.read_text(encoding="utf-8").splitlines()[-30:]) if lp.exists() else "(empty)")
    st.subheader(t("API"))
    st.code("pip install -r requirements-api.txt\nuvicorn firesentinel.api:app --host 127.0.0.1 --port 8000   # docs at /docs\n# non-local access requires FIRESENTINEL_API_KEY and HTTPS in front")
    st.subheader(t("Known limitations"))
    for x in LIMITATIONS:
        st.markdown(f"- {x}")
