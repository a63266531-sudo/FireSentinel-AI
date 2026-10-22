#!/usr/bin/env python3
"""Environment + end-to-end self check.
   python scripts/check_setup.py            offline checks (no internet needed)
   python scripts/check_setup.py --network  also probes FIRMS (needs FIRMS_MAP_KEY), Open-Meteo, NASA ORNL MODIS, Overpass"""
import argparse
import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
results = []


def check(name, fn, required=True):
    try:
        detail = fn()
        results.append((name, "PASS", str(detail or "")))
    except Exception as e:  # report, never crash
        results.append((name, "FAIL" if required else "WARN", f"{type(e).__name__}: {e}"[:160]))


def py_version():
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, found {sys.version.split()[0]}"
    return sys.version.split()[0]


def imports():
    out = []
    for m in ("numpy", "pandas", "sklearn", "scipy", "plotly", "streamlit", "requests", "joblib", "reportlab", "matplotlib", "markdown"):
        mod = importlib.import_module(m)
        out.append(f"{m} {getattr(mod, '__version__', '?')}")
    return ", ".join(out)


def offline_pipeline():
    import pandas as pd
    from firesentinel import contract, data, pipeline
    raw, _ = pipeline.make_demo("Greece", 112, 3)
    df, rep = data.validate(raw, "SYNTHETIC")
    g, res, msg = pipeline.prepare_and_train(df, "synthetic", max_cells=30)
    assert res is not None, msg
    c, s = contract.build_contract(contract.info_from_res(res), g, df, pd.Timestamp.now(), synthetic=True)
    assert len(c) > 0 and set(c["status"]) <= {"ok", "unreliable", "unavailable"}
    return f"{rep['rows_clean']} synthetic detections, best model {res['best']}, risk-contract statuses {s['status_counts']}"


def security():
    import security_check
    f = security_check.scan(ROOT)
    assert not f, f"{len(f)} finding(s): {f[:2]}"
    return "no secrets / risky calls"


def net_firms():
    import os
    from firesentinel import data
    key = os.environ.get("FIRMS_MAP_KEY")
    assert key, "FIRMS_MAP_KEY not set (put it in .env)"
    d = data.fetch_chunk(key, "VIIRS_SNPP_NRT", (24.7, 21.9, 36.9, 31.7), 1)
    return f"{len(d)} rows, columns: {', '.join(d.columns[:6])}"


def net_openmeteo():
    import pandas as pd
    from firesentinel import weather
    end = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize() - timedelta(days=15)
    d = weather.fetch_open_meteo(30.0, 31.0, end - timedelta(days=6), end)
    assert len(d) >= 5
    return f"{len(d)} daily rows, columns: {', '.join(d.columns)}"


def net_ornl():
    from firesentinel import vegetation
    j = vegetation._get("MOD13Q1/dates", {"latitude": 30.0, "longitude": 31.0})
    return f"{len(j['dates'])} NDVI composite dates"


def net_overpass():
    from firesentinel import impact
    d = impact.overpass_assets(30.04, 31.23, 3)
    return f"{len(d)} OSM assets"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true")
    a = ap.parse_args()
    check("python", py_version)
    check("dependencies", imports)
    check("offline demo pipeline", offline_pipeline)
    check("security scan", security)
    if a.network:
        check("NASA FIRMS (needs key)", net_firms)
        check("Open-Meteo weather", net_openmeteo)
        check("NASA ORNL MODIS NDVI", net_ornl)
        check("OpenStreetMap Overpass", net_overpass)
    w = max(len(r[0]) for r in results)
    for n, s, d in results:
        print(f"[{s}] {n.ljust(w)}  {d}")
    sys.exit(1 if any(s == "FAIL" for _, s, _ in results) else 0)
