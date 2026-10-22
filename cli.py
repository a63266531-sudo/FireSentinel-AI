"""Command line: python -m firesentinel.cli demo | fetch | train | schedule"""
import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import data, experiments, model, pipeline, providers, scheduler, store
from .config import REGIONS, SOURCES


def main(argv=None):
    ap = argparse.ArgumentParser(prog="firesentinel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    key_kw = dict(default=os.environ.get("FIRMS_MAP_KEY"), help="FIRMS MAP_KEY (or set FIRMS_MAP_KEY / .env)")
    d = sub.add_parser("demo", help="end-to-end run on SYNTHETIC data (offline)")
    d.add_argument("--region", default="Mediterranean", choices=list(REGIONS))
    d.add_argument("--weeks", type=int, default=26)
    f = sub.add_parser("fetch", help="download from NASA FIRMS into the local database")
    f.add_argument("--key", **key_kw)
    f.add_argument("--source", default="VIIRS_SNPP_NRT", choices=list(SOURCES))
    f.add_argument("--region", default="Egypt", choices=list(REGIONS))
    f.add_argument("--days", type=int, default=5)
    t = sub.add_parser("train", help="train on the local database")
    t.add_argument("--region", default="Egypt", choices=list(REGIONS))
    t.add_argument("--weather", default="open-meteo", choices=["open-meteo", "synthetic"])
    t.add_argument("--tune", action="store_true")
    s = sub.add_parser("schedule", help="fetch -> store -> alerts -> notify, repeatedly or once")
    s.add_argument("--key", **key_kw)
    s.add_argument("--sources", nargs="+", default=["VIIRS_SNPP_NRT"], choices=list(SOURCES))
    s.add_argument("--region", default="Egypt", choices=list(REGIONS))
    s.add_argument("--every-hours", type=float, default=3.0)
    s.add_argument("--notify", nargs="*", default=[], choices=["email", "telegram"])
    s.add_argument("--send", action="store_true", help="really send (default is a dry run)")
    s.add_argument("--once", action="store_true")
    x = sub.add_parser("experiment", help="reproducible experiment with full record (demo or local database)")
    x.add_argument("--demo", action="store_true", help="SYNTHETIC data (offline)")
    x.add_argument("--region", default="Greece", choices=list(REGIONS))
    x.add_argument("--weeks", type=int, default=26)
    x.add_argument("--demo-seed", type=int, default=7)
    x.add_argument("--end", default=None, help="demo end date (a Sunday), for exact reproducibility")
    x.add_argument("--weather", default="synthetic", choices=["open-meteo", "synthetic"])
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--tune", action="store_true")
    x.add_argument("--name", default="experiment")
    v = sub.add_parser("verify-experiment", help="re-run a DEMO experiment and compare metrics")
    v.add_argument("path")
    sub.add_parser("providers", help="write docs/PROVIDERS.md from the provider registry")
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        raw, _ = pipeline.make_demo(a.region, a.weeks * 7)
        df, rep = data.validate(raw, "SYNTHETIC")
        g, res, msg = pipeline.prepare_and_train(df, "synthetic")
        print("SYNTHETIC DEMO DATA - not a scientific result")
        print(rep)
        print(msg or res["metrics"].to_string(index=False))
        return 0
    if a.cmd in ("fetch", "schedule") and not a.key:
        ap.error("a FIRMS MAP_KEY is required (--key, FIRMS_MAP_KEY or .env)")
    if a.cmd == "fetch":
        now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
        raw, errs = data.fetch_range(a.key, a.source, REGIONS[a.region], now - timedelta(days=a.days - 1), now)
        if len(raw):
            data.save_raw(raw, a.source, REGIONS[a.region], now)
            df, _ = data.validate(raw, a.source)
            ins = store.upsert(df, now)
            store.log_run(now, a.source, REGIONS[a.region], len(raw), ins, "ok" if not errs else "partial", "; ".join(errs))
            print(f"fetched {len(raw)}, inserted {ins} new, errors: {errs}")
            return 0
        store.log_run(now, a.source, REGIONS[a.region], 0, 0, "failed", "; ".join(errs))
        print("no data", errs)
        return 1
    if a.cmd == "schedule":
        kw = dict(key=a.key, sources=a.sources, region=a.region, channels=a.notify, dry_run=not a.send)
        if a.once:
            print(scheduler.run_once(**kw))
        else:
            scheduler.loop(a.every_hours, **kw)
        return 0
    if a.cmd == "providers":
        from .config import ROOT
        (ROOT / "docs" / "PROVIDERS.md").write_text(providers.providers_markdown(), encoding="utf-8")
        print("docs/PROVIDERS.md written")
        return 0
    if a.cmd == "verify-experiment":
        out = experiments.verify_reproduction(a.path)
        print(out)
        return 0 if out["match"] else 1
    if a.cmd == "experiment":
        cfg = {"name": a.name, "seed": a.seed, "tune": a.tune, "wx_mode": a.weather if not a.demo else "synthetic"}
        if a.demo:
            exp, out = experiments.run_demo_experiment(a.region, a.weeks, a.demo_seed, a.end, cfg)
        else:
            df = store.load(REGIONS[a.region])
            if df.empty:
                print("Local database is empty for this region. Run the fetch command first.")
                return 1
            exp, out = experiments.run_experiment(data.validate(df)[0], cfg, "database")
        print("experiment written to", out)
        print("beats baselines (95% CI excludes 0):", exp["beats_baselines"]["ci_excludes_zero"], "| calibration supported:", exp["calibration"]["supported"])
        print("reproduce:", exp["reproduce"])
        return 0
    df = store.load(REGIONS[a.region])
    if df.empty:
        print("Local database is empty for this region. Run the fetch command first.")
        return 1
    df, _ = data.validate(df)
    g, res, msg = pipeline.prepare_and_train(df, a.weather, tune=a.tune)
    print(msg or res["metrics"].to_string(index=False))
    if res:
        print("saved:", model.save_model(res)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
