"""High-level workflows used by the app, the API and the CLI."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import features, model, synthetic
from .config import CELL, REGIONS, get_logger

log = get_logger()


def demo_end():
    """Last complete Sunday (UTC)."""
    today = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()
    return today - timedelta(days=(today.weekday() + 1))


def make_demo(region="Mediterranean", days=140, seed=7, end=None):
    """Synthetic FIRMS-like data ending on a Sunday (so weeks are complete). Pass `end` for exact reproducibility."""
    end = pd.Timestamp(end).normalize() if end is not None else demo_end()
    start = end - timedelta(days=days - 1)
    bbox = REGIONS[region]
    return synthetic.generate_fires(bbox, start, end, seed=seed), bbox


def flags_from_features(feats):
    return {"use_weather": any(f in feats for f in features.WX), "use_veg": any(f in feats for f in features.VEG),
            "use_nextwx": any(f in feats for f in features.NX), "use_spatial": any(f in feats for f in features.SP)}


def build_only(df, wx_mode="synthetic", cell=CELL, max_cells=60, neg_ratio=0.3, use_weather=True,
               use_veg=True, use_nextwx=False, use_spatial=True, progress=None, selection_fraction=0.55):
    """Cell-week dataset + feature list (no training)."""
    bbox = features.data_bbox(df)
    start, end = df["acq_date"].min(), df["acq_date"].max()
    cutoff = start + (end - start) * selection_fraction  # cells are chosen from history BEFORE this date only
    cells = features.select_cells(df, bbox, cell, max_cells, neg_ratio, before=cutoff)
    fallback = len(cells) < 5
    if fallback:
        log.warning("Too few cells active before %s; selecting from all data (selection leakage possible).", cutoff)
        cells = features.select_cells(df, bbox, cell, max_cells, neg_ratio)
    wx = None
    if use_weather or use_nextwx:
        wx = features.build_weather(cells, start, end + pd.Timedelta(days=7 if use_nextwx else 0), wx_mode, cell, progress)
        if wx.empty:
            log.warning("No weather could be loaded; continuing without weather features.")
            wx, use_weather, use_nextwx = None, False, False
    veg = lc = None
    if use_veg:
        veg, lc = features.build_vegetation(cells, start, end, wx_mode, cell)
        if veg.empty and lc.empty:
            log.warning("No vegetation data could be loaded; continuing without it.")
            use_veg = False
    g = features.build_dataset(df, cells, wx, cell, veg, lc, nextwx=use_nextwx, spatial=use_spatial)
    g.attrs.update({"selection_cutoff": str(cutoff), "selection_fallback": fallback, "n_cells": int(len(cells)),
                    "weather_source": wx_mode if (use_weather or use_nextwx) else "none"})
    return g, features.feature_list(g, use_weather, use_veg, use_nextwx, use_spatial)


def prepare_and_train(df, wx_mode="synthetic", cell=CELL, max_cells=60, neg_ratio=0.3, use_weather=True,
                      progress=None, use_veg=True, use_nextwx=False, use_spatial=True, tune=False):
    g, feats = build_only(df, wx_mode, cell, max_cells, neg_ratio, use_weather, use_veg, use_nextwx, use_spatial, progress)
    res, msg = model.evaluate(g, feats, tune=tune)
    return g, res, msg
