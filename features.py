"""Cell-week dataset: fire history + weather + vegetation + neighbours; label = fire NEXT week."""
import numpy as np
import pandas as pd

from .config import CELL, get_logger
from .geo import cell_ids
from .vegetation import LC_FEATS, get_landcover, get_ndvi, weekly_ndvi
from .weather import get_weather, weekly_weather

log = get_logger()
BASE = ["count", "lag1", "lag2", "roll4", "hist_mean", "frp", "frp_sum", "m_sin", "m_cos"]
WX = ["tmax_mean", "precip_sum", "wind_max", "et0_sum", "rh_mean", "dry_days", "precip14", "fwi", "elevation"]
VEG = ["ndvi", "ndvi_anom", "ndvi_chg"]
NX = ["nx_tmax", "nx_wind", "nx_precip", "nx_fwi"]
SP = ["nb_count", "nb_lag1"]
WEEK = pd.Timedelta(days=7)


def assign_cells(df, cell=CELL):
    d = cell_ids(df, cell)
    d["week"] = d["acq_datetime"].dt.to_period("W").dt.start_time.astype("datetime64[ns]")
    return d


def data_bbox(df, pad=0.5):
    return (df["longitude"].min() - pad, df["latitude"].min() - pad,
            df["longitude"].max() + pad, df["latitude"].max() + pad)


def select_cells(df, bbox, cell=CELL, max_cells=60, neg_ratio=0.3, seed=0, before=None):
    """Cells with detections (most active first) + optional random never-fire cells.
    Extra negative cells reduce selection bias but make the task easier.
    `before`: only detections earlier than this timestamp may influence the selection, so that cells are never
    chosen using information from the validation/test period (selection leakage)."""
    d = cell_ids(df if before is None else df[df["acq_datetime"] < before], cell)
    fc = d.groupby(["cy", "cx"]).size().sort_values(ascending=False).reset_index(name="n")
    n_pos = min(len(fc), max(1, int(max_cells / (1 + neg_ratio))))
    pos = fc.head(n_pos)[["cy", "cx"]].assign(fire_cell=True)
    w, s, e, n = bbox
    cy0, cy1 = int(np.floor(s / cell)), int(np.floor(n / cell))
    cx0, cx1 = int(np.floor(w / cell)), int(np.floor(e / cell))
    rng, fire_set, neg = np.random.default_rng(seed), set(zip(fc["cy"], fc["cx"])), set()
    n_neg, tries = int(n_pos * neg_ratio), 0
    while cy1 > cy0 and cx1 > cx0 and len(neg) < n_neg and tries < n_neg * 50 + 10:
        c = (int(rng.integers(cy0, cy1)), int(rng.integers(cx0, cx1)))
        tries += 1
        if c not in fire_set:
            neg.add(c)
    if not neg:
        return pos
    return pd.concat([pos, pd.DataFrame(sorted(neg), columns=["cy", "cx"]).assign(fire_cell=False)], ignore_index=True)


def build_weather(cells, start, end, mode, cell=CELL, progress=None):
    out, n = [], len(cells)
    for i, (cy, cx) in enumerate(zip(cells["cy"], cells["cx"])):
        try:
            wk = weekly_weather(get_weather((cy + 0.5) * cell, (cx + 0.5) * cell, start, end, mode))
            wk["cy"], wk["cx"] = cy, cx
            out.append(wk)
        except Exception as e:
            log.warning("Weather failed for cell (%s,%s): %s", cy, cx, e)
        if progress:
            progress((i + 1) / n)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_vegetation(cells, start, end, mode, cell=CELL):
    """Weekly NDVI per cell + static land-cover one-hot. Returns (veg_df, lc_df); empty if unavailable."""
    vs, lcs = [], []
    for cy, cx in zip(cells["cy"], cells["cx"]):
        lat, lon = (cy + 0.5) * cell, (cx + 0.5) * cell
        try:
            w = weekly_ndvi(get_ndvi(lat, lon, start, end, mode), start, end)
            w["cy"], w["cx"] = cy, cx
            vs.append(w)
        except Exception as e:
            log.warning("NDVI failed for cell (%s,%s): %s", cy, cx, e)
        try:
            grp = get_landcover(lat, lon, pd.Timestamp(end).year, mode)
            lcs.append({"cy": cy, "cx": cx, **{f: float(f == f"lc_{grp}") for f in LC_FEATS}})
        except Exception as e:
            log.warning("Land cover failed for cell (%s,%s): %s", cy, cx, e)
    return (pd.concat(vs, ignore_index=True) if vs else pd.DataFrame(),
            pd.DataFrame(lcs) if lcs else pd.DataFrame())


def _neighbour_counts(g, all_cnt, lag=0):
    tot = np.zeros(len(g))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                idx = pd.MultiIndex.from_arrays([g["cy"] + dy, g["cx"] + dx, g["week"] - WEEK * lag])
                tot += all_cnt.reindex(idx).fillna(0).to_numpy()
    return tot


def build_dataset(df, cells, wx=None, cell=CELL, veg=None, lc=None, nextwx=False, spatial=True):
    """One row per cell-week. Features use only information up to that week.
    label = any detection in the following week; label_frp = log1p(total FRP next week).
    nextwx=True adds NEXT week's weather (perfect-prognosis: assumes a perfect forecast when evaluated)."""
    d = assign_cells(df, cell)
    first_partial = d["acq_datetime"].min().weekday() != 0
    last_partial = d["acq_datetime"].max().weekday() != 6
    g = d.groupby(["cy", "cx", "week"]).agg(count=("latitude", "size"), frp=("frp", "mean"),
                                           frp_sum=("frp", "sum")).reset_index()
    all_cnt = g.set_index(["cy", "cx", "week"])["count"]  # all detections, also outside selected cells
    weeks = pd.date_range(d["week"].min(), d["week"].max(), freq="W-MON").astype("datetime64[ns]")
    if first_partial and len(weeks) > 1:
        weeks = weeks[1:]  # a partial first week would understate counts
    full = cells[["cy", "cx"]].drop_duplicates().merge(pd.DataFrame({"week": weeks}), how="cross")
    g = full.merge(g, on=["cy", "cx", "week"], how="left")
    g[["count", "frp", "frp_sum"]] = g[["count", "frp", "frp_sum"]].fillna(0)

    if wx is not None and len(wx):
        g = g.merge(wx, on=["cy", "cx", "week"], how="left")
        if nextwx:
            nx = wx[["cy", "cx", "week", "tmax_mean", "wind_max", "precip_sum", "fwi"]].copy()
            nx["week"] = nx["week"] - WEEK
            nx = nx.rename(columns={"tmax_mean": "nx_tmax", "wind_max": "nx_wind", "precip_sum": "nx_precip", "fwi": "nx_fwi"})
            g = g.merge(nx, on=["cy", "cx", "week"], how="left")
        # Missing values are left as NaN on purpose: imputation is fitted inside each model pipeline on the
        # training rows only, so no statistic of the validation/test period leaks into preprocessing.
    g = g.sort_values(["cy", "cx", "week"]).reset_index(drop=True)

    if veg is not None and len(veg):
        g = g.merge(veg[["cy", "cx", "week", "ndvi"]], on=["cy", "cx", "week"], how="left").sort_values(["cy", "cx", "week"]).reset_index(drop=True)
        g["ndvi"] = g.groupby(["cy", "cx"])["ndvi"].transform(lambda s: s.ffill())  # past values only (no back-fill)
        g["ndvi_anom"] = g["ndvi"] - g.groupby(["cy", "cx"])["ndvi"].transform(lambda s: s.expanding().mean())
        g["ndvi_chg"] = g.groupby(["cy", "cx"])["ndvi"].diff().fillna(0)
    if lc is not None and len(lc):
        g = g.merge(lc, on=["cy", "cx"], how="left")
        g[LC_FEATS] = g[LC_FEATS].fillna(0)

    grp = g.groupby(["cy", "cx"])["count"]
    g["lag1"], g["lag2"] = grp.shift(1), grp.shift(2)
    g["roll4"] = grp.transform(lambda s: s.rolling(4, min_periods=1).mean())
    g["hist_mean"] = grp.transform(lambda s: s.expanding().mean())
    g["m_sin"] = np.sin(2 * np.pi * g["week"].dt.month / 12)
    g["m_cos"] = np.cos(2 * np.pi * g["week"].dt.month / 12)
    if spatial:
        g["nb_count"] = _neighbour_counts(g, all_cnt, 0)
        g["nb_lag1"] = _neighbour_counts(g, all_cnt, 1)

    nxt = grp.shift(-1)
    nxt_f = g.groupby(["cy", "cx"])["frp_sum"].shift(-1)
    g["label"] = np.where(nxt.isna(), np.nan, (nxt > 0).astype(float))
    g["label_frp"] = np.where(nxt_f.isna(), np.nan, np.log1p(nxt_f.fillna(0)))
    if last_partial and len(weeks) > 1:
        g.loc[g["week"] == weeks[-2], ["label", "label_frp"]] = np.nan  # next week incomplete -> unknown
    return g.dropna(subset=["lag1", "lag2"]).reset_index(drop=True)


def feature_list(g, use_weather=True, use_veg=True, use_nextwx=False, use_spatial=True):
    fs = list(BASE)
    fs += [c for c in WX if use_weather and c in g and g[c].notna().any()]
    if use_veg:
        fs += [c for c in VEG if c in g] + [c for c in LC_FEATS if c in g and g[c].sum() > 0]
    fs += [c for c in NX if use_nextwx and c in g and g[c].notna().any()]
    fs += [c for c in SP if use_spatial and c in g]
    return fs
