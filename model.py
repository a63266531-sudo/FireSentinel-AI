"""Training, honest evaluation, calibration, explainability and error analysis."""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CELL, LIMITATIONS, MODELS, RISK_BINS, RISK_LABELS
from .security import sha256_file


def _models(seed=42):
    return {
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                             LogisticRegression(class_weight="balanced", max_iter=2000)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                       RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                              n_jobs=-1, random_state=seed)),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200,
                                                            class_weight="balanced", random_state=seed),
    }


def _safe(fn, y, s):
    try:
        return float(fn(y, s))
    except ValueError:
        return float("nan")


def _best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
    return float(t[int(np.argmax(f1))])


GRIDS = {
    "Logistic Regression": [{"logisticregression__C": c} for c in (0.1, 1.0, 10.0)],
    "Random Forest": [{"randomforestclassifier__min_samples_leaf": l, "randomforestclassifier__max_depth": d}
                      for l in (3, 8) for d in (None, 8)],
    "Gradient Boosting": [{"max_depth": d, "learning_rate": lr} for d in (3, 5) for lr in (0.05, 0.1)],
}


def _tune(name, tr, feats, seed):
    """Small grid search with an expanding-window (forward-chaining) CV INSIDE the training period."""
    grid, wk = GRIDS.get(name), np.sort(tr["week"].unique())
    if not grid or len(wk) < 8:
        return None, []
    edges = [int(len(wk) * q) for q in (0.5, 0.67, 0.84, 1.0)]
    best, best_s, rows = None, -1.0, []
    for p in grid:
        sc = []
        for a, b in zip(edges[:-1], edges[1:]):
            trn, val = tr[tr["week"].isin(wk[:a])], tr[tr["week"].isin(wk[a:b])]
            if trn["label"].nunique() < 2 or val["label"].nunique() < 2:
                continue
            m = _models(seed)[name]
            m.set_params(**p)
            for k in [k for k in m.get_params() if k.endswith("n_estimators")]:
                m.set_params(**{k: 150})  # faster during search only
            m.fit(trn[feats], trn["label"])
            sc.append(_safe(average_precision_score, val["label"], m.predict_proba(val[feats])[:, 1]))
        s = float(np.nanmean(sc)) if sc and not np.all(np.isnan(sc)) else float("nan")
        rows.append({"Model": name, "params": str(p), "CV PR-AUC": round(s, 3)})
        if not np.isnan(s) and s > best_s:
            best, best_s = p, s
    return best, rows


def evaluate(g, feats, seed=42, tune=False):
    """Time-ordered split: 60% train / 20% validation (calibration + threshold) / 20% test.
    The test period is never used for fitting, tuning, calibrating or model selection."""
    g = g[g["label"].notna()].copy()
    weeks = np.sort(g["week"].unique())
    if len(weeks) < 10:
        return None, f"Need at least 10 labeled weeks (have {len(weeks)}). Load a longer history."
    t1, t2 = weeks[int(len(weeks) * 0.6)], weeks[int(len(weeks) * 0.8)]
    tr = g[g["week"] < t1]
    va = g[(g["week"] >= t1) & (g["week"] < t2)]
    te = g[g["week"] >= t2]
    if tr["label"].nunique() < 2 or te["label"].nunique() < 2:
        return None, "Train or test split contains a single class. Use a larger region or a longer period."
    has_val = va["label"].nunique() == 2 and len(va) >= 30
    y = te["label"]
    rows, models, cals, ptest = [], {}, {}, {}

    def add(name, kind, sva, ste, brier=np.nan):
        thr = _best_threshold(va["label"], sva) if has_val else 0.5
        pred = (ste >= thr).astype(int)
        rows.append({"Model": name, "Type": kind,
                     "PR-AUC": _safe(average_precision_score, y, ste), "ROC-AUC": _safe(roc_auc_score, y, ste),
                     "Precision": precision_score(y, pred, zero_division=0),
                     "Recall": recall_score(y, pred, zero_division=0),
                     "F1": f1_score(y, pred, zero_division=0), "Brier": brier,
                     "Threshold": thr,
                     "Val PR-AUC": _safe(average_precision_score, va["label"], sva) if has_val else np.nan})

    baselines = {"Baseline: persistence": lambda d: (d["count"] > 0).astype(float).to_numpy(),
                 "Baseline: climatology": lambda d: d["hist_mean"].to_numpy()}
    for n, f in baselines.items():
        add(n, "baseline", f(va), f(te))
    tuning = []
    for n, m in _models(seed).items():
        if tune:
            bp, trows = _tune(n, tr, feats, seed)
            if bp:
                m.set_params(**bp)
            tuning += trows
        m.fit(tr[feats], tr["label"])
        pva, pte = m.predict_proba(va[feats])[:, 1], m.predict_proba(te[feats])[:, 1]
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pva, va["label"]) if has_val else None
        if cal is not None:  # class weights distort probabilities -> recalibrate on validation only
            pva, pte = cal.predict(pva), cal.predict(pte)
        models[n], cals[n], ptest[n] = m, cal, pte
        add(n, "ml", pva, pte, brier=_safe(brier_score_loss, y, pte))

    met = pd.DataFrame(rows)
    ml = met[met["Type"] == "ml"]
    key = "Val PR-AUC" if has_val else "PR-AUC"
    best = ml.loc[ml[key].idxmax(), "Model"]
    best_row = met[met["Model"] == best].iloc[0]
    base_best = met.loc[met["Type"] == "baseline", "PR-AUC"].max()
    te = te.assign(proba=ptest[best], pred=(ptest[best] >= best_row["Threshold"]).astype(int),
                   score_persistence=(te["count"] > 0).astype(float), score_climatology=te["hist_mean"])
    info = {"train_weeks": (str(pd.Timestamp(weeks[0]).date()), str(pd.Timestamp(t1).date())),
            "val_weeks": (str(pd.Timestamp(t1).date()), str(pd.Timestamp(t2).date())),
            "test_weeks": (str(pd.Timestamp(t2).date()), str(pd.Timestamp(weeks[-1]).date())),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "test_positive_rate": float(y.mean()),
            "selection": "validation PR-AUC" if has_val else "TEST PR-AUC (validation too small - optimistic)"}
    res = {"metrics": met.round(3), "info": info, "models": models, "calibrators": cals, "best": best,
           "threshold": float(best_row["Threshold"]), "feats": feats, "test": te,
           "beats_baseline": bool(best_row["PR-AUC"] > base_best),
           "tuning": pd.DataFrame(tuning) if tuning else None, "ref": tr[feats].median()}
    yv, pv = res["test"]["label"], res["test"]["proba"]
    res["ece"] = _ece(yv, pv)
    base_brier = float(yv.mean() * (1 - yv.mean()))
    res["calibration_supported"] = bool(cals[best] is not None and res["ece"] <= 0.10
                                        and _safe(brier_score_loss, yv, pv) < base_brier)
    res["train_stats"] = _train_stats(tr, feats)
    res["ci"] = bootstrap_ci(res, seed=seed)
    res["calib_table"] = calibration_table(yv, pv)
    res["train_ref"] = tr[feats].sample(min(len(tr), 1000), random_state=seed)
    d = res["ci"].loc[res["ci"]["metric"] == "PR-AUC minus best baseline"]
    res["beats_baseline_ci"] = bool(len(d) and d["lo"].iloc[0] > 0)
    return res, ""


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def calibration_table(y, p, bins=10):
    """Equal-width probability bins on the TEST period with the observed rate and a Wilson 95% interval."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows, edges = [], np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        n = int(m.sum())
        k = int(y[m].sum()) if n else 0
        wl, wh = wilson_interval(k, n)
        rows.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": n, "mean_pred": float(p[m].mean()) if n else np.nan,
                     "observed": k / n if n else np.nan, "wilson_lo": wl, "wilson_hi": wh})
    return pd.DataFrame(rows).round(3)


def confusion_and_pr(res):
    """Confusion matrix at the validation-chosen threshold and the PR curve (test period)."""
    te = res["test"]
    tn = int(((te["pred"] == 0) & (te["label"] == 0)).sum()); fp = int(((te["pred"] == 1) & (te["label"] == 0)).sum())
    fn = int(((te["pred"] == 0) & (te["label"] == 1)).sum()); tp = int(((te["pred"] == 1) & (te["label"] == 1)).sum())
    p, r, _ = precision_recall_curve(te["label"], te["proba"])
    return {"matrix": pd.DataFrame([[tn, fp], [fn, tp]], index=["actual: no detection", "actual: detection"],
                                   columns=["predicted: no", "predicted: yes"]),
            "pr_curve": pd.DataFrame({"recall": r, "precision": p}), "base_rate": float(te["label"].mean())}


def _ece(y, p, bins=10):
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def _train_stats(tr, feats):
    q = tr[feats].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T
    q.columns = ["q01", "q05", "q50", "q95", "q99"]
    q.insert(0, "min", tr[feats].min())
    q["max"] = tr[feats].max()
    return q


def bootstrap_ci(res, n=300, seed=0, alpha=0.05):
    """Block bootstrap over test WEEKS (weeks are the dependent unit). With few test weeks the intervals are
    unreliable; n_test_weeks is reported. Includes the PR-AUC difference to the best baseline."""
    te = res["test"].reset_index(drop=True)
    wk = te["week"].to_numpy()
    uw = np.unique(wk)
    idx_by_week = [np.where(wk == w)[0] for w in uw]
    y, p, pred = te["label"].to_numpy(), te["proba"].to_numpy(), te["pred"].to_numpy()
    sp, sc = te["score_persistence"].to_numpy(), te["score_climatology"].to_numpy()
    rng = np.random.default_rng(seed)

    def metrics(ix):
        yy = y[ix]
        if yy.min() == yy.max():
            return None
        pr = average_precision_score(yy, p[ix])
        base = max(average_precision_score(yy, sp[ix]), average_precision_score(yy, sc[ix]))
        return {"PR-AUC": pr, "Brier": brier_score_loss(yy, p[ix]), "Precision": precision_score(yy, pred[ix], zero_division=0),
                "Recall": recall_score(yy, pred[ix], zero_division=0), "F1": f1_score(yy, pred[ix], zero_division=0),
                "PR-AUC minus best baseline": pr - base}

    full = metrics(np.arange(len(te)))
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(uw), len(uw))
        m = metrics(np.concatenate([idx_by_week[i] for i in pick]))
        if m:
            draws.append(m)
    rows = []
    for k, v in (full or {}).items():
        vals = [d[k] for d in draws]
        rows.append({"metric": k, "estimate": v, "lo": np.percentile(vals, 100 * alpha / 2) if len(vals) >= 20 else np.nan,
                     "hi": np.percentile(vals, 100 * (1 - alpha / 2)) if len(vals) >= 20 else np.nan,
                     "valid_resamples": len(draws), "n_test_weeks": len(uw)})
    return pd.DataFrame(rows).round(3)


def evaluate_spatiotemporal(g, feats, model_name=None, seed=42, holdout_frac=0.25):
    """Strictest split: some spatial blocks are held out for ALL time, and the test is the LATER period in those
    blocks. The model never saw the places nor the period."""
    g = g[g["label"].notna()].copy()
    blocks = (g["cy"] // 4).astype(str) + "_" + (g["cx"] // 4).astype(str)
    ub = np.array(sorted(blocks.unique()))
    if len(ub) < 4:
        return None
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(ub)[:max(1, int(np.ceil(len(ub) * holdout_frac)))])
    weeks = np.sort(g["week"].unique())
    if len(weeks) < 10:
        return None
    cut = weeks[int(len(weeks) * 0.7)]
    is_held = blocks.isin(held)
    tr, te = g[~is_held & (g["week"] < cut)], g[is_held & (g["week"] >= cut)]
    if tr["label"].nunique() < 2 or te["label"].nunique() < 2:
        return None
    rows = []
    sc = {"Baseline: persistence": (te["count"] > 0).astype(float), "Baseline: climatology": te["hist_mean"]}
    for name, s in sc.items():
        rows.append({"Model": name, "PR-AUC": _safe(average_precision_score, te["label"], s), "ROC-AUC": _safe(roc_auc_score, te["label"], s)})
    for name in ([model_name] if model_name else list(_models(seed))):
        m = _models(seed)[name].fit(tr[feats], tr["label"])
        s = m.predict_proba(te[feats])[:, 1]
        rows.append({"Model": name, "PR-AUC": _safe(average_precision_score, te["label"], s), "ROC-AUC": _safe(roc_auc_score, te["label"], s)})
    out = pd.DataFrame(rows).round(3)
    out.attrs.update({"train_blocks": int((~is_held).groupby(blocks).any().sum() - len(held)), "held_out_blocks": sorted(held),
                      "test_positive_rate": float(te["label"].mean()), "n_train": len(tr), "n_test": len(te),
                      "split_week": str(pd.Timestamp(cut).date())})
    return out


def walk_forward(g, feats, model_name="Logistic Regression", n_origins=8, min_train_weeks=8, seed=42):
    """Hindcast: at the end of each of the last n weeks, train ONLY on rows whose labels were already known at that
    time, predict the following week, then score it. Returns (per_origin_df, pooled_dict)."""
    g = g[g["label"].notna()].copy()
    wk = np.sort(g["week"].unique())
    origins = [w for w in wk[-n_origins:] if (wk < w).sum() >= min_train_weeks]
    rows, ys, ps, base_p, base_c = [], [], [], [], []
    for k in origins:
        tr = g[g["week"] <= k - pd.Timedelta(days=7)]  # labels about weeks <= k are known at the end of week k
        te = g[g["week"] == k]                           # forecast about week k+1
        if tr["label"].nunique() < 2 or te.empty:
            continue
        m = _models(seed)[model_name].fit(tr[feats], tr["label"])
        p = m.predict_proba(te[feats])[:, 1]
        pers = (te["count"] > 0).astype(float).to_numpy()
        rows.append({"forecast_made_end_of_week": str(pd.Timestamp(k).date()), "train_rows": len(tr), "n_cells": len(te),
                     "positives": int(te["label"].sum()), "PR-AUC": _safe(average_precision_score, te["label"], p),
                     "persistence PR-AUC": _safe(average_precision_score, te["label"], pers)})
        ys.append(te["label"].to_numpy()); ps.append(p); base_p.append(pers); base_c.append(te["hist_mean"].to_numpy())
    if not rows:
        return None, None
    y, p = np.concatenate(ys), np.concatenate(ps)
    pooled = {"model": model_name, "origins": len(rows), "pooled_PR-AUC": _safe(average_precision_score, y, p),
              "pooled_persistence_PR-AUC": _safe(average_precision_score, y, np.concatenate(base_p)),
              "pooled_climatology_PR-AUC": _safe(average_precision_score, y, np.concatenate(base_c)),
              "pooled_positive_rate": float(y.mean())}
    return pd.DataFrame(rows).round(3), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pooled.items()}


def geo_cv(g, feats, model_name, k=5, seed=42):
    """Spatial block CV: trains on some regions, tests on unseen regions (all weeks)."""
    g = g[g["label"].notna()].copy()
    blocks = (g["cy"] // 4).astype(str) + "_" + (g["cx"] // 4).astype(str)
    nb = blocks.nunique()
    if nb < 3:
        return None
    rows = []
    for i, (a, b) in enumerate(GroupKFold(n_splits=min(k, nb)).split(g, g["label"], blocks)):
        tr, te = g.iloc[a], g.iloc[b]
        if tr["label"].nunique() < 2 or te["label"].nunique() < 2:
            continue
        m = _models(seed)[model_name].fit(tr[feats], tr["label"])
        p = m.predict_proba(te[feats])[:, 1]
        rows.append({"fold": i + 1, "test_rows": len(te), "PR-AUC": _safe(average_precision_score, te["label"], p),
                     "persistence PR-AUC": _safe(average_precision_score, te["label"], (te["count"] > 0).astype(float)),
                     "positive rate": float(te["label"].mean())})
    return pd.DataFrame(rows).round(3) if rows else None


def importance(res, n=2000, seed=0):
    """Permutation importance on the test set (drop in PR-AUC). Not causal."""
    te = res["test"]
    te = te.sample(n, random_state=seed) if len(te) > n else te
    try:
        pi = permutation_importance(res["models"][res["best"]], te[res["feats"]], te["label"],
                                    scoring="average_precision", n_repeats=5, random_state=seed)
    except ValueError:
        return None
    return (pd.DataFrame({"feature": res["feats"], "importance": pi.importances_mean, "std": pi.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True).round(4))


def calibration_df(res, bins=8):
    te = res["test"]
    try:
        obs, pred = calibration_curve(te["label"], te["proba"], n_bins=bins, strategy="quantile")
    except ValueError:
        return None
    return pd.DataFrame({"mean predicted": pred, "observed rate": obs})


def error_analysis(res, cell=CELL):
    te = res["test"].copy()
    te["month"] = te["week"].dt.month
    te["region_block"] = (te["cy"] // 4).astype(str) + "_" + (te["cx"] // 4).astype(str)
    te["fp"] = ((te["pred"] == 1) & (te["label"] == 0)).astype(int)
    te["fn"] = ((te["pred"] == 0) & (te["label"] == 1)).astype(int)
    agg = lambda by: te.groupby(by).agg(rows=("label", "size"), positives=("label", "sum"),
                                         false_pos=("fp", "sum"), false_neg=("fn", "sum")).reset_index()
    cols = ["cy", "cx", "week", "count", "proba", "label"]
    return {"by_month": agg("month"), "by_region": agg("region_block"),
            "top_false_positives": te[te["fp"] == 1].sort_values("proba", ascending=False).head(10)[cols],
            "top_false_negatives": te[te["fn"] == 1].sort_values("proba").head(10)[cols]}


def score_with_bundle(bundle, g, cell=CELL):
    """Risk of any detection in the week AFTER the latest week in g, using a model bundle."""
    missing = [f for f in bundle["features"] if f not in g.columns]
    if missing:
        raise ValueError(f"Dataset lacks features the model needs: {missing}")
    last = g[g["week"] == g["week"].max()].copy()
    p = bundle["model"].predict_proba(last[bundle["features"]])[:, 1]
    if bundle.get("calibrator") is not None:
        p = bundle["calibrator"].predict(p)
    last["risk_prob"] = p
    last["risk_level"] = pd.cut(p, RISK_BINS, labels=RISK_LABELS, right=False)
    last["lat"], last["lon"] = (last["cy"] + 0.5) * cell, (last["cx"] + 0.5) * cell
    return last[["cy", "cx", "lat", "lon", "week", "count", "risk_prob", "risk_level"]].reset_index(drop=True)


def score_latest(res, g, cell=CELL):
    b = res["best"]
    return score_with_bundle({"model": res["models"][b], "calibrator": res["calibrators"][b],
                              "features": res["feats"]}, g, cell)


def load_latest(models_dir=MODELS):
    files = sorted(models_dir.glob("fs-*.joblib"))
    if not files:
        return None, None
    meta_path = files[-1].with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not meta.get("sha256") or meta["sha256"] != sha256_file(files[-1]):
        raise ValueError(f"Model file {files[-1].name} failed its integrity check; refusing to load it.")
    return joblib.load(files[-1]), meta


def explain_cell(res, g, cy, cx):
    """Local explanation by occlusion: how much does the risk change if one feature is set to its
    typical (training-median) value? Positive = that feature pushes risk UP for this cell.
    A transparent approximation, not SHAP, and not causal."""
    row = g[(g["cy"] == cy) & (g["cx"] == cx)].sort_values("week").tail(1)
    if row.empty:
        return None
    feats, b = res["feats"], res["best"]
    m, cal = res["models"][b], res["calibrators"][b]
    pr = lambda x: (lambda p: float(cal.predict(p)[0]) if cal is not None else float(p[0]))(m.predict_proba(x[feats])[:, 1])
    p0, out = pr(row), []
    for f in feats:
        alt = row.copy()
        alt[f] = res["ref"][f]
        out.append({"feature": f, "value": float(row[f].iloc[0]), "typical": float(res["ref"][f]), "effect_on_risk": p0 - pr(alt)})
    d = pd.DataFrame(out)
    d.attrs["risk"] = p0
    return d.reindex(d["effect_on_risk"].abs().sort_values(ascending=False).index).reset_index(drop=True).round(4)


def evaluate_intensity(g, feats, seed=42):
    """Second target: how much fire activity (log1p total FRP) will a cell have next week?
    Same chronological test period as the classifier; compared with persistence."""
    g = g[g["label_frp"].notna()].copy()
    weeks = np.sort(g["week"].unique())
    if len(weeks) < 10:
        return None
    t2 = weeks[int(len(weeks) * 0.8)]
    tr, te = g[g["week"] < t2], g[g["week"] >= t2]
    if len(tr) < 50 or len(te) < 10:
        return None
    y = te["label_frp"]
    rows = []

    def add(name, kind, pred):
        act = y > 0
        rho = spearmanr(y, pred)[0] if np.std(pred) > 0 and y.nunique() > 1 else float("nan")
        rows.append({"Model": name, "Type": kind, "MAE": mean_absolute_error(y, pred),
                     "MAE (active cells)": mean_absolute_error(y[act], np.asarray(pred)[act.to_numpy()]) if act.any() else float("nan"),
                     "R2": r2_score(y, pred), "Spearman": rho})

    add("Baseline: persistence", "baseline", np.log1p(te["frp_sum"]).to_numpy())
    for n, m in {"Random Forest": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                                RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=seed)),
                 "Gradient Boosting": HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=200, random_state=seed)}.items():
        m.fit(tr[feats], tr["label_frp"])
        add(n, "ml", m.predict(te[feats]))
    return pd.DataFrame(rows).round(3)


def save_model(res, models_dir=MODELS, extra=None):
    """Persist model + calibrator + metadata (model card)."""
    version = datetime.now(timezone.utc).strftime("fs-%Y%m%d-%H%M%S")
    b = res["best"]
    mpath = models_dir / f"{version}.joblib"
    joblib.dump({"model": res["models"][b], "calibrator": res["calibrators"][b], "threshold": res["threshold"],
                 "features": res["feats"], "version": version, "train_stats": res["train_stats"],
                 "train_ref": res["train_ref"]}, mpath)
    meta = {"version": version, "sha256": sha256_file(mpath), "trained_at_utc": datetime.now(timezone.utc).isoformat(), "model": b,
            "features": res["feats"], "split": res["info"], "beats_baseline": res["beats_baseline"],
            "beats_baseline_ci": res["beats_baseline_ci"], "calibration_supported": res["calibration_supported"],
            "ece": res["ece"], "ci": res["ci"].to_dict("records"), "calib_table": res["calib_table"].to_dict("records"),
            "test_metrics": res["metrics"][res["metrics"]["Model"] == b].iloc[0].to_dict(),
            "limitations": LIMITATIONS, **(extra or {})}
    p = models_dir / f"{version}.json"
    p.write_text(json.dumps(meta, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return version, meta
