"""Reports: Markdown, standalone HTML (print-ready) and PDF with charts."""
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd

from .config import LIMITATIONS


def _md(df):
    if df is None or len(df) == 0:
        return "_n/a_\n"
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def build_report(origin, rep, df, res=None, geo_cv=None, imp=None, wx_mode=None, alerts=None,
                 intensity=None, tuning=None, exposure=None):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# FireSentinel AI - Analysis Report\n\nGenerated: {now}\n"]
    if origin == "demo":
        L.append("> **WARNING: SYNTHETIC DEMO DATA. Nothing below is a real-world finding.**\n")
    L += ["## 1. Data\n", f"- Origin: {origin}", f"- Sources: {', '.join(sorted(map(str, df['source'].unique())))}",
          f"- Observation window: {rep.get('first_observation')} to {rep.get('last_observation')}",
          f"- Clean detections: {rep.get('rows_clean')}\n", "## 2. Data quality\n",
          _md(pd.DataFrame([rep]).T.reset_index().rename(columns={"index": "check", 0: "value"}))]
    if alerts is not None and len(alerts):
        L += ["## 3. Alerts (latest run)\n", _md(alerts[["severity", "category", "type", "explanation"]].head(15))]
    if exposure is not None and len(exposure):
        L += ["## 4. Exposure near clusters\n", _md(exposure)]
    if res:
        i = res["info"]
        L += ["## 5. Risk model\n",
              "Target: any detection in a grid cell during the next calendar week. "
              f"Weather source: {wx_mode}. Features ({len(res['feats'])}): {', '.join(res['feats'])}.\n",
              f"- Train {i['train_weeks']}, validation {i['val_weeks']}, test {i['test_weeks']}",
              f"- Rows: {i['n_train']} / {i['n_val']} / {i['n_test']}; test positive rate {i['test_positive_rate']:.1%}",
              f"- Model selection by: {i['selection']}",
              f"- Selected: **{res['best']}**; beats baselines on held-out test: **{res['beats_baseline']}**\n",
              "### Test-set metrics\n", _md(res["metrics"])]
        if tuning is not None:
            L += ["### Hyper-parameter search (forward-chaining CV inside training period)\n", _md(tuning)]
        if geo_cv is not None:
            L += ["### Geographic (unseen-region) validation\n", _md(geo_cv)]
        if imp is not None:
            L += ["### Permutation importance (not causal)\n", _md(imp.head(10))]
        if intensity is not None:
            L += ["### Fire-intensity forecast (log1p total FRP next week)\n", _md(intensity)]
    L += ["## 6. Limitations\n"] + [f"- {x}" for x in LIMITATIONS]
    L.append("\n## 7. Reproducibility\n\nRaw downloads are stored unchanged under data/raw with retrieval metadata; "
             "model versions and model cards under models/. No performance figure is valid beyond the stated test period.")
    return "\n".join(L)


def to_html(md: str) -> str:
    try:
        import markdown
        body = markdown.markdown(md, extensions=["tables"])
    except Exception:
        body = f"<pre>{escape(md)}</pre>"
    css = ("body{font-family:system-ui,Arial;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
           "table{border-collapse:collapse;font-size:12px;margin:1rem 0}td,th{border:1px solid #ccc;padding:3px 6px}"
           "th{background:#f3f3f3}blockquote{background:#fff3cd;margin:0;padding:.5rem 1rem}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>FireSentinel AI report</title><style>{css}</style></head><body>{body}</body></html>"


def build_pdf(origin, rep, df, res=None, geo_cv=None, imp=None, wx_mode=None, alerts=None, risk=None,
              intensity=None, exposure=None) -> bytes:
    """PDF with charts (English text only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from .geo import daily_counts

    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=7, leading=8.5)
    W = A4[0] - 3 * cm
    story = []
    P = lambda t, s="BodyText": story.append(Paragraph(escape(str(t)), ss[s]))

    def table(d, widths=None, max_rows=15):
        if d is None or len(d) == 0:
            return
        d = d.head(max_rows).copy()
        rows = [[Paragraph(f"<b>{escape(str(c))}</b>", small) for c in d.columns]]
        for r in d.itertuples(index=False):
            rows.append([Paragraph(escape(f"{v:.3f}" if isinstance(v, float) else str(v)), small) for v in r])
        t = Table(rows, colWidths=widths or [W / len(d.columns)] * len(d.columns), repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.extend([t, Spacer(1, 8)])

    def chart(make, h=6.5):
        try:
            fig, ax = plt.subplots(figsize=(8, h / 2.54 * 8 / 16 * 2.0 / 2.0 * 2))
            make(ax)
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=130)
            plt.close(fig)
            buf.seek(0)
            ratio = fig.get_figheight() / fig.get_figwidth()
            story.extend([Image(buf, width=W, height=W * ratio), Spacer(1, 6)])
        except Exception as e:  # a failing chart must never break the report
            P(f"(chart unavailable: {e})")

    P("FireSentinel AI - Analysis Report", "Title")
    P(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    if origin == "demo":
        P("WARNING: SYNTHETIC DEMO DATA - nothing below is a real-world finding.", "Heading3")
    P("1. Data", "Heading2")
    P(f"Origin: {origin}. Sources: {', '.join(sorted(map(str, df['source'].unique())))}. "
      f"Window: {rep.get('first_observation')} to {rep.get('last_observation')}. Clean detections: {rep.get('rows_clean')}.")
    dc = daily_counts(df)
    chart(lambda ax: (ax.bar(dc.index, dc.values, width=0.9, color="#d9541e"), ax.set_title("Detections per day")))
    P("2. Data quality", "Heading2")
    table(pd.DataFrame([rep]).T.reset_index().rename(columns={"index": "check", 0: "value"}), [W * 0.4, W * 0.6], 20)
    if alerts is not None and len(alerts):
        P("3. Alerts", "Heading2")
        table(alerts[["severity", "category", "type", "explanation"]], [W * 0.1, W * 0.12, W * 0.2, W * 0.58])
    if exposure is not None and len(exposure):
        P("4. Exposure near clusters", "Heading2")
        table(exposure)
    if res:
        i = res["info"]
        P("5. Risk model", "Heading2")
        P(f"Target: any detection in a grid cell next week. Weather: {wx_mode}. {len(res['feats'])} features. "
          f"Train {i['train_weeks']}, validation {i['val_weeks']}, test {i['test_weeks']}. Test positive rate {i['test_positive_rate']:.1%}. "
          f"Selected {res['best']}; beats baselines on the held-out test: {res['beats_baseline']}.")
        table(res["metrics"])
        m = res["metrics"]
        chart(lambda ax: (ax.barh(m["Model"], m["PR-AUC"], color=["#888" if t == "baseline" else "#1f77b4" for t in m["Type"]]),
                          ax.axvline(i["test_positive_rate"], ls="--", color="k", lw=0.8), ax.set_title("Test PR-AUC (dashed = random guess)")), 5)
        if imp is not None:
            top = imp.head(10).iloc[::-1]
            chart(lambda ax: (ax.barh(top["feature"], top["importance"]), ax.set_title("Permutation importance (not causal)")), 5.5)
        if geo_cv is not None:
            P("Geographic validation (unseen regions)", "Heading3")
            table(geo_cv)
        if intensity is not None:
            P("Fire-intensity forecast", "Heading3")
            table(intensity)
        if risk is not None and len(risk):
            def rmap(ax):
                ax.scatter(df["longitude"], df["latitude"], s=1, c="#bbb")
                col = "model_score" if "model_score" in risk else "risk_prob"
                sc = ax.scatter(risk["lon"], risk["lat"], c=risk[col], cmap="viridis", s=40, vmin=0, vmax=1, edgecolor="k", linewidth=0.3)
                plt.colorbar(sc, ax=ax, label="model score (not necessarily a probability)")
                ax.set_title("Model score, next week")
            chart(rmap, 7)
    P("6. Limitations", "Heading2")
    for x in LIMITATIONS:
        P("- " + x)
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      title="FireSentinel AI report").build(story)
    return buf.getvalue()


def build_experiment_md(e: dict) -> str:
    """Experiment report (Markdown) from an experiment dict."""
    df = lambda recs: _md(pd.DataFrame(recs)) if recs else "_n/a_\n"
    L = [f"# Experiment report: {e['name']} (`{e['id']}`)\n", f"Created: {e['created_at_utc']}\n"]
    if e["origin"] == "demo":
        L.append("> **SYNTHETIC DEMONSTRATION DATA. These numbers are not evidence of real-world predictive skill.**\n")
    d = e["data"]
    L += ["## Data\n", f"- Fingerprint (sha256): `{d['fingerprint_sha256']}`", f"- Rows: {d['rows']}; window {d['first_observation']} to {d['last_observation']}",
          f"- Sources: {', '.join(d['sources'])}", f"- Context data source: {d.get('provenance_weather')}", f"- Unit: {e['unit']}",
          f"- Cell selection: {e['cell_selection']['rule']} (cutoff {e['cell_selection']['cutoff']}, {e['cell_selection']['n_cells']} cells, fallback: {e['cell_selection']['fallback_used']})\n",
          "## Label\n", e["label_definition"] + "\n", "## Split\n",
          f"- Train {e['split']['train_weeks']}, validation {e['split']['val_weeks']}, test {e['split']['test_weeks']}; rows {e['split']['n_train']}/{e['split']['n_val']}/{e['split']['n_test']}",
          f"- Test positive rate {e['split']['test_positive_rate']:.1%}", f"- Preprocessing: {e['preprocessing']}\n",
          "## Model\n", f"- Selected **{e['model']['selected']}** by {e['model']['selection_rule']} ({e['model']['estimator']}, tuned: {e['model']['tuned']})",
          f"- Seeds: {e['seeds']}", f"- Features ({len(e['features'])}): {', '.join(e['features'])}\n", "## Test metrics\n", df(e["metrics_test"]),
          "### 95% block-bootstrap intervals (resampling test weeks)\n", df(e["bootstrap_ci_test"]),
          f"Beats baselines: point estimate **{e['beats_baselines']['point_estimate']}**, 95% interval excludes 0 **{e['beats_baselines']['ci_excludes_zero']}**.\n",
          f"Calibration: ECE {e['calibration']['ECE']}, calibrated-probability display supported: **{e['calibration']['supported']}**\n", df(e["calibration"]["table"]),
          "## Stricter validation\n"]
    st = e.get("spatiotemporal_holdout")
    L += ["### Spatio-temporal hold-out (unseen blocks AND later period)\n", df(st["table"]) if st else "_not enough blocks/weeks_\n",
          "### Walk-forward hindcast (train only on labels known at the time)\n", df(e["walk_forward_hindcast"]["per_origin"]),
          f"Pooled: {e['walk_forward_hindcast']['pooled']}\n", "### Spatial block cross-validation\n", df(e["spatial_block_cv"]),
          "### Permutation importance (not causal)\n", df(e["permutation_importance"]), "### Input drift, last 3 weeks vs training (PSI; seasonal change is expected)\n",
          df(e["input_drift_last_3_weeks"]), "## Reproduce\n", f"`{e['reproduce']}`\n", f"Software: {e['software']}\n", "## Limitations\n"]
    L += [f"- {x}" for x in e["limitations"]]
    return "\n".join(L)
