## طريقة التشغيل

### 1. تحميل المشروع
git clone https://github.com/a63266531-sudo/FireSentinel-AI.git
cd FireSentinel-AI

### 2. إنشاء بيئة افتراضية
python -m venv venv
source venv/bin/activate
(على ويندوز: venv\Scripts\activate)

### 3. تثبيت المكتبات
pip install -r requirements.txt

### 4. تشغيل الاختبارات
pytest


# AI Fire Risk Analyzer - ELITE (FireSentinel AI)

A research-grade wildfire **observation + uncertainty-aware next-week risk** workspace built on NASA FIRMS detections, weather, NDVI,
land cover and neighbouring activity. It is a strong foundation for investigation, **not** an operational system:
it does not prevent fires, has **no demonstrated real-world skill yet**, and must not replace official warnings.

## 1. What is implemented (and how sure we are)
Executed here (Python 3.12.3, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0): **55 tests passed, 0 failed, 4 skipped** (API runtime tests need fastapi/httpx).
Everything ran offline against **synthetic** data or mocks. See `docs/AUDIT.md` for the per-capability status (V/U/P/M/B) and the defects fixed.

| Area | What you get |
|---|---|
| Scientific tasks kept separate | A observation, B next-week occurrence risk, C spread = **extension point only**, D walk-forward hindcast |
| Providers | registry with product IDs, docs, access, resolution, units, CRS, time convention, licence, quality flags, limits (`docs/PROVIDERS.md`), schema validation, sha256 provenance log |
| Geospatial | explicit WGS84, bbox validation, projected-coordinate detection, true spherical cell areas, haversine index, coverage summaries, chunked CSV reader with row cap |
| Leakage-safe ML | chronological split, cells chosen from pre-cutoff history, imputation fitted on training rows only, no back-fill, baselines (persistence, climatology), LR / RF / gradient boosting, optional tuning |
| Uncertainty | block-bootstrap CIs, calibration (isotonic, ECE, Wilson table), **risk contract**: score vs probability, status ok / unreliable / unavailable, completeness, freshness, out-of-distribution flags |
| Validation | spatial CV, spatio-temporal hold-out, walk-forward hindcast, error analysis by month/region, PSI drift |
| Explainability | permutation importance, local occlusion explanation, **Scenario Lab** (sensitivity analysis with extrapolation warnings) |
| Experiments | one-command reproducible record + report + `verify-experiment` |
| Ops | CLI, scheduler, alerts (observation / data / prediction), email + Telegram (dry-run default), read-only API with key/loopback guard, hardened Docker, CI matrix |
| Dashboard | 9 views (Overview, Observations, Risk Explorer, Historical Trends, Model Diagnostics, Scenario Lab, Data & Provenance, Experiment Reports, System Health), colour-blind-safe palette with text legends, English/Arabic (Arabic covers main labels; rest falls back to English) |

**Not verified here:** real Streamlit/Plotly rendering (app executed against a stub), live FIRMS / Open-Meteo / NASA ORNL / Overpass calls, FastAPI runtime, Docker build, CI, Python 3.10/3.11, pandas 2.x.
**Blocked:** training/evaluating on real data (needs your FIRMS history). **Not implemented by design:** fire-spread simulation.

## 2. Quick start - Kali Linux + VS Code (details: `docs/VSCODE_KALI.md`)
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip unzip
sudo apt install -y code-oss                      # or the Microsoft .deb from code.visualstudio.com
unzip AI_Fire_Risk_Analyzer_ELITE.zip && cd AI_Fire_Risk_Analyzer_ELITE
code .
bash scripts/setup.sh        # venv + dependencies (incl. API + test tools) + self check
bash scripts/run.sh          # http://localhost:8501  -> "One-click demo (offline)"
bash scripts/test.sh         # pytest + security scan
```
Real data: `cp .env.example .env`, set `FIRMS_MAP_KEY`, run `.venv/bin/python scripts/check_setup.py --network` (verifies FIRMS, Open-Meteo, NASA ORNL, Overpass), then use *NASA FIRMS API* or
`python -m firesentinel.cli fetch --region Egypt`, repeat daily (or `schedule`), and when you have >= ~12 weeks run an experiment.

## 3. Commands
```bash
python -m firesentinel.cli demo                      # offline synthetic run
python -m firesentinel.cli experiment --demo --region Greece --weeks 26 --end 2026-10-18   # SYNTHETIC reproducible experiment
python -m firesentinel.cli verify-experiment experiments/exp-...                           # reproduction check
python -m firesentinel.cli fetch|train|schedule|providers ...
uvicorn firesentinel.api:app --host 127.0.0.1 --port 8000                                  # API (docs/API.md)
```

## 4. Demo path vs real-data path
* **Demo:** synthetic, labelled everywhere, generated from a known rule; proves the software runs, **never** predictive skill.
* **Real:** FIRMS history in `data/firesentinel.db` + Open-Meteo + NASA MODIS (via ORNL) -> *Model Diagnostics* or `experiment`. Interpret results only with the CIs, baselines, hold-outs and limitations shown.

## 5. Layout
```
app.py  firesentinel/{providers,data,store,geodesy,geo,weather,vegetation,features,model,contract,monitoring,scenario,spread,
        experiments,alerts,notify,impact,report,service,api,scheduler,pipeline,synthetic,security,i18n,cli,config}.py
tests/ (55 executed tests)  scripts/ (setup, run, test, check_setup, security_check)  docs/  .vscode/  Dockerfile docker-compose.yml constraints.txt
```
Docs: `AUDIT`, `METHODOLOGY` (hypotheses, label, leakage, validation), `ARCHITECTURE`, `PROVIDERS`, `EXPERIMENTS`, `API`, `SECURITY`, `REFERENCES`, `DEMO_SCRIPT`, `PITCH`, `VSCODE_KALI`.

## 6. Known defects and limitations
* Weather/NDVI/land cover are point samples at cell centres; NDVI quality flags are unused; labels are detections, not ground truth.
* The demo's latest week lies outside the summer training range, so its estimates are (correctly) flagged unreliable; with few test weeks the CIs are wide.
* Selecting cells from pre-cutoff history means the model covers only cells already active before the cutoff.
* PDF report is English only; Arabic UI is partial.
* Code is tested only on Python 3.12.3 / pandas 3.0.2; CI for 3.10-3.12 is written but unrun.

## 7. Roadmap
Real multi-season training and publication of results; NDVI pixel-quality filtering; ERA5-Land / station weather; fuel and terrain layers; a validated spread model (e.g. Rothermel-based) with perimeter validation; probabilistic spatial models if data justify them; full Arabic localisation; Docker/CI verification.
License: MIT.
