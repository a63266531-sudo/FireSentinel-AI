<div align="center">

# 🔥 FireSentinel AI

### Elite Wildfire Risk Forecasting Platform

**From NASA satellite fire data to a verifiable, reproducible weekly risk forecast, built with production-grade engineering.**



![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-3776AB?logo=python&logoColor=white)




![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)




![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)




![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)




![Tests](https://img.shields.io/badge/tests-55-brightgreen)




![License](https://img.shields.io/badge/License-MIT-yellow)



</div>

---

## 📖 Overview

FireSentinel AI is an end-to-end platform that monitors wildfire activity and forecasts **next-week fire risk** by fusing multiple data sources:

| Source | Purpose |
|---|---|
| 🛰️ **NASA FIRMS** | Active fire detections |
| 🌦️ **Open-Meteo** | Weather conditions |
| 🌿 **NASA MODIS (via ORNL)** | NDVI vegetation index |
| 🗺️ **Land cover** | Terrain and fuel context |
| 🏘️ **Overpass (OSM)** | Nearby human activity |

Every result is designed to be **auditable, reproducible, and honest about its limits**.

---

## ✨ Key Features

### 🧩 Large, Modular Architecture
- **27+ modules** covering providers, features, modeling, monitoring, alerts, reports, API, and scheduling
- Clean separation between data, model, service, and presentation layers

### 📊 9-View Interactive Dashboard
Overview · Notes · Risk Explorer · Historical Trends · Model Diagnostics · Scenario Lab · Data & Provenance · Experiment Reports · System Health

- Color-blind-safe palette with text symbols
- English / Arabic interface

### 🔬 Reproducible Experiments
- Every experiment is recorded and can be re-verified with a single command
- Confidence intervals, baselines, and held-out test sets are reported

### 🔔 Alerts & Notifications
- Monitoring, data, and forecast alerts
- Email + Telegram delivery

### 🔐 Secure Read-Only API
- FastAPI service with API-key protection and rate limiting
- Interactive docs (see `docs/API.md`)

### 🐳 Deployment Ready
- Optimized Dockerfile + `docker-compose.yml`
- CI matrix for Python 3.10 – 3.12
- Pinned `constraints.txt`

### ✅ Quality & Security
- **55 automated tests** (pytest)
- Built-in security scan (`scripts/security_check.py`)

---

## 🏗️ Architecture

```text
Data Providers ──► Ingestion & Store ──► Feature Engineering ──► Model
 (FIRMS, Weather,      (SQLite)           (geo, weather,          │
  NDVI, OSM)                               vegetation)            ▼
                                                          Risk Forecast
                                                                  │
        ┌──────────────┬──────────────┬─────────────┬─────────────┘
        ▼              ▼              ▼             ▼
   Dashboard         REST API       Alerts       PDF Reports
  (Streamlit)       (FastAPI)   (Email/Telegram)
```

### Project Layout

```text
app.py                      # Streamlit dashboard entry point
firesentinel/
├── providers, data, store  # Data access & storage
├── geodesy, geo            # Spatial utilities
├── weather, vegetation     # Environmental features
├── features, model         # ML pipeline
├── contract, monitoring    # Data contracts & health checks
├── scenario, spread        # Scenario analysis
├── experiments             # Reproducible experiments
├── alerts, notify          # Alerting
├── impact, report          # Impact estimates & PDF reports
├── service, api            # Service layer & REST API
├── scheduler, pipeline     # Automation
├── synthetic               # Offline demo data
├── security, i18n, cli     # Security, translations, CLI
└── config
tests/                      # 55 executed tests
scripts/                    # setup, run, test, check_setup, security_check
docs/                       # Full documentation
Dockerfile · docker-compose.yml · constraints.txt
```

---

## 🚀 Quick Start

### Requirements
Python 3.10+ (tested on 3.12), `pip`, `venv`

### Install

```bash
git clone https://github.com/a63266531-sudo/FireSentinel-AI.git
cd FireSentinel-AI

bash scripts/setup.sh     # venv + dependencies + self-check
bash scripts/run.sh       # dashboard at http://localhost:8501
bash scripts/test.sh      # pytest + security scan
```

Then open **http://localhost:8501** and choose **"One-click demo (offline)"**.

### Manual install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-api.txt   # API
pip install -r requirements-dev.txt   # development & tests
```

### Docker

```bash
docker compose up --build
```

---

## 💻 Command Line

```bash
# Offline synthetic run
python -m firesentinel.cli demo

# Reproducible SYNTHETIC experiment
python -m firesentinel.cli experiment --demo --region Greece --weeks 26 --end 2026-10-18

# Verify an experiment reproduces
python -m firesentinel.cli verify-experiment experiments/exp-...

# Other commands
python -m firesentinel.cli fetch|train|schedule|providers ...
```

### Run the API

```bash
uvicorn firesentinel.api:app --host 127.0.0.1 --port 8000
```

---

## 🌍 Using Real Data

```bash
cp .env.example .env                       # set FIRMS_MAP_KEY
.venv/bin/python scripts/check_setup.py --network
python -m firesentinel.cli fetch --region Egypt
```

Repeat daily (or use `schedule`). Once you have **≥ 12 weeks** of history, run an experiment.

> Get a free FIRMS key from the NASA FIRMS website.

---

## 🧪 Demo Path vs Real-Data Path

| | Demo Path | Real-Data Path |
|---|---|---|
| **Data** | Synthetic, labeled everywhere | NASA FIRMS + Open-Meteo + MODIS |
| **Purpose** | Proves the software works | Model diagnostics & experiments |
| **Predictive skill** | None (by design) | Interpret only with confidence intervals, baselines, held-out sets |

---

## 🔬 Scientific Transparency

This project deliberately separates **what is verified** from **what is not**:

- ✅ **Verified:** test suite runs on Python 3.12.3 / pandas 3.0.2
- ⚠️ **Not yet verified here:** live provider calls, FastAPI runtime, Docker build, CI runs, Python 3.10/3.11, pandas 2.x
- ⏸️ **Disabled:** training on real data (requires your own FIRMS history)

### Known Limitations
- Weather, NDVI, and land cover are point samples at cell centers
- Labels are detections, not ground truth
- The model only covers cells already active before the cutoff
- PDF reports are English only; the Arabic UI is partial

---

## 📚 Documentation

`AUDIT` · `METHODOLOGY` · `ARCHITECTURE` · `PROVIDERS` · `EXPERIMENTS` · `API` · `SECURITY` · `REFERENCES` · `DEMO_SCRIPT` · `PITCH` · `VSCODE_KALI`

All in the `docs/` folder.

---

## 🗺️ Roadmap

- [ ] Real multi-season training and published results
- [ ] NDVI pixel-quality filtering
- [ ] ERA5-Land / station weather data
- [ ] Fuel and terrain layers
- [ ] Validated spread model (e.g. Rothermel) with perimeter validation
- [ ] Probabilistic spatial models
- [ ] Full Arabic translation
- [ ] Docker / CI verification

---

## ⚠️ Disclaimer

FireSentinel AI is a research and engineering platform. It is **not** an operational warning system and must not replace official fire-danger warnings.

---

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">

**Built with 🔥 for safer forests**

</div>
