"""Central configuration, paths and logging."""
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW, PROC, CACHE = DATA / "raw", DATA / "processed", DATA / "cache"
MODELS, LOGS, EXPERIMENTS = ROOT / "models", ROOT / "logs", ROOT / "experiments"
DB_PATH = DATA / "firesentinel.db"
for _p in (RAW, PROC, CACHE, MODELS, LOGS, EXPERIMENTS):
    _p.mkdir(parents=True, exist_ok=True)

CELL = 0.5  # grid cell size in degrees

REGIONS = {  # west, south, east, north
    "Egypt": (24.7, 21.9, 36.9, 31.7),
    "Mediterranean": (-10.0, 30.0, 40.0, 46.0),
    "California": (-124.5, 32.5, -114.0, 42.0),
    "Australia (East)": (140.0, -39.0, 154.0, -10.0),
    "Amazon (Brazil)": (-74.0, -18.0, -44.0, 5.0),
    "Central Africa": (10.0, -15.0, 40.0, 10.0),
    "Greece": (19.0, 34.5, 28.5, 42.0),
}

SOURCES = {
    "VIIRS_SNPP_NRT": "VIIRS S-NPP, near real-time",
    "VIIRS_NOAA20_NRT": "VIIRS NOAA-20, near real-time",
    "VIIRS_NOAA21_NRT": "VIIRS NOAA-21, near real-time",
    "MODIS_NRT": "MODIS, near real-time",
    "VIIRS_SNPP_SP": "VIIRS S-NPP, standard processing (archive)",
    "MODIS_SP": "MODIS, standard processing (archive)",
}

RISK_BINS = [0.0, 0.10, 0.30, 0.60, 1.0001]
RISK_LABELS = ["Low", "Moderate", "High", "Very high"]

LIMITATIONS = [
    "A satellite thermal anomaly is not necessarily a confirmed wildfire; absence of a detection does not prove absence of fire.",
    "Detections are affected by cloud cover, overpass timing and sensor resolution.",
    "Risk estimates are relative, apply to the grid cell and the next calendar week only, and are valid only for the data they were evaluated on.",
    "Feature importance shows model reliance, not causation.",
    "Weather values are point estimates at grid-cell centres and may differ from local conditions.",
    "This is a research and decision-support tool, not an official emergency warning service.",
]


class RedactingFilter(logging.Filter):
    """Removes known secret values (API keys, tokens, passwords) from every log record."""

    def filter(self, record):
        from .security import redact
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def get_logger(name: str = "firesentinel") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fh = logging.FileHandler(LOGS / "pipeline.log", encoding="utf-8")
        fh.addFilter(RedactingFilter())
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    return log


def load_env(path=None):
    """Load KEY=VALUE pairs from .env into os.environ (never overrides existing variables)."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


load_env()
