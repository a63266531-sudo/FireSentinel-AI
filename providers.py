"""Data-provider interface with explicit metadata, schema validation and provenance records.
Endpoints/products are implemented from public documentation and were NOT verified live from the build environment:
`verified_live` stays False until you run `python scripts/check_setup.py --network`."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib

import numpy as np
import pandas as pd

from . import data, vegetation, weather, synthetic
from .security import check_upload_size


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str
    product: str
    official_docs: str
    access: str
    auth: str
    spatial_resolution: str
    temporal_resolution: str
    units: str
    crs: str
    time_convention: str
    license: str
    quality_flags: str
    update_frequency: str
    known_limitations: str
    missing_data_behavior: str
    verified_live: bool = False


SCHEMAS = {  # required column -> kind ('f' float, 'M' datetime, 'O' object)
    "fire": {"latitude": "f", "longitude": "f", "acq_datetime": "M", "frp": "f", "source": "O", "conf_level": "O"},
    "weather": {"date": "M", "temp_max": "f", "precip": "f", "wind_max": "f", "et0": "f", "elevation": "f"},
    "ndvi": {"date": "M", "ndvi": "f"},
}
RANGES = {"latitude": (-90, 90), "longitude": (-180, 180), "ndvi": (-0.2, 1.0), "precip": (0, 2000), "rh_mean": (0, 100),
          "frp": (0, 1e5), "temp_max": (-90, 60), "wind_max": (0, 400)}


def validate_schema(df: pd.DataFrame, schema: str) -> list:
    """Returns a list of problems (empty = valid). Never silently coerces."""
    probs, req = [], SCHEMAS[schema]
    for col, kind in req.items():
        if col not in df.columns:
            probs.append(f"missing column {col}")
        elif kind == "f" and not pd.api.types.is_numeric_dtype(df[col]):
            probs.append(f"{col} must be numeric")
        elif kind == "M" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            probs.append(f"{col} must be datetime")
    for col, (lo, hi) in RANGES.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                probs.append(f"{col}: {bad} value(s) outside [{lo}, {hi}]")
    return probs


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy().tobytes()).hexdigest()


def make_provenance(provider: "Provider", request: dict, df: pd.DataFrame, problems=(), note="") -> dict:
    return {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "provider": provider.info.id, "product": provider.info.product,
            "request": {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in request.items() if k != "key"},
            "rows": int(len(df)), "sha256": fingerprint(df), "schema_ok": not problems, "problems": list(problems), "note": note}


class Provider(ABC):
    info: ProviderInfo
    schema: str

    @abstractmethod
    def _fetch(self, **kw) -> pd.DataFrame: ...

    def fetch(self, **kw):
        """Returns (validated_df, provenance). Raises ValueError when the schema is violated."""
        df = self._fetch(**kw)
        problems = validate_schema(df, self.schema) if len(df) else []
        prov = make_provenance(self, kw, df, problems)
        if problems:
            raise ValueError(f"{self.info.id}: schema validation failed: {problems}")
        return df, prov


class FirmsProvider(Provider):
    schema = "fire"
    info = ProviderInfo(
        "nasa_firms", "NASA FIRMS active fire (area API)", "VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT, MODIS_NRT, VIIRS_SNPP_SP, MODIS_SP",
        "https://firms.modaps.eosdis.nasa.gov/api/area/", "HTTPS CSV, area endpoint, max 5 days per request", "Free MAP_KEY (env FIRMS_MAP_KEY)",
        "VIIRS ~375 m, MODIS ~1 km (pixel footprints)", "Near-real-time, several overpasses per day; SP = archived standard processing",
        "FRP in MW; brightness in K", "WGS84 lat/lon (EPSG:4326)", "acq_date + acq_time in UTC", "NASA open data (EOSDIS data-use policy); acknowledge NASA FIRMS",
        "VIIRS confidence l/n/h; MODIS confidence 0-100; day/night flag", "NRT: typically hours after overpass; see FIRMS docs",
        "Thermal anomalies are not confirmed wildfires (industrial heat, gas flares, agricultural burns); cloud and overpass gaps hide fires; transaction limits apply",
        "Empty response -> no rows (not an error); HTTP/network failure -> retries then error listed per 5-day chunk, partial results kept; invalid key -> stops immediately")

    def _fetch(self, key, source, bbox, start, end, **_):
        raw, errs = data.fetch_range(key, source, bbox, start, end)
        if errs and raw.empty:
            raise RuntimeError("; ".join(errs))
        return data.validate(raw, source)[0] if len(raw) else pd.DataFrame()


class OpenMeteoProvider(Provider):
    schema = "weather"
    info = ProviderInfo(
        "open_meteo", "Open-Meteo historical weather + forecast", "archive-api.open-meteo.com/v1/archive and api.open-meteo.com/v1/forecast (daily)",
        "https://open-meteo.com/en/docs/historical-weather-api", "HTTPS JSON, per point", "None (free for non-commercial use; see terms)",
        "Reanalysis/model grid, roughly 0.1-0.25 degrees depending on the underlying model", "Daily aggregates (from hourly)",
        "deg C, mm, km/h, %, MJ/m2 (et0 mm), elevation m", "WGS84 lat/lon", "UTC (timezone=UTC requested)", "Data CC BY 4.0 - attribution required; API terms apply",
        "None exposed by this app", "Archive lags a few days; forecast up to 16 days", "Point value at the grid-cell centre, not local station data; relative humidity may be unavailable for some ranges",
        "Request failure -> exception; humidity retried without; missing humidity left NaN and imputed on training rows only")

    def _fetch(self, lat, lon, start, end, mode="open-meteo", **_):
        return weather.get_weather(lat, lon, start, end, mode)


class NdviProvider(Provider):
    schema = "ndvi"
    info = ProviderInfo(
        "modis_ndvi", "MODIS Terra NDVI via ORNL DAAC web service", "MOD13Q1 v061, band 250m_16_days_NDVI",
        "https://modis.ornl.gov/data/modis_webservice.html", "HTTPS JSON subset, per point", "None",
        "250 m (pixel at the point)", "16-day composites", "NDVI (scale factor 0.0001, fill -3000)", "MODIS sinusoidal grid, point queried by lat/lon",
        "Composite start date; this app delays availability by 16 days to avoid look-ahead", "NASA LP DAAC / ORNL DAAC data policies; cite DOI 10.5067/MODIS/MOD13Q1.061",
        "Pixel reliability / SummaryQA bands are NOT used yet", "Composites become available days after the window",
        "Single pixel; cloud-contaminated composites possible because quality bands are not filtered; point may be water/urban",
        "Failure -> cell skipped, NDVI features NaN (imputed on training rows only) or the feature group is disabled")

    def _fetch(self, lat, lon, start, end, mode="ornl", **_):
        return vegetation.get_ndvi(lat, lon, start, end, "synthetic" if mode == "synthetic" else mode)


class SyntheticProvider(Provider):
    schema = "fire"
    info = ProviderInfo(
        "synthetic_demo", "Synthetic demonstration generator", "firesentinel.synthetic (rule-based)", "docs/METHODOLOGY.md", "local, offline", "None",
        "0.5 degree cells", "daily", "FIRMS-like columns", "WGS84", "UTC", "MIT (project)", "n/a", "on demand",
        "NOT real observations. Fire rate is generated from a known rule, so any skill measured on it proves only that the software runs.",
        "n/a")

    def _fetch(self, region, days, seed, end=None, **_):
        from . import pipeline
        raw, _ = pipeline.make_demo(region, days, seed, end)
        return data.validate(raw, "SYNTHETIC")[0]


class LocalCsvProvider(Provider):
    schema = "fire"
    info = ProviderInfo(
        "local_csv", "User-supplied FIRMS CSV (archive download)", "FIRMS Download tool CSV", "https://firms.modaps.eosdis.nasa.gov/download/",
        "Local file, 50 MB UI limit; chunked reader for larger files", "None", "as downloaded", "as downloaded", "as FIRMS", "WGS84", "UTC",
        "Per FIRMS terms", "as FIRMS", "static", "Your file is trusted only after validation; truncated at max_rows for memory safety",
        "Invalid rows dropped and counted in the quality report")

    def _fetch(self, path, bbox=None, max_rows=2_000_000, **_):
        import os
        check_upload_size(os.path.getsize(path), 5000)
        return data.validate(data.load_csv_chunked(path, bbox, max_rows), "local_csv")[0]


REGISTRY = {p.info.id: p for p in (FirmsProvider(), OpenMeteoProvider(), NdviProvider(), SyntheticProvider(), LocalCsvProvider())}
EXTRA_SOURCES = [
    ProviderInfo("modis_landcover", "MODIS land cover via ORNL DAAC", "MCD12Q1 v061, band LC_Type1 (IGBP)", "https://modis.ornl.gov/data/modis_webservice.html",
                 "HTTPS JSON subset", "None", "500 m", "Annual", "class codes 1-17 (grouped to forest/shrub/grass/crop/other)", "MODIS sinusoidal",
                 "Year", "NASA LP DAAC data policies; DOI 10.5067/MODIS/MCD12Q1.061", "Quality layers not used", "Annual, ~1 year latency",
                 "Single pixel; classes are generalised", "Failure -> land-cover features omitted"),
    ProviderInfo("osm_overpass", "OpenStreetMap Overpass (exposure assets)", "place / boundary=protected_area / highway=motorway|trunk|primary",
                 "https://wiki.openstreetmap.org/wiki/Overpass_API", "HTTPS POST", "None", "vector", "continuously edited", "names, tags", "WGS84", "n/a",
                 "ODbL - attribute OpenStreetMap contributors", "Completeness varies by country", "live", "Crowd-sourced, incomplete in places; public servers are rate-limited",
                 "Failure -> exposure estimate unavailable; CSV/demo assets still work"),
]


def describe_all() -> pd.DataFrame:
    return pd.DataFrame([asdict(p.info) for p in REGISTRY.values()] + [asdict(i) for i in EXTRA_SOURCES])


def providers_markdown() -> str:
    L = ["# Data providers (generated from `firesentinel/providers.py`)\n",
         "`verified_live=False` everywhere: the build environment had no internet. Run `python scripts/check_setup.py --network` to verify.\n"]
    for r in describe_all().to_dict("records"):
        L.append(f"## {r['name']} (`{r['id']}`)\n")
        for k, v in r.items():
            if k not in ("id", "name"):
                L.append(f"- **{k.replace('_', ' ')}**: {v}")
        L.append("")
    return "\n".join(L)
