"""Explicit geographic-coordinate handling. Everything is WGS84 degrees (EPSG:4326); areas and distances are computed on
a sphere, never as 'degrees squared'. No silent CRS conversion: projected (metre) coordinates are detected and refused."""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import CELL

R_EARTH_KM = 6371.0088


def validate_bbox(bbox):
    w, s, e, n = (float(x) for x in bbox)
    if not (-180 <= w < e <= 180):
        raise ValueError("bbox needs -180 <= west < east <= 180 (boxes crossing the antimeridian are not supported; split them)")
    if not (-90 <= s < n <= 90):
        raise ValueError("bbox needs -90 <= south < north <= 90")
    return (w, s, e, n)


def check_geographic_degrees(df, lat="latitude", lon="longitude") -> dict:
    """Detects projected (metre) coordinates and swapped lat/lon before they silently corrupt distances."""
    la, lo = pd.to_numeric(df[lat], errors="coerce"), pd.to_numeric(df[lon], errors="coerce")
    out_lat, out_lon = int((la.abs() > 90).sum()), int((lo.abs() > 180).sum())
    return {"rows": int(len(df)), "lat_out_of_range": out_lat, "lon_out_of_range": out_lon,
            "looks_projected_metres": bool(len(df) and (la.abs().max() > 360 or lo.abs().max() > 360)),
            "possible_lat_lon_swap": bool(len(df) and ((la.abs() > 90) & (lo.abs() <= 90)).mean() > 0.5), "crs": "EPSG:4326"}


def km_per_degree(lat):
    """(km per degree latitude, km per degree longitude) on the WGS84 ellipsoid (standard series)."""
    r = np.radians(np.asarray(lat, float))
    return (111.132954 - 0.559822 * np.cos(2 * r) + 0.001175 * np.cos(4 * r),
            111.41513 * np.cos(r) - 0.09455 * np.cos(3 * r) + 0.00012 * np.cos(5 * r))


def cell_area_km2(cy, cell=CELL):
    """Spherical area of a lat/lon grid cell whose row index is cy: R^2 * dlon * (sin(lat_n) - sin(lat_s))."""
    cy = np.asarray(cy, float)
    s, n = np.radians(cy * cell), np.radians((cy + 1) * cell)
    return R_EARTH_KM ** 2 * np.radians(cell) * (np.sin(n) - np.sin(s))


def to_local_km(lat, lon, lat0, lon0):
    """Local equirectangular projection to km. Valid only for small extents (< ~500 km); use for display/short distances."""
    kl, kn = km_per_degree(lat0)
    return (np.asarray(lon, float) - lon0) * kn, (np.asarray(lat, float) - lat0) * kl


def coverage_summary(df, bbox, cell=CELL, start=None, end=None) -> dict:
    """Spatial, temporal and per-source coverage. Area is the sum of whole grid cells (bbox edges are not clipped)."""
    w, s, e, n = validate_bbox(bbox)
    cys = np.arange(int(np.floor(s / cell)), int(np.floor(n / cell)) + 1)
    n_cx = int(np.floor(e / cell)) - int(np.floor(w / cell)) + 1
    total_area = float(cell_area_km2(cys, cell).sum() * n_cx)
    cy = np.floor(df["latitude"] / cell).astype(int)
    cx = np.floor(df["longitude"] / cell).astype(int)
    cells = pd.DataFrame({"cy": cy, "cx": cx}).drop_duplicates()
    covered = float(cell_area_km2(cells["cy"].to_numpy(), cell).sum())
    d0 = pd.Timestamp(start) if start is not None else df["acq_date"].min()
    d1 = pd.Timestamp(end) if end is not None else df["acq_date"].max()
    n_days = (d1 - d0).days + 1
    per_src = {str(k): {"first": str(g["acq_date"].min().date()), "last": str(g["acq_date"].max().date()),
                        "days_with_data": int(g["acq_date"].nunique()), "detections": int(len(g))}
               for k, g in df.groupby("source")}
    return {"cells_in_bbox": int(len(cys) * n_cx), "cells_with_detections": int(len(cells)),
            "area_km2_in_bbox": round(total_area), "area_km2_cells_with_detections": round(covered),
            "share_of_area_with_detections": round(covered / total_area, 4) if total_area else None,
            "days_in_window": int(n_days), "days_with_detections": int(df["acq_date"].nunique()),
            "days_without_detections": int(n_days - df["acq_date"].nunique()), "per_source": per_src,
            "note": "Cells without detections can mean no fire OR no usable observation (cloud, overpass, outage)."}


class SpatialIndex:
    """Haversine BallTree for fast radius queries over many points (O(log n) instead of scanning all rows)."""

    def __init__(self, lat, lon):
        self._xy = np.radians(np.c_[np.asarray(lat, float), np.asarray(lon, float)])
        self._tree = BallTree(self._xy, metric="haversine") if len(self._xy) else None

    def within(self, lat, lon, radius_km):
        if self._tree is None:
            return np.array([], dtype=int)
        ind = self._tree.query_radius(np.radians([[lat, lon]]), r=radius_km / R_EARTH_KM)
        return ind[0]

    def nearest(self, lat, lon, k=1):
        if self._tree is None:
            return np.array([]), np.array([], dtype=int)
        d, i = self._tree.query(np.radians([[lat, lon]]), k=min(k, len(self._xy)))
        return d[0] * R_EARTH_KM, i[0]
