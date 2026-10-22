"""SQLite storage: accumulate detections across fetches, log every pipeline run."""
import sqlite3

import pandas as pd

from .config import DB_PATH

COLS = ["latitude", "longitude", "acq_datetime", "source", "satellite", "instrument",
        "confidence", "conf_level", "brightness", "frp", "daynight"]
SCHEMA = """
CREATE TABLE IF NOT EXISTS detections(
  latitude REAL, longitude REAL, acq_datetime TEXT, source TEXT, satellite TEXT,
  instrument TEXT, confidence TEXT, conf_level TEXT, brightness REAL, frp REAL,
  daynight TEXT, retrieved_at TEXT,
  UNIQUE(latitude, longitude, acq_datetime, source));
CREATE TABLE IF NOT EXISTS runs(
  retrieved_at TEXT, source TEXT, bbox TEXT, rows INTEGER, inserted INTEGER,
  status TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS sent_alerts(hash TEXT PRIMARY KEY, sent_at TEXT, channel TEXT);
CREATE TABLE IF NOT EXISTS provenance(
  retrieved_at TEXT, provider TEXT, product TEXT, request TEXT, rows INTEGER, sha256 TEXT, schema_ok INTEGER, note TEXT);
"""


def connect(path=DB_PATH):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def _clean(v):
    return None if pd.isna(v) else (v if isinstance(v, (int, float, str)) else str(v))


def upsert(df, retrieved_at, path=DB_PATH) -> int:
    """Insert new detections (duplicates ignored). Returns number inserted."""
    d = df.copy()
    for c in COLS:
        if c not in d.columns:
            d[c] = None
    d["acq_datetime"] = pd.to_datetime(d["acq_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    d["confidence"] = d["confidence"].map(lambda v: None if pd.isna(v) else str(v))
    d["retrieved_at"] = retrieved_at.isoformat()
    rows = [tuple(_clean(v) for v in r) for r in d[COLS + ["retrieved_at"]].itertuples(index=False, name=None)]
    con = connect(path)
    try:
        before = con.total_changes
        con.executemany("INSERT OR IGNORE INTO detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        return con.total_changes - before
    finally:
        con.close()


def load(bbox=None, start=None, end=None, path=DB_PATH) -> pd.DataFrame:
    q, p = "SELECT * FROM detections WHERE 1=1", []
    if bbox:
        w, s, e, n = bbox
        q += " AND longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?"
        p += [w, e, s, n]
    if start is not None:
        q += " AND acq_datetime >= ?"
        p.append(pd.Timestamp(start).strftime("%Y-%m-%d 00:00:00"))
    if end is not None:
        q += " AND acq_datetime <= ?"
        p.append(pd.Timestamp(end).strftime("%Y-%m-%d 23:59:59"))
    con = connect(path)
    try:
        df = pd.read_sql_query(q + " ORDER BY acq_datetime", con, params=p)
    finally:
        con.close()
    df["acq_datetime"] = pd.to_datetime(df["acq_datetime"])
    df["acq_date"] = df["acq_datetime"].dt.normalize()
    return df


def log_run(retrieved_at, source, bbox, rows, inserted, status, message="", path=DB_PATH):
    con = connect(path)
    try:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                    (retrieved_at.isoformat(), source, str(list(bbox)), rows, inserted, status, message[:500]))
        con.commit()
    finally:
        con.close()


def recent_runs(n=20, path=DB_PATH) -> pd.DataFrame:
    con = connect(path)
    try:
        return pd.read_sql_query("SELECT * FROM runs ORDER BY retrieved_at DESC LIMIT ?", con, params=[n])
    finally:
        con.close()


def was_sent(h, path=DB_PATH) -> bool:
    con = connect(path)
    try:
        return con.execute("SELECT 1 FROM sent_alerts WHERE hash=?", (h,)).fetchone() is not None
    finally:
        con.close()


def mark_sent(h, channel, path=DB_PATH):
    con = connect(path)
    try:
        con.execute("INSERT OR IGNORE INTO sent_alerts VALUES (?,?,?)", (h, pd.Timestamp.now().isoformat(), channel))
        con.commit()
    finally:
        con.close()


def log_provenance(p: dict, path=DB_PATH):
    con = connect(path)
    try:
        con.execute("INSERT INTO provenance VALUES (?,?,?,?,?,?,?,?)",
                    (p["retrieved_at_utc"], p["provider"], p["product"], str(p["request"])[:500], int(p["rows"]),
                     p["sha256"], int(bool(p["schema_ok"])), str(p.get("note", ""))[:300]))
        con.commit()
    finally:
        con.close()


def recent_provenance(n=30, path=DB_PATH) -> pd.DataFrame:
    con = connect(path)
    try:
        return pd.read_sql_query("SELECT * FROM provenance ORDER BY retrieved_at DESC LIMIT ?", con, params=[n])
    finally:
        con.close()
