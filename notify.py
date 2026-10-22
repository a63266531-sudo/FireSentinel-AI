"""Alert delivery (email / Telegram) with de-duplication. Secrets come from environment
variables only - never commit them.

  Email:    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO, [ALERT_EMAIL_FROM]
  Telegram: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import hashlib
import os
import smtplib
from email.message import EmailMessage

import pandas as pd
import requests

from . import store
from .alerts import ORDER
from .config import DB_PATH, get_logger

log = get_logger()
DISCLAIMER = ("FireSentinel AI is a research/decision-support tool. Detections are unconfirmed satellite "
              "thermal anomalies; risk values are statistical estimates. Not an official emergency warning.")


def configured():
    e = os.environ.get
    return {"email": all(e(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "ALERT_EMAIL_TO")),
            "telegram": all(e(k) for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"))}


def alert_hash(row):
    key = f"{row['type']}|{round(float(row['lat']), 2)}|{round(float(row['lon']), 2)}|{str(row['time'])[:13]}"
    return hashlib.sha1(key.encode()).hexdigest()


def format_message(alerts_df, max_items=10):
    lines = [f"FireSentinel AI - {len(alerts_df)} new alert(s)", ""]
    for r in alerts_df.head(max_items).itertuples():
        lines.append(f"[{r.severity.upper()}] {r.category} / {r.type} @ {r.lat:.2f}, {r.lon:.2f} ({str(r.time)[:16]})")
        lines.append(f"  {r.explanation}")
    if len(alerts_df) > max_items:
        lines.append(f"... and {len(alerts_df) - max_items} more")
    return "\n".join(lines + ["", DISCLAIMER])


def send_email(subject, body):
    e = os.environ
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, e.get("ALERT_EMAIL_FROM", e["SMTP_USER"]), e["ALERT_EMAIL_TO"]
    msg.set_content(body)
    with smtplib.SMTP(e["SMTP_HOST"], int(e.get("SMTP_PORT", 587)), timeout=30) as s:
        s.starttls()
        s.login(e["SMTP_USER"], e["SMTP_PASS"])
        s.send_message(msg)


def send_telegram(subject, body):
    e = os.environ
    r = requests.post(f"https://api.telegram.org/bot{e['TELEGRAM_BOT_TOKEN']}/sendMessage",
                      json={"chat_id": e["TELEGRAM_CHAT_ID"], "text": f"{subject}\n\n{body}"[:4000]}, timeout=30)
    r.raise_for_status()


SENDERS = {"email": send_email, "telegram": send_telegram}


def dispatch(alerts_df, channels, min_severity="warning", dry_run=True, path=DB_PATH, senders=None):
    """Send only alerts not sent before. dry_run=True (default) only previews and remembers nothing."""
    senders = senders or SENDERS
    out = {"new": 0, "channels": [], "errors": [], "dry_run": dry_run, "preview": ""}
    if alerts_df is None or alerts_df.empty or not channels:
        return out
    df = alerts_df[alerts_df["severity"].map(ORDER) <= ORDER[min_severity]].copy()
    df["_h"] = df.apply(alert_hash, axis=1)
    fresh = df[[not store.was_sent(h, path) for h in df["_h"]]]
    out["new"], out["preview"] = len(fresh), format_message(fresh) if len(fresh) else ""
    if fresh.empty or dry_run:
        return out
    subject = f"FireSentinel AI: {len(fresh)} new alert(s)"
    ok = []
    for ch in channels:
        try:
            senders[ch](subject, out["preview"])
            ok.append(ch)
        except Exception as ex:
            out["errors"].append(f"{ch}: {ex}")
            log.error("Notify %s failed: %s", ch, ex)
    if ok:
        for h in fresh["_h"]:
            store.mark_sent(h, ",".join(ok), path)
    out["channels"] = ok
    return out
