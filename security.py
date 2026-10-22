"""Small security helpers: path safety, upload limits, integrity hashes, secret redaction."""
import hashlib
import os
from pathlib import Path

MAX_UPLOAD_MB = 50
SECRET_ENV = ("FIRMS_MAP_KEY", "SMTP_PASS", "TELEGRAM_BOT_TOKEN")


def safe_join(base: Path, name: str) -> Path:
    """Join a user-supplied name onto base; refuse anything that escapes base."""
    base = Path(base).resolve()
    p = (base / name).resolve()
    if p != base and base not in p.parents:
        raise ValueError("Unsafe path")
    return p


def check_upload_size(size_bytes: int, max_mb: int = MAX_UPLOAD_MB):
    if size_bytes > max_mb * 1024 * 1024:
        raise ValueError(f"File is larger than the {max_mb} MB limit")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def redact(text, extra=()) -> str:
    """Remove known secret values from a message before logging or showing it."""
    out = str(text)
    for v in [os.environ.get(k) for k in SECRET_ENV] + list(extra):
        if v and len(str(v)) >= 4:
            out = out.replace(str(v), "***")
    return out
