#!/usr/bin/env python3
"""Static secret / risky-call scan of the project tree. Exit code 1 when something is found."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "data", "models", "logs", ".pytest_cache", "node_modules"}
SKIP_FILES = {"security_check.py"}
EXT = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".sh", ".txt", ".example", ".cfg", ".ini"}
SECRET = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"), "Telegram bot token"),
    (re.compile(r"""(?i)(api[_-]?key|token|passw(?:or)?d|secret)\s*[:=]\s*["'][A-Za-z0-9_\-]{16,}["']"""), "hard-coded credential"),
]
RISKY = [
    (re.compile(r"(?<![\w.])eval\("), "eval()"), (re.compile(r"(?<![\w.])exec\("), "exec()"),
    (re.compile(r"shell\s*=\s*True"), "shell=True"), (re.compile(r"os\.system\("), "os.system()"),
    (re.compile(r"pickle\.loads?\("), "pickle load"),
]


def scan(root: Path = ROOT):
    findings = []
    if (root / ".env").exists():
        findings.append((".env", 0, "a real .env file must not be shipped"))
    for p in root.rglob("*"):
        if not p.is_file() or set(p.relative_to(root).parts) & SKIP_DIRS or p.name in SKIP_FILES:
            continue
        if p.suffix not in EXT and p.name != ".env.example":
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for rx, what in SECRET + (RISKY if p.suffix == ".py" else []):
                if rx.search(line):
                    findings.append((str(p.relative_to(root)), n, what))
    return findings


if __name__ == "__main__":
    f = scan()
    for path, n, what in f:
        print(f"{path}:{n}: {what}")
    print("security_check:", "FAILED" if f else "no findings")
    sys.exit(1 if f else 0)
