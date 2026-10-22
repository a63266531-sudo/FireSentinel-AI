#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && . .venv/bin/activate
python -m pytest -q
python scripts/security_check.py
