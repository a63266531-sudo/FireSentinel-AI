#!/usr/bin/env bash
# One-time setup (Kali / Debian / Ubuntu). Creates .venv, installs dependencies, runs the self check.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v python3 >/dev/null || { echo "python3 missing: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "Python 3.10+ is required"; exit 1; }
python3 -m venv .venv || { echo "venv failed: sudo apt install -y python3-venv"; exit 1; }
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
[ -f .env ] || cp .env.example .env
python scripts/check_setup.py
echo
echo "Setup finished. Start the dashboard with:  bash scripts/run.sh"
