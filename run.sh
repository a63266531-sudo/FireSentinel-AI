#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || { echo "Run  bash scripts/setup.sh  first"; exit 1; }
# shellcheck disable=SC1091
. .venv/bin/activate
exec streamlit run app.py "$@"
