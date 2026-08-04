#!/usr/bin/env bash
# API only, on http://127.0.0.1:8000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/semanticSearch"

exec "$ROOT/.venv/bin/python" -m uvicorn api:app --port "${API_PORT:-8000}" --reload
