#!/usr/bin/env bash
# The terminal chat, no API and no UI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/semanticSearch"

exec "$ROOT/.venv/bin/python" main.py
