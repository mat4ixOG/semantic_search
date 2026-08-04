#!/usr/bin/env bash
# UI only, on http://localhost:3000. Needs the API already running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/frontend"

if [ -s "$HOME/.nvm/nvm.sh" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.nvm/nvm.sh"
  nvm use "${NODE_VERSION:-20.20.1}" >/dev/null
fi

[ -d node_modules ] || npm install --no-audit --no-fund

exec npm run dev -- --port "${UI_PORT:-3000}"
