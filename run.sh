#!/usr/bin/env bash
# Starts the API and the UI together. Ctrl+C stops both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-3000}"
NODE_VERSION="${NODE_VERSION:-20.20.1}"

if [ ! -x "$PYTHON" ]; then
  echo "no virtualenv at $ROOT/.venv"
  exit 1
fi

# Starting on top of a server that is already running fails with a stack
# trace that hides what actually happened.
port_busy() {
  ss -ltn 2>/dev/null | grep -q ":$1 " || lsof -ti:"$1" >/dev/null 2>&1
}

if port_busy "$API_PORT"; then
  echo "port $API_PORT is already in use, the API may already be running"
  echo "  open http://127.0.0.1:$API_PORT/api/health to check"
  echo "  or stop it with: pkill -f 'uvicorn api:app'"
  exit 1
fi

if port_busy "$UI_PORT"; then
  echo "port $UI_PORT is already in use, the UI may already be running"
  echo "  open http://localhost:$UI_PORT to check"
  echo "  or stop it with: pkill -f 'next dev'"
  exit 1
fi

cleanup() {
  echo
  echo "stopping..."
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> starting API on http://127.0.0.1:$API_PORT"
cd "$ROOT/semanticSearch"
"$PYTHON" -m uvicorn api:app --port "$API_PORT" --reload &
API_PID=$!

# Wait for it to answer before opening the UI, otherwise the first page load
# shows "API unreachable" for no reason.
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then
    echo "==> API is up"
    break
  fi
  sleep 1
done

echo "==> starting UI on http://localhost:$UI_PORT"
cd "$ROOT/frontend"

# The default node on this machine is too old for Next.js.
if [ -s "$HOME/.nvm/nvm.sh" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.nvm/nvm.sh"
  nvm use "$NODE_VERSION" >/dev/null
fi

if [ ! -d node_modules ]; then
  echo "==> installing frontend dependencies"
  npm install --no-audit --no-fund
fi

BACKEND_URL="http://127.0.0.1:$API_PORT" npm run dev -- --port "$UI_PORT"
