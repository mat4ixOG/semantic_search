#!/usr/bin/env bash
# Keeps both models resident so the pipeline stops paying a cold load on
# every call. Run with: sudo ./tune-ollama.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "needs root, run: sudo $0"
  exit 1
fi

OVERRIDE_DIR=/etc/systemd/system/ollama.service.d
mkdir -p "$OVERRIDE_DIR"

cat > "$OVERRIDE_DIR/override.conf" <<'EOF'
[Service]
# A question fires several calls in a row. The default 5 minute keep alive
# means the answer model is often evicted between questions and reloaded
# from disk, which costs more than the generation itself.
Environment="OLLAMA_KEEP_ALIVE=30m"

# The answer model and the utility model, both resident at once.
Environment="OLLAMA_MAX_LOADED_MODELS=2"

# One request at a time. Parallel slots multiply the KV cache, and nothing
# here issues concurrent requests.
Environment="OLLAMA_NUM_PARALLEL=1"

Environment="OLLAMA_FLASH_ATTENTION=1"
EOF

systemctl daemon-reload
systemctl restart ollama

echo "applied:"
sed -n 's/^Environment="\(.*\)"$/  \1/p' "$OVERRIDE_DIR/override.conf"
echo
echo "verify with: ollama ps   (models should stay listed for 30m)"
