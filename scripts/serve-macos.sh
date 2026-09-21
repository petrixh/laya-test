#!/usr/bin/env bash
# Run the Laya service natively on Apple silicon using the MLX backend.
# Docker is not used: MLX needs Metal, which containers on macOS cannot reach.
#
#   ./scripts/serve-macos.sh                      # English checkpoint
#   LAYA_SUBFOLDER=multilingual ./scripts/serve-macos.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script is for Apple silicon. On Linux use: docker compose up -d" >&2
  exit 1
fi

PY=${PY:-python3.11}
command -v "$PY" >/dev/null || { echo "need $PY (laya-mlx requires Python 3.11+)" >&2; exit 1; }

if [[ ! -d .venv ]]; then
  echo "creating .venv with $PY"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements-mlx.txt

export LAYA_BACKEND=${LAYA_BACKEND:-mlx}
export LAYA_DTYPE=${LAYA_DTYPE:-float16}
# 0.0.0.0 so another machine on the LAN can point its browser at this service
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "serving Laya (backend=$LAYA_BACKEND subfolder=${LAYA_SUBFOLDER:-none}) on $HOST:$PORT"
exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
