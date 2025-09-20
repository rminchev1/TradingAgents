#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=5173
DO_INSTALL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      DO_INSTALL=true
      shift
      ;;
    --backend-port)
      BACKEND_PORT="$2"; shift 2;
      ;;
    --frontend-port)
      FRONTEND_PORT="$2"; shift 2;
      ;;
    *)
      echo "Unknown option: $1" >&2; exit 1;
      ;;
  esac
done

echo "[start] Root: $ROOT_DIR"
echo "[start] Backend port: $BACKEND_PORT | Frontend port: $FRONTEND_PORT"

if $DO_INSTALL; then
  echo "[start] Installing Python deps (requirements.txt)";
  pip install -r "$ROOT_DIR/requirements.txt"
  echo "[start] Installing frontend deps (web)";
  (cd "$ROOT_DIR/web" && npm install)
fi

cleanup() {
  local code=$?
  if [[ -n "${BACKEND_PID:-}" ]]; then
    echo "[start] Stopping backend (pid $BACKEND_PID)";
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  exit $code
}
trap cleanup EXIT INT TERM

echo "[start] Launching backend (uvicorn)"
(cd "$ROOT_DIR" && uvicorn server.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload) &
BACKEND_PID=$!
echo "[start] Backend pid: $BACKEND_PID"

echo "[start] Waiting for backend readiness at http://127.0.0.1:$BACKEND_PORT/healthz"
for i in {1..30}; do
  if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:$BACKEND_PORT/healthz" >/dev/null; then
    echo "[start] Backend is ready"
    break
  fi
  sleep 1
done

echo "[start] Launching frontend (Vite dev server) at http://localhost:$FRONTEND_PORT"
echo "[start] Tip: export OPENAI_API_KEY / GOOGLE_API_KEY before starting if needed."

# Ensure Vite dev server proxies to the backend port
cd "$ROOT_DIR/web"
VITE_API_URL="http://127.0.0.1:$BACKEND_PORT" PORT="$FRONTEND_PORT" npm run dev
