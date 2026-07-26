#!/usr/bin/env bash
# One-shot setup + run for macOS / Linux.
# Installs Python and Node dependencies (first run only), then starts the
# LangGraph backend (http://127.0.0.1:2024) and the atlas-web UI
# (http://localhost:3000). Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

# ── Prerequisites ───────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "ERROR: no .env file found in the project root."
    echo "Copy .env.example to .env and fill in the API keys, then re-run."
    exit 1
fi

PYTHON=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Install Python 3.12 from https://www.python.org/downloads/"
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: npm not found. Install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

# ── Install (skipped when already done) ─────────────────────────────────────
if [ ! -x .venv/bin/langgraph ]; then
    echo ">> Creating virtualenv with $PYTHON and installing Python dependencies (takes a few minutes)..."
    "$PYTHON" -m venv .venv
    .venv/bin/pip install -r requirements-server.txt
fi

if [ ! -d atlas-web/node_modules ]; then
    echo ">> Installing web UI dependencies..."
    (cd atlas-web && npm install --no-audit --no-fund)
fi

# ── Run ─────────────────────────────────────────────────────────────────────
echo ">> Starting backend (LangGraph dev server) on http://127.0.0.1:2024 ..."
.venv/bin/langgraph dev --allow-blocking --no-browser &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT

echo ">> Waiting for the backend to come up..."
BACKEND_UP=0
for _ in $(seq 1 60); do
    if curl -s -o /dev/null http://127.0.0.1:2024/ok; then
        BACKEND_UP=1
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "ERROR: backend exited during startup — see the log output above."
        exit 1
    fi
    sleep 2
done
if [ "$BACKEND_UP" -ne 1 ]; then
    echo "ERROR: backend did not respond on http://127.0.0.1:2024 within 2 minutes."
    exit 1
fi

echo ""
echo ">> Backend is up. Starting the web UI ..."
echo ">> Open http://localhost:3000 in your browser once Next.js reports 'Ready'."
echo ">> Press Ctrl+C to stop both servers."
echo ""
cd atlas-web && npm run dev
