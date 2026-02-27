#!/usr/bin/env bash
# =============================================================================
# EditomeCopilot — One-click start (Linux / macOS)
# Prerequisites: Python 3.10+, Node.js 18+, npm
# First-time setup is handled automatically below.
# =============================================================================
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

# ── 1. Python virtual environment ─────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "[1/4] Creating Python virtual environment (.venv) ..."
    python3 -m venv "$VENV"
fi

echo "[1/4] Activating .venv ..."
source "${VENV}/bin/activate"

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo "[2/4] Installing Python dependencies ..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ── 3. Frontend build ─────────────────────────────────────────────────────────
echo "[3/4] Building frontend ..."
cd frontend
npm install --silent
npm run build --silent
cd "$ROOT"

# ── 4. Launch backend ─────────────────────────────────────────────────────────
echo "[4/4] Starting EditomeCopilot on http://localhost:6006 ..."
exec uvicorn app:app --host 0.0.0.0 --port 6006
