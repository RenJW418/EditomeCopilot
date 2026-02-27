@echo off
REM =============================================================================
REM  EditomeCopilot -- One-click start (Windows)
REM  Prerequisites: Python 3.10+, Node.js 18+, npm
REM =============================================================================
setlocal enabledelayedexpansion
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV=%ROOT%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
set "ACTIVATE=%VENV%\Scripts\activate.bat"

REM -- 1. Python virtual environment --------------------------------------------
if not exist "%PYTHON%" (
    echo [1/4] Creating Python virtual environment (.venv) ...
    python -m venv "%VENV%"
)

echo [1/4] Activating .venv ...
call "%ACTIVATE%"

REM -- 2. Python dependencies ---------------------------------------------------
echo [2/4] Installing Python dependencies ...
pip install -q --upgrade pip
pip install -q -r requirements.txt

REM -- 3. Frontend build --------------------------------------------------------
echo [3/4] Building frontend ...
cd frontend
call npm install --silent
call npm run build --silent
cd /d "%ROOT%"

REM -- 4. Launch backend --------------------------------------------------------
echo [4/4] Starting EditomeCopilot on http://localhost:6006 ...
uvicorn app:app --host 0.0.0.0 --port 6006

endlocal
