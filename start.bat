@echo off
title Meeting Intelligence Platform

echo.
echo  ================================================
echo   Meeting Intelligence Platform
echo   Milestone 1, 2 and 4
echo  ================================================
echo.

:: ── Check that the virtual environment exists ─────────────────────────────
if not exist "%~dp0myenv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found.
    echo.
    echo  Run this first:
    echo    python -m venv myenv
    echo    myenv\Scripts\activate.bat
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── Check that streamlit is installed inside the venv ────────────────────
if not exist "%~dp0myenv\Scripts\streamlit.exe" (
    echo  [ERROR] Streamlit not found in the virtual environment.
    echo.
    echo  Run this first:
    echo    myenv\Scripts\pip.exe install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── Check for .env file ───────────────────────────────────────────────────
if not exist "%~dp0.env" (
    echo  [WARNING] .env file not found.
    echo  The AI summary and Zoom features require API keys.
    echo.
    echo  To set up:
    echo    copy .env.example .env
    echo    Then edit .env and add your keys.
    echo.
    echo  The app will still run without keys (mock mode for providers).
    echo.
)

:: ── Run DB init / check ────────────────────────────────────────────────────
echo  Initialising database...
"%~dp0myenv\Scripts\python.exe" -c "import database; database.init_db(); print('  Database ready.')" 2>nul
if errorlevel 1 (
    echo  [WARNING] Could not pre-initialise database - app will handle it on startup.
)

:: ── Start the app ─────────────────────────────────────────────────────────
echo.
echo  Starting app...
echo  Open your browser at: http://localhost:8501
echo.
echo  Press Ctrl+C to stop.
echo.

"%~dp0myenv\Scripts\streamlit.exe" run "%~dp0app.py"

echo.
echo  App stopped.
pause
