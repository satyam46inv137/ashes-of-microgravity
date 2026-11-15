@echo off
title Ashes of Microgravity - Combustion Lab
echo ============================================
echo   Ashes of Microgravity - Combustion Lab
echo ============================================
echo.

REM
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%b"=="" set "%%a=%%~b"
    )
)

if "%GROQ_API_KEY%"=="" (
    echo [WARNING] GROQ_API_KEY not set - RAG will use fallback mode
    echo           Get a free key: https://console.groq.com
    echo.
)

REM
set "PY_CMD=python"
if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"

start "" cmd /c "timeout /t 3 >nul & start http://localhost:8000"
"%PY_CMD%" -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

pause