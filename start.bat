@echo off
echo ========================================
echo   Numerical Analysis Platform
echo ========================================
echo.

echo Starting Backend (FastAPI)...
cd /d "%~dp0backend"
start "Backend" cmd /k "set PYTHONPATH=%~dp0backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --log-level error"

echo Starting Frontend (Vite Dev Server)...
cd /d "%~dp0frontend"
start "Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo   Services are starting...
echo   Backend:  http://localhost:8002
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8002/docs
echo ========================================
echo.
pause