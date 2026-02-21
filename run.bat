@echo off
setlocal

title FabricGuard – Hole Detection

echo.
echo =====================================================
echo   FabricGuard  -  Hole Detection System
echo =====================================================
echo.

:: ── Locate Python ─────────────────────────────────────────────
:: Try venv first, then fall back to system Python
set VENV_PYTHON=venv\Scripts\python.exe
set VENV_ACTIVATE=venv\Scripts\activate.bat

if exist "%VENV_PYTHON%" (
    echo [OK] Virtual environment found
    call "%VENV_ACTIVATE%"
    set PYTHON=python
) else (
    echo [WARN] No venv found - checking system Python...
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python not found!
        echo         Run setup.bat first, or install Python from python.org
        pause
        exit /b 1
    )
    set PYTHON=python
    echo [OK] Using system Python
)

:: ── Check main.py exists ───────────────────────────────────────
if not exist "main.py" (
    echo [ERROR] main.py not found!
    echo         Make sure you are running this from the project folder:
    echo         d:\Root\FabricGuard AI\fabricguard-pro\
    pause
    exit /b 1
)

:: ── Check key packages ─────────────────────────────────────────
echo.
echo Checking dependencies...
%PYTHON% -c "import fastapi, uvicorn, ultralytics, cv2, torch" 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Some packages are missing. Running setup...
    echo.
    %PYTHON% -m pip install fastapi uvicorn ultralytics opencv-python pillow numpy pyyaml python-multipart jinja2 torch torchvision --quiet
)

:: ── Show device info (GPU or CPU, both are fine) ────────────────
echo.
%PYTHON% -c "import torch; g=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None; print('[OK] Device: GPU -',g) if g else print('[OK] Device: CPU (no GPU found - will run on CPU)')"

:: ── Launch ─────────────────────────────────────────────────────
echo.
echo [OK] Starting backend server...
echo.
echo   UI  ^>  http://127.0.0.1:8000
echo.
echo   Press Ctrl+C to stop the server.
echo =====================================================
echo.

%PYTHON% main.py

echo.
echo Server stopped.
pause
