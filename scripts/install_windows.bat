@echo off
REM LumaViewPro Windows install script
REM Installs Python dependencies. Camera SDK (Basler Pylon) must be installed separately.
REM
REM Usage:
REM   scripts\install_windows.bat          Install directly
REM   scripts\install_windows.bat --venv   Install in a virtual environment

setlocal

set "PROJECT_DIR=%~dp0.."
set "VENV_DIR=%PROJECT_DIR%\venv"
set "USE_VENV=0"

if "%~1"=="--venv" set "USE_VENV=1"

REM --- Check Python is available ---
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found.
    echo Install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

REM --- Check Python version ---
for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VERSION=%%v
for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version_info.minor)"') do set PY_MINOR=%%v

if %PY_MINOR% LSS 11 (
    echo Error: Python %PY_VERSION% found, but LumaViewPro requires Python 3.11, 3.12, or 3.13.
    echo Install a supported version from https://www.python.org/downloads/
    pause
    exit /b 1
)
if %PY_MINOR% GTR 13 (
    echo Error: Python %PY_VERSION% found, but LumaViewPro requires Python 3.11, 3.12, or 3.13.
    echo Install a supported version from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Found Python %PY_VERSION%

REM --- Install dependencies ---
if "%USE_VENV%"=="1" (
    if exist "%VENV_DIR%" (
        echo Virtual environment already exists at %VENV_DIR%
    ) else (
        echo Creating virtual environment...
        python -m venv "%VENV_DIR%"
    )
    echo Installing dependencies in virtual environment...
    "%VENV_DIR%\Scripts\pip" install --upgrade pip
    "%VENV_DIR%\Scripts\pip" install -r "%PROJECT_DIR%\requirements.txt"
) else (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r "%PROJECT_DIR%\requirements.txt"
)

REM --- Verify installation ---
echo.
echo Verifying installation...
if "%USE_VENV%"=="1" (
    "%VENV_DIR%\Scripts\python" -c "import kivy; import numpy; import cv2; import serial; print('All core packages verified.')"
) else (
    python -c "import kivy; import numpy; import cv2; import serial; print('All core packages verified.')"
)

echo.
echo =========================================
echo  LumaViewPro installation complete!
echo =========================================
echo.
if "%USE_VENV%"=="1" (
    echo To run LumaViewPro:
    echo   %VENV_DIR%\Scripts\activate
    echo   python lumaviewpro.py
) else (
    echo To run LumaViewPro:
    echo   python lumaviewpro.py
)
echo.
echo To run in simulate mode (no hardware):
echo   python lumaviewpro.py --simulate
echo.
echo Note: Basler Pylon SDK must be installed separately
echo   https://docs.baslerweb.com/pylon-software-suite
echo.
pause
