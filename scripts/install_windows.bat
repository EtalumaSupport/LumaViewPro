@echo off
REM LumaViewPro Windows install script
REM Scans for Python 3.11-3.13, creates a venv, installs dependencies.
REM Camera SDK (Basler Pylon or IDS Peak) must be installed separately.
REM
REM Usage: Double-click or run from command prompt.
REM After install, use run.bat to launch LumaViewPro.

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "VENV_DIR=%PROJECT_DIR%\venv"

echo =========================================
echo  LumaViewPro Installer
echo =========================================
echo.

REM --- Use Python helper script for version discovery ---
REM Try py launcher first, then python on PATH
set "BOOTSTRAP_PY="

py --version >nul 2>&1
if not errorlevel 1 (
    set "BOOTSTRAP_PY=py"
    goto :found_bootstrap
)

python --version >nul 2>&1
if not errorlevel 1 (
    set "BOOTSTRAP_PY=python"
    goto :found_bootstrap
)

python3 --version >nul 2>&1
if not errorlevel 1 (
    set "BOOTSTRAP_PY=python3"
    goto :found_bootstrap
)

echo Error: No Python installation found.
echo.
echo Install Python 3.11+ from https://www.python.org/downloads/
echo Make sure to check "Add python.exe to PATH" during installation.
echo.
pause
exit /b 1

:found_bootstrap
REM Run the Python discovery helper — it prints the path to the best Python
for /f "tokens=*" %%p in ('%BOOTSTRAP_PY% "%SCRIPT_DIR%_find_python.py"') do set "BEST_PY=%%p"

if "%BEST_PY%"=="" (
    echo Error: No suitable Python 3.11-3.13 found.
    echo Install Python 3.11+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

if "%BEST_PY%"=="ERROR" (
    echo Error: No suitable Python 3.11-3.13 found.
    echo Install Python 3.11+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Get the version string for display
for /f "tokens=*" %%v in ('%BEST_PY% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "BEST_VER=%%v"
echo.
echo Using Python %BEST_VER%
echo   (%BEST_PY%)
echo.

REM --- Check for existing venv ---
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment already exists at %VENV_DIR%
    echo Updating existing virtual environment...
    goto :install_deps
)

REM --- Create virtual environment ---
echo Creating virtual environment...
%BEST_PY% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Error: Failed to create virtual environment.
    echo.
    pause
    exit /b 1
)

:install_deps
echo.
echo Installing dependencies...
"%VENV_DIR%\Scripts\python" -m pip install --upgrade pip --quiet
"%VENV_DIR%\Scripts\pip" install -r "%PROJECT_DIR%\requirements.txt"

if errorlevel 1 (
    echo.
    echo Error: pip install failed. Check the output above for details.
    echo.
    pause
    exit /b 1
)

REM --- Verify installation ---
echo.
echo Verifying core packages...
"%VENV_DIR%\Scripts\python" -c "import kivy; import numpy; import cv2; import serial; print('All core packages verified.')"

if errorlevel 1 (
    echo.
    echo Warning: Some packages failed to import. LumaViewPro may not run correctly.
    echo.
)

REM --- Create run.bat ---
(
echo @echo off
echo cd /d "%%~dp0"
echo "venv\Scripts\python" lumaviewpro.py %%*
) > "%PROJECT_DIR%\run.bat"

REM --- Create run_simulate.bat ---
(
echo @echo off
echo cd /d "%%~dp0"
echo "venv\Scripts\python" lumaviewpro.py --simulate %%*
) > "%PROJECT_DIR%\run_simulate.bat"

echo.
echo =========================================
echo  LumaViewPro installation complete!
echo =========================================
echo.
echo  Python:  %BEST_VER%
echo  Venv:    %VENV_DIR%
echo.
echo  To run LumaViewPro:
echo    Double-click run.bat
echo    or: venv\Scripts\python lumaviewpro.py
echo.
echo  To run in simulate mode (no hardware):
echo    Double-click run_simulate.bat
echo.
echo  Note: Camera SDK must be installed separately
echo    Basler: https://docs.baslerweb.com/pylon-software-suite
echo    IDS:    https://en.ids-imaging.com/ids-peak.html
echo.
pause
