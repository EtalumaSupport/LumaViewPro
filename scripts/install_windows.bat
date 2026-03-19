@echo off
REM LumaViewPro Windows install script
REM Scans for Python 3.11-3.13, creates a venv, installs dependencies.
REM Camera SDK (Basler Pylon or IDS Peak) must be installed separately.
REM
REM Usage:
REM   Double-click or: scripts\install_windows.bat
REM
REM After install, use run.bat to launch LumaViewPro.

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0.."
set "VENV_DIR=%PROJECT_DIR%\venv"

echo =========================================
echo  LumaViewPro Installer
echo =========================================
echo.

REM --- Scan for available Python versions ---
REM Uses the Windows Python Launcher (py.exe) if available, falls back to PATH
set "FOUND_COUNT=0"
set "BEST_PY="
set "BEST_MINOR=0"

REM Try py launcher first (handles multiple Python installs cleanly)
py --list >nul 2>&1
if not errorlevel 1 (
    echo Scanning for Python installations...
    echo.
    for %%m in (13 12 11) do (
        py -3.%%m --version >nul 2>&1
        if not errorlevel 1 (
            for /f "tokens=*" %%v in ('py -3.%%m -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do (
                set /a FOUND_COUNT+=1
                echo   [!FOUND_COUNT!] Python %%v
                set "PY_!FOUND_COUNT!=py -3.%%m"
                set "PY_VER_!FOUND_COUNT!=%%v"
                if !BEST_MINOR! LSS %%m (
                    set "BEST_PY=py -3.%%m"
                    set "BEST_MINOR=%%m"
                    set "BEST_VER=%%v"
                )
            )
        )
    )
)

REM If py launcher not found or found nothing, try python on PATH
if %FOUND_COUNT% EQU 0 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo Error: No Python installation found.
        echo.
        echo Install Python 3.11+ from https://www.python.org/downloads/
        echo Make sure to check "Add python.exe to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
    for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version_info.minor)"') do set "PATH_MINOR=%%v"
    if !PATH_MINOR! LSS 11 (
        echo Error: Python 3.!PATH_MINOR! found, but LumaViewPro requires 3.11, 3.12, or 3.13.
        echo Install a supported version from https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
    if !PATH_MINOR! GTR 13 (
        echo Error: Python 3.!PATH_MINOR! found, but LumaViewPro requires 3.11, 3.12, or 3.13.
        echo Install a supported version from https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
    for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "BEST_VER=%%v"
    set "BEST_PY=python"
    set "FOUND_COUNT=1"
    set "PY_1=python"
    set "PY_VER_1=!BEST_VER!"
    echo   [1] Python !BEST_VER! (from PATH)
)

echo.

REM --- Let user choose if multiple versions found ---
if %FOUND_COUNT% GTR 1 (
    echo Found %FOUND_COUNT% Python versions. Which one should LumaViewPro use?
    echo   Press Enter for the recommended version [Python %BEST_VER%]
    echo.
    set /p "CHOICE=Choose [1-%FOUND_COUNT%] or Enter for default: "

    if "!CHOICE!"=="" (
        echo.
        echo Using Python %BEST_VER%
    ) else (
        set "BEST_PY=!PY_%%CHOICE%%!"
        REM Resolve the choice
        if "!CHOICE!"=="1" set "BEST_PY=!PY_1!" & set "BEST_VER=!PY_VER_1!"
        if "!CHOICE!"=="2" set "BEST_PY=!PY_2!" & set "BEST_VER=!PY_VER_2!"
        if "!CHOICE!"=="3" set "BEST_PY=!PY_3!" & set "BEST_VER=!PY_VER_3!"
        echo.
        echo Using Python !BEST_VER!
    )
) else (
    echo Using Python %BEST_VER%
)

echo.

REM --- Check for existing venv ---
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment already exists at %VENV_DIR%
    echo.
    set /p "RECREATE=Recreate it? [y/N]: "
    if /i "!RECREATE!"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo Updating existing virtual environment...
        goto :install_deps
    )
)

REM --- Create virtual environment ---
echo Creating virtual environment with Python %BEST_VER%...
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
echo @echo off> "%PROJECT_DIR%\run.bat"
echo REM Launch LumaViewPro>> "%PROJECT_DIR%\run.bat"
echo cd /d "%%~dp0">> "%PROJECT_DIR%\run.bat"
echo "venv\Scripts\python" lumaviewpro.py %%*>> "%PROJECT_DIR%\run.bat"

REM --- Create run_simulate.bat ---
echo @echo off> "%PROJECT_DIR%\run_simulate.bat"
echo REM Launch LumaViewPro in simulate mode (no hardware needed)>> "%PROJECT_DIR%\run_simulate.bat"
echo cd /d "%%~dp0">> "%PROJECT_DIR%\run_simulate.bat"
echo "venv\Scripts\python" lumaviewpro.py --simulate %%*>> "%PROJECT_DIR%\run_simulate.bat"

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
echo    or: venv\Scripts\python lumaviewpro.py --simulate
echo.
echo  Note: Camera SDK must be installed separately
echo    Basler: https://docs.baslerweb.com/pylon-software-suite
echo    IDS:    https://en.ids-imaging.com/ids-peak.html
echo.
pause
