@echo off
REM LumaViewPro Windows install script
REM Scans for Python 3.11-3.13, creates a venv, installs dependencies.
REM
REM Usage: Double-click or run from command prompt.
REM After install, use run.bat to launch LumaViewPro.

setlocal

echo =========================================
echo  LumaViewPro Installer
echo =========================================
echo.

REM --- Find project root (parent of scripts/) ---
pushd "%~dp0.."
set "PROJECT_DIR=%CD%"
popd
set "VENV_DIR=%PROJECT_DIR%\venv"
set "FIND_SCRIPT=%~dp0_find_python.py"

REM --- Find a Python to bootstrap with ---
set "BOOT_PY="
py --version >nul 2>&1 && set "BOOT_PY=py" && goto :have_boot
python --version >nul 2>&1 && set "BOOT_PY=python" && goto :have_boot
python3 --version >nul 2>&1 && set "BOOT_PY=python3" && goto :have_boot

echo Error: No Python installation found.
echo Install Python 3.11+ from https://www.python.org/downloads/
echo Make sure to check "Add python.exe to PATH" during installation.
pause
exit /b 1

:have_boot
REM --- Run Python discovery helper ---
REM _find_python.py prints the best Python command to stdout
REM and informational messages to stderr
%BOOT_PY% "%FIND_SCRIPT%" > "%TEMP%\lvp_python_cmd.txt" 2>nul
set /p BEST_PY=<"%TEMP%\lvp_python_cmd.txt"
del "%TEMP%\lvp_python_cmd.txt" >nul 2>nul

if "%BEST_PY%"=="" goto :no_python
if "%BEST_PY%"=="ERROR" goto :no_python
goto :have_python

:no_python
echo Error: No suitable Python 3.11-3.13 found.
echo Install Python 3.11+ from https://www.python.org/downloads/
pause
exit /b 1

:have_python
REM Show what we found
%BEST_PY% -c "import sys; v=sys.version_info; print('Python '+str(v.major)+'.'+str(v.minor)+'.'+str(v.micro))"
echo Using: %BEST_PY%
echo.

REM --- Check for existing venv ---
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment already exists.
    echo Updating dependencies...
    goto :install_deps
)

REM --- Create virtual environment ---
echo Creating virtual environment...
%BEST_PY% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Error: Failed to create virtual environment.
    pause
    exit /b 1
)

:install_deps
echo.
echo Installing dependencies...
call "%VENV_DIR%\Scripts\python" -m pip install --upgrade pip --quiet
call "%VENV_DIR%\Scripts\pip" install -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 (
    echo.
    echo Error: pip install failed.
    pause
    exit /b 1
)

REM --- Verify installation ---
echo.
echo Verifying core packages...
call "%VENV_DIR%\Scripts\python" -c "import kivy; import numpy; import cv2; import serial; print('All core packages verified.')"
if errorlevel 1 (
    echo.
    echo Warning: Some core packages failed to import. Check output above.
)

REM --- Create run.bat in project root ---
>"%PROJECT_DIR%\run.bat" (
    echo @echo off
    echo cd /d "%%~dp0"
    echo call "%%~dp0venv\Scripts\python.exe" lumaviewpro.py %%*
)

>"%PROJECT_DIR%\run_simulate.bat" (
    echo @echo off
    echo cd /d "%%~dp0"
    echo call "%%~dp0venv\Scripts\python.exe" lumaviewpro.py --simulate %%*
)

echo.
echo =========================================
echo  LumaViewPro installation complete!
echo =========================================
echo.
echo  To run LumaViewPro:
echo    Double-click run.bat in %PROJECT_DIR%
echo.
echo  To run in simulate mode (no hardware):
echo    Double-click run_simulate.bat
echo.
echo  Note: Camera SDK must be installed separately
echo    Basler: https://docs.baslerweb.com/pylon-software-suite
echo    IDS:    https://en.ids-imaging.com/ids-peak.html
echo.
pause
