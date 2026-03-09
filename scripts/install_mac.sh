#!/bin/bash
# LumaViewPro macOS install script
# Installs Python dependencies. Camera SDK (Basler Pylon) must be installed separately.
#
# Usage:
#   bash scripts/install_mac.sh          # Install directly
#   bash scripts/install_mac.sh --venv   # Install in a virtual environment

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MIN_MAJOR=3
MIN_MINOR=11
MAX_MINOR=13

USE_VENV=false
if [ "$1" = "--venv" ]; then
    USE_VENV=true
fi

# --- Check Python version ---
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found."
    echo "Install Python 3.11+ from https://www.python.org/downloads/macos/"
    echo "  or: brew install python@3.13"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -ne "$MIN_MAJOR" ] || [ "$PY_MINOR" -lt "$MIN_MINOR" ] || [ "$PY_MINOR" -gt "$MAX_MINOR" ]; then
    echo "Error: Python $PY_VERSION found, but LumaViewPro requires Python 3.11, 3.12, or 3.13."
    echo "Install a supported version from https://www.python.org/downloads/macos/"
    exit 1
fi

echo "Found Python $PY_VERSION"

# --- Install dependencies ---
if [ "$USE_VENV" = true ]; then
    VENV_DIR="$PROJECT_DIR/venv"
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment already exists at $VENV_DIR"
    else
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    echo "Installing dependencies in virtual environment..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
else
    echo "Installing dependencies..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -r "$PROJECT_DIR/requirements.txt"
fi

# --- Verify installation ---
echo ""
echo "Verifying installation..."
if [ "$USE_VENV" = true ]; then
    "$VENV_DIR/bin/python" -c "import kivy; import numpy; import cv2; import serial; print('All core packages verified.')"
else
    python3 -c "import kivy; import numpy; import cv2; import serial; print('All core packages verified.')"
fi

echo ""
echo "========================================="
echo " LumaViewPro installation complete!"
echo "========================================="
echo ""
if [ "$USE_VENV" = true ]; then
    echo "To run LumaViewPro:"
    echo "  cd $PROJECT_DIR"
    echo "  source venv/bin/activate"
    echo "  python lumaviewpro.py"
else
    echo "To run LumaViewPro:"
    echo "  cd $PROJECT_DIR"
    echo "  python3 lumaviewpro.py"
fi
echo ""
echo "To run in simulate mode (no hardware):"
echo "  python lumaviewpro.py --simulate"
echo ""
echo "Note: Basler Pylon SDK must be installed separately"
echo "  https://docs.baslerweb.com/pylon-software-suite"
