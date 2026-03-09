#!/bin/bash
# LumaViewPro Linux install script
# Installs system dependencies and Python packages.
# Camera SDK (Basler Pylon) must be installed separately.
#
# Usage:
#   bash scripts/install_linux.sh          # Install directly
#   bash scripts/install_linux.sh --venv   # Install in a virtual environment

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

# --- Install system dependencies ---
echo "Checking system dependencies..."

if command -v apt-get &> /dev/null; then
    echo "Detected apt package manager (Debian/Ubuntu)"
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv \
        libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
elif command -v dnf &> /dev/null; then
    echo "Detected dnf package manager (Fedora/RHEL)"
    sudo dnf install -y python3 python3-pip \
        SDL2-devel SDL2_image-devel SDL2_mixer-devel SDL2_ttf-devel
elif command -v pacman &> /dev/null; then
    echo "Detected pacman package manager (Arch)"
    sudo pacman -S --needed python python-pip \
        sdl2 sdl2_image sdl2_mixer sdl2_ttf
else
    echo "Warning: Could not detect package manager."
    echo "Please ensure Python 3.11+ and SDL2 development libraries are installed."
fi

# --- Check Python version ---
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found after install attempt."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -ne "$MIN_MAJOR" ] || [ "$PY_MINOR" -lt "$MIN_MINOR" ] || [ "$PY_MINOR" -gt "$MAX_MINOR" ]; then
    echo "Error: Python $PY_VERSION found, but LumaViewPro requires Python 3.11, 3.12, or 3.13."
    echo "You may need to install a newer Python version from your package manager or https://www.python.org/downloads/"
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

# --- USB permissions ---
if ! groups | grep -q dialout; then
    echo ""
    echo "Adding $USER to dialout group for USB serial access..."
    sudo usermod -a -G dialout "$USER"
    echo "You will need to log out and back in for this to take effect."
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
