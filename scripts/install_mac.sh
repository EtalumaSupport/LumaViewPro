#!/bin/bash
# LumaViewPro macOS install script
# Scans for Python 3.11-3.13, creates a venv, installs dependencies.
# Camera SDK (Basler Pylon) must be installed separately.
#
# Usage:
#   bash scripts/install_mac.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

echo "========================================="
echo " LumaViewPro Installer"
echo "========================================="
echo ""

# --- Scan for available Python versions ---
FOUND=()
FOUND_CMDS=()

for minor in 13 12 11; do
    # Check python3.XX first (brew, pyenv, etc.)
    cmd="python3.$minor"
    if command -v "$cmd" &>/dev/null; then
        ver=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
        FOUND+=("$ver")
        FOUND_CMDS+=("$cmd")
    fi
done

# If nothing found via versioned commands, try python3 on PATH
if [ ${#FOUND[@]} -eq 0 ]; then
    if command -v python3 &>/dev/null; then
        py_minor=$(python3 -c "import sys; print(sys.version_info.minor)")
        if [ "$py_minor" -ge 11 ] && [ "$py_minor" -le 13 ]; then
            ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
            FOUND+=("$ver")
            FOUND_CMDS+=("python3")
        else
            echo "Error: Python 3.$py_minor found, but LumaViewPro requires 3.11, 3.12, or 3.13."
            echo "Install a supported version:"
            echo "  brew install python@3.13"
            echo "  or: https://www.python.org/downloads/macos/"
            exit 1
        fi
    else
        echo "Error: No Python installation found."
        echo "Install Python 3.11+:"
        echo "  brew install python@3.13"
        echo "  or: https://www.python.org/downloads/macos/"
        exit 1
    fi
fi

echo "Found Python installations:"
echo ""
for i in "${!FOUND[@]}"; do
    n=$((i + 1))
    echo "  [$n] Python ${FOUND[$i]}  (${FOUND_CMDS[$i]})"
done

# Default to first found (newest)
BEST_IDX=0
BEST_VER="${FOUND[0]}"
BEST_CMD="${FOUND_CMDS[0]}"

echo ""

# --- Let user choose if multiple versions found ---
if [ ${#FOUND[@]} -gt 1 ]; then
    echo "Which Python should LumaViewPro use?"
    echo "  Press Enter for the recommended version [Python $BEST_VER]"
    echo ""
    read -p "Choose [1-${#FOUND[@]}] or Enter for default: " CHOICE

    if [ -n "$CHOICE" ]; then
        idx=$((CHOICE - 1))
        if [ "$idx" -ge 0 ] && [ "$idx" -lt ${#FOUND[@]} ]; then
            BEST_IDX=$idx
            BEST_VER="${FOUND[$idx]}"
            BEST_CMD="${FOUND_CMDS[$idx]}"
        fi
    fi

    echo "Using Python $BEST_VER"
else
    echo "Using Python $BEST_VER"
fi

echo ""

# --- Check for existing venv ---
if [ -f "$VENV_DIR/bin/python" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo ""
    read -p "Recreate it? [y/N]: " RECREATE
    if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
        echo "Removing old virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Updating existing virtual environment..."
        "$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
        "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
        echo ""
        echo "Update complete!"
        exit 0
    fi
fi

# --- Create virtual environment ---
echo "Creating virtual environment with Python $BEST_VER..."
$BEST_CMD -m venv "$VENV_DIR"

# --- Install dependencies ---
echo ""
echo "Installing dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# --- Verify installation ---
echo ""
echo "Verifying core packages..."
"$VENV_DIR/bin/python" -c "import kivy; import numpy; import cv2; import serial; print('All core packages verified.')"

# --- Create run script ---
cat > "$PROJECT_DIR/run.sh" << 'RUNEOF'
#!/bin/bash
cd "$(dirname "$0")"
venv/bin/python lumaviewpro.py "$@"
RUNEOF
chmod +x "$PROJECT_DIR/run.sh"

cat > "$PROJECT_DIR/run_simulate.sh" << 'RUNEOF'
#!/bin/bash
cd "$(dirname "$0")"
venv/bin/python lumaviewpro.py --simulate "$@"
RUNEOF
chmod +x "$PROJECT_DIR/run_simulate.sh"

echo ""
echo "========================================="
echo " LumaViewPro installation complete!"
echo "========================================="
echo ""
echo "  Python:  $BEST_VER"
echo "  Venv:    $VENV_DIR"
echo ""
echo "  To run LumaViewPro:"
echo "    ./run.sh"
echo "    or: venv/bin/python lumaviewpro.py"
echo ""
echo "  To run in simulate mode (no hardware):"
echo "    ./run_simulate.sh"
echo "    or: venv/bin/python lumaviewpro.py --simulate"
echo ""
echo "  Note: Basler Pylon SDK must be installed separately"
echo "    https://docs.baslerweb.com/pylon-software-suite"
