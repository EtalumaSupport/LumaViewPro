# LumaViewPro

LumaViewPro is an open-source fluorescence microscope control application written in Python. Sponsored by [Etaluma, Inc.](https://www.etaluma.com/), it provides a full-featured GUI and a Python API for controlling Etaluma Lumascope microscopes.

**Current version:** 4.0.0-beta

![lvpscreenshot2](https://user-images.githubusercontent.com/108957480/179601967-8c2f3be7-5371-4091-9f07-fd34e1c8f9bb.png)

## Hardware Overview

LumaViewPro controls the **Etaluma Lumascope** family of inverted fluorescence microscopes and Etaluma OEM optics modules.

### Microscope Models

| Model | Stage | Turret | Description |
|-------|-------|--------|-------------|
| **Lumi** | Z only | No | Luminescence-only imager |
| **LS820** | Z only | No | Fixed-stage fluorescence microscope |
| **LS850** | XYZ | No | Motorized XY stage |
| **LS850T** | XYZ + Turret | 4-position | Full-featured with objective turret |
| **OEM Modules** | Configurable | Configurable | Etaluma optics modules for integration into custom instruments |

### Capabilities

- **LED Illumination** -- Up to 6 channels: Blue, Green, Red fluorescence; Brightfield, Phase Contrast, Darkfield; Luminescence mode
- **Motion Control** -- Motorized X/Y stage, Z focus, and 4-position objective turret (model-dependent)
- **Camera** -- Basler (Pylon) camera with 12-bit depth, configurable ROI, binning, gain, and exposure

## Software Overview

- **GUI** with live image display, protocol editor, and full microscope controls
- **Protocol Engine** for automated multi-well, multi-channel, time-lapse, Z-stack, and video capture
- **Autofocus** with frame-based settling
- **Simulate Mode** -- run without hardware using `python lumaviewpro.py --simulate`
- **Python API** -- `lumascope_api.py` provides headless microscope control for scripting and automation

API documentation and example scripts are in the `docs/` folder.

## Getting Started

The easiest way to run LumaViewPro is to use the Windows installer, which includes the application, camera driver, and all dependencies. Download the latest release from the [GitHub Releases](https://github.com/EtalumaSupport/LumaViewPro/releases) page.

To run from source on any platform (Windows, macOS, or Linux), use the install scripts or follow the manual instructions below.

## Troubleshooting

**"No camera found" or camera not detected**
- Make sure the Basler Pylon Runtime is installed (the Windows installer includes it; for source installs, download from [Basler](https://docs.baslerweb.com/pylon-software-suite))
- Verify the camera is connected via USB 3.0 (not USB 2.0)
- On Linux, check that your user is in the `dialout` group: `groups $USER`

**"Python not found" or wrong version**
- LumaViewPro requires Python 3.11, 3.12, or 3.13
- On Windows, make sure "Add python.exe to PATH" was checked during installation
- On macOS/Linux, try `python3 --version` to check your installed version

**Application crashes on startup**
- Try running from the command line to see error messages: `python lumaviewpro.py`
- Try simulate mode to rule out hardware issues: `python lumaviewpro.py --simulate`

**Permission denied on Linux (serial port)**
```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

## Support

To report bugs or request features, please open an issue on [GitHub Issues](https://github.com/EtalumaSupport/LumaViewPro/issues). Please include your LumaViewPro version and, if possible, attach a zipped copy of your logs folder:
- **Windows (installed):** `Documents\LumaViewPro {version}\logs\`
- **Running from source:** `logs\` in the LumaViewPro folder

## Requirements (running from source)

- **Python**: 3.11, 3.12, or 3.13
- **Camera SDK**: [Basler Pylon](https://docs.baslerweb.com/pylon-software-suite) (included in the Windows installer, must be installed separately when running from source)
- **OS**: Windows 10/11, macOS, Linux (see note below)

**macOS note:** pypylon 4.0.0 wheels are limited to macOS 14 (Sonoma) on Apple Silicon (ARM64) — macOS 15 (Sequoia) and later are **not yet supported**. Intel Macs support macOS 11+. Check the [Basler Pylon page](https://www.baslerweb.com/en-us/software/pylon/) for the latest compatibility information.

## Installation from Source

### Quick Install

Install scripts are provided in the `scripts/` folder. They check your Python version and install all dependencies:

| Platform | Script | Usage |
|----------|--------|-------|
| Windows | `scripts\install_windows.bat` | Double-click or run from Command Prompt |
| macOS | `scripts/install_mac.sh` | `bash scripts/install_mac.sh` |
| Linux | `scripts/install_linux.sh` | `bash scripts/install_linux.sh` |

Add `--venv` to install in a virtual environment instead of system Python (e.g. `bash scripts/install_mac.sh --venv`).

**Note:** Camera SDK (Basler Pylon) must still be installed separately -- see platform instructions below.

### Manual Install: Windows

1. **Install Python 3.11+** from [python.org](https://www.python.org/downloads/)
   - Check "Add python.exe to PATH"
   - Check "Install launcher for all users"
   - Click "Install Now"
   - Restart your computer after installation

2. **Install camera SDK**
   - [Basler Pylon](https://docs.baslerweb.com/pylon-software-suite)
   - [IDS Peak](https://en.ids-imaging.com/ids-peak.html) (if using an IDS camera)

3. **Download LumaViewPro**
   - Download the ZIP from GitHub and extract it, or:
   ```
   git clone https://github.com/EtalumaSupport/LumaViewPro.git
   ```

4. **Install Python dependencies**
   - Open Command Prompt and navigate to the LumaViewPro folder:
   ```
   cd path\to\LumaViewPro
   pip install -r requirements.txt
   ```

5. **Run LumaViewPro**
   ```
   python lumaviewpro.py
   ```

### Manual Install: macOS

1. **Install Python 3.11+**
   - Download from [python.org](https://www.python.org/downloads/macos/), or:
   ```bash
   brew install python@3.13
   ```

2. **Install camera SDK**
   - [Basler Pylon for macOS](https://docs.baslerweb.com/pylon-software-suite)
   - [IDS Peak for macOS](https://en.ids-imaging.com/ids-peak.html) (if using an IDS camera)

3. **Download LumaViewPro**
   ```bash
   git clone https://github.com/EtalumaSupport/LumaViewPro.git
   cd LumaViewPro
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or use a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Run LumaViewPro**
   ```bash
   python lumaviewpro.py
   ```

### Manual Install: Linux

1. **Install Python 3.11+ and system dependencies**

   Ubuntu/Debian:
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip python3-venv
   ```

   Fedora:
   ```bash
   sudo dnf install python3 python3-pip
   ```

   Kivy may require additional system packages:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
   ```

2. **Install camera SDK**
   - [Basler Pylon for Linux](https://docs.baslerweb.com/pylon-software-suite)
   - [IDS Peak for Linux](https://en.ids-imaging.com/ids-peak.html) (if using an IDS camera)

3. **Download LumaViewPro**
   ```bash
   git clone https://github.com/EtalumaSupport/LumaViewPro.git
   cd LumaViewPro
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or use a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **USB device permissions** (required for microscope communication)
   ```bash
   sudo usermod -a -G dialout $USER
   # Log out and back in for the change to take effect
   ```

6. **Run LumaViewPro**
   ```bash
   python lumaviewpro.py
   ```

## Optional: Java and ImageJ Integration

LumaViewPro supports ImageJ/FIJI integration for advanced image analysis. This is optional and not required for normal operation.

1. Install [Azul Java 8 JDK+FX](https://www.azul.com/downloads/?version=java-8-lts&package=jdk-fx#zulu) for your OS
   - During installation, select the option to set the `JAVA_HOME` environment variable
   - On Apple Silicon (ARM64) Macs, install the ARM64 JDK — the JVM architecture must match your Python architecture
2. Install [Apache Maven 3.9.8+](https://maven.apache.org/download.cgi) (Binary Zip Archive)
   - Add the Maven `bin/` folder to your `PATH`

## Development

To run the test suite or build release packages, install the development dependencies:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Updating

**Windows installer:** Download and run the latest installer from the [GitHub Releases](https://github.com/EtalumaSupport/LumaViewPro/releases) page.

**From source (git):**
```bash
cd LumaViewPro
git pull
pip install -r requirements.txt
```

**From source (ZIP):** Download and extract the latest version from GitHub and re-run `pip install -r requirements.txt`.

## License

Copyright 2023-2026, Etaluma, Inc. MIT License. See [docs/LICENSE](docs/LICENSE) for details.

**Third-party dependencies:** LumaViewPro bundles and uses software under several licenses. Full per-library attribution, license texts, and upstream source pointers are in [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md). Summary:

- **Permissive (MIT / BSD / Apache-2.0):** Kivy, NumPy, SciPy, pandas, scikit-image, OpenCV, Matplotlib, xarray, tifffile, imagecodecs, pyserial, psutil, numba, and the ImageJ-integration packages (JPype1, pyimagej, scyjava, jgo) — all compatible with LVP's MIT license.
- **LGPL-2.1 (FFmpeg via PyAV):** The video I/O path links dynamically against FFmpeg's LGPL-licensed shared libraries, which are bundled alongside the application. FFmpeg source is available at https://ffmpeg.org/download.html and the LGPL-2.1 text is at `docs/licenses/LICENSE.LGPL-2.1.txt`.
- **Proprietary camera SDKs (Basler Pylon, IDS Peak):** Separately installed by the user or chained via the Windows installer; each surfaces its own EULA during install. The `pypylon` and `ids-peak` Python wrappers are BSD-3.
- **Optional ImageJ runtime (Amazon Corretto JDK, Apache Maven):** Corretto is OpenJDK under GPLv2 with the Classpath Exception; Maven is Apache-2.0. Both are bundled only when the Windows Bundle installer is used.
