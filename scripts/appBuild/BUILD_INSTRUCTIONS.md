# LumaViewPro Build Instructions

## One-Time Setup (Windows)

### 1. Install prerequisites
- **Python 3.12+** with `pip install -r requirements.txt` and `pip install pyinstaller`
- **Git** (in PATH)
- **WiX Toolset v6**: `dotnet tool install --global wix`
- **.NET SDK** (required for WiX)

### 2. Create the build folder
```powershell
mkdir C:\LumaViewPro
mkdir C:\LumaViewPro\prereqs
```

### 3. Add dependency MSIs to prereqs folder
Download and place in `C:\LumaViewPro\prereqs\`:
- `pylon_USB_Camera_Driver.msi` — Basler Pylon camera driver
- `amazon-corretto-8-xxx-jdk.msi` — Amazon Corretto 8 JDK (for ImageJ)

The script auto-detects these by filename pattern. If they're missing, the Bundle installer is skipped but the MSI still builds.

### 4. Get the build script
```powershell
cd C:\LumaViewPro
git clone --depth 1 --branch 4.0.0-beta https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1
rmdir _getscript -Recurse -Force
```

Your folder should look like:
```
C:\LumaViewPro\
├── build.ps1
└── prereqs\
    ├── pylon_USB_Camera_Driver.msi
    └── amazon-corretto-8-xxx-jdk.msi
```

---

## Building a Release

```powershell
cd C:\LumaViewPro
.\build.ps1
```

When prompted, enter the branch name (e.g., `4.0.0-beta`).

The script:
1. Clones the branch from GitHub
2. Reads the version from `version.txt` (e.g., `4.0.0-beta2`)
3. Builds the EXE with PyInstaller
4. Builds the MSI with WiX
5. Builds the Bundle installer (if prereqs are present)
6. Cleans up temp files

### Output
```
C:\LumaViewPro\builds\LumaViewPro-4.0.0-beta2\
├── LumaViewPro-4.0.0-beta2.msi          ← standalone installer
└── LumaViewPro-4.0.0-beta2-setup.exe    ← bundle with Pylon + Corretto
```

---

## Before Each Build

The version in `version.txt` determines the build name. To bump the version:

1. Edit `version.txt` in the repo (e.g., `4.0.0-beta3 (2026-03-28 09:00)`)
2. Commit and push
3. Run `.\build.ps1`

The beta number should increase with each EXE build so testers can identify which build they're running.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `wix not found` | `dotnet tool install --global wix`, restart PowerShell |
| `python not found` | Install Python 3.12+, make sure it's in PATH |
| `git clone failed` | Check branch name exists, check network |
| `PyInstaller failed` | Run `pip install -r requirements.txt` and `pip install --upgrade pyinstaller` |
| `Bundle skipped` | Put Pylon + Corretto MSIs in `C:\LumaViewPro\prereqs\` |
| `Permission denied` | Close any running LumaViewPro, try running PowerShell as Administrator |

---

## Notes

- Each build clones fresh from GitHub — you always build exactly what's pushed
- Previous builds are preserved in `C:\LumaViewPro\builds\` (not auto-deleted)
- Temp files in `C:\LumaViewPro\_tmp\` are auto-cleaned after each build
- The build script can also accept the branch as a parameter: `.\build.ps1 -Branch 4.0.0-beta`
