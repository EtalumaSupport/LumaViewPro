# LumaViewPro Build Instructions

**Updated:** 2026-03-31

## One-Time Setup (Windows)

### 1. Install prerequisites
- **Python 3.12+**
- **Git** (in PATH)
- **WiX Toolset v5**: `dotnet tool install --global wix --version 5.0.2`
  - Then add UI extension: `wix extension add -g WixToolset.UI.wixext`
  - Note: WiX 6.0.2 broke the UI extension (no Install Complete dialog). Use v5.
- **.NET SDK** (required for WiX)

### 2. Create the build folder
Pick any location; the build script runs relative to itself, with no hardcoded paths.

```powershell
mkdir D:\Builds\LumaViewPro
mkdir D:\Builds\LumaViewPro\dependencies
```

### 3. Add dependencies
Download and place these in your build folder's `dependencies\`:

**Required:**
- `apache-maven-*\` — Extract from the Apache Maven binary zip (any version). Bundled into the installed app for ImageJ support. The build script detects the folder name automatically.

**Optional (for Bundle installer):**
- `pylon_USB_Camera_Driver.msi` — Basler Pylon USB Camera Driver MSI
- `amazon-corretto-8-xxx-jdk.msi` — Amazon Corretto 8 JDK Windows x64 MSI

If the Pylon or Corretto MSIs are missing, the Bundle installer is skipped but the standalone MSI still builds.

### 4. Get the build script

```powershell
cd D:\Builds\LumaViewPro
git clone --depth 1 --branch 4.0.0-beta https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1
rmdir _getscript -Recurse -Force
```

Your folder should look like:

```text
D:\Builds\LumaViewPro\
|-- build.ps1
`-- dependencies\
    |-- apache-maven-3.9.8\   (or any version)
    |-- pylon_USB_Camera_Driver.msi      (optional)
    `-- amazon-corretto-8-xxx-jdk.msi    (optional)
```

## Building a Package

```powershell
cd D:\Builds\LumaViewPro
.\build.ps1
```

The script:
1. Prompts for package type (`Dev` or `Release`). Dev reuses cached `buildvenv`; Release recreates from scratch.
2. Shows available branches — select one (e.g., `4.0.0-beta`)
3. Clones the selected branch from GitHub
4. Reads the version from `version.txt` (e.g., `4.0.0-beta3`)
5. Creates/reuses `buildvenv` and installs `requirements.txt` + `requirements-dev.txt`
6. Builds the EXE with PyInstaller 6.19
7. Copies Apache Maven into the install directory
8. Builds the MSI with WiX (includes Install Complete dialog via WixUI_Minimal)
9. Builds the Bundle installer if Pylon and Corretto MSIs are present
10. Cleans up temp files

### Output
All output is created next to `build.ps1`:

```text
D:\Builds\LumaViewPro\exe_artifacts\LumaViewPro-4.0.0-beta3\
|-- LumaViewPro-4.0.0-beta3.msi          (standalone installer)
`-- LumaViewPro-4.0.0-beta3-setup.exe    (bundle with Pylon + Corretto)
```

Previous builds are preserved in `exe_artifacts\` and are not auto-deleted.

## Before Each Build

The version in `version.txt` determines the build name. To bump the version:

1. Edit `version.txt` line 1 in the repo (e.g., `4.0.0-beta4`)
2. Commit and push
3. Run `.\build.ps1`

The beta number should increase with each EXE build so testers can identify which build they are running. The pre-commit hook automatically updates the timestamp on line 2.

## Key Dependencies

The build installs these automatically via pip:

| Package | Purpose |
|---------|---------|
| PyInstaller 6.19 | EXE packaging |
| av (PyAV) | H.264/MP4 video encoding |
| Kivy 2.3.1 | GUI framework |
| numpy, scipy, scikit-image | Image processing |
| opencv-python-headless | Image/video processing (no GUI) |
| pypylon, ids-peak | Camera SDKs |

## WiX Installer Details

The installer uses WiX v5 with three .wxs files:
- `Package.wxs` — MSI definition, shortcuts, environment variables, Install Complete dialog
- `Folders.wxs` — Directory structure
- `Bundle.wxs` — All-in-one setup.exe wrapping LVP + Pylon + Corretto

The build script passes dynamic variables to WiX:
- `ProductName` — from version.txt (e.g., "LumaViewPro 4.0.0-beta3")
- `Version` — numeric portion (e.g., "4.0.0")
- `MavenFolderName` — detected from dependencies/ (e.g., "apache-maven-3.9.8")
- `InstallFolderDir` — PyInstaller output directory

Bundle installs in order: Corretto → Pylon → LumaViewPro.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `wix not found` | Run `dotnet tool install --global wix --version 5.0.2`, then restart PowerShell |
| `UI extension error` | Run `wix extension add -g WixToolset.UI.wixext` |
| `python not found` | Install Python 3.12+ and make sure it is available through `py`, `python`, or `python3` |
| `git clone failed` | Check that the branch exists and that the machine has network access |
| `pip install failed` | Check internet access; use `Release` package type to force clean venv |
| `PyInstaller failed` | Re-run after dependency install succeeds |
| `Apache Maven not found` | Download Maven and extract to `dependencies\apache-maven-*\` |
| `Bundle skipped` | Put the Pylon and Corretto MSIs in `dependencies\` |
| `Permission denied` | Close any running LumaViewPro instance; try PowerShell as Administrator |

## Updating the Build Script

If `build.ps1` has been updated in the repo, re-grab it:

```powershell
git clone --depth 1 --branch 4.0.0-beta https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1 -Force
rmdir _getscript -Recurse -Force
```

There is no separate `update_build_script.ps1` — just re-clone and copy.

## Future: Code Signing

Once Etaluma has code signing certificates:
- Windows: `signtool sign` on EXE, MSI, and Bundle after build
- macOS: `codesign` + `xcrun notarytool` + `xcrun stapler staple`

This eliminates SmartScreen/Gatekeeper warnings for end users.
