# LumaViewPro Build Instructions

## One-Time Setup (Windows)

### 1. Install prerequisites
- **Python 3.12+**
- **Git** (in PATH)
- **WiX Toolset v6**: `dotnet tool install --global wix`
- **.NET SDK** (required for WiX)

`build.ps1` manages the build virtual environment itself and installs `requirements-dev.txt`, so there is no separate `setup.ps1` step and no global `pip install` step required for the build machine.
`build.ps1` also prompts for package type:
- `Dev` reuses a cached `buildvenv` for faster repeat packaging runs
- `Release` recreates that build environment from scratch before packaging

### 2. Create the build folder
Pick any location; the build script runs relative to itself, with no hardcoded paths.

```powershell
mkdir D:\Builds\LumaViewPro
mkdir D:\Builds\LumaViewPro\dependencies
```

### 3. Add dependencies
Download and place these in your build folder's `dependencies\`:

**Required:**
- `apache-maven-3.9.8\` - Extract from the Apache Maven binary zip. It is bundled into the installed app for ImageJ support.

**Optional (for Bundle installer):**
- `pylon_USB_Camera_Driver.msi` - Basler Pylon USB Camera Driver MSI
- `amazon-corretto-8-xxx-jdk.msi` - Amazon Corretto 8 JDK Windows x64 MSI

If the Pylon or Corretto MSIs are missing, the Bundle installer is skipped but the standalone MSI still builds.

### 4. Get the build script

```powershell
cd D:\Builds\LumaViewPro
git clone --depth 1 --branch main https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1
copy _getscript\scripts\appBuild\dependencies\README.md .\dependencies\README.md
rmdir _getscript -Recurse -Force
```

Your folder should look like:

```text
D:\Builds\LumaViewPro\
|-- build.ps1
`-- dependencies\
    |-- README.md
    |-- apache-maven-3.9.8\
    |-- pylon_USB_Camera_Driver.msi      (optional)
    `-- amazon-corretto-8-xxx-jdk.msi    (optional)
```

## Building a Package

```powershell
cd D:\Builds\LumaViewPro
.\build.ps1
```

The script first shows the saved build directory and asks `Update build directory? [y/N]` so you can move the build root off the old `C:\LumaViewPro` path if you want. It remembers that choice in `.build_config` next to `build.ps1`.
It then asks whether this is a `Dev` package or `Release` package. Dev builds reuse the cached `buildvenv`; release builds recreate it from scratch.

Select the branch when prompted, for example `4.0.0-beta` or `main`.

The script:
1. Prompts for the build directory and remembers it
2. Prompts for package type (`Dev` or `Release`)
3. Clones the selected branch from GitHub
4. Reads the version from `version.txt` (for example `4.0.0-beta2`)
5. Reuses or recreates `buildvenv` depending on package type
6. Installs `requirements-dev.txt` into that venv
7. Builds the EXE with PyInstaller
8. Copies Apache Maven into the install directory
9. Builds the MSI with WiX
10. Builds the Bundle installer if Pylon and Corretto MSIs are present
11. Cleans up temp files

### Output
All output is created next to `build.ps1`:

```text
D:\Builds\LumaViewPro\exe_artifacts\LumaViewPro-4.0.0-beta2\
|-- LumaViewPro-4.0.0-beta2.msi
`-- LumaViewPro-4.0.0-beta2-setup.exe
```

Previous builds are preserved in `exe_artifacts\` and are not auto-deleted.

## Before Each Build

The version in `version.txt` determines the build name. To bump the version:

1. Edit `version.txt` in the repo, for example `4.0.0-beta3`
2. Commit and push
3. Run `.\build.ps1`

The beta number should increase with each EXE build so testers can identify which build they are running.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `wix not found` | Run `dotnet tool install --global wix`, then restart PowerShell |
| `python not found` | Install Python 3.12+ and make sure it is available through `py`, `python`, or `python3` |
| `git clone failed` | Check that the branch exists and that the machine has network access |
| `pip install failed` | Verify internet access and that the pinned packages in `requirements.txt` and `requirements-dev.txt` are still available; use a `Release` package run to force a clean build environment if needed |
| `PyInstaller failed` | Re-run the build after the dependency install step succeeds; PyInstaller is installed into the managed build venv automatically |
| `Apache Maven not found` | Download Maven 3.9.8 and extract it to `dependencies\apache-maven-3.9.8\` |
| `Bundle skipped` | Put the Pylon and Corretto MSIs in `dependencies\` |
| `Permission denied` | Close any running LumaViewPro instance and try PowerShell as Administrator |

## Updating the Build Script

If `build.ps1` has been updated in the repo, re-grab it manually:

```powershell
git clone --depth 1 --branch main https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1 -Force
copy _getscript\scripts\appBuild\dependencies\README.md .\dependencies\README.md -Force
rmdir _getscript -Recurse -Force
```

## Notes

- The build script uses paths relative to where `build.ps1` lives, with no hardcoded locations
- Dev builds reuse a cached `buildvenv`, while release builds recreate it from scratch
- Each build clones fresh from GitHub, so you always build exactly what is pushed
- Temp files in `_tmp\` are auto-cleaned after each build
- The selected build root is remembered in `.build_config` next to `build.ps1`
- The build script also accepts the branch as a parameter: `.\build.ps1 -Branch 4.0.0-beta`
