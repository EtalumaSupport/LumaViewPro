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
mkdir C:\LumaViewPro\dependencies
```

### 3. Add dependencies
Download and place in `C:\LumaViewPro\dependencies\`:

**Required:**
- `apache-maven-3.9.8\` — Extract from [Apache Maven download](https://maven.apache.org/download.cgi) (Binary Zip Archive). Bundled into the installed app for ImageJ support.

**Optional (for Bundle installer):**
- `pylon_USB_Camera_Driver.msi` — [Basler Pylon SDK](https://docs.baslerweb.com/pylon-software-suite) USB Camera Driver
- `amazon-corretto-8-xxx-jdk.msi` — [Amazon Corretto 8 JDK](https://docs.aws.amazon.com/corretto/latest/corretto-8-ug/downloads-list.html)

If the Pylon/Corretto MSIs are missing, the Bundle installer is skipped but the standalone MSI still builds.

### 4. Get the build script
```powershell
cd C:\LumaViewPro
git clone --depth 1 --branch main https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1
rmdir _getscript -Recurse -Force
```

Your folder should look like:
```
C:\LumaViewPro\
├── build.ps1
└── dependencies\
    ├── README.md
    ├── apache-maven-3.9.8\
    ├── pylon_USB_Camera_Driver.msi      (optional)
    └── amazon-corretto-8-xxx-jdk.msi    (optional)
```

---

## Building a Release

```powershell
cd C:\LumaViewPro
.\build.ps1
```

Select the branch when prompted (e.g., `4.0.0-beta` or `main`).

The script:
1. Clones the branch from GitHub
2. Reads the version from `version.txt` (e.g., `4.0.0-beta2`)
3. Builds the EXE with PyInstaller
4. Copies Apache Maven into the install directory
5. Builds the MSI with WiX
6. Builds the Bundle installer (if Pylon + Corretto MSIs are present)
7. Cleans up temp files

### Output
```
C:\LumaViewPro\builds\LumaViewPro-4.0.0-beta2\
├── LumaViewPro-4.0.0-beta2.msi          ← standalone installer
└── LumaViewPro-4.0.0-beta2-setup.exe    ← bundle with Pylon + Corretto
```

---

## Before Each Build

The version in `version.txt` determines the build name. To bump the version:

1. Edit `version.txt` in the repo (e.g., `4.0.0-beta3`)
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
| `Apache Maven not found` | Download Maven 3.9.8 and extract to `dependencies\apache-maven-3.9.8\` |
| `Bundle skipped` | Put Pylon + Corretto MSIs in `dependencies\` |
| `Permission denied` | Close any running LumaViewPro, try running PowerShell as Administrator |

---

## Updating the Build Script

If `build.ps1` has been updated in the repo, re-grab it:
```powershell
cd C:\LumaViewPro
.\update_build_script.ps1
```

Or manually:
```powershell
git clone --depth 1 --branch main https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1 -Force
rmdir _getscript -Recurse -Force
```

---

## Notes

- Each build clones fresh from GitHub — you always build exactly what's pushed
- Previous builds are preserved in `C:\LumaViewPro\builds\` (not auto-deleted)
- Temp files in `C:\LumaViewPro\_tmp\` are auto-cleaned after each build
- The build script can also accept the branch as a parameter: `.\build.ps1 -Branch 4.0.0-beta`
