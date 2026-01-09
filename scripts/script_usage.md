# LumaViewPro Build Script Usage Guide

This guide provides complete instructions for creating Windows installers (MSI and Bundle/EXE) for LumaViewPro using the automated build script.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Configuration](#initial-configuration)
3. [Running the Build Script](#running-the-build-script)
4. [Understanding the Output](#understanding-the-output)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. WiX Toolset v4
WiX Toolset is required to build MSI and Bundle installers.

**Check if installed:**
```powershell
wix --version
```

**If not installed:**
```powershell
dotnet tool install --global wix
```

**Verify installation:**
```powershell
wix --version
```
Expected output: `v4.x.x` or similar

**Note about WiX BAL extension:**
The WiX Bootstrapper Application Library (BAL) extension required for Bundle creation is **automatically included** in the `scripts/build_exe/deps/` folder. The build script will use this bundled extension automatically. No manual installation is required.

### 2. Python & PyInstaller
- Python 3.x must be installed and in PATH
- PyInstaller is required to create the executable
- All Python dependencies from `requirements.txt` must be installed

**Verify Python:**
```powershell
python --version
```

**Install PyInstaller (if needed):**
```powershell
pip install pyinstaller
```

### 3. Git
Git must be installed and in PATH for the script to clone the repository.

**Verify Git:**
```powershell
git --version
```

### 4. External MSI Dependencies (for Bundle creation)
To create the complete bundle installer, you need:
- **Pylon Camera Driver MSI** - For Basler camera support
- **Amazon Corretto 8 JDK MSI** - Java runtime for ImageJ functionality

Download these and note their file paths for configuration.

---

## Initial Configuration

### 1. Configure Dependency Paths
Edit `scripts/config/build_dependencies.json` to specify paths to external MSI files:

```json
{
    "pylon_driver_msi": "C:\\Path\\To\\pylon_USB_Camera_Driver.msi",
    "corretto_jdk_msi": "C:\\Path\\To\\amazon-corretto-8.442.06.1-windows-x64-jdk.msi",
    "wix_bal_extension": ""
}
```

**Notes:**
- Use double backslashes (`\\`) in paths
- Use absolute paths
- `pylon_driver_msi` and `corretto_jdk_msi`: Required for Bundle creation. If empty, you'll be prompted or bundle will be skipped
- `wix_bal_extension`: Optional. Leave empty to use the bundled extension from deps folder (recommended)
- If paths are configured and valid, you won't be prompted during build

**WiX BAL Extension Priority:**
1. **Bundled** - Uses `scripts/build_exe/deps/WixToolset.BootstrapperApplications.wixext.dll` (default)
2. **Config** - Uses custom path from `wix_bal_extension` field if specified
3. **Package Manager** - Falls back to `wix extension add` if neither above are available

### 2. Important: Working Directory
⚠️ **Run the script from a directory path that does NOT contain spaces or special characters (like dashes in "OneDrive - Sample").**

**Good:**
- `C:\Users\username\Downloads\scripts\`
- `C:\Build\LVP\`
- `D:\Projects\`

**Problematic:**
- `C:\Users\username\OneDrive - Sample\...` (contains dash)
- `C:\My Projects\...` (contains space)

If you encounter WiX argument parsing errors, this is likely the cause.

---

## Running the Build Script

### Basic Usage
Navigate to the scripts directory and run:

```powershell
cd C:\Path\To\LumaViewPro\scripts
.\build_win_release.ps1
```

### With ZIP Creation
To also create source and executable ZIP archives:

```powershell
.\build_win_release.ps1 -CreateZip
```

### Interactive Prompts
The script will prompt you for:

1. **Branch name** - Git branch to build from (e.g., `beta-3.0`, `main`)
2. **Version** - Version string (e.g., `3.0.0-beta`, `3.0.0`)
   - For WiX compatibility, use format: `X.X.X` or `X.X.X-suffix`
   - The numeric portion must be in `X.X.X` format
3. **Pylon MSI path** - Only prompted if not configured in JSON
4. **Corretto MSI path** - Only prompted if not configured in JSON

### What the Script Does

1. ✓ Verifies WiX Toolset v4 is installed
2. ✓ Clones the specified branch from GitHub
3. ✓ Validates version matches `version.txt` in repository
4. ✓ Copies license files to top level
5. ✓ (Optional) Creates source ZIP and TAR.GZ archives
6. ✓ Runs PyInstaller to build executable
7. ✓ (Optional) Creates executable ZIP archive
8. ✓ Copies Apache Maven from deps to build output
9. ✓ Builds MSI package using WiX
10. ✓ (Optional) Builds Bundle/EXE installer if dependencies are provided

---

## Understanding the Output

### Directory Structure
After a successful build, you'll find:

```
scripts/
├── tmp/                              # Temporary build files (can be deleted)
│   ├── LumaViewPro-X.X.X/           # Cloned repository
│   └── artifacts/                    # Source archives (if -CreateZip used)
│       ├── LumaViewPro-X.X.X-source.zip
│       ├── LumaViewPro-X.X.X-source.tar.gz
│       └── LumaViewPro-X.X.X.zip    # Executable archive
│
└── exe_artifacts/                    # MAIN OUTPUT DIRECTORY
    ├── package/
    │   └── LumaViewPro-X.X.X.msi    # ✓ MSI Installer
    └── bundle/
        └── LumaViewPro-X.X.X-setup.exe  # ✓ Bundle Installer (if created)
```

### Output Files

#### MSI Package (`exe_artifacts/package/`)
- **Standalone installer** for LumaViewPro only
- Does NOT include Java or Pylon drivers
- Users must install dependencies separately
- Installs to: `C:\Program Files\LumaViewPro-X.X.X`

#### Bundle Setup (`exe_artifacts/bundle/`)
- **Complete installer** with all dependencies
- Includes: LumaViewPro MSI, Pylon Driver MSI, Amazon Corretto JDK MSI
- Recommended for end-user distribution
- Single executable that installs everything

---

## Troubleshooting

### WiX Error: "Additional argument '-' was unexpected"
**Cause:** Working directory path contains spaces or dashes (e.g., "OneDrive - Cal Poly")

**Solution:** Run the script from a directory without spaces or special characters in the path.

### Error: "WiX Toolset v4 not found"
**Solution:** 
```powershell
dotnet tool install --global wix
```
Then restart your PowerShell session.

### Error: "version.txt contents do not match supplied version"
**Cause:** The version you entered doesn't match the version in the repository's `version.txt` file.

**Solution:** Check the `version.txt` file in the branch you're building and enter that exact version.

### Error: "Apache Maven not found"
**Cause:** The `scripts/build_exe/deps/apache-maven-3.9.8` directory doesn't exist.

**Solution:** Download Apache Maven 3.9.8 and extract it to `scripts/build_exe/deps/apache-maven-3.9.8/`

### Bundle Creation Skipped
**Cause:** Either Pylon or Corretto MSI path is missing or invalid.

**Solution:** 
1. Verify paths in `config/build_dependencies.json` are correct
2. Check that the MSI files actually exist at those locations
3. Use absolute paths with double backslashes

### WiX BAL Extension Errors
**Cause:** The bundled WiX BAL extension in deps folder is missing or corrupted.

**Solution:** 
1. Verify `scripts/build_exe/deps/WixToolset.BootstrapperApplications.wixext.dll` exists
2. If missing, re-download or copy from a working WiX 5.0.1 installation
3. Alternatively, specify a custom extension path in `config/build_dependencies.json`:
   ```json
   {
       "wix_bal_extension": "C:\\Path\\To\\WixToolset.BootstrapperApplications.wixext.dll"
   }
   ```
4. As last resort, leave `wix_bal_extension` empty and ensure `wix extension add -g WixToolset.Bal.wixext` works

### PyInstaller Errors
**Cause:** Missing Python dependencies or PyInstaller issues.

**Solution:**
```powershell
pip install -r requirements.txt
pip install --upgrade pyinstaller
```

### Permission Errors
**Cause:** Trying to overwrite files or directories in use.

**Solution:** 
- Close LumaViewPro if it's running
- Delete the `tmp` and `exe_artifacts` directories manually
- Run PowerShell as Administrator if needed

---

## Advanced Options

### Custom Output Location
The script creates `exe_artifacts` in the directory where `build_win_release.ps1` is located. To change this:
1. Copy the script to your desired location
2. Run from there

### Build Configuration
The WiX files are located in `scripts/build_exe/wix/`:
- `Package.wxs` - MSI package configuration
- `Bundle.wxs` - Bundle installer configuration
- `Folders.wxs` - Installation directory structure

Modify these files to customize:
- Installation directory structure
- Shortcuts and registry keys
- Upgrade behavior
- UI customization

---

## Notes

- The script creates absolute paths internally, so the output location is predictable
- Temporary files in `tmp/` can be safely deleted after build completes
- The script uses the WiX files from your **current working directory**, not from the cloned repo
- Version string must contain numeric version in `X.X.X` format for WiX compatibility
- Product name can include suffixes (e.g., "LumaViewPro-3.0.0-beta") but version field in MSI must be numeric only

---

## Quick Reference

**Minimal build (MSI only):**
```powershell
.\build_win_release.ps1
```

**Full build with archives:**
```powershell
.\build_win_release.ps1 -CreateZip
```

**Main outputs:**
- MSI: `scripts/exe_artifacts/package/LumaViewPro-X.X.X.msi`
- Bundle: `scripts/exe_artifacts/bundle/LumaViewPro-X.X.X-setup.exe`