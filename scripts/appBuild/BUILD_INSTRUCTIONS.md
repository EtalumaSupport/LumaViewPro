# LumaViewPro Build Instructions

These instructions describe the Windows release package build driven by
`build.ps1`. The script clones the selected branch from GitHub, creates or
reuses a build virtual environment, packages LumaViewPro with PyInstaller, then
builds WiX installer outputs.

## Outputs

All build outputs are written under `exe_artifacts\LumaViewPro-<version>\` next
to `build.ps1`.

| Output | Purpose | Ship to customers? |
|--------|---------|--------------------|
| `LumaViewPro-<version>.msi` | Standalone LumaViewPro installer. It has minimal install UI and installs the LVP application folder, Start Menu shortcut, environment variables, bundled Apache Maven folder, and app files. | Usually no. Use for internal testing, debugging, or cases where prerequisites are already handled separately. |
| `LumaViewPro-<version>-setup.exe` | Main customer installer. This is the WiX Bundle with the full installer UI. It runs the LVP MSI and also includes the Basler Pylon USB driver installer and Amazon Corretto Java SDK installer in one install flow. | Yes. This is the primary file to ship when building a release package. |

The `-setup.exe` bundle is only created when both dependency MSIs are present:

- Basler Pylon USB Camera Driver MSI
- Amazon Corretto 8 JDK Windows x64 MSI

Apache Maven is not a separate MSI. The build script copies the
`apache-maven-3.9.8\` folder into the LumaViewPro install directory, and the
LVP MSI adds Maven's `bin` folder to PATH.

## One-Time Setup

### 1. Install Tools

- Python 3.12 or newer
- Git, available in PATH
- .NET SDK, required by WiX
- WiX Toolset v6 — pinned. **Do not use WiX v7.** This build's `Bundle.wxs`
  is authored for the v4–v6 `WixToolset.Bal.wixext` API; v7 restructured
  `WixStdBA` and the Burn ↔ BootstrapperApplication protocol added a
  required `scope` field, so `-setup.exe` bundles built against v7 fail at
  runtime with `0x80070057: Failed to read plan scope of BAEnginePlan args`
  the moment the user clicks Install. Install the pinned version:

```powershell
dotnet tool install --global wix --version 6.0.0
wix --version    # confirm 6.x — must NOT be 7.x
```

If you already have a newer version installed (for example v7 from
`dotnet tool install --global wix` without a version pin), downgrade:

```powershell
dotnet tool uninstall --global wix
dotnet tool install --global wix --version 6.0.0
wix extension remove --global WixToolset.Bal.wixext   # clear v7 BAL cache
wix --version
```

The build script auto-installs `WixToolset.Bal.wixext` to match whatever
`wix` it finds, so once `wix` is back on v6 the bundle build picks up the
matching v6 BAL automatically.

Do not downgrade to WiX v5 either. In this build flow the standalone MSI
intentionally has minimal UI, and the customer-facing install UI lives in
the WiX Bundle `-setup.exe`. v6 is the only supported lane.

`build.ps1` manages its own build virtual environment and installs
`requirements-dev.txt`. Do not install PyInstaller globally for this build.

### 2. Create A Build Folder

Use a simple local path outside OneDrive. The build script can run from any
folder, but long or synced paths are more likely to cause Windows build issues.

```powershell
mkdir C:\Users\user\LVP\appBuild
mkdir C:\Users\user\LVP\appBuild\dependencies
```

### 3. Add Build Dependencies

Put these files/folders in `dependencies\` next to `build.ps1`:

```text
dependencies\
|-- README.md
|-- apache-maven-3.9.8\
|-- pylon_USB_Camera_Driver.msi
`-- amazon-corretto-8-xxx-jdk.msi
```

Required for the standalone MSI:

- `apache-maven-3.9.8\` extracted from the Apache Maven binary zip

Required for the customer `-setup.exe` bundle:

- `pylon_USB_Camera_Driver.msi`
- `amazon-corretto-8-xxx-jdk.msi`

If Pylon or Corretto are missing, the build still creates the standalone MSI but
skips the customer `-setup.exe` bundle.

### 4. Copy The Build Script

From your build folder, clone the branch that contains the build script version
you want, then copy `build.ps1` and the dependency README out of the clone:

```powershell
cd C:\Users\dovyd\LVP\appBuild
git clone --depth 1 --branch 4.1.0-dev-exe https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1
copy _getscript\scripts\appBuild\dependencies\README.md .\dependencies\README.md
rmdir _getscript -Recurse -Force
```

Use the branch name that should supply the build script. The build script itself
also clones a branch during packaging, so official builds must be made from
committed and pushed code.

Important: build-system changes must be committed before testing the package
flow. There are two separate copies involved:

- The `build.ps1` you run is the copy in the external build folder.
- During packaging, that script clones the selected branch and uses the WiX
  files and PyInstaller spec from the cloned branch.

If a fix to `build.ps1`, `build_exe\wix\*.wxs`, or
`config\lumaviewpro_win_release.spec` only exists as an uncommitted IDE change,
the package build will not see it. Commit the build-system change to the branch,
refresh the external `build.ps1` copy if that file changed, then build that same
branch.

## Building A Package

Run:

```powershell
cd C:\Users\dovyd\LVP\appBuild
.\build.ps1
```

The script asks for:

1. Build directory: normally keep the current folder.
2. Package type:
   - `Dev` reuses cached `buildvenv` for faster repeat builds.
   - `Release` deletes and recreates `buildvenv` for a clean package.
3. Branch to build, for example `4.1.0-dev-exe`, `4.0.0-beta`, or `main`.

The selected branch is cloned fresh from GitHub. Local uncommitted changes are
not included in the package, including local edits to the WiX files or
PyInstaller spec.

The script then:

1. Reads the version from `version.txt`.
2. Installs build dependencies into `buildvenv`.
3. Builds the app folder with PyInstaller.
4. Copies Apache Maven into the app install folder.
5. Builds `LumaViewPro-<version>.msi`.
6. Builds `LumaViewPro-<version>-setup.exe` if Pylon and Corretto are present.
7. Removes temporary clone/build files.

Example output:

```text
C:\Users\dovyd\LVP\appBuild\exe_artifacts\LumaViewPro-4.1.0-dev\
|-- LumaViewPro-4.1.0-dev.msi
`-- LumaViewPro-4.1.0-dev-setup.exe
```

Previous completed packages are preserved in `exe_artifacts\`.

## Release Checklist

Before a customer build:

1. Confirm `version.txt` has the release version testers should see.
2. Commit and push the branch to be packaged.
3. Confirm `dependencies\` contains Maven, Pylon, and Corretto.
4. Run `.\build.ps1` and choose `Release`.
5. Verify both output files exist.
6. Ship `LumaViewPro-<version>-setup.exe`.
7. Keep the `.msi` for internal testing or troubleshooting.

## Updating The Build Script

If `build.ps1` changes in the repo, refresh the copy in your build folder:

```powershell
git clone --depth 1 --branch 4.1.0-dev-exe https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1 -Force
copy _getscript\scripts\appBuild\dependencies\README.md .\dependencies\README.md -Force
rmdir _getscript -Recurse -Force
```

Use the branch that contains the desired build script.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `wix not found` | Install WiX v6 with `dotnet tool install --global wix --version 6.0.0`, then restart PowerShell. |
| `wix --version` reports `7.x.x` (or build aborts with "WiX 7 is not supported") | Downgrade per the One-Time Setup section. v7 changed the BootstrapperApplication API and produces bundles that fail at runtime with `0x80070057`. |
| Bundle `-setup.exe` fails at runtime with `0x80070057: Failed to read plan scope of BAEnginePlan args` (or `Failed to load splash screen bitmap`) | The bundle was built against WiX v7. Confirm `wix --version` shows 6.x, then rebuild. |
| WiX `WixUI` or UI extension errors | This build should not require `WixToolset.UI.wixext` for the standalone MSI. Make sure the branch being packaged uses the restored minimal-UI `Package.wxs`; the full install UI belongs to the `-setup.exe` bundle. |
| `python not found` | Install Python 3.12+ and make sure `py`, `python`, or `python3` is available. |
| `git clone failed` | Check network access and confirm the selected branch exists on GitHub. |
| `pip install failed` | Check internet access and package pins. Try a `Release` build to recreate `buildvenv`. |
| `PyInstaller failed` | Fix dependency install errors first, then rerun the build. |
| `Apache Maven not found` | Extract Maven to `dependencies\apache-maven-3.9.8\`. |
| `Bundle skipped` | Add both Pylon and Corretto MSIs to `dependencies\`. The MSI will still build, but the customer `-setup.exe` will not. |
| `Permission denied` | Close running LumaViewPro instances and use a local non-OneDrive build folder. If needed, run PowerShell as Administrator. |

## Notes

- The build script uses paths relative to the folder containing `build.ps1`.
- The selected build root is remembered in `.build_config` next to `build.ps1`.
- The build script accepts parameters, for example:

```powershell
.\build.ps1 -Branch 4.1.0-dev-exe -BuildType Release
```
