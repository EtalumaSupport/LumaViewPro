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
| `LumaViewPro-<version>.msi` | Standalone LumaViewPro installer. It has minimal install UI and installs the LVP application folder, Start Menu shortcut, Desktop shortcut, environment variables, and app files. | Usually no. Use for internal testing, debugging, or cases where prerequisites are already handled separately. |
| `LumaViewPro-<version>-setup.exe` | Main customer installer. This is the WiX Bundle with the full installer UI. It runs the LVP MSI and also chains the Basler Pylon USB driver installer and (when present) the IDS Peak runtime installer in one install flow. | Yes. This is the primary file to ship when building a release package. |

The `-setup.exe` bundle is only created when the **required** dependency MSI is
present: the Basler Pylon USB Camera Driver MSI. Optional dependencies (IDS
Peak, FX2 driver) are silently skipped when their files are absent.

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

Use **`C:\LVP\appbuild`**. Avoid placing the build folder under your user
profile (`C:\Users\<user>\...`):

- A username with spaces (e.g. `Etaluma Microscope`) breaks several build
  steps that don't quote the path correctly — observed when this layout was
  tried previously.
- OneDrive-synced paths under the user profile cause Windows file-locking
  errors during PyInstaller and WiX runs.
- Long paths (`Documents\Projects\...`) trip `MAX_PATH` in some tools.

Create the folder once:

```powershell
mkdir C:\LVP\appbuild
mkdir C:\LVP\appbuild\dependencies
```

### 3. Add Build Dependencies

All build inputs go in **`C:\LVP\appbuild\dependencies\`** next to `build.ps1`.
Layout when fully populated:

```text
dependencies\
|-- README.md                            # ships with the build script
|-- pylon_USB_Camera_Driver.msi          # required for the bundle
|-- ids_peak_<version>.exe               # optional — IDS cameras
|-- setup.iss                            # optional — paired with ids_peak EXE
`-- fx2\
    |-- LumaScope_WinUSB.inf             # optional — LVC FX2 driver bind
    `-- libusb-1.0.dll                   # optional — pyusb backend on Windows
```

#### Where each dependency comes from

| File | Source | Notes |
|------|--------|-------|
| `pylon_USB_Camera_Driver.msi` | [baslerweb.com/en/downloads/software-downloads](https://www.baslerweb.com/en/downloads/software-downloads/) | Free MyBasler account required. Pick the standalone USB driver MSI (not the full Pylon SDK installer). |
| `ids_peak_<version>.exe` | [en.ids-imaging.com/download-peak.html](https://en.ids-imaging.com/download-peak.html) → Runtime variant (~26 MB) | Free MyIDS account required. Pick a runtime version that matches the `ids-peak` PyPI binding pinned in `requirements.txt` (`1.13.0.0.6` → runtime ≥ 2.18). |
| `setup.iss` | Generated locally, once per IDS Peak runtime version | Run `ids_peak_<version>.exe /r` on a Windows host with a clean install of that exact runtime; it records your interactive choices into `%WINDIR%\setup.iss`. Copy that file into `dependencies\` next to the EXE. |
| `fx2\LumaScope_WinUSB.inf` | Firmware repo, `fx2_firmware/build_deps/LumaScope_WinUSB.inf` | ~2 KB text file. Copy as-is. |
| `fx2\libusb-1.0.dll` | Firmware repo, `fx2_firmware/build_deps/libusb-1.0.dll` | x64 build, ~150 KB. Pinned to libusb v1.0.29 (VS2019/MS64). To upgrade, replace the file in the Firmware repo and update `fx2_firmware/build_deps/README.md`. |

Keep a local mirror of every account-gated download (Pylon, IDS Peak) in
your build artefact store. Do not depend on being logged into MyBasler /
MyIDS at build time.

#### Required for the standalone MSI

The standalone MSI builds from the LumaViewPro source alone -- no files in
`dependencies\` are required for it.

#### Required for the `-setup.exe` bundle

- `pylon_USB_Camera_Driver.msi`

If this is missing, the build still creates the standalone MSI but skips the
customer `-setup.exe` bundle.

### Optional Dependencies

The build script auto-detects these. If absent, the build skips the
relevant component silently — no failure. Drop them into `dependencies\` to
enable.

#### IDS Peak Runtime (USB3 driver for IDS cameras)

```text
dependencies\
|-- ids_peak_<version>.exe          # InstallShield runtime installer (~26 MB)
`-- setup.iss                        # Recorded silent-install response file
```

`ids_peak_*.exe` is the **Runtime** variant, not the full IDS peak
Comprehensive package. It contains drivers + GenTL transport layers only —
no Cockpit, no IPL Viewer, no SDK headers.

`setup.iss` must be recorded once per runtime version. On a clean Windows
host, run `ids_peak_<version>.exe /r` and step through the interactive
installer. It writes `%WINDIR%\setup.iss` capturing every choice. Copy that
file alongside the EXE in `dependencies\`. Re-record whenever you bump the
runtime version or change install options.

When both are present, the bundle chains an `ExePackage` that runs
`ids_peak_<version>.exe /s /f1"setup.iss"` per machine. When either is
missing, the bundle is built without IDS Peak (IDS cameras still need a
manual driver install on customer machines).

**Version pairing constraint:** IDS hard-couples the Python binding's
`genericAPI` major.minor to the runtime's. Mismatched runtime/binding
versions fail silently at camera enumeration. Confirm against the runtime's
`readme.html` shipped inside the EXE before locking in a pair.

#### FX2 WinUSB Driver (LVC cameras: LS560 / LS620 / LS720)

```text
dependencies\fx2\
|-- LumaScope_WinUSB.inf            # ~2 KB text file
`-- libusb-1.0.dll                  # ~150 KB native library
```

Two files together enable end-to-end FX2 support. Both are
build-time-optional and runtime-decoupled — `drivers/fx2driver.py` gates
its imports with `_FX2_AVAILABLE`, so absence degrades to "FX2 not
supported" rather than a crash.

- **`LumaScope_WinUSB.inf`** — binds inbox `WinUSB.sys` to the FX2 VID/PID
  `0x04B4:0x8613` / `0x04B4:0xEA17` so `pyusb` can open the device without
  Zadig. When present, the LVP MSI installs it to
  `<InstallFolder>\drivers\fx2\` and runs `pnputil /add-driver <inf> /install`
  during `InstallFiles` (deferred, runs as SYSTEM, errors ignored).

- **`libusb-1.0.dll`** — the native USB library that pyusb's `libusb1`
  backend loads via `ctypes` on Windows. PyInstaller bundles it next to
  `lumaviewpro.exe` so `ctypes.cdll.LoadLibrary("libusb-1.0.dll")` finds
  it inside the installed app.

When either file is absent, the build still succeeds but FX2 cameras
won't work on the resulting installer:

| INF | DLL | Result |
|---|---|---|
| ✓ | ✓ | Full FX2 support — driver auto-installed, app can talk to camera |
| ✗ | ✓ | App has FX2 libraries but customer must install driver via Zadig |
| ✓ | ✗ | Driver installed but bundled app fails at first FX2 access |
| ✗ | ✗ | FX2 unsupported (current 4.0.0-beta default) |

### 4. Copy The Build Script

From your build folder, clone the branch that contains the build script
version you want, then copy `build.ps1` and the dependency README out of
the clone:

```powershell
cd C:\LVP\appbuild
git clone --depth 1 --branch 4.0.0-beta https://github.com/EtalumaSupport/LumaViewPro.git _getscript
copy _getscript\scripts\appBuild\build.ps1 .\build.ps1
copy _getscript\scripts\appBuild\dependencies\README.md .\dependencies\README.md
rmdir _getscript -Recurse -Force
```

Use the branch name that should supply the build script. The build script
itself also clones a branch during packaging, so official builds must be
made from committed and pushed code.

**Important: build-system changes must be committed before testing the
package flow.** There are two separate copies involved:

- The `build.ps1` you run is the copy in the external build folder
  (`C:\LVP\appbuild\build.ps1`).
- During packaging, that script clones the selected branch and uses the
  WiX files and PyInstaller spec from the cloned branch.

If a fix to `build.ps1`, `build_exe\wix\*.wxs`, or
`config\lumaviewpro_win_release.spec` only exists as an uncommitted IDE
change, the package build will not see it. Commit the build-system change
to the branch, refresh the external `build.ps1` copy if that file changed,
then build that same branch.

## Building A Package

Run:

```powershell
cd C:\LVP\appbuild
.\build.ps1
```

The script asks for:

1. Build directory: normally keep the current folder.
2. Package type:
   - `Dev` reuses cached `buildvenv` for faster repeat builds.
   - `Release` deletes and recreates `buildvenv` for a clean package.
3. Branch to build. The interactive picker offers:
   - `[1] 4.0.0-beta` — current shipping beta line
   - `[2] 4.1.0-dev` — active development
   - `[3] main` — release
   - `[0] Enter custom branch` — anything else by name

The selected branch is cloned fresh from GitHub. Local uncommitted changes
are not included in the package, including local edits to the WiX files
or PyInstaller spec.

The script then:

1. Reads the version from `version.txt` and derives a 4-part installer
   version (e.g. `4.0.0-beta6` → `4.0.0.6`) so beta-to-beta upgrades are
   visible to Windows Installer / Burn.
2. Installs build dependencies into `buildvenv`.
3. Builds the app folder with PyInstaller. Bundles `libusb-1.0.dll` if
   present in `dependencies\fx2\`.
4. Builds `LumaViewPro-<version>.msi`. Adds an FX2 driver-install
   custom action if `LumaScope_WinUSB.inf` is present.
5. Builds `LumaViewPro-<version>-setup.exe` if the Pylon MSI is present.
   Chains the IDS Peak runtime EXE if its files are present.
6. Removes temporary clone/build files.

Example output:

```text
C:\LVP\appbuild\exe_artifacts\LumaViewPro-4.0.0-beta6\
|-- LumaViewPro-4.0.0-beta6.msi
`-- LumaViewPro-4.0.0-beta6-setup.exe
```

Previous completed packages are preserved in `exe_artifacts\`.

## Release Checklist

Before a customer build:

1. Confirm `version.txt` has the release version testers should see.
2. Commit and push the branch to be packaged.
3. Confirm `dependencies\` contains the Pylon MSI. Confirm any optional
   dependencies you intend to ship (IDS Peak, FX2 INF + DLL) are also present.
4. Run `.\build.ps1` and choose `Release`.
5. Verify both output files exist.
6. Ship `LumaViewPro-<version>-setup.exe`.
7. Keep the `.msi` for internal testing or troubleshooting.

## Updating The Build Script

If `build.ps1` changes in the repo, refresh the copy in your build folder:

```powershell
cd C:\LVP\appbuild
git clone --depth 1 --branch 4.0.0-beta https://github.com/EtalumaSupport/LumaViewPro.git _getscript
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
| `Bundle skipped` | Add the Pylon MSI to `dependencies\`. The MSI will still build, but the customer `-setup.exe` will not. |
| Beta-to-beta upgrade leaves the older beta installed alongside the newer one | Both betas were built before the 4-part version derivation landed (commit `2c8805f`). One-shot fix: uninstall the old beta manually via Settings → Apps. From that build forward, the bundle's related-bundle Detect handles the swap automatically. |
| New build doesn't replace the old beta install — installer says "already installed" | The MSI's `MajorUpgrade` only compares the first 3 parts of `ProductVersion`. Always install via the `-setup.exe` bundle (which compares all 4 parts), not the standalone MSI. |
| `Permission denied` during PyInstaller or MSI build | Close any running LumaViewPro instance (including the previous `-setup.exe`'s installer-modify dialog). Use `C:\LVP\appbuild\` instead of OneDrive-synced paths. |
| Path-with-spaces errors in the build log (e.g. `C:\Users\Etaluma Microscope\...`) | You're building from your user profile. Move the build to `C:\LVP\appbuild\` per One-Time Setup §2. |

## Notes

- The build script uses paths relative to the folder containing `build.ps1`.
- The selected build root is remembered in `.build_config` next to `build.ps1`.
- The build script accepts parameters, for example:

```powershell
.\build.ps1 -Branch 4.1.0-dev -BuildType Release
```
