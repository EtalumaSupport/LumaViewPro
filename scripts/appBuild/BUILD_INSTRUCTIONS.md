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
| The build record: `build.log` (the console transcript), `pyinstaller_<version>_<time>.log` (PyInstaller's full DEBUG output: where every collected binary came from), `pip_freeze_<version>_<time>.txt` (every package and version the bundle was packed from), `pyinstaller_warn_*.txt` and `pyinstaller_*.toc` (what PyInstaller flagged, and what shipped) | What this build did, readable after the build's temporary folder is gone. The console shows PyInstaller at INFO and above; the DEBUG lines are in its log. | No. Keep it with the installer it describes. |

The `-setup.exe` bundle is only created when both **required** dependencies
are present: the Basler Pylon USB Camera Driver MSI and the Microsoft VC++
Redistributable (`vc_redist.x64.exe` -- the app deliberately ships no
`msvcp140.dll` of its own, so the bundle must chain the redistributable).
The optional dependency (IDS Peak) is silently skipped when its files are
absent. The FX2 driver package is carried by every build, from the repo.

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
`requirements-build.txt`: the app's requirements plus PyInstaller and its
hooks, both pinned, and nothing else, since anything installed there can be
packed into the installer.
Do not install PyInstaller globally for this build.

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
|-- vc_redist.x64.exe                    # required for the bundle
|-- ids_peak_<version>.exe               # optional — IDS cameras
`-- setup.iss                            # optional — paired with ids_peak EXE
```

#### Where each dependency comes from

| File | Source | Notes |
|------|--------|-------|
| `pylon_USB_Camera_Driver.msi` | [baslerweb.com/en/downloads/software-downloads](https://www.baslerweb.com/en/downloads/software-downloads/) | Free MyBasler account required. Pick the standalone USB driver MSI (not the full Pylon SDK installer). |
| `vc_redist.x64.exe` | [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) | Microsoft's latest VC++ x64 redistributable, no account needed. The build fails if its version is older than the newest VC runtime any bundled wheel carries (build.ps1 prints both in the CRT census). |
| `ids_peak_<version>.exe` | [en.ids-imaging.com/download-peak.html](https://en.ids-imaging.com/download-peak.html) → Runtime variant | Free MyIDS account required. **The download is a package; extract it and use the inner `ids_peak_<version>.exe` (underscores). The hyphenated outer file cannot be driven silently.** Chaining IDS is not currently supported — see the IDS Peak section below before using these two rows. |
| `setup.iss` | Generated locally, once per IDS Peak runtime version | Run the **inner** `ids_peak_<version>.exe /r`; it records your interactive choices into `%WINDIR%\setup.iss`. The choices matter — the USB3 Vision Transport Layer must be selected. See the IDS Peak section below. |

Keep a local mirror of every account-gated download (Pylon, IDS Peak) in
your build artefact store. Do not depend on being logged into MyBasler /
MyIDS at build time.

#### Required for the standalone MSI

The standalone MSI builds from the LumaViewPro source alone -- no files in
`dependencies\` are required for it.

#### Required for the `-setup.exe` bundle

- `pylon_USB_Camera_Driver.msi`
- `vc_redist.x64.exe`

If either is missing, the build still creates the standalone MSI but skips
the customer `-setup.exe` bundle.

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

> **Status: chaining IDS is NOT currently supported, and not a shipping
> requirement.** It was attempted on 2026-08-17 and failed; the unresolved
> defects are listed at the end of this section. Ship without it — IDS
> customers install the IDS peak runtime themselves, which works. Read the
> rest of this section before attempting it again.

##### The two executables — the trap that costs a bench trip

IDS ships **a download package that contains the installer**, and only the
inner one can be driven silently. From the IDS peak 26.06 readme:

> Extract the IDS peak setup package to a local directory. **The required
> silent setup command line parameters can only be processed by the
> `ids_peak_<version>.exe`.**

| File | Naming | Accepts `/s /f1`? |
|---|---|---|
| `ids-peak-win-<variant>-setup-64-<ver>.exe` — the download | **hyphens** | **No** |
| `ids_peak_<version>.exe` — inside it | **underscores** | Yes |

The underscore spelling in the selector is therefore load-bearing, not an
accident to be "fixed" by widening. Putting the wrapper in `dependencies\`
builds a bundle whose IDS package exits **`0x80042000`** at install time.
The build script detects the wrapper by name and fails with instructions.

Pick the **Runtime** variant (drivers + GenTL transport layers only — no
Cockpit, no SDK headers), not Standard or Extended.

##### Recording `setup.iss`

Required by IDS, not by us: silent mode reads its choices from a recorded
response file. Record once per runtime version, **from the extracted inner
exe**, on a host with a clean install of that version:

```powershell
ids_peak_<version>.exe /r      # writes %WINDIR%\setup.iss
```

The interactive installer presents **many component choices** — transport
layers (USB3 Vision / GigE Vision socket vs kernel / uEye), SDKs, examples,
DirectShow, tools. Whatever you tick is what every silent install replays,
so tick deliberately: **the USB3 Vision Transport Layer (U3VK) is the one
that makes IDS cameras enumerate.** Without it, the app still fails with
`GENICAM_GENTLN_PATH environment variable not found`.

Copy the file next to the EXE. Re-record on any version bump.

**No spaces in the `/f1` path** — the IDS readme states this twice, and it
is why the documented invocation is the relative form
`/s /f1".\setup.iss"` (the response file sits in the setup directory).

##### Unresolved defects — fix these before chaining is viable

1. **The `setup.iss` path is baked in from the build machine.**
   `build.ps1` passes `$ids_iss_files[0].FullName` into `Bundle.wxs`'s
   `InstallArguments`, so the installer looks for e.g.
   `C:\LVP\appbuild\dependencies\setup.iss` on the *customer's* machine.
   It resolves nowhere but the build host. The `.iss` must ship as a
   `<Payload>` and be referenced relatively — and note Burn's package cache
   lives under `C:\ProgramData\Package Cache\…`, which contains a space, so
   an absolute path there would violate the no-spaces rule anyway.
2. **The IDS `ExePackage` is vital.** `Vital` defaults to yes, so a failed
   optional camera driver **rolls back the entire LumaViewPro install** —
   observed twice on 2026-08-17, leaving the machine with no LVP and an
   "unspecified error". An optional driver must never be able to do that.
3. **Re-runs every time.** `DetectCondition="0"` means the IDS installer
   runs on every install, including repairs and upgrades.
4. **A reboot is required** after IDS peak installs, which the bundle does
   not currently account for.

Diagnosing a failure: the IDS installer writes its own `Setup.log` with the
real error, and Burn names its per-package log in the bundle log
(`WixBundleLog_IDSPeakRuntime`). Neither is in a support bundle.

**Version pairing constraint:** IDS couples the Python bindings to the
runtime. Check the runtime's own `readme.html` for the exact binding
versions it ships and compare against `requirements.txt` before pairing —
the readme lists them explicitly. IDS peak 26.06 also introduced a new
`ids_peak_common` package that the readme puts **first** in the install
order and that `requirements.txt` does not currently carry. IDS renumbered
releases from `2.x` to `YY.MM` (2.21 → 26.06), so any "runtime ≥ 2.18"
guidance predates the change and cannot be compared numerically.

#### FX2 WinUSB Driver (Lumascope Classic: LS560 / LS620 / LS720)

Nothing to drop in. The driver package is
`scripts\appBuild\build_exe\fx2_driver\Lumascope_Classic.inf` in the repo: it
binds inbox `WinUSB.sys` to both of the FX2's identities, `0x04B4:0x8613` (the
bootloader) and `0x04B4:0xEA17` (the running firmware), named "Lumascope
Classic (Bootloader)" and "Lumascope Classic" in Device Manager. The build
stamps its `DriverVer` with the build's date and the installer version
(`scripts\appBuild\config\fx2_driver_package.py`), writes the stamped copy to
the output folder's `fx2_driver\`, and the MSI installs it to
`<InstallFolder>\drivers\fx2\`. Windows ranks equally signed packages for one
device by that date, so the stamp is what lets ours outrank an older Zadig
binding.

The MSI then runs the driver step, `lvp_fx2_driver_step.exe install` (a second
executable the release spec builds from `lvp_fx2_driver_step.py`, deferred, as
SYSTEM, its result ignored): it adds our package to the driver store, and only
once it is there exports and removes every other vendor's package for the
running firmware (`0xEA17`), binds the present devices to ours, removes our
older packages and the past devices of both identities, and logs every step
to `<InstallFolder>\logs\driver_install_<version>_<time>.log`. Uninstall runs
`lvp_fx2_driver_step.exe uninstall`, which removes our package and restores
what install exported. Until the package's catalog is signed, it does not
enter the store, the step removes nothing, and the customer must still bind
the driver with Zadig.

The rest of what the FX2 needs is not a dependency to drop in either: the release
spec ships the firmware hex from the clone's `firmware\` folder and the
native libusb library from the `libusb-package` wheel (`requirements.txt`),
inside the bundle's `libusb_package\` folder where the driver looks for it.
The build is refused, naming the file, when either is missing, before
PyInstaller runs and again in the built tree. A `libusb*.dll` or an `.inf`
left in `dependencies\` or `dependencies\fx2\` from the old route also refuses
the build: nothing reads it any more.

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
   - `Dev` reuses cached `buildvenv` for faster repeat builds, and rebuilds
     it when the branch's `requirements*.txt` differ from the ones it was
     built from (the log says why).
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
3. Builds the app folder with PyInstaller, with the FX2's firmware hex and
   `libusb-package`'s library.
4. Stamps the FX2 driver package and builds `LumaViewPro-<version>.msi`,
   which carries it and runs the FX2 driver step at install and uninstall.
5. Builds `LumaViewPro-<version>-setup.exe` if the Pylon MSI is present.
   Chains the IDS Peak runtime EXE if its files are present.
6. Removes temporary clone/build files.

Example output:

```text
C:\LVP\appbuild\exe_artifacts\LumaViewPro-4.0.0-beta6\
|-- LumaViewPro-4.0.0-beta6.msi
|-- LumaViewPro-4.0.0-beta6-setup.exe
`-- fx2_driver\
    `-- Lumascope_Classic.inf          # the stamped copy the MSI carries
```

Previous completed packages are preserved in `exe_artifacts\`.

## Release Checklist

Before a customer build:

1. Confirm `version.txt` has the release version testers should see.
2. Commit and push the branch to be packaged.
3. Confirm `dependencies\` contains the Pylon MSI. Confirm any optional
   dependency you intend to ship (IDS Peak) is also present.
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
