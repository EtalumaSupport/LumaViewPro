# Build Dependencies

Place the following files in this directory before running `build.ps1`.
The build script auto-detects them by filename pattern -- the patterns are
given below because the *names matter*: a file that does not match is
treated as absent.

`build.ps1` announces what it found and what it did not, near the start of
every run. Read that block before walking away from a build; a missing
dependency silently changes what gets produced.

## Required for the MSI

Nothing. The standalone MSI builds from LumaViewPro source alone.

## Required for the Bundle installer (`-setup.exe`)

**If either of these is missing, no bundle is built.** The MSI still builds
and the run still exits 0, so a bundle-less build is easy to miss if you
are not reading the dependency block.

| File | Pattern matched | Source |
|---|---|---|
| `pylon_USB_Camera_Driver.msi` | `*pylon*USB*.msi` | [Basler Pylon SDK](https://docs.baslerweb.com/pylon-software-suite) -- USB Camera Driver MSI only |
| `vc_redist.x64.exe` | `vc_redist.x64.exe` (exact) | Microsoft VC++ Redistributable, x64 |

The redistributable is chained rather than bundled: the app deliberately
does not ship `msvcp140`, so the installer must supply it system-wide.

## Required to build the bundle at all

| File | Pattern matched | Source |
|---|---|---|
| `WixToolset.BootstrapperApplications.wixext.dll` | exact name | Archived in the **Firmware** repo at `tools/appbuild/deps/` |

The WiX BAL extension that builds the bundle bootstrapper, pinned to a
v4-v6-compatible version on purpose: a v7-era BAL produces a `-setup.exe`
that installs fine on the build box and then **fails at every customer
install with `0x80070057`**. `build.ps1` will not fetch it from the nuget
feed, because the feed serves whatever is current and nothing here
version-checks the extension. If it is absent the build stops and says so.

## Optional

| File | Pattern matched | Effect when absent |
|---|---|---|
| `ids_peak_*.exe` plus `setup.iss` | `ids_peak_*.exe`, `setup.iss` | IDS cameras need a manual driver install on customer machines |
Nothing for the FX2 (LS560, LS620, LS720) goes here: its driver INF is in
the repo (`scripts\appBuild\build_exe\fx2_driver\`), and its libusb library
and firmware come with the clone and the `libusb-package` wheel. An `.inf` or
a `libusb*.dll` left here or in an `fx2\` subfolder from the old route
refuses the build.

## Notes

- Contents of this directory are gitignored, except this README.
- None of these files is tracked in the LumaViewPro repo. They are placed
  by hand on the build box.
