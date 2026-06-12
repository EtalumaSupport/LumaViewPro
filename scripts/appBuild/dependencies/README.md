# Build Dependencies

Place the following files in this directory before running `build.ps1`:

## Required for MSI build
The standalone MSI builds from the LumaViewPro source alone -- no extra files
are required in this directory for it.

## Required for Bundle installer (optional)
Without this, the standalone MSI still builds but the all-in-one setup.exe is skipped.

- `pylon_USB_Camera_Driver.msi` — [Basler Pylon SDK](https://docs.baslerweb.com/pylon-software-suite) (USB Camera Driver MSI only)

## Notes
- Contents of this directory are gitignored (except this README)
- The build script auto-detects files by filename pattern
- The Pylon MSI is chained into the Bundle installer; the optional IDS Peak
  runtime is chained when present
