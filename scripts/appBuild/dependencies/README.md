# Build Dependencies

Place the following files in this directory before running `build.ps1`:

## Required for MSI build
- `apache-maven-3.9.8/` — Extract from [Apache Maven download](https://maven.apache.org/download.cgi) (Binary Zip Archive). Bundled into the installed app for ImageJ/PyImageJ support.

## Required for Bundle installer (optional)
Without these, the standalone MSI still builds but the all-in-one setup.exe is skipped.

- `pylon_USB_Camera_Driver.msi` — [Basler Pylon SDK](https://docs.baslerweb.com/pylon-software-suite) (USB Camera Driver MSI only)
- `amazon-corretto-8-xxx-jdk.msi` — [Amazon Corretto 8 JDK](https://docs.aws.amazon.com/corretto/latest/corretto-8-ug/downloads-list.html) (Windows x64 MSI)

## Notes
- Contents of this directory are gitignored (except this README)
- The build script auto-detects files by filename pattern
- Maven is copied into the installed app; Pylon and Corretto MSIs are chained into the Bundle installer
