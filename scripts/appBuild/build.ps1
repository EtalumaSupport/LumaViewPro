# LumaViewPro Build Script
#
# SETUP (one time):
#   1. Install tools: Python 3.12+, Git, WiX (dotnet tool install --global wix)
#   2. Put dependencies in dependencies\ next to this script (see dependencies\README.md)
#
# USAGE:
#   .\build.ps1                          # interactive branch selection
#   .\build.ps1 -Branch 4.0.0-beta      # specific branch
#
# Output: <build_dir>\exe_artifacts\LumaViewPro-X.X.X\

param(
    [string]$Branch = "",
    [ValidateSet("Dev", "Release")]
    [string]$BuildType = ""
)

$ErrorActionPreference = "Stop"
$repo_url = "https://github.com/EtalumaSupport/LumaViewPro.git"
$script_dir = Split-Path -Parent $PSCommandPath
$config_file = Join-Path $script_dir ".build_config"

function Get-BuildPython {
    $probe = "import json, sys; print(json.dumps({'executable': sys.executable, 'version': [sys.version_info[0], sys.version_info[1], sys.version_info[2]]}))"
    $candidates = @(
        @{ Label = "py -3.13"; Command = "py"; Args = @("-3.13") }
        @{ Label = "py -3.12"; Command = "py"; Args = @("-3.12") }
        @{ Label = "python"; Command = "python"; Args = @() }
        @{ Label = "python3"; Command = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        try {
            $result = & $candidate.Command @($candidate.Args + @("-c", $probe)) 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $result) { continue }

            $info = $result | ConvertFrom-Json
            $major = [int]$info.version[0]
            $minor = [int]$info.version[1]
            $patch = [int]$info.version[2]

            if ($major -eq 3 -and $minor -ge 12) {
                return [PSCustomObject]@{
                    Label = $candidate.Label
                    Command = $candidate.Command
                    Args = $candidate.Args
                    Executable = $info.executable
                    Version = "$major.$minor.$patch"
                }
            }
        } catch {
            continue
        }
    }

    return $null
}

# ---------------------------------------------------------------------------
# Build directory selection
# ---------------------------------------------------------------------------
# Default to script location; user can override and it's saved for next time
$build_dir = $script_dir
if (Test-Path $config_file) {
    $saved = (Get-Content $config_file -TotalCount 1).Trim()
    if ($saved -and (Test-Path $saved)) { $build_dir = $saved }
}

Write-Host "`nBuild directory: $build_dir"
$change = Read-Host -Prompt "Update build directory? [y/N]"
if ($change -eq "y" -or $change -eq "Y") {
    $new_dir = Read-Host -Prompt "Build directory"
    if ($new_dir) {
        New-Item -Path $new_dir -ItemType Directory -Force | Out-Null
        $build_dir = (Resolve-Path $new_dir).Path
    }
}
# Save preference
Set-Content $config_file $build_dir

$build_type_prompt = $BuildType
if (-not $build_type_prompt) {
    Write-Host "`nPackage type:"
    Write-Host "  [1] Dev package (reuse cached build environment when possible)"
    Write-Host "  [2] Release package (recreate build environment from scratch)"
    $build_type_choice = Read-Host -Prompt "Select package type [1/2] (default 1)"

    switch ($build_type_choice) {
        "2" { $build_type_prompt = "Release" }
        default { $build_type_prompt = "Dev" }
    }
}

$BuildType = $build_type_prompt
Write-Host "Package type: $BuildType"

# All build paths relative to build_dir
$tmp = Join-Path $build_dir "_tmp"
$artifacts = Join-Path $build_dir "exe_artifacts"
$deps = Join-Path $script_dir "dependencies"
$venv = Join-Path $build_dir "buildvenv"

# Make sure we're not stuck inside a previous build
Set-Location $build_dir

# ---------------------------------------------------------------------------
# Select branch
# ---------------------------------------------------------------------------
if (-not $Branch) {
    $branches = @(
        "4.0.0-beta"
        "4.1.0-dev"
        "main"
    )
    Write-Host "`nAvailable branches:"
    for ($i = 0; $i -lt $branches.Count; $i++) {
        Write-Host "  [$($i+1)] $($branches[$i])"
    }
    Write-Host "  [0] Enter custom branch"
    $choice = Read-Host -Prompt "Select branch (1-$($branches.Count), or 0 for custom)"
    if ($choice -eq "0" -or -not $choice) {
        $Branch = Read-Host -Prompt "Branch name"
    } else {
        $idx = [int]$choice - 1
        if ($idx -ge 0 -and $idx -lt $branches.Count) {
            $Branch = $branches[$idx]
        } else {
            Write-Host "Invalid selection"
            Exit 1
        }
    }
}
Write-Host "Building branch: $Branch"

# ---------------------------------------------------------------------------
# Find dependencies
# ---------------------------------------------------------------------------
$pylon_msi = ""
$corretto_msi = ""
$maven_dir = ""

if (-not (Test-Path $deps)) {
    New-Item $deps -ItemType Directory -Force | Out-Null
    Write-Host "`nCreated dependencies\ folder. See dependencies\README.md for what to put there."
}

$pylon_files = Get-ChildItem -Path $deps -Filter "*pylon*USB*.msi" -ErrorAction SilentlyContinue
if ($pylon_files) { $pylon_msi = $pylon_files[0].FullName; Write-Host "Found Pylon: $pylon_msi" }

$corretto_files = Get-ChildItem -Path $deps -Filter "*corretto*.msi" -ErrorAction SilentlyContinue
if ($corretto_files) { $corretto_msi = $corretto_files[0].FullName; Write-Host "Found Corretto: $corretto_msi" }

$maven_files = Get-ChildItem -Path $deps -Directory -Filter "apache-maven*" -ErrorAction SilentlyContinue
if ($maven_files) { $maven_dir = $maven_files[0].FullName; Write-Host "Found Maven: $maven_dir" }

# ---------------------------------------------------------------------------
# Optional dependencies — bundle uses them when present, skips silently when
# absent. Drop the matching files into dependencies\ to enable.
# ---------------------------------------------------------------------------

# IDS Peak runtime (USB3 transport layer for IDS cameras). InstallShield
# .exe — needs a recorded setup.iss for silent install. See
# BUILD_INSTRUCTIONS.md > Optional Dependencies.
$ids_peak_exe = ""
$ids_peak_iss = ""
$ids_files = Get-ChildItem -Path $deps -Filter "ids_peak_*.exe" -ErrorAction SilentlyContinue
if ($ids_files) {
    $ids_iss_files = Get-ChildItem -Path $deps -Filter "setup.iss" -ErrorAction SilentlyContinue
    if ($ids_iss_files) {
        $ids_peak_exe = $ids_files[0].FullName
        $ids_peak_iss = $ids_iss_files[0].FullName
        Write-Host "Found IDS Peak: $ids_peak_exe"
        Write-Host "Found IDS Peak setup.iss: $ids_peak_iss"
    } else {
        Write-Host "Found IDS Peak EXE but no setup.iss in dependencies\ - IDS Peak will NOT be bundled (silent install needs both)"
    }
}

# FX2 WinUSB INF + native libusb-1.0.dll for LVC (LS560/LS620/LS720)
# cameras. Two files needed for end-to-end FX2 support:
#   1. LumaScope_WinUSB.inf — binds inbox WinUSB.sys to the FX2 VID/PID;
#      installed by pnputil custom action in Package.wxs.
#   2. libusb-1.0.dll — pyusb's libusb1 backend on Windows; loaded by
#      ctypes inside the bundled LVP exe. Without it, pyusb fails to
#      find a USB backend and the FX2 driver silently no-ops.
# See BUILD_INSTRUCTIONS.md > Optional Dependencies.
$fx2_inf = ""
$fx2_libusb_dll = ""
$fx2_dir = Join-Path $deps "fx2"
if (Test-Path $fx2_dir) {
    $fx2_files = Get-ChildItem -Path $fx2_dir -Filter "*WinUSB*.inf" -ErrorAction SilentlyContinue
    if ($fx2_files) {
        $fx2_inf = $fx2_files[0].FullName
        Write-Host "Found FX2 WinUSB INF: $fx2_inf"
    }
    $fx2_dll_files = Get-ChildItem -Path $fx2_dir -Filter "libusb-1.0.dll" -ErrorAction SilentlyContinue
    if ($fx2_dll_files) {
        $fx2_libusb_dll = $fx2_dll_files[0].FullName
        Write-Host "Found FX2 libusb-1.0.dll: $fx2_libusb_dll"
    }
}

if (-not $pylon_msi) { Write-Host "No Pylon MSI in dependencies\ - bundle will be skipped" }
if (-not $corretto_msi) { Write-Host "No Corretto MSI in dependencies\ - bundle will be skipped" }
if (-not $maven_dir) { Write-Host "Warning: Apache Maven not found in dependencies\ - ImageJ will not work in installed app" }
if (-not $ids_peak_exe) { Write-Host "No IDS Peak runtime in dependencies\ - IDS cameras will need manual driver install on customer machines" }
if (-not $fx2_inf) { Write-Host "No FX2 WinUSB INF in dependencies\fx2\ - LVC FX2 driver will need manual install (Zadig)" }
if (-not $fx2_libusb_dll) { Write-Host "No FX2 libusb-1.0.dll in dependencies\fx2\ - bundled LVP will not be able to talk to FX2 cameras even if WinUSB driver is installed" }

# ---------------------------------------------------------------------------
# Check tools
# ---------------------------------------------------------------------------
Write-Host "`nChecking tools..."
try { $wix_version_raw = & wix --version 2>&1; Write-Host "  WiX: $wix_version_raw" } catch { Write-Host "ERROR: WiX not found. Run: dotnet tool install --global wix --version 6.0.0"; Exit 1 }

# Refuse WiX v7+: Bundle.wxs is authored for the v4-v6 WixToolset.Bal.wixext
# API. v7 restructured WixStdBA and added a required scope field in the
# Burn-BA plan protocol; bundles built with v7 still produce a -setup.exe
# but fail at customer install with 0x80070057 "Failed to read plan scope
# of BAEnginePlan args". Catch the wrong WiX up front, before the build
# wastes 5+ minutes producing a broken bundle.
$wix_major = $null
if ($wix_version_raw -match '^\s*(\d+)\.') { $wix_major = [int]$matches[1] }
if ($null -ne $wix_major -and $wix_major -ge 7) {
    Write-Host ""
    Write-Host "ERROR: WiX $wix_version_raw is not supported by this build."
    Write-Host "  This build's Bundle.wxs requires WiX v6.x."
    Write-Host "  Downgrade with:"
    Write-Host "    dotnet tool uninstall --global wix"
    Write-Host "    dotnet tool install --global wix --version 6.0.0"
    Write-Host "    wix extension remove --global WixToolset.Bal.wixext"
    Write-Host "  See scripts\appBuild\BUILD_INSTRUCTIONS.md for details."
    Exit 1
}

try { $v = & git --version 2>&1; Write-Host "  Git: $v" } catch { Write-Host "ERROR: Git not found"; Exit 1 }

$python = Get-BuildPython
if (-not $python) {
    Write-Host "ERROR: Python 3.12+ not found. Install Python 3.12 or 3.13 and make sure it is available via py, python, or python3."
    Exit 1
}
Write-Host "  Build Python: $($python.Version) [$($python.Executable)]"
$wix_exe = (Get-Command wix).Source

# ---------------------------------------------------------------------------
# Clean previous temp, clone fresh
# ---------------------------------------------------------------------------
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item $tmp -ItemType Directory -Force | Out-Null

Write-Host "`nCloning $Branch..."
$clone = Join-Path $tmp "src"
$ErrorActionPreference = "Continue"
git clone --depth 1 --branch $Branch $repo_url $clone
$clone_exit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($clone_exit -ne 0) { Write-Host "ERROR: Clone failed"; Exit 1 }
Remove-Item "$clone\.git*" -Recurse -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# Read version
# ---------------------------------------------------------------------------
$ver_raw = (Get-Content "$clone\version.txt" -TotalCount 1).Trim()
if ($ver_raw -match '^\S+') { $version = $matches[0] } else { Write-Host "ERROR: Can't parse version.txt"; Exit 1 }

$product = "LumaViewPro-$version"

# Build a 4-part Major.Minor.Patch.Revision installer version. WiX Bundle
# ProductVersion + MSI ProductVersion are what Windows Installer compares to
# decide upgrade-vs-no-op. Without a unique 4th component every beta gets
# version "4.0.0" and beta3 -> beta6 is invisible to the upgrade engine
# (related-bundle Detect doesn't fire). Encoding rules:
#
#   X.Y.Z-betaN     -> X.Y.Z.N    (beta number in revision slot)
#   X.Y.Z           -> X.Y.Z.99   (release > any beta in same series)
#   X.Y.Z-<other>   -> X.Y.Z.0    (dev/rc/etc — lowest, never shipped)
#
# When the next major series (4.1.0) starts shipping, X.Y.Z.99 of the older
# series stays below 4.1.0.* automatically, so no transition logic needed.
if ($version -match '^(\d+\.\d+\.\d+)-beta(\d+)$') {
    $wix_ver = "$($matches[1]).$($matches[2])"
} elseif ($version -match '^(\d+\.\d+\.\d+)$') {
    $wix_ver = "$($matches[1]).99"
} elseif ($version -match '^(\d+\.\d+\.\d+)-') {
    $wix_ver = "$($matches[1]).0"
} else {
    Write-Host "ERROR: Can't derive installer version from version.txt: $version"
    Exit 1
}
Write-Host "Installer version (4-part): $wix_ver"

Write-Host "`n======================================="
Write-Host "  Building $product"
Write-Host "  Installer version: $wix_ver"
Write-Host "======================================="

# Rename source dir
$src = Join-Path $tmp $product
Rename-Item $clone $product

# ---------------------------------------------------------------------------
# Create build venv and install dependencies
# ---------------------------------------------------------------------------
Write-Host "`n--- Build Environment ---"
$recreate_build_env = $BuildType -eq "Release"

if ($recreate_build_env -and (Test-Path $venv)) {
    Write-Host "Removing cached build environment for release build..."
    Remove-Item $venv -Recurse -Force
}

$venv_python = Join-Path $venv "Scripts\python.exe"
$venv_exists = Test-Path $venv_python

if (-not $venv_exists) {
    Write-Host "Creating build venv..."
    & $python.Command @($python.Args + @("-m", "venv", $venv))
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to create venv"; Exit 1 }
} else {
    Write-Host "Reusing cached build environment: $venv"
}

$venv_python = Join-Path $venv "Scripts\python.exe"

Write-Host "Upgrading pip..."
& $venv_python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to upgrade pip in build venv"; Set-Location $build_dir; Exit 1 }

if (Test-Path "$src\requirements-dev.txt") {
    Write-Host "Installing build dependencies..."
    & $venv_python -m pip install -r "$src\requirements-dev.txt"
} else {
    Write-Host "Installing runtime dependencies..."
    & $venv_python -m pip install -r "$src\requirements.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pip install failed"; Set-Location $build_dir; Exit 1 }

    Write-Host "Installing PyInstaller..."
    & $venv_python -m pip install pyinstaller
}
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pip install failed"; Set-Location $build_dir; Exit 1 }

# Verify PyInstaller is available
& $venv_python -m PyInstaller --version
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: PyInstaller not available in build venv"; Set-Location $build_dir; Exit 1 }

# ---------------------------------------------------------------------------
# Build EXE
# ---------------------------------------------------------------------------
Write-Host "`n--- PyInstaller ---"
Set-Location $src
# License files may be in licenses/ (old) or docs/licenses/ (current)
if (Test-Path ".\licenses") {
    Copy-Item ".\licenses\*" -Destination ".\" -Force
} elseif (Test-Path ".\docs\licenses") {
    Copy-Item ".\docs\licenses\*" -Destination ".\" -Force
}
if (Test-Path ".\docs\LICENSE") {
    Copy-Item ".\docs\LICENSE" -Destination ".\" -Force
}

# The .spec file must be in the repo under scripts/appBuild/config/
$spec = ".\scripts\appBuild\config\lumaviewpro_win_release.spec"
if (-not (Test-Path $spec)) { Write-Host "ERROR: Spec file not found: $spec"; Set-Location $build_dir; Exit 1 }
$spec_contents = Get-Content $spec -Raw
if ($spec_contents -notmatch 'contents_directory\s*=\s*[''"]\.[''"]') {
    Write-Host "ERROR: The cloned spec file does not set contents_directory='.'."
    Write-Host "Push or build from a branch that contains the updated PyInstaller spec before creating a release build."
    Set-Location $build_dir
    Exit 1
}
Copy-Item $spec ".\lumaviewpro.spec"

Write-Host "Building executable..."
# Hand the FX2 libusb DLL path to the PyInstaller spec via env var. The
# spec checks $env:FX2_LIBUSB_DLL and conditionally adds the binary to
# the Analysis. Empty/unset -> spec skips bundling, FX2 stays unsupported
# in the resulting exe.
if ($fx2_libusb_dll) {
    $env:FX2_LIBUSB_DLL = $fx2_libusb_dll
    Write-Host "Bundling FX2 libusb-1.0.dll: $fx2_libusb_dll"
} else {
    $env:FX2_LIBUSB_DLL = ""
}
& $venv_python -m PyInstaller --log-level WARN .\lumaviewpro.spec
$pyi_exit = $LASTEXITCODE
$env:FX2_LIBUSB_DLL = $null
if ($pyi_exit -ne 0) { Write-Host "ERROR: PyInstaller failed"; Set-Location $build_dir; Exit 1 }

# Create install directory
$install = ".\dist\$product"
New-Item $install -ItemType Directory -Force | Out-Null
Copy-Item ".\dist\lumaviewpro\*" -Destination $install -Recurse
$install = (Resolve-Path $install).Path

# PyInstaller 6 may keep bundled resources under _internal even when the
# application executable stays at the install root. WiX only needs these
# assets at build time for branding, so detect whichever layout was produced.
$installer_assets_dir = $install
$installer_icon = Join-Path $installer_assets_dir "data\icons\icon.ico"
if (-not (Test-Path $installer_icon)) {
    $internal_assets_dir = Join-Path $install "_internal"
    $internal_icon = Join-Path $internal_assets_dir "data\icons\icon.ico"
    if (Test-Path $internal_icon) {
        $installer_assets_dir = $internal_assets_dir
        $installer_icon = $internal_icon
        Write-Host "Using PyInstaller _internal assets for installer branding"
    }
}

if (-not (Test-Path $installer_icon)) {
    Write-Host "ERROR: Installer icon not found in either $install\data\icons or $install\_internal\data\icons"
    Set-Location $build_dir
    Exit 1
}

# Copy Maven if available
if ($maven_dir) {
    $maven_name = Split-Path $maven_dir -Leaf
    Copy-Item $maven_dir -Destination "$install\$maven_name" -Recurse -Force
    Write-Host "Maven copied to install directory"
}

# ---------------------------------------------------------------------------
# Build MSI
# ---------------------------------------------------------------------------
Write-Host "`n--- WiX MSI ---"
$wix_dir = Join-Path $src "scripts\appBuild\build_exe\wix"
Set-Location $wix_dir

$output_dir = Join-Path $artifacts $product
New-Item $output_dir -ItemType Directory -Force | Out-Null
$msi = Join-Path $output_dir "$product.msi"

Write-Host "Building MSI..."
$msi_args = @(
    'build', '-arch', 'x64',
    '-d', "InstallFolderDir=$install",
    '-d', "InstallerAssetsDir=$installer_assets_dir",
    '-d', "ProjectDir=$wix_dir\",
    '-d', "ProductName=$product",
    '-d', "Version=$wix_ver"
)
# Pass optional FX2 INF path. Package.wxs gates the FX2 driver-install
# component on this define being set, so absence -> no FX2 install action
# in the MSI.
if ($fx2_inf) {
    $msi_args += @('-d', "Fx2InfPath=$fx2_inf")
}
$msi_args += @('-out', $msi, 'Package.wxs', 'Folders.wxs')
& $wix_exe @msi_args

if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: MSI build failed"; Set-Location $build_dir; Exit 1 }
Write-Host "MSI: $msi"

# ---------------------------------------------------------------------------
# Build Bundle (if dependencies available)
# ---------------------------------------------------------------------------
$bundle = ""
if ($pylon_msi -and $corretto_msi) {
    Write-Host "`n--- WiX Bundle ---"
    $bundle = Join-Path $output_dir "$product-setup.exe"

    # Find BAL extension
    $bal_dep = Join-Path $src "scripts\appBuild\build_exe\deps\WixToolset.BootstrapperApplications.wixext.dll"
    $bal_script = Join-Path $script_dir "build_exe\deps\WixToolset.BootstrapperApplications.wixext.dll"
    if (Test-Path $bal_dep) { $ext = $bal_dep }
    elseif (Test-Path $bal_script) { $ext = $bal_script }
    else { & wix extension add -g WixToolset.Bal.wixext 2>&1 | Out-Null; $ext = "WixToolset.Bal.wixext" }

    Write-Host "Building bundle..."
    $bundle_args = @(
        'build', '-arch', 'x64',
        '-ext', $ext,
        '-d', "LVPInstallFolderDir=$install",
        '-d', "InstallerAssetsDir=$installer_assets_dir",
        '-d', "LVPMsiDir=$msi",
        '-d', "PylonDriverDir=$pylon_msi",
        '-d', "CorretoMsiDir=$corretto_msi",
        '-d', "ProductName=$product",
        '-d', "ProductVersion=$wix_ver"
    )
    # Pass optional IDS Peak runtime EXE + setup.iss. Bundle.wxs gates the
    # IDS Peak ExePackage on these defines, so absence -> not chained.
    if ($ids_peak_exe -and $ids_peak_iss) {
        $bundle_args += @(
            '-d', "IDSPeakExe=$ids_peak_exe",
            '-d', "IDSPeakSetupIss=$ids_peak_iss"
        )
    }
    $bundle_args += @('-out', $bundle, 'Bundle.wxs')
    & $wix_exe @bundle_args

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Bundle build failed"
        $bundle = ""
    } else {
        Write-Host "Bundle: $bundle"
    }
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Set-Location $build_dir

# Clean temp (clone + logs + build artifacts)
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`n======================================="
Write-Host "  BUILD COMPLETE"
Write-Host "======================================="
Write-Host "  MSI:    $msi"
if ($bundle -and (Test-Path $bundle)) {
    Write-Host "  Bundle: $bundle"
}
Write-Host "  Output: $output_dir"
Write-Host "======================================="
