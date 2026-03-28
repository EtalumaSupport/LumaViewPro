# LumaViewPro Build Script
#
# SETUP (one time):
#   1. Create C:\LumaViewPro\
#   2. Copy this script to C:\LumaViewPro\build.ps1
#   3. Put Pylon and Corretto MSIs in C:\LumaViewPro\prereqs\
#
# USAGE:
#   cd C:\LumaViewPro
#   .\build.ps1
#
# That's it. It clones, builds EXE, builds MSI, builds Bundle.
# Output: C:\LumaViewPro\builds\LumaViewPro-X.X.X\

param(
    [string]$Branch = ""
)

$ErrorActionPreference = "Stop"
$root = "C:\LumaViewPro"
$repo_url = "https://github.com/EtalumaSupport/LumaViewPro.git"

# Make sure we're not stuck inside a previous build
Set-Location $root

# ---------------------------------------------------------------------------
# Ask for branch
# ---------------------------------------------------------------------------
if (-not $Branch) {
    $Branch = Read-Host -Prompt "Branch (e.g., 4.0.0-beta)"
}

# ---------------------------------------------------------------------------
# Find prereqs (optional — bundle skipped if not found)
# ---------------------------------------------------------------------------
$prereqs = Join-Path $root "prereqs"
$pylon_msi = ""
$corretto_msi = ""

if (Test-Path $prereqs) {
    $pylon_files = Get-ChildItem -Path $prereqs -Filter "*pylon*USB*.msi" -ErrorAction SilentlyContinue
    if ($pylon_files) { $pylon_msi = $pylon_files[0].FullName; Write-Host "Found Pylon: $pylon_msi" }

    $corretto_files = Get-ChildItem -Path $prereqs -Filter "*corretto*.msi" -ErrorAction SilentlyContinue
    if ($corretto_files) { $corretto_msi = $corretto_files[0].FullName; Write-Host "Found Corretto: $corretto_msi" }
}

if (-not $pylon_msi) { Write-Host "No Pylon MSI in prereqs\ — bundle will be skipped" }
if (-not $corretto_msi) { Write-Host "No Corretto MSI in prereqs\ — bundle will be skipped" }

# ---------------------------------------------------------------------------
# Check tools
# ---------------------------------------------------------------------------
Write-Host "`nChecking tools..."
try { $v = & wix --version 2>&1; Write-Host "  WiX: $v" } catch { Write-Host "ERROR: WiX not found. Run: dotnet tool install --global wix"; Exit 1 }
try { $v = & python --version 2>&1; Write-Host "  Python: $v" } catch { Write-Host "ERROR: Python not found"; Exit 1 }
try { $v = & git --version 2>&1; Write-Host "  Git: $v" } catch { Write-Host "ERROR: Git not found"; Exit 1 }

# ---------------------------------------------------------------------------
# Clean previous temp, clone fresh
# ---------------------------------------------------------------------------
$tmp = Join-Path $root "_tmp"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item $tmp -ItemType Directory -Force | Out-Null

Write-Host "`nCloning $Branch..."
$clone = Join-Path $tmp "src"
# Git writes progress to stderr which PowerShell treats as errors with ErrorActionPreference=Stop.
# Temporarily relax error handling for the clone command.
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
$wix_ver = $version
if ($version -match '^(\d+\.\d+\.\d+)') { $wix_ver = $matches[1] }

Write-Host "`n======================================="
Write-Host "  Building $product"
Write-Host "  WiX version: $wix_ver"
Write-Host "======================================="

# Rename source dir
$src = Join-Path $tmp $product
Rename-Item $clone $product

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
if (-not (Test-Path $spec)) { Write-Host "ERROR: Spec file not found: $spec"; Exit 1 }
Copy-Item $spec ".\lumaviewpro.spec"
# Use python -m PyInstaller in case pyinstaller.exe isn't in PATH
python -m PyInstaller --log-level INFO .\lumaviewpro.spec
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: PyInstaller failed"; Set-Location $root; Exit 1 }

# Prepare install directory
$install = ".\dist\$product"
New-Item $install -ItemType Directory -Force | Out-Null
Copy-Item ".\dist\lumaviewpro\*" -Destination $install -Recurse

# Verify critical files exist in dist
$icon_check = ".\dist\lumaviewpro\data\icons\icon.ico"
if (-not (Test-Path $icon_check)) {
    Write-Host "WARNING: icon.ico not found in PyInstaller output. Checking dist contents..."
    Write-Host "dist\lumaviewpro\ top-level:"
    Get-ChildItem ".\dist\lumaviewpro\" -ErrorAction SilentlyContinue | Select-Object -First 20
    Write-Host "Looking for icon.ico anywhere in dist:"
    Get-ChildItem ".\dist\" -Recurse -Filter "icon.ico" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  Found: $($_.FullName)" }

    # PyInstaller may put data files inside _internal/ in v6.x
    $internal_icon = ".\dist\lumaviewpro\_internal\data\icons\icon.ico"
    if (Test-Path $internal_icon) {
        Write-Host "Found icon in _internal — copying data folder to top level"
        Copy-Item ".\dist\lumaviewpro\_internal\data" -Destination ".\dist\lumaviewpro\data" -Recurse -Force
        # Re-copy to install dir
        Copy-Item ".\dist\lumaviewpro\data" -Destination "$install\data" -Recurse -Force
    }
}

$install = (Resolve-Path $install).Path

# Copy Maven if available
$script_dir = Split-Path -Parent $PSCommandPath
$maven = Join-Path $script_dir "build_exe\deps\apache-maven-3.9.8"
# Also check the cloned repo's copy
$maven_repo = Join-Path $src "scripts\appBuild\build_exe\deps\apache-maven-3.9.8"
if (Test-Path $maven) {
    Copy-Item $maven -Destination "$install\apache-maven-3.9.8" -Recurse -Force
} elseif (Test-Path $maven_repo) {
    Copy-Item $maven_repo -Destination "$install\apache-maven-3.9.8" -Recurse -Force
} else {
    Write-Host "Warning: Apache Maven not found"
}

# ---------------------------------------------------------------------------
# Build MSI
# ---------------------------------------------------------------------------
Write-Host "`n--- WiX MSI ---"
# WiX files are in the cloned repo
$wix_dir = Join-Path $src "scripts\appBuild\build_exe\wix"
Set-Location $wix_dir

$output_dir = Join-Path $root "builds\$product"
New-Item $output_dir -ItemType Directory -Force | Out-Null
$msi = Join-Path $output_dir "$product.msi"

$wixExe = (Get-Command wix).Source
& $wixExe build -arch x64 `
    -d "InstallFolderDir=$install" `
    -d "ProjectDir=$wix_dir\" `
    -d "ProductName=$product" `
    -d "Version=$wix_ver" `
    -out $msi `
    Package.wxs Folders.wxs

if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: MSI build failed"; Set-Location $root; Exit 1 }
Write-Host "MSI: $msi"

# ---------------------------------------------------------------------------
# Build Bundle (if prereqs available)
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

    & $wixExe build -arch x64 `
        -ext $ext `
        -d "LVPInstallFolderDir=$install" `
        -d "LVPMsiDir=$msi" `
        -d "PylonDriverDir=$pylon_msi" `
        -d "CorretoMsiDir=$corretto_msi" `
        -d "ProductName=$product" `
        -d "ProductVersion=$wix_ver" `
        -out $bundle `
        Bundle.wxs

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
Set-Location $root

# Clean temp
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
