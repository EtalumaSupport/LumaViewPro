# LumaViewPro Build Script
#
# SETUP (one time):
#   1. Create C:\LumaViewPro\
#   2. Copy this script to C:\LumaViewPro\build.ps1
#   3. Put dependencies in C:\LumaViewPro\dependencies\ (see dependencies\README.md)
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
$script_dir = Split-Path -Parent $PSCommandPath

# Make sure we're not stuck inside a previous build
Set-Location $root

# ---------------------------------------------------------------------------
# Select branch
# ---------------------------------------------------------------------------
if (-not $Branch) {
    $branches = @(
        "4.0.0-beta"
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
# When run from C:\LumaViewPro\build.ps1, $script_dir = C:\LumaViewPro\
# When run from repo, $script_dir = scripts\appBuild\
# Either way, dependencies\ is next to the script.
$deps = Join-Path $script_dir "dependencies"
if (-not (Test-Path $deps)) {
    # Also check repo layout (build.ps1 copied to $root, deps in repo)
    $deps = Join-Path $root "dependencies"
}

$pylon_msi = ""
$corretto_msi = ""
$maven_dir = ""

if (Test-Path $deps) {
    $pylon_files = Get-ChildItem -Path $deps -Filter "*pylon*USB*.msi" -ErrorAction SilentlyContinue
    if ($pylon_files) { $pylon_msi = $pylon_files[0].FullName; Write-Host "Found Pylon: $pylon_msi" }

    $corretto_files = Get-ChildItem -Path $deps -Filter "*corretto*.msi" -ErrorAction SilentlyContinue
    if ($corretto_files) { $corretto_msi = $corretto_files[0].FullName; Write-Host "Found Corretto: $corretto_msi" }

    $maven_files = Get-ChildItem -Path $deps -Directory -Filter "apache-maven*" -ErrorAction SilentlyContinue
    if ($maven_files) { $maven_dir = $maven_files[0].FullName; Write-Host "Found Maven: $maven_dir" }
}

if (-not $pylon_msi) { Write-Host "No Pylon MSI in dependencies\ - bundle will be skipped" }
if (-not $corretto_msi) { Write-Host "No Corretto MSI in dependencies\ - bundle will be skipped" }
if (-not $maven_dir) { Write-Host "Warning: Apache Maven not found in dependencies\ - ImageJ will not work in installed app" }

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

# Create install directory
$install = ".\dist\$product"
New-Item $install -ItemType Directory -Force | Out-Null
Copy-Item ".\dist\lumaviewpro\*" -Destination $install -Recurse
$install = (Resolve-Path $install).Path

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
