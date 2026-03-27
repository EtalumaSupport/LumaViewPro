# LumaViewPro Windows Release Build Script
#
# Usage: .\build_win_release.ps1 [-CreateZip] [-Branch <name>]
#
# Clones the repo, reads version from version.txt, builds EXE via PyInstaller,
# builds MSI via WiX, optionally builds Bundle with Pylon + Corretto MSIs.
#
# Prerequisites: Python 3, PyInstaller, WiX Toolset v6, Git

param(
    [switch]$CreateZip = $false,
    [string]$Branch = ""
)

$ErrorActionPreference = "Stop"

$repo_url = "https://github.com/EtalumaSupport/LumaViewPro.git"
$script_dir = Split-Path -Parent $PSCommandPath
$build_root = Join-Path -Path $script_dir -ChildPath "build_output"

# ---------------------------------------------------------------------------
# 1. Get branch name
# ---------------------------------------------------------------------------
if (-not $Branch) {
    $Branch = Read-Host -Prompt "Branch to build (e.g., 4.0.0-beta)"
}

# ---------------------------------------------------------------------------
# 2. Load dependency paths from config (optional)
# ---------------------------------------------------------------------------
$config_file = Join-Path -Path $script_dir -ChildPath "config\build_dependencies.json"
$pylon_msi = ""
$corretto_msi = ""
$wix_bal_ext = ""

if (Test-Path $config_file) {
    try {
        $config = Get-Content -Path $config_file -Raw | ConvertFrom-Json
        if ($config.pylon_driver_msi -and (Test-Path $config.pylon_driver_msi)) {
            $pylon_msi = $config.pylon_driver_msi
            Write-Host "Pylon MSI: $pylon_msi"
        }
        if ($config.corretto_jdk_msi -and (Test-Path $config.corretto_jdk_msi)) {
            $corretto_msi = $config.corretto_jdk_msi
            Write-Host "Corretto MSI: $corretto_msi"
        }
        if ($config.wix_bal_extension -and (Test-Path $config.wix_bal_extension)) {
            $wix_bal_ext = $config.wix_bal_extension
        }
    } catch {
        Write-Host "Warning: Could not read config file: $_"
    }
}

if (-not $pylon_msi) {
    $pylon_msi = Read-Host -Prompt "Path to Pylon driver MSI (blank to skip bundle)"
}
if (-not $corretto_msi) {
    $corretto_msi = Read-Host -Prompt "Path to Amazon Corretto JDK MSI (blank to skip bundle)"
}

# ---------------------------------------------------------------------------
# 3. Verify prerequisites
# ---------------------------------------------------------------------------
Write-Host "`n--- Checking prerequisites ---"

try {
    $wix_ver = & wix --version 2>&1
    Write-Host "WiX Toolset: $wix_ver"
} catch {
    Write-Host "ERROR: WiX Toolset v6 not found. Install with: dotnet tool install --global wix"
    Exit 1
}

try {
    $py_ver = & python --version 2>&1
    Write-Host "Python: $py_ver"
} catch {
    Write-Host "ERROR: Python not found in PATH"
    Exit 1
}

# ---------------------------------------------------------------------------
# 4. Clean and create build directory
# ---------------------------------------------------------------------------
if (Test-Path $build_root) {
    Write-Host "Cleaning previous build..."
    Remove-Item $build_root -Recurse -Force
}
New-Item -Path $build_root -ItemType Directory -Force | Out-Null

# ---------------------------------------------------------------------------
# 5. Clone repo
# ---------------------------------------------------------------------------
$clone_dir = Join-Path -Path $build_root -ChildPath "repo"
Write-Host "`n--- Cloning $Branch ---"
git clone --depth 1 --branch $Branch $repo_url $clone_dir
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: git clone failed"
    Exit 1
}
Remove-Item "$clone_dir/.git*" -Recurse -Force

# ---------------------------------------------------------------------------
# 6. Read version from version.txt
# ---------------------------------------------------------------------------
$version_raw = (Get-Content -Path "$clone_dir/version.txt" -TotalCount 1).Trim()
if ($version_raw -match '^\S+') {
    $version = $matches[0]
} else {
    Write-Host "ERROR: Could not parse version from version.txt: '$version_raw'"
    Exit 1
}

$product_name = "LumaViewPro-$version"

# WiX needs numeric-only version (x.x.x)
$wix_version = $version
if ($version -match '^(\d+\.\d+\.\d+)') {
    $wix_version = $matches[1]
}

# Rename clone directory to include version
$repo_dir = Join-Path -Path $build_root -ChildPath $product_name
Rename-Item -Path $clone_dir -NewName $product_name
$repo_dir = Join-Path -Path $build_root -ChildPath $product_name

Write-Host "`n=========================================="
Write-Host "Building: $product_name"
Write-Host "Version:  $version (WiX: $wix_version)"
Write-Host "Source:   $repo_dir"
Write-Host "=========================================="

# ---------------------------------------------------------------------------
# 7. Copy license files
# ---------------------------------------------------------------------------
Set-Location -Path $repo_dir
Copy-Item '.\licenses\*' -Destination '.\' -Force

# ---------------------------------------------------------------------------
# 8. Create source archives (optional)
# ---------------------------------------------------------------------------
$artifact_dir = Join-Path -Path $build_root -ChildPath "artifacts"
New-Item -Path $artifact_dir -ItemType Directory -Force | Out-Null

if ($CreateZip) {
    Write-Host "`n--- Creating source archives ---"
    Set-Location -Path $build_root
    Compress-Archive -Path ".\$product_name" -DestinationPath "$artifact_dir\$product_name-source.zip" -CompressionLevel Optimal
    tar czf "$artifact_dir\$product_name-source.tar.gz" ".\$product_name"
}

# ---------------------------------------------------------------------------
# 9. Build EXE with PyInstaller
# ---------------------------------------------------------------------------
Write-Host "`n--- Building EXE with PyInstaller ---"
Set-Location -Path $repo_dir
Copy-Item '.\scripts\appBuild\config\lumaviewpro_win_release.spec' '.\lumaviewpro.spec'
pyinstaller --log-level INFO .\lumaviewpro.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyInstaller build failed"
    Set-Location $script_dir
    Exit 1
}

# Rename output directory
$pyinstaller_output = ".\dist\lumaviewpro"
$install_dir = ".\dist\$product_name"
New-Item -Path $install_dir -ItemType Directory -Force | Out-Null
Copy-Item -Path "$pyinstaller_output\*" -Destination $install_dir -Recurse
$install_dir = Resolve-Path -Path $install_dir | Select-Object -ExpandProperty Path

# Copy Apache Maven from deps
$maven_source = Join-Path -Path $script_dir -ChildPath "build_exe\deps\apache-maven-3.9.8"
if (Test-Path $maven_source) {
    Copy-Item -Path $maven_source -Destination (Join-Path -Path $install_dir -ChildPath "apache-maven-3.9.8") -Recurse -Force
    Write-Host "Apache Maven copied to build output"
} else {
    Write-Host "Warning: Apache Maven not found at $maven_source"
}

if ($CreateZip) {
    Compress-Archive -Path $install_dir -DestinationPath "$artifact_dir\$product_name.zip" -CompressionLevel Optimal
}

# ---------------------------------------------------------------------------
# 10. Build MSI with WiX
# ---------------------------------------------------------------------------
Write-Host "`n--- Building MSI with WiX ---"
$wix_dir = Join-Path -Path $script_dir -ChildPath "build_exe\wix"
$msi_output_dir = Join-Path -Path $build_root -ChildPath "installers"
New-Item -Path $msi_output_dir -ItemType Directory -Force | Out-Null
$msi_path = Join-Path -Path $msi_output_dir -ChildPath "$product_name.msi"

Set-Location -Path $wix_dir

$wixExe = (Get-Command wix).Source
& $wixExe build -arch x64 `
    -d "InstallFolderDir=$install_dir" `
    -d "ProjectDir=$wix_dir\" `
    -d "ProductName=$product_name" `
    -d "Version=$wix_version" `
    -out $msi_path `
    Package.wxs Folders.wxs

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: MSI build failed"
    Set-Location $script_dir
    Exit 1
}
Write-Host "MSI built: $msi_path"

# ---------------------------------------------------------------------------
# 11. Build Bundle/EXE with WiX (optional — needs Pylon + Corretto MSIs)
# ---------------------------------------------------------------------------
$bundle_path = ""
if ($pylon_msi -and $corretto_msi -and (Test-Path $pylon_msi) -and (Test-Path $corretto_msi)) {
    Write-Host "`n--- Building Bundle installer ---"
    $bundle_path = Join-Path -Path $msi_output_dir -ChildPath "$product_name-setup.exe"

    # Find BAL extension
    $deps_bal = Join-Path -Path $script_dir -ChildPath "build_exe\deps\WixToolset.BootstrapperApplications.wixext.dll"
    if (Test-Path $deps_bal) {
        $ext_ref = $deps_bal
    } elseif ($wix_bal_ext) {
        $ext_ref = $wix_bal_ext
    } else {
        # Try package manager
        & wix extension add -g WixToolset.Bal.wixext 2>&1 | Out-Null
        $ext_ref = "WixToolset.Bal.wixext"
    }

    & $wixExe build -arch x64 `
        -ext $ext_ref `
        -d "LVPInstallFolderDir=$install_dir" `
        -d "LVPMsiDir=$msi_path" `
        -d "PylonDriverDir=$pylon_msi" `
        -d "CorretoMsiDir=$corretto_msi" `
        -d "ProductName=$product_name" `
        -d "ProductVersion=$wix_version" `
        -out $bundle_path `
        Bundle.wxs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Bundle build failed"
        $bundle_path = ""
    } else {
        Write-Host "Bundle built: $bundle_path"
    }
} else {
    Write-Host "`nSkipping bundle (Pylon and/or Corretto MSI not provided)"
}

# ---------------------------------------------------------------------------
# 12. Done
# ---------------------------------------------------------------------------
Set-Location $script_dir

Write-Host "`n=========================================="
Write-Host "BUILD COMPLETE: $product_name"
Write-Host "=========================================="
Write-Host "MSI:    $msi_path"
if ($bundle_path -and (Test-Path $bundle_path)) {
    Write-Host "Bundle: $bundle_path"
}
if ($CreateZip) {
    Write-Host "Source: $artifact_dir\$product_name-source.zip"
    Write-Host "EXE:   $artifact_dir\$product_name.zip"
}
Write-Host "=========================================="
