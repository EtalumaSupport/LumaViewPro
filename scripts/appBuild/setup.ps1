# LumaViewPro Build Environment Setup
#
# Run this ONCE on a new build machine:
#   1. Open PowerShell
#   2. Paste this one line:
#      irm https://raw.githubusercontent.com/EtalumaSupport/LumaViewPro/main/scripts/appBuild/setup.ps1 | iex
#
# Or save this file and run it:
#   .\setup.ps1

$root = "C:\LumaViewPro"
$branch = "main"
$repo_url = "https://github.com/EtalumaSupport/LumaViewPro.git"

Write-Host "Setting up LumaViewPro build environment in $root"

# Create folder structure
New-Item -Path $root -ItemType Directory -Force | Out-Null
New-Item -Path "$root\dependencies" -ItemType Directory -Force | Out-Null
Set-Location $root

# Clone and extract build scripts
if (Test-Path "_getscript") { Remove-Item "_getscript" -Recurse -Force }
git clone --depth 1 --branch $branch $repo_url _getscript
Copy-Item "_getscript\scripts\appBuild\build.ps1" ".\build.ps1" -Force
Copy-Item "_getscript\scripts\appBuild\update_build_script.ps1" ".\update_build_script.ps1" -Force
Copy-Item "_getscript\scripts\appBuild\BUILD_INSTRUCTIONS.md" ".\BUILD_INSTRUCTIONS.md" -Force
Copy-Item "_getscript\scripts\appBuild\dependencies\README.md" ".\dependencies\README.md" -Force -ErrorAction SilentlyContinue

# Install Python dependencies (includes PyInstaller)
Write-Host "`nInstalling Python dependencies..."
pip install -r "_getscript\requirements-dev.txt"

Remove-Item "_getscript" -Recurse -Force

Write-Host ""
Write-Host "======================================="
Write-Host "  Setup complete!"
Write-Host "======================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Put dependencies in $root\dependencies\ (see dependencies\README.md)"
Write-Host "     - apache-maven-3.9.8\  (extract from Maven binary zip)"
Write-Host "     - Pylon USB Camera Driver MSI  (optional, for bundle)"
Write-Host "     - Amazon Corretto 8 JDK MSI    (optional, for bundle)"
Write-Host "  2. Run: .\build.ps1"
Write-Host ""
Write-Host "To update build scripts later: .\update_build_script.ps1"
Write-Host "======================================="
