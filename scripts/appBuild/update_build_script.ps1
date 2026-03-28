# Updates build.ps1 from the latest repo
# Run from C:\LumaViewPro\

param(
    [string]$Branch = "4.0.0-beta"
)

$root = "C:\LumaViewPro"
Set-Location $root

$repo_url = "https://github.com/EtalumaSupport/LumaViewPro.git"

if (Test-Path "_getscript") { Remove-Item "_getscript" -Recurse -Force }

$ErrorActionPreference = "Continue"
git clone --depth 1 --branch $Branch $repo_url _getscript
$ErrorActionPreference = "Stop"

Copy-Item "_getscript\scripts\appBuild\build.ps1" ".\build.ps1" -Force
Copy-Item "_getscript\scripts\appBuild\update_build_script.ps1" ".\update_build_script.ps1" -Force
Remove-Item "_getscript" -Recurse -Force

Write-Host "build.ps1 updated from $Branch"
