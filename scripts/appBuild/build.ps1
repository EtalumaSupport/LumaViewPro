# LumaViewPro Build Script
#
# SETUP (one time):
#   1. Install tools: Python 3.12+, Git, WiX 6.x
#      (dotnet tool install --global wix --version 6.0.0)
#      The version is required, not advisory: this script refuses anything
#      but 6.x, because a v7-built bundle installs fine on the build box
#      and then fails at every customer install with 0x80070057.
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
# Build log capture
# ---------------------------------------------------------------------------
# Tee every byte that follows (Write-Host, native command stdout/stderr,
# Python warnings, WiX output, exit codes) into a timestamped log file
# alongside the console echo. Bench evidence 2026-05-08: when beta6
# silently shipped a broken imagecodecs and the soak failed, post-mortem
# required operator-captured console output; now the capture happens
# automatically and the operator just attaches the file.
$build_log_ts = (Get-Date -Format 'yyyyMMdd_HHmmss')
if (-not (Test-Path $artifacts)) { New-Item -ItemType Directory -Path $artifacts -Force | Out-Null }
$build_log = Join-Path $artifacts "build_$build_log_ts.log"
# If a prior build.ps1 invocation in this PowerShell session crashed without
# reaching its Stop-Transcript, an orphan transcript is still active and the
# next Start-Transcript silently writes to nothing. Drain any stale transcript
# stack before opening this run's log.
while ($true) {
    try { Stop-Transcript -ErrorAction Stop | Out-Null }
    catch { break }
}
Start-Transcript -Path $build_log -IncludeInvocationHeader | Out-Null

function Write-Phase {
    param([string]$Name)
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] === $Name ==="
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  LumaViewPro build starting at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "  Build host:   $env:COMPUTERNAME (user $env:USERNAME)"
Write-Host "  Script path:  $PSCommandPath"
Write-Host "  PowerShell:   $($PSVersionTable.PSVersion)"
Write-Host "  Branch param: '$Branch'"
Write-Host "  BuildType:    '$BuildType'"
Write-Host "  Build dir:    $build_dir"
Write-Host "  Log file:     $build_log"
Write-Host "================================================================"

# ---------------------------------------------------------------------------
# Select branch
# ---------------------------------------------------------------------------
if (-not $Branch) {
    # Built from the live remote, never hardcoded. The previous hardcoded
    # four offered 4.0.0-wave7-decomp and 4.1.0-dev, neither of which
    # exists on origin -- two of four entries failed at the clone.
    #
    # Ordering by recency needs commit DATES, and ls-remote cannot supply
    # them here: `git ls-remote --heads --sort=-committerdate` fatals with
    # "requires access to object data" outside a repo, and with "missing
    # object" inside an empty one. A blobless bare clone carries the
    # commits without any file contents -- measured 1.7 s / 3.5 MB against
    # this repo -- and is deleted again as soon as the list is read.
    $refcache = Join-Path $build_dir "_branchrefs.git"
    if (Test-Path $refcache) { Remove-Item $refcache -Recurse -Force }
    $ErrorActionPreference = "Continue"
    git clone --bare --filter=blob:none --no-tags -q $repo_url $refcache 2>&1 | Out-Null
    $refcache_exit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($refcache_exit -ne 0) {
        Write-Host "ERROR: could not read the branch list from $repo_url (git exit $refcache_exit)."
        Write-Host "  The build clones this same remote a few steps later, so this is"
        Write-Host "  almost certainly a network or credential problem that would stop"
        Write-Host "  the build anyway."
        Write-Host "  To skip the menu entirely:  .\build.ps1 -Branch <name>"
        Set-Location $build_dir
        Exit 1
    }

    # Release lines first so a release build is never buried under active
    # development, then the ten most recently updated branches. The release
    # pattern is derived rather than listed, so 4.0.1 / 4.1.0 age in with
    # no edit here; anything older or oddly named is still reachable by
    # typing it, or via -Branch.
    $release_pattern = '^[0-9]+\.[0-9]+\.[0-9]+(-beta[0-9]*)?$'
    $all_refs = @(& git --git-dir=$refcache for-each-ref --sort=-committerdate --format='%(refname:short)' refs/heads/)
    $releases = @($all_refs | Where-Object { $_ -match $release_pattern })
    $recent = @($all_refs | Where-Object { $releases -notcontains $_ } | Select-Object -First 10)
    $branches = @($releases) + @($recent)
    Remove-Item $refcache -Recurse -Force

    Write-Host "`nAvailable branches (releases first, then 10 most recent):"
    for ($i = 0; $i -lt $branches.Count; $i++) {
        Write-Host "  [$($i+1)] $($branches[$i])"
    }
    Write-Host "  [0] Enter custom branch"
    $choice = Read-Host -Prompt "Select branch (1-$($branches.Count), 0 for custom, or type a branch name directly)"
    if ($choice -eq "0" -or -not $choice) {
        $Branch = Read-Host -Prompt "Branch name"
    } elseif ($choice -match '^\d+$') {
        $idx = [int]$choice - 1
        if ($idx -ge 0 -and $idx -lt $branches.Count) {
            $Branch = $branches[$idx]
        } else {
            Write-Host "Invalid selection"
            Exit 1
        }
    } else {
        $Branch = $choice
    }
}
Write-Host "Building branch: $Branch"

# ---------------------------------------------------------------------------
# Find dependencies
# ---------------------------------------------------------------------------
$pylon_msi = ""

if (-not (Test-Path $deps)) {
    New-Item $deps -ItemType Directory -Force | Out-Null
    Write-Host "`nCreated dependencies\ folder. See dependencies\README.md for what to put there."
}

$pylon_files = Get-ChildItem -Path $deps -Filter "*pylon*USB*.msi" -ErrorAction SilentlyContinue
if ($pylon_files) { $pylon_msi = $pylon_files[0].FullName; Write-Host "Found Pylon: $pylon_msi" }

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

# VC++ Redistributable (x64). REQUIRED for the bundle: the app no longer
# ships msvcp140/concrt140 app-local (a stale bundled copy shadowed the
# system runtime and broke the IDS SDK's DLL init on client machines), so
# the installer must guarantee the system runtime exists on machines
# where LVP is the first thing installed. Download vc_redist.x64.exe
# from Microsoft and drop it in dependencies\. See BUILD_INSTRUCTIONS.md.
$vc_redist_exe = ""
$vc_redist_files = Get-ChildItem -Path $deps -Filter "vc_redist.x64.exe" -ErrorAction SilentlyContinue
if ($vc_redist_files) {
    $vc_redist_exe = $vc_redist_files[0].FullName
    Write-Host "Found VC++ Redistributable: $vc_redist_exe ($((Get-Item $vc_redist_exe).VersionInfo.FileVersion))"
}

if (-not $pylon_msi) { Write-Host "No Pylon MSI in dependencies\ - bundle will be skipped" }
if (-not $vc_redist_exe) { Write-Host "No vc_redist.x64.exe in dependencies\ - bundle will NOT be built (the app does not ship msvcp140; the installer must chain the redistributable)" }
if (-not $ids_peak_exe) { Write-Host "No IDS Peak runtime in dependencies\ - IDS cameras will need manual driver install on customer machines" }
if (-not $fx2_inf) { Write-Host "No FX2 WinUSB INF in dependencies\fx2\ - LVC FX2 driver will need manual install (Zadig)" }
if (-not $fx2_libusb_dll) { Write-Host "No FX2 libusb-1.0.dll in dependencies\fx2\ - bundled LVP will not be able to talk to FX2 cameras even if WinUSB driver is installed" }

# ---------------------------------------------------------------------------
# Check tools
# ---------------------------------------------------------------------------
Write-Host "`nChecking tools..."
try { $wix_version_raw = & wix --version 2>&1; Write-Host "  WiX: $wix_version_raw" } catch { Write-Host "ERROR: WiX not found. Run: dotnet tool install --global wix --version 6.0.0"; Exit 1 }

# Require WiX v6.x. Bundle.wxs is authored for the v4-v6
# WixToolset.Bal.wixext API. v7 restructured WixStdBA and added a required
# scope field in the Burn-BA plan protocol; bundles built with v7 still
# produce a -setup.exe but fail at customer install with 0x80070057 "Failed
# to read plan scope of BAEnginePlan args". Catch the wrong WiX up front,
# before the build wastes 5+ minutes producing a broken bundle.
#
# Written as an allowlist rather than a refusal of v7, because a refusal has
# to enumerate every way it can be fooled. `wix --version` goes through
# Out-String first: with 2>&1 it can arrive as an ARRAY, and -match against
# an array does not populate $Matches -- so a version test reading
# $matches[1] silently picks up whatever the branch menu above left there,
# and [int]$null is 0, which passes a "-ge 7" test. Array, empty, null and
# unparseable inputs now land in the same refusal as v5 and v7.
$wix_version_text = ($wix_version_raw | Out-String).Trim()
if ($wix_version_text -notmatch '^6\.') {
    Write-Host ""
    Write-Host "ERROR: WiX version '$wix_version_text' is not supported by this build."
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

# Capture HEAD SHA before .git is wiped; useful for tying a released
# installer back to a specific commit when the post-mortem starts from
# the bundle exe rather than the build log.
$ErrorActionPreference = "Continue"
$git_sha = (& git -C $clone rev-parse HEAD 2>$null).Trim()
$ErrorActionPreference = "Stop"
$git_sha_short = if ($git_sha) { $git_sha.Substring(0, 7) } else { '<unknown>' }
Write-Host "Git SHA: $git_sha ($git_sha_short)"

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
Write-Phase "Build Environment"
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
Write-Phase "PyInstaller"
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
# DEBUG level so the transcript names PyInstaller's binary-dependency
# search directories -- the record of WHERE each collected DLL came
# from. At WARN those lines are suppressed and a bad collected binary
# (e.g. a stale C runtime scavenged from the build box) is
# undiagnosable after the fact.
& $venv_python -m PyInstaller --log-level DEBUG .\lumaviewpro.spec
$pyi_exit = $LASTEXITCODE
$env:FX2_LIBUSB_DLL = $null
if ($pyi_exit -ne 0) { Write-Host "ERROR: PyInstaller failed"; Set-Location $build_dir; Exit 1 }

# The transcript is the ONLY artifact that survives the _tmp cleanup, so
# every freeze diagnostic must land in it (and a copy of the warn file
# lands next to the build log). Losing the warn file cost a full client
# round-trip diagnosing a module PyInstaller had flagged at build time.
Write-Host "--- PyInstaller warn file ---"
$warn_file = Get-ChildItem ".\build\lumaviewpro\warn-*.txt" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($warn_file) {
    Get-Content $warn_file.FullName | Write-Host
    Copy-Item $warn_file.FullName (Join-Path $artifacts "pyinstaller_warn_${version}_$build_log_ts.txt") -Force
} else {
    Write-Host "WARNING: no PyInstaller warn file found under .\build\lumaviewpro\"
}

# Archive the PyInstaller TOC manifests beside the warn file. They record
# the absolute SOURCE path of every collected binary -- COLLECT-00.toc is
# the post-dedup set that actually ships (Analysis-00.toc alone can name a
# losing duplicate; the Splash and Tree channels bypass it entirely) --
# and the temp tree that holds them is deleted at the end of every build.
# Without this copy, "which directory did that DLL come from" is
# unanswerable for any shipped artifact.
#
# Named by build timestamp as well as version, because the version alone
# is not a build identity: consecutive builds of one version overwrote
# each other's manifests, leaving the survivor unattributable and
# defeating the purpose above. ${version} is braced deliberately -- a bare
# $version_ parses the underscore as part of the variable name and
# expands to nothing, silently dropping the version from the filename.
foreach ($toc in Get-ChildItem ".\build\lumaviewpro\*.toc" -ErrorAction SilentlyContinue) {
    Copy-Item $toc.FullName (Join-Path $artifacts ("pyinstaller_" + $toc.BaseName + "_${version}_$build_log_ts.toc")) -Force
}

# Dist census + hard gate: an exe missing a camera-SDK package cannot see
# that camera class and MUST NOT ship silently -- the gap only surfaces on
# a client machine otherwise. Folder presence covers the binary halves;
# the spec asserts the pure-Python halves inside the frozen archive.
Write-Host "--- Dist camera-stack census (.\dist\lumaviewpro) ---"
$critical_pkgs = @('pypylon', 'ids_peak', 'ids_peak_ipl', 'ids_peak_afl', 'ids_peak_icv')
$census_missing = @()
foreach ($pkg in $critical_pkgs) {
    if (Test-Path ".\dist\lumaviewpro\$pkg") {
        Write-Host "  ${pkg}: present"
    } else {
        Write-Host "  ${pkg}: MISSING"
        $census_missing += $pkg
    }
}
if ($census_missing.Count -gt 0) {
    Write-Host "ERROR: critical camera packages missing from the frozen dist: $($census_missing -join ', ')"
    Write-Host "The exe would silently lack support for those cameras on client machines."
    Set-Location $build_dir
    Exit 1
}

# Content census: folder presence is not completeness -- a package dir can
# exist while the DLLs beside its extension module were dropped or altered,
# and the exe then fails the import only on a client machine. The build
# venv's installed wheels are the single source of truth: every pyd/dll in
# the venv package dir must exist in dist with the same byte size (size
# inequality also catches post-collection mangling, e.g. compression).
# pypylon is deliberately NOT content-checked: its PyInstaller hook prunes
# files legitimately, so a venv diff would false-fail; folder presence
# above still covers it.
$content_pkgs = @('ids_peak', 'ids_peak_ipl', 'ids_peak_afl', 'ids_peak_icv')
$content_bad = @()
foreach ($pkg in $content_pkgs) {
    $src_dir = Join-Path $venv "Lib\site-packages\$pkg"
    if (-not (Test-Path $src_dir)) { continue }
    $binaries = Get-ChildItem -Path $src_dir -File | Where-Object { $_.Extension -in '.pyd', '.dll' }
    foreach ($bin in $binaries) {
        $dist_file = ".\dist\lumaviewpro\$pkg\$($bin.Name)"
        if (-not (Test-Path $dist_file)) {
            $content_bad += "$pkg\$($bin.Name) (missing from dist)"
        } elseif ((Get-Item $dist_file).Length -ne $bin.Length) {
            $content_bad += "$pkg\$($bin.Name) (dist $((Get-Item $dist_file).Length) bytes vs wheel $($bin.Length))"
        }
    }
}
if ($content_bad.Count -gt 0) {
    Write-Host "ERROR: camera-SDK package content incomplete or altered in dist:"
    $content_bad | ForEach-Object { Write-Host "  $_" }
    Set-Location $build_dir
    Exit 1
}
Write-Host "  content census: ids_peak* pyd/dll name+size verified against the build venv"
# CRT policy: the spec's post-COLLECT gate removes app-root
# msvcp140/concrt140 (an app-local copy shadows System32 for the whole
# process, and a stale one broke the IDS SDK's DLL init on client
# machines); the chained VC++ Redistributable supplies them system-wide
# instead. Re-assert here on the exact tree WiX packages: a build whose
# spec gate did not run must die here, not on a client machine.
foreach ($crt in @('msvcp140.dll', 'concrt140.dll')) {
    if (Test-Path ".\dist\lumaviewpro\$crt") {
        Write-Host "ERROR: $crt present at dist top level - the CRT policy gate in the spec did not run; refusing to package a bundle that shadows the system runtime"
        Set-Location $build_dir
        Exit 1
    }
}
if (-not (Test-Path ".\dist\lumaviewpro\vcruntime140.dll")) {
    Write-Host "ERROR: vcruntime140.dll missing at dist top level - python3xx.dll imports it; the exe cannot start"
    Set-Location $build_dir
    Exit 1
}
# Redist-version floor: the chained vc_redist must be at least as new as
# the newest VC runtime any bundled wheel carries (pypylon ships its own
# msvcp140 inside its package dir). With no app-root msvcp, those
# extensions resolve msvcp from System32 -- resolving to an OLDER copy
# than they were built against is a missing-entry-point crash waiting on
# exactly the machines the redist chain is meant to protect.
$vc_dlls = Get-ChildItem ".\dist\lumaviewpro" -Recurse -Include 'msvcp140*.dll', 'vcruntime140*.dll', 'concrt140*.dll', 'vcomp140*.dll', 'ucrtbase.dll'
Write-Host "  CRT census (path : FileVersion) - the build record of every C-runtime file that ships:"
foreach ($dll in $vc_dlls) {
    $rel = $dll.FullName.Substring((Resolve-Path ".\dist\lumaviewpro").Path.Length + 1)
    Write-Host "    $rel : $($dll.VersionInfo.FileVersion)"
}
if ($vc_redist_exe) {
    # Order on VS_FIXEDFILEINFO's numeric fields, never on the FileVersion
    # display string. Vendors stamp free text into that string -- a shipped
    # runtime here reads "14.16.27052.0 built by: cloudtest" -- and casting
    # such a string to [version] yields $null, so filtering the nulls away
    # dropped that file out of the maximum without a word in the log.
    # Dropping can only LOWER the maximum, so what it hid was a false PASS;
    # and had no candidate parsed at all the maximum would go $null, which
    # -lt compares as False, letting the gate approve a build whose floor it
    # never computed. The numeric fields cannot carry free text and cannot
    # fail to parse, so both holes close at the source.
    # ucrtbase is excluded because it versions on Windows build numbers
    # (10.x) against this family's 14.x and would always dominate. vcomp140
    # is censused but not yet floored: its version has never been recorded
    # in a build, and gating on an unmeasured value is how a green build
    # turns red for the wrong reason.
    $vc_candidates = @($vc_dlls | Where-Object {
        $_.Name -notlike 'ucrtbase*' -and $_.Name -notlike 'vcomp140*'
    })
    $max_vc = ($vc_candidates | ForEach-Object {
        [version]::new($_.VersionInfo.FileMajorPart, $_.VersionInfo.FileMinorPart,
                       $_.VersionInfo.FileBuildPart, $_.VersionInfo.FilePrivatePart)
    } | Measure-Object -Maximum).Maximum
    if ($null -eq $max_vc) {
        Write-Host "ERROR: CRT floor check derived no version from any of the $($vc_candidates.Count) censused VC runtime file(s) - refusing to certify a floor that was never computed"
        Set-Location $build_dir
        Exit 1
    }
    $redist_info = (Get-Item $vc_redist_exe).VersionInfo
    $redist_ver = [version]::new($redist_info.FileMajorPart, $redist_info.FileMinorPart,
                                 $redist_info.FileBuildPart, $redist_info.FilePrivatePart)
    Write-Host "  CRT floor check: newest bundled VC runtime $max_vc; chained redist $redist_ver"
    if ($redist_ver -lt $max_vc) {
        Write-Host "ERROR: chained vc_redist ($redist_ver) is older than the newest bundled VC runtime ($max_vc) - update dependencies\vc_redist.x64.exe"
        Set-Location $build_dir
        Exit 1
    }
}

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

# ---------------------------------------------------------------------------
# Build MSI
# ---------------------------------------------------------------------------
Write-Phase "WiX MSI"
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
    # No trailing backslash: a quoted native-command arg ending in '\'
    # makes the closing quote escape, which corrupts the whole arg list
    # once $wix_dir contains a space (e.g. a user profile with a space).
    # Package.wxs supplies the path separator, same as InstallFolderDir.
    '-d', "ProjectDir=$wix_dir",
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
# Distinguishes "no bundle was attempted" (a deliberate MSI-only build) from
# "a bundle was attempted and failed". Both leave $bundle empty, so $bundle
# alone cannot tell the summary below which happened.
$bundle_failed = $false
if ($pylon_msi -and $vc_redist_exe) {
    Write-Host "`n--- WiX Bundle ---"
    $bundle = Join-Path $output_dir "$product-setup.exe"

    # Find BAL extension
    $bal_dep = Join-Path $src "scripts\appBuild\build_exe\deps\WixToolset.BootstrapperApplications.wixext.dll"
    $bal_script = Join-Path $script_dir "build_exe\deps\WixToolset.BootstrapperApplications.wixext.dll"
    # No feed fallback. The vendored DLL is version-pinned; `wix extension
    # add -g` fetches whatever the feed currently serves, and nothing here
    # inspects the extension's version -- the check above reads the WiX
    # TOOLCHAIN version, which a v7-era BAL extension passes untouched. A
    # feed-fetched extension is therefore the one way a bundle can look
    # clean here and still fail 0x80070057 on every customer machine.
    if (Test-Path $bal_dep) { $ext = $bal_dep }
    elseif (Test-Path $bal_script) { $ext = $bal_script }
    else {
        Write-Host "ERROR: WiX BAL extension not found. Looked in:"
        Write-Host "  $bal_dep"
        Write-Host "  $bal_script"
        Write-Host "  Copy WixToolset.BootstrapperApplications.wixext.dll to the second path."
        Write-Host "  It is tracked in the LumaViewPro repo under scripts\appBuild\build_exe\deps\,"
        Write-Host "  and is byte-identical to the WixToolset.BootstrapperApplications.wixext nuget package."
        Set-Location $build_dir
        Exit 1
    }

    Write-Host "Building bundle..."
    $bundle_args = @(
        'build', '-arch', 'x64',
        '-ext', $ext,
        '-d', "LVPInstallFolderDir=$install",
        '-d', "InstallerAssetsDir=$installer_assets_dir",
        '-d', "LVPMsiDir=$msi",
        '-d', "PylonDriverDir=$pylon_msi",
        '-d', "VCRedistExe=$vc_redist_exe",
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
        Write-Host "ERROR: Bundle build failed (wix exit $LASTEXITCODE)"
        $bundle = ""
        $bundle_failed = $true
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
if ($bundle_failed) {
    Write-Host "  BUILD INCOMPLETE - the bundle failed"
} else {
    Write-Host "  BUILD COMPLETE"
}
Write-Host "======================================="
Write-Host "  MSI:      $msi"
# A failed bundle must appear here. Omitting the line is how an attempted-
# and-failed bundle became indistinguishable from a deliberate MSI-only
# build, which prints no Bundle line either.
if ($bundle_failed) {
    Write-Host "  Bundle:   FAILED - see the wix errors above; the MSI above is still usable"
} elseif ($bundle -and (Test-Path $bundle)) {
    Write-Host "  Bundle:   $bundle"
}
Write-Host "  Output:   $output_dir"
Write-Host "  Version:  $version"
Write-Host "  Git SHA:  $git_sha_short  ($git_sha)"
Write-Host "  Log:      $build_log"
Write-Host "  Ended:    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "======================================="

# Copy the build log into the version-specific output dir so it ships
# alongside the MSI / bundle. The primary copy in $artifacts root
# remains for cross-build sorting by timestamp.
if (Test-Path $output_dir) {
    try {
        Copy-Item $build_log -Destination (Join-Path $output_dir "build.log") -Force
    } catch {
        Write-Host "Note: build.log copy into $output_dir failed: $_"
    }
}

Stop-Transcript | Out-Null

# Non-zero ONLY when a bundle was attempted and failed. A deliberate MSI-only
# build never attempted one and still exits 0, which several documented
# workflows rely on.
#
# Placed after Stop-Transcript deliberately: every Exit above this point
# bypasses both the transcript close and the build.log copy, so exiting at the
# failure site would destroy the log for the one run that most needs it.
if ($bundle_failed) { Exit 1 }
