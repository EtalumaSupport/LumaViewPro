# handle_snapshot_loop.ps1 — periodic per-handle-type snapshot of the LVP
# python.exe process, for INS-2 handle-leak shape diagnosis.
#
# Lifetime: TEMPORARY instrumentation, lives on perf-instrumentation-4.0.0-beta.
# Remove once INS-2 handle accumulation shape is characterized and the
# investigation closes.
#
# Usage (run AFTER LVP is launched + protocol is started):
#   .\scripts\handle_snapshot_loop.ps1 -OutDir D:\handles_overnight_2026-04-30
#
# Optional:
#   -HandleExe <path>          Override handle64.exe location (default: PATH search + common SysInternals dirs)
#   -MaxDurationHours <n>      Auto-stop after N hours (default 24, safety net)
#   -PythonPid <int>           Skip auto-detection, use this PID directly
#
# Output (in -OutDir):
#   handles_T{HhMMmSSs}.txt    one snapshot per file, sortable by filename
#   manifest.csv               timestamp_iso,elapsed_seconds,filename,total_handles
#   run.log                    script's own status messages
#
# Cadence: dense early to capture startup curve, sparse later.
#   T0, T1min, T2min, T5min, T10min, T20min, T30min, then every 30 min
#
# Ctrl-C to stop cleanly. Snapshots already written are preserved.

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$OutDir,
    [string]$HandleExe = '',
    [double]$MaxDurationHours = 24,
    [int]$PythonPid = 0
)

# --- Resolve handle64.exe location ---
function Find-HandleExe {
    param([string]$Override)
    if ($Override -and (Test-Path $Override)) { return $Override }
    # Try PATH
    $cmd = Get-Command handle64.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    # Try common SysInternals install dirs
    $candidates = @(
        "$env:USERPROFILE\Tools\SysInternals\handle64.exe",
        "$env:USERPROFILE\Downloads\SysInternals\handle64.exe",
        "C:\Tools\SysInternals\handle64.exe",
        "C:\Program Files\SysInternals\handle64.exe",
        "C:\SysInternals\handle64.exe",
        "$PSScriptRoot\handle64.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

$handleExe = Find-HandleExe -Override $HandleExe
if (-not $handleExe) {
    Write-Error "handle64.exe not found. Pass -HandleExe <path> or place it on PATH."
    exit 2
}

# --- Resolve LVP python.exe PID ---
function Find-LvpPid {
    $procs = Get-Process python -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -like '*LumaViewPro*' }
    if (-not $procs) { return 0 }
    if ($procs.Count -gt 1) {
        Write-Warning "Multiple LumaViewPro python processes found; using the first one."
        $procs = $procs | Select-Object -First 1
    }
    return $procs.Id
}

if ($PythonPid -le 0) {
    $PythonPid = Find-LvpPid
    if ($PythonPid -le 0) {
        Write-Error "LumaViewPro python.exe not found. Launch LVP first, or pass -PythonPid <id>."
        exit 3
    }
}

# Confirm the PID still exists
try {
    $procCheck = Get-Process -Id $PythonPid -ErrorAction Stop
    Write-Host "Targeting PID $PythonPid ($($procCheck.ProcessName), title='$($procCheck.MainWindowTitle)')"
} catch {
    Write-Error "PID $PythonPid is not running."
    exit 4
}

# --- Set up output dir ---
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}
$resolvedOut = (Resolve-Path $OutDir).Path
$manifestPath = Join-Path $resolvedOut 'manifest.csv'
$logPath = Join-Path $resolvedOut 'run.log'

if (-not (Test-Path $manifestPath)) {
    'timestamp_iso,elapsed_seconds,filename,total_handles' | Out-File -FilePath $manifestPath -Encoding ascii
}

function Write-Log {
    param([string]$Msg)
    $line = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')  $Msg"
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

Write-Log "handle_snapshot_loop.ps1 starting. PID=$PythonPid OutDir=$resolvedOut HandleExe=$handleExe MaxDurationHours=$MaxDurationHours"

# --- Cadence schedule (seconds from t0) ---
# Dense at startup, sparse later.
$schedule = @(0, 60, 120, 300, 600, 1200, 1800)  # 0, 1m, 2m, 5m, 10m, 20m, 30m
$next = 1800
$maxSec = [int]($MaxDurationHours * 3600)
while ($next -le $maxSec) {
    $next += 1800  # every 30 min after T30
    $schedule += $next
}

# --- One-time EULA acceptance via dummy invocation ---
try {
    & $handleExe -accepteula 2>&1 | Out-Null
} catch {
    Write-Log "Warning: EULA pre-accept failed (may be already accepted): $_"
}

# --- Snapshot loop ---
$t0 = Get-Date

function Take-Snapshot {
    param([int]$ElapsedSec)
    $hours = [int]([Math]::Floor($ElapsedSec / 3600))
    $minutes = [int]([Math]::Floor(($ElapsedSec % 3600) / 60))
    $seconds = [int]($ElapsedSec % 60)
    $stem = ('handles_T{0:D2}h{1:D2}m{2:D2}s' -f $hours, $minutes, $seconds)
    $outFile = Join-Path $resolvedOut "$stem.txt"

    try {
        & $handleExe -p $PythonPid -s 2>&1 | Out-File -FilePath $outFile -Encoding ascii
    } catch {
        Write-Log "Snapshot at T+${ElapsedSec}s FAILED: $_"
        return
    }

    # Parse total handles from the snapshot
    $total = 'NA'
    try {
        $totalLine = Get-Content $outFile | Where-Object { $_ -match 'Total handles:' } | Select-Object -First 1
        if ($totalLine -match 'Total handles:\s*(\d+)') {
            $total = $Matches[1]
        }
    } catch {
        # leave NA
    }

    $iso = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'
    "$iso,$ElapsedSec,$stem.txt,$total" | Add-Content -Path $manifestPath
    Write-Log "Snapshot T+${ElapsedSec}s ($stem.txt, total=$total)"
}

try {
    foreach ($targetSec in $schedule) {
        $now = (Get-Date) - $t0
        $waitSec = $targetSec - [int]$now.TotalSeconds
        if ($waitSec -gt 0) {
            Start-Sleep -Seconds $waitSec
        }

        # Re-check process is still running
        $stillRunning = Get-Process -Id $PythonPid -ErrorAction SilentlyContinue
        if (-not $stillRunning) {
            Write-Log "PID $PythonPid no longer running; stopping."
            break
        }

        Take-Snapshot -ElapsedSec $targetSec
    }
} finally {
    Write-Log "handle_snapshot_loop.ps1 exiting. Total snapshots in $resolvedOut."
}
