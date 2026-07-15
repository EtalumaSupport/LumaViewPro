# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Profile a running LumaViewPro process: py-spy self-time x psutil total CPU
-> a stamped, ranked absolute-per-function CPU artifact.

Out-of-process by construction -- it attaches to the live PID, so it adds no
load to LVP's hot path. Run it on the bench box against a live session:

    python -m tools.profiling.profile_session --pid <LVP_PID> \
        --duration 60 --rate 50 --scenario liveview-fit \
        --settings-json data/current.json

py-spy attaches without elevation on Windows; on macOS it needs root, so the
real runs happen on the bench (the aggregation math is unit-tested off-bench).
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from tools.profiling.aggregate import compute_absolute_cpu, parse_folded

# py-spy is a sampling profiler: a low rate is cheap but coarse, a long window
# recovers the resolution. 50 Hz over 60 s = 3000 votes (~+-2% on a 10% fn).
DEFAULT_RATE_HZ = 50
DEFAULT_DURATION_S = 60


def _repo_root() -> Path:
    # tools/profiling/profile_session.py -> repo root two levels up.
    return Path(__file__).resolve().parents[2]


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ['git', '-C', str(root), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else 'unknown'
    except (subprocess.SubprocessError, OSError):
        return 'unknown'


def _build_identity(root: Path) -> dict:
    version = 'unknown'
    version_txt = root / 'version.txt'
    if version_txt.exists():
        version = version_txt.read_text(encoding='utf-8', errors='replace').splitlines()[0].strip()
    return {
        'git_sha': _git(root, 'rev-parse', 'HEAD'),
        'git_branch': _git(root, 'rev-parse', '--abbrev-ref', 'HEAD'),
        'version': version,
    }


def _config_snapshot(settings_json: Path | None) -> dict:
    # A profile is only comparable to another at the SAME config. Snapshot the
    # knobs that move CPU so a later compare can refuse mismatched runs.
    keys = ('live_view_fps', 'preview_host_downscale')
    snap: dict = {'settings_json': str(settings_json) if settings_json else None}
    if settings_json and settings_json.exists():
        try:
            data = json.loads(settings_json.read_text(encoding='utf-8', errors='replace'))
            for key in keys:
                if key in data:
                    snap[key] = data[key]
        except (json.JSONDecodeError, OSError):
            snap['settings_read_error'] = True
    return snap


def _sample_total_cpu(pid: int, duration_s: float, out: list[float], stop: threading.Event) -> None:
    """Sample the target process's total CPU (all threads, all cores) once per
    second for the window. psutil percent is relative to one core (>100% on
    multiple cores); the first read primes the baseline and is discarded."""
    try:
        proc = psutil.Process(pid)
        proc.cpu_percent(interval=None)  # prime; first value is always 0.0
    except psutil.Error:
        return
    deadline = time.monotonic() + duration_s
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            out.append(proc.cpu_percent(interval=1.0))
        except psutil.Error:
            break


def _pyspy_path(interpreter_dir: Path | None = None) -> str:
    """Absolute path to the py-spy binary.

    A bare ``py-spy`` fails under ``sudo`` (needed to attach on macOS) because
    sudo replaces PATH with a sanitized secure_path that omits the interpreter's
    bin dir. py-spy is installed alongside the running interpreter, so resolve it
    there first, then fall back to PATH for the case where it lives elsewhere.
    """
    interpreter_dir = interpreter_dir or Path(sys.executable).parent
    candidate = interpreter_dir / 'py-spy'
    if candidate.exists():
        return str(candidate)
    found = shutil.which('py-spy')
    if found:
        return found
    raise FileNotFoundError(
        f'py-spy not found next to the interpreter ({candidate}) or on PATH. '
        f'Install it (pip install py-spy).'
    )


def _run_pyspy(pid: int, duration_s: int, rate_hz: int, raw_path: Path) -> None:
    # -f raw = folded stacks; no --native (Python frames only, the cheap mode).
    result = subprocess.run(
        [
            _pyspy_path(),
            'record',
            '--pid',
            str(pid),
            '--format',
            'raw',
            '--rate',
            str(rate_hz),
            '--duration',
            str(duration_s),
            '--output',
            str(raw_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Surface py-spy's own reason (permission, dead pid, non-Python target)
        # instead of a bare CalledProcessError -- the failure must be legible.
        hint = ''
        if 'root' in result.stderr.lower() or 'permission' in result.stderr.lower():
            hint = ' -- on macOS run the whole command under sudo.'
        raise RuntimeError(
            f'py-spy failed (exit {result.returncode}): {result.stderr.strip()}{hint}'
        )


def profile(
    pid: int,
    duration_s: int,
    rate_hz: int,
    scenario: str,
    outdir: Path,
    settings_json: Path | None,
) -> Path:
    """Run one profiling capture and write the artifact. Returns its path."""
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    raw_path = outdir / f'pyspy_{stamp}.raw'

    cpu_samples: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_total_cpu, args=(pid, duration_s, cpu_samples, stop), daemon=True
    )
    sampler.start()
    _run_pyspy(pid, duration_s, rate_hz, raw_path)
    stop.set()
    sampler.join(timeout=3)

    folded = raw_path.read_text(encoding='utf-8', errors='replace')
    self_counts, total_samples, skipped = parse_folded(folded)
    mean_pct = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    total_cores = mean_pct / 100.0
    functions = compute_absolute_cpu(self_counts, total_samples, total_cores)

    try:
        cmdline = ' '.join(psutil.Process(pid).cmdline())
    except psutil.Error:
        cmdline = 'unknown'

    root = _repo_root()
    artifact = {
        'manifest': {
            'timestamp': stamp,
            'machine': platform.node(),
            'os': platform.platform(),
            'scenario': scenario,
            **_build_identity(root),
            'pid': pid,
            'cmdline': cmdline,
            'duration_s': duration_s,
            'rate_hz': rate_hz,
            'total_samples': total_samples,
            'skipped_lines': skipped,
            'total_process_cpu_pct': mean_pct,
            'total_process_cpu_cores': total_cores,
            'cpu_samples_pct': cpu_samples,
            'config': _config_snapshot(settings_json),
        },
        'functions': [asdict(f) for f in functions],
    }
    artifact_path = outdir / f'profile_{scenario}_{stamp}.json'
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    _print_ranked(artifact)
    return artifact_path


def _print_ranked(artifact: dict, top_n: int = 20) -> None:
    m = artifact['manifest']
    print(
        f'\nProfile: {m["scenario"]} | {m["version"]} {m["git_sha"][:8]} | {m["machine"]}\n'
        f'  {m["total_samples"]} samples @ {m["rate_hz"]} Hz over {m["duration_s"]}s | '
        f'process CPU {m["total_process_cpu_pct"]:.0f}% ({m["total_process_cpu_cores"]:.2f} cores)'
    )
    if m['skipped_lines']:
        print(f'  WARNING: {m["skipped_lines"]} folded lines skipped (format drift?)')
    print(f'\n  {"self CPU%":>9}  {"ms/s":>7}  {"+-95%":>6}  function')
    for fn in artifact['functions'][:top_n]:
        print(
            f'  {fn["cpu_cores"] * 100:8.1f}%  {fn["cpu_ms_per_s"]:7.1f}  '
            f'{fn["err_cores_95"] * 100:5.1f}%  {fn["function"]}'
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Absolute per-function CPU profile of a live LVP.')
    parser.add_argument('--pid', type=int, required=True, help='PID of the running LumaViewPro')
    parser.add_argument(
        '--duration', type=int, default=DEFAULT_DURATION_S, help='seconds to sample'
    )
    parser.add_argument('--rate', type=int, default=DEFAULT_RATE_HZ, help='py-spy samples/sec')
    parser.add_argument('--scenario', required=True, help='label, e.g. liveview-fit (for compare)')
    parser.add_argument(
        '--settings-json',
        type=Path,
        default=None,
        help='LVP current.json/settings.json to snapshot',
    )
    parser.add_argument('--outdir', type=Path, default=Path('logs/cpu_profile'))
    args = parser.parse_args(argv)
    profile(args.pid, args.duration, args.rate, args.scenario, args.outdir, args.settings_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
