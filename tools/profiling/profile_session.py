# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Profile a running LumaViewPro process: sampled self-time x psutil total CPU
-> a stamped, ranked absolute-per-function CPU artifact.

Out-of-process by construction -- it attaches to the live PID, so it adds no
load to LVP's hot path. Run it on the bench box against a live session:

    python -m tools.profiling.profile_session --pid <LVP_PID> \
        --duration 60 --rate 50 --scenario liveview-fit \
        --settings-json data/current.json

The sampler is py-spy, which attaches without elevation on Windows, and on
macOS austin's on-CPU mode, run under sudo against a LumaViewPro started from a
Homebrew Python: macOS refuses any attach to python.org's hardened interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from tools.profiling.aggregate import compute_absolute_cpu, parse_folded

# The samplers are sampling profilers: a low rate is cheap but coarse, a long window
# recovers the resolution. 50 Hz over 60 s = 3000 votes (~+-2% on a 10% fn).
DEFAULT_RATE_HZ = 50
DEFAULT_DURATION_S = 60


def _repo_root() -> Path:
    # tools/profiling/profile_session.py -> repo root two levels up.
    return Path(__file__).resolve().parents[2]


# The launch banner's lines (lvp_logger.log_environment_banner), by label. The
# banner opens with Version and closes with Git. The round-trip test writes a
# banner with the real function and reads it back here, so a relabelled banner
# fails a test rather than every artifact.
_BANNER_LINE = re.compile(
    r'^\[\w+\] \[[^\]]*\] (\d\d/\d\d/\d{4} \d\d:\d\d:\d\d\.\d{3}) - lvp_logger\.py - '
    r'\[LVP Main  \] (\w+):\s+(.*)$'
)
_BANNER_FIELDS = {
    'Version': 'version',
    'Built': 'built',
    'CommitGUID': 'commit_guid',
    'BuildID': 'build_id',
    'Runtime': 'runtime',
    'PID': 'pid',
    'Git': 'git',
}
# A banner's time is read to the second, in local wall time; the process's
# start is psutil's, finer. A banner written in the process's first second counts.
_START_SLACK_S = 1.0


def _banners(path: Path) -> list[tuple[float, dict]]:
    """Every complete banner in one log file, oldest first, with its time."""
    banners = []
    current: dict | None = None
    started = 0.0
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = _BANNER_LINE.match(line)
        if not match or match.group(2) not in _BANNER_FIELDS:
            continue
        stamp, label, value = match.groups()
        if label == 'Version':
            current = {}
            started = time.mktime(time.strptime(stamp[:-4], '%m/%d/%Y %H:%M:%S'))
        if current is None:
            continue
        current[_BANNER_FIELDS[label]] = value.strip()
        if label == 'Git':
            banners.append((started, current))
            current = None
    return banners


def _build_of(pid: int) -> dict:
    """The build the process names in its own launch banner, or why there is none.

    The profiled process's log is the one it holds open, so the profiler never
    guesses a data folder. Its banner is the newest one naming its PID and
    written after it started: a recycled PID's banner in an old backup is not
    it, and a process that wrote no banner (any LumaViewPro host but the GUI
    today) has none.
    """
    try:
        proc = psutil.Process(pid)
        started = proc.create_time()
        logs = [Path(f.path) for f in proc.open_files() if Path(f.path).name == 'lumaviewpro.log']
    except psutil.Error as e:
        return {'build': None, 'build_identity_source': f'process {pid} not readable: {e}'}
    if len(logs) != 1:
        held = 'no LumaViewPro log' if not logs else f'{len(logs)} LumaViewPro logs'
        return {'build': None, 'build_identity_source': f'{held} open in process {pid}'}
    # The live log and its rotated backups, newest first.
    files = sorted(logs[0].parent.glob('lumaviewpro.*'), key=lambda p: p.stat().st_mtime)
    for path in reversed(files):
        for when, banner in reversed(_banners(path)):
            if banner.get('pid') == str(pid) and when >= started - _START_SLACK_S:
                return {'build': banner, 'build_identity_source': str(path)}
    return {
        'build': None,
        'build_identity_source': f'no banner for process {pid} in {logs[0].parent}',
    }


def describe_build(manifest: dict) -> str:
    """The profiled build in a few words, or why it is not known."""
    build = manifest['build']
    if build is None:
        return f'build unknown: {manifest["build_identity_source"]}'
    return f'{build["version"]} {build["git"]}'


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


def _tool_path(name: str, interpreter_dir: Path | None = None) -> str:
    """Absolute path to a sampling tool's binary (py-spy, austin, mojo2austin).

    A bare name fails under ``sudo`` (needed to attach on macOS) because sudo
    replaces PATH with a sanitized secure_path that omits the interpreter's bin
    dir. The tools are installed alongside the running interpreter, so resolve
    there first, then fall back to PATH for the case where one lives elsewhere.
    """
    interpreter_dir = interpreter_dir or Path(sys.executable).parent
    candidate = interpreter_dir / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f'{name} not found next to the interpreter ({candidate}) or on PATH. '
        f'Install the dev requirements (pip install -r requirements-dev.txt).'
    )


# The sampler, by host. On macOS py-spy counts a sleeping thread as CPU (its
# idle filter never acts there; ground truth 2026-10-09: a sleeper at 0.47
# cores), and austin's on-CPU mode does not. Elsewhere py-spy, the measured
# reference, until austin's ground truth is run on Windows.
AUSTIN = 'austin -c'
PY_SPY = 'py-spy'


def _sampler(host: str = sys.platform) -> str:
    return AUSTIN if host == 'darwin' else PY_SPY


def _sampler_command(sampler: str, pid: int, duration_s: int, rate_hz: int, out: Path) -> list[str]:
    """The command that records ``duration_s`` of the process's stacks to ``out``."""
    if sampler == AUSTIN:
        interval_us = round(1_000_000 / rate_hz)
        return [
            _tool_path('austin'),
            '-c',
            '-i',
            f'{interval_us}us',
            '-x',
            str(duration_s),
            '-p',
            str(pid),
            '-o',
            str(out),
        ]
    # -f raw = folded stacks; no --native (Python frames only, the cheap mode).
    return [
        _tool_path('py-spy'),
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
        str(out),
    ]


# The leaf of an on-CPU sample with no Python frame on its thread's stack, so
# its CPU is counted rather than skipped.
NO_PYTHON_FRAME = '<no Python frame>'


def austin_to_folded(text: str) -> str:
    """Austin's text samples as the folded lines ``parse_folded`` reads.

    ``mojo2austin`` writes one line per sample: ``P<pid>;T<iid>:<tid>;`` then
    frames ``file:qualname:line`` and the sample's microseconds. Each becomes
    one folded sample with its frames written ``qualname (file:line)``. The
    microseconds are dropped: a line is a sample, as py-spy counts them.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        head = line.rpartition(' ')[0]
        frames = head.split(';')[2:]
        folded = []
        for frame in frames:
            parts = frame.rsplit(':', 2)
            folded.append(f'{parts[1]} ({parts[0]}:{parts[2]})' if len(parts) == 3 else frame)
        lines.append(f'{";".join(folded) or NO_PYTHON_FRAME} 1')
    return '\n'.join(lines) + '\n'


# austin's exits (austin 4.0.0 src/austin.c, error.h): its -x window ends by
# emulating Ctrl-C, so a full recording returns -SIGINT, 254; a target that
# exits mid-window fails its next sample and returns austin's error, as does a
# refused attach, AUSTIN_EPERM, 2.
_AUSTIN_RECORDED = 254
_AUSTIN_EPERM = 2


def _recorded(sampler: str, returncode: int) -> bool:
    return returncode == (_AUSTIN_RECORDED if sampler == AUSTIN else 0)


def _refused_attach(sampler: str, returncode: int, stderr: str) -> bool:
    if sampler == AUSTIN:
        return returncode == _AUSTIN_EPERM
    return 'root' in stderr.lower() or 'permission' in stderr.lower()


def _attach_hint(host: str, is_root: bool) -> str:
    """Why an attach was refused, where the sampler's own words cannot say."""
    if host != 'darwin':
        return ''
    if not is_root:
        return ' -- on macOS run the whole command under sudo.'
    return (
        ' -- macOS refuses an attach, even to root, to a Python signed with the hardened'
        " runtime (python.org's installer); start LumaViewPro from a Homebrew Python"
        ' (README, macOS).'
    )


def _run_sampler(sampler: str, pid: int, duration_s: int, rate_hz: int, raw_path: Path) -> None:
    """Record the process's stacks and leave them at ``raw_path`` as folded lines."""
    recorded = raw_path.with_suffix('.mojo') if sampler == AUSTIN else raw_path
    result = subprocess.run(
        _sampler_command(sampler, pid, duration_s, rate_hz, recorded),
        capture_output=True,
        text=True,
    )
    if not _recorded(sampler, result.returncode):
        # Surface the sampler's own reason (permission, dead pid, non-Python
        # target) instead of a bare CalledProcessError -- the failure must be
        # legible.
        hint = ''
        if _refused_attach(sampler, result.returncode, result.stderr):
            hint = _attach_hint(sys.platform, hasattr(os, 'geteuid') and os.geteuid() == 0)
        raise RuntimeError(
            f'{sampler} failed (exit {result.returncode}): {result.stderr.strip()}{hint}'
        )
    if sampler == AUSTIN:
        text_path = raw_path.with_suffix('.austin')
        converted = subprocess.run(
            [_tool_path('mojo2austin'), str(recorded), str(text_path)],
            capture_output=True,
            text=True,
        )
        if converted.returncode != 0:
            raise RuntimeError(
                f'mojo2austin failed (exit {converted.returncode}): {converted.stderr.strip()}'
            )
        raw_path.write_text(
            austin_to_folded(text_path.read_text(encoding='utf-8', errors='replace')),
            encoding='utf-8',
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
    build = _build_of(pid)
    sampler_name = _sampler()
    raw_path = outdir / f'stacks_{stamp}.folded'

    cpu_samples: list[float] = []
    stop = threading.Event()
    cpu_thread = threading.Thread(
        target=_sample_total_cpu, args=(pid, duration_s, cpu_samples, stop), daemon=True
    )
    cpu_thread.start()
    _run_sampler(sampler_name, pid, duration_s, rate_hz, raw_path)
    stop.set()
    cpu_thread.join(timeout=3)

    folded = raw_path.read_text(encoding='utf-8', errors='replace')
    self_counts, total_samples, skipped = parse_folded(folded)
    mean_pct = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    total_cores = mean_pct / 100.0
    functions = compute_absolute_cpu(self_counts, total_samples, total_cores)

    try:
        cmdline = ' '.join(psutil.Process(pid).cmdline())
    except psutil.Error:
        cmdline = 'unknown'

    artifact = {
        'manifest': {
            'timestamp': stamp,
            'machine': platform.node(),
            'os': platform.platform(),
            'scenario': scenario,
            **build,
            'sampler': sampler_name,
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
        f'\nProfile: {m["scenario"]} | {describe_build(m)} | {m["machine"]} | {m["sampler"]}\n'
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
    parser.add_argument('--rate', type=int, default=DEFAULT_RATE_HZ, help='samples/sec')
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
