# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Ground-truth check for the CPU profiler: run the profiler against a synthetic
workload with a KNOWN split and assert it recovers that split within error.

This closes the one premise the off-bench unit tests cannot: that the sampler's
self-sample shares, joined to the psutil total, reflect the TRUE CPU-time split
on a real running process -- not just that the aggregation arithmetic is correct.
It is the acceptance gate for trusting any absolute number the profiler reports.

Two checks, each on its own run of the workload: the known 70 / 30 split with an
idle sleeper beside it, and the GIL-free run, whose one busy function works in C
with the GIL released while the main thread sleeps. On macOS run it under sudo
with a Homebrew Python (the workload is launched with this interpreter, and
macOS refuses an attach to python.org's); Windows attaches without elevation.
One command:

    python -m tools.profiling.ground_truth --split 0.7 --duration 30

It prints PASS / FAIL for each. The pure check logic (``recovered_split``,
``check_identity``, ``check_gil_free``) is unit-tested off-bench against
synthetic artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tools.profiling._synthetic_workload import HOT_A, HOT_B, HOT_GIL_FREE, IDLE
from tools.profiling.profile_session import _repo_root, profile

# The idle sleeper must read as ~no CPU. A ceiling well above sampling noise but
# far below any real thread's cost: crossing it means idle time was miscounted.
_IDLE_CEILING_CORES = 0.05

# Systematic floor on the split tolerance, on top of the sampling-error band:
# absorbs small asymmetries (per-call overhead, GC) without masking a real
# mis-attribution. A recovered split off by more than this is a genuine failure.
_SPLIT_TOLERANCE_FLOOR = 0.03

# Extra seconds the workload runs beyond the profiling window so it is already
# burning when the sampler attaches and outlives the last sample.
_WORKLOAD_MARGIN_S = 5.0


@dataclass(frozen=True)
class GroundTruthResult:
    """Outcome of comparing the profiler's recovered split to the known split."""

    passed: bool
    expected_a_share: float
    recovered_a_share: float
    tolerance: float
    hot_a_cores: float
    hot_b_cores: float
    sleeper_cores: float
    total_cpu_cores: float
    report: str


@dataclass(frozen=True)
class GilFreeResult:
    """Outcome of the GIL-free run: the one busy function kept its CPU."""

    passed: bool
    gil_free_share: float
    sleeper_cores: float
    total_cpu_cores: float
    report: str


# In the GIL-free run one function does all the work; the interpreter's own
# bookkeeping is the rest. A sampler that counts only the GIL holder gives it ~0.
_GIL_FREE_FLOOR = 0.9


def _sampler_of(artifact: dict) -> str:
    return artifact['manifest']['sampler']


def _leaf_cores(artifact: dict, name: str) -> tuple[float, float]:
    """Sum absolute CPU (cores) and 95% error over every leaf containing ``name``.

    Leaf strings carry the location too (``_hot_a (file.py:63)``), so match
    on the name as a substring. Errors add in quadrature (independent samples).
    """
    cores = 0.0
    var = 0.0
    for fn in artifact['functions']:
        if name in fn['function']:
            cores += fn['cpu_cores']
            var += fn.get('err_cores_95', 0.0) ** 2
    return cores, var**0.5


def recovered_split(artifact: dict, hot_a: str = HOT_A, hot_b: str = HOT_B) -> float:
    """Fraction of the two hot functions' CPU attributed to ``hot_a``.

    Normalized over just the hot pair, so outer-loop / interpreter overhead
    (which lands on other leaves and is symmetric between the two) does not bias
    the ratio. Returns 0.0 if neither hot function drew any samples.
    """
    a_cores, _ = _leaf_cores(artifact, hot_a)
    b_cores, _ = _leaf_cores(artifact, hot_b)
    hot_total = a_cores + b_cores
    return a_cores / hot_total if hot_total > 0 else 0.0


def check_identity(
    artifact: dict,
    expected_a_share: float,
    hot_a: str = HOT_A,
    hot_b: str = HOT_B,
) -> GroundTruthResult:
    """Assert the profiler recovered the known split and did not count idle CPU.

    Args:
        artifact: a profile artifact (manifest + ranked functions).
        expected_a_share: the configured fraction of CPU in ``hot_a`` (the truth).
        hot_a: leaf name of the higher-share hot function.
        hot_b: leaf name of the lower-share hot function.

    Returns:
        A ``GroundTruthResult`` whose ``passed`` is True only when the recovered
        split is within tolerance AND the sleeper drew negligible CPU.
    """
    a_cores, a_err = _leaf_cores(artifact, hot_a)
    b_cores, b_err = _leaf_cores(artifact, hot_b)
    sleeper_cores, _ = _leaf_cores(artifact, IDLE)
    hot_total = a_cores + b_cores
    recovered = recovered_split(artifact, hot_a, hot_b)
    # Move the component errors onto the normalized ratio, then floor it.
    err_on_ratio = (a_err + b_err) / hot_total if hot_total > 0 else 1.0
    tolerance = max(_SPLIT_TOLERANCE_FLOOR, err_on_ratio)

    split_ok = hot_total > 0 and abs(recovered - expected_a_share) <= tolerance
    idle_ok = sleeper_cores <= _IDLE_CEILING_CORES
    passed = split_ok and idle_ok

    total_cpu = artifact['manifest'].get('total_process_cpu_cores', 0.0)
    lines = [
        f'{"PASS" if passed else "FAIL"}: {_sampler_of(artifact)} share x psutil total vs known split',
        f'  known {hot_a} share : {expected_a_share:.3f}',
        f'  recovered           : {recovered:.3f}  (tol +-{tolerance:.3f})',
        f'  {hot_a} / {hot_b} CPU  : {a_cores:.3f} / {b_cores:.3f} cores',
        f'  idle sleeper CPU    : {sleeper_cores:.3f} cores (ceiling {_IDLE_CEILING_CORES})',
        f'  process total       : {total_cpu:.2f} cores',
    ]
    if not split_ok:
        lines.append(
            '  -> recovered split outside tolerance: the sampled share does NOT map to CPU'
        )
    if not idle_ok:
        lines.append('  -> idle thread counted as CPU: the total or the sampling is wrong')
    return GroundTruthResult(
        passed=passed,
        expected_a_share=expected_a_share,
        recovered_a_share=recovered,
        tolerance=tolerance,
        hot_a_cores=a_cores,
        hot_b_cores=b_cores,
        sleeper_cores=sleeper_cores,
        total_cpu_cores=total_cpu,
        report='\n'.join(lines),
    )


def check_gil_free(artifact: dict) -> GilFreeResult:
    """Assert the GIL-free function kept its CPU and the sleeping thread did not gain any."""
    total_cpu = artifact['manifest'].get('total_process_cpu_cores', 0.0)
    hot_cores, _ = _leaf_cores(artifact, HOT_GIL_FREE)
    sleeper_cores, _ = _leaf_cores(artifact, IDLE)
    share = hot_cores / total_cpu if total_cpu > 0 else 0.0
    share_ok = share >= _GIL_FREE_FLOOR
    idle_ok = sleeper_cores <= _IDLE_CEILING_CORES
    passed = share_ok and idle_ok
    lines = [
        f'{"PASS" if passed else "FAIL"}: {_sampler_of(artifact)} keeps CPU spent with the GIL released',
        f'  {HOT_GIL_FREE} share : {share:.3f} of the process (floor {_GIL_FREE_FLOOR})',
        f'  idle sleeper CPU    : {sleeper_cores:.3f} cores (ceiling {_IDLE_CEILING_CORES})',
        f'  process total       : {total_cpu:.2f} cores',
    ]
    if not share_ok:
        lines.append(
            '  -> CPU spent with the GIL released was lost: numpy and OpenCV would read low'
        )
    if not idle_ok:
        lines.append('  -> idle thread counted as CPU: the total or the sampling is wrong')
    return GilFreeResult(
        passed=passed,
        gil_free_share=share,
        sleeper_cores=sleeper_cores,
        total_cpu_cores=total_cpu,
        report='\n'.join(lines),
    )


def _profile_workload(
    workload_args: list[str], scenario: str, duration_s: int, rate_hz: int, outdir: Path
) -> dict:
    """Launch the synthetic workload with ``workload_args``, profile it, return the artifact."""
    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'tools.profiling._synthetic_workload',
            *workload_args,
            '--duration',
            str(duration_s + _WORKLOAD_MARGIN_S),
        ],
        cwd=str(_repo_root()),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        # Block on the workload's PID line so the sampler attaches to a process
        # that is already burning, not one still importing.
        startup = proc.stdout.readline() if proc.stdout else ''
        if not startup.startswith('synthetic_workload pid='):
            raise RuntimeError(f'synthetic workload did not start (got {startup!r})')
        artifact_path = profile(
            pid=proc.pid,
            duration_s=duration_s,
            rate_hz=rate_hz,
            scenario=scenario,
            outdir=outdir,
            settings_json=None,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return json.loads(Path(artifact_path).read_text(encoding='utf-8'))


def run(split: float, duration_s: int, rate_hz: int, outdir: Path) -> list:
    """Both checks, each on its own run of the workload; their results, in order."""
    split_artifact = _profile_workload(
        ['--split', str(split)], f'ground-truth-{split:g}', duration_s, rate_hz, outdir
    )
    gil_free_artifact = _profile_workload(
        ['--gil-free'], 'ground-truth-gil-free', duration_s, rate_hz, outdir
    )
    results = [check_identity(split_artifact, split), check_gil_free(gil_free_artifact)]
    for result in results:
        print('\n' + result.report)
    print(f'\nVerdicts written to {write_verdicts(results, outdir)}')
    return results


def write_verdicts(results: list, outdir: Path) -> Path:
    """The checks' reports, saved beside their profiles, so the verdict is a file."""
    path = outdir / f'ground_truth_{time.strftime("%Y%m%d_%H%M%S")}.txt'
    path.write_text('\n\n'.join(result.report for result in results) + '\n', encoding='utf-8')
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validate the CPU profiler against a known split.')
    parser.add_argument('--split', type=float, default=0.7, help='known fraction of CPU in _hot_a')
    parser.add_argument('--duration', type=int, default=30, help='profiling window seconds')
    parser.add_argument('--rate', type=int, default=50, help='samples/sec')
    parser.add_argument('--outdir', type=Path, default=Path('logs/cpu_profile/ground_truth'))
    args = parser.parse_args(argv)
    results = run(args.split, args.duration, args.rate, args.outdir)
    return 0 if all(result.passed for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
