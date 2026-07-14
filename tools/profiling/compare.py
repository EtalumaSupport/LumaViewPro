# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Compare two profile artifacts -> per-function CPU deltas beyond sampling noise.

The regression / before-after engine. Two runs are comparable only when the
scenario, machine, and CPU-relevant config match -- otherwise a delta reflects
the setup, not the code (the classic "compared runs that differ in more than
the variable under test" trap). The git SHA is EXPECTED to differ: that is the
variable under test. A delta counts as real only when it exceeds the two runs'
combined 95% sampling error.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

# Config keys that move CPU; two runs must agree on these to be comparable.
_COMPAT_CONFIG_KEYS = ('live_view_fps', 'preview_host_downscale')


@dataclass(frozen=True)
class FunctionDelta:
    function: str
    before_cores: float
    after_cores: float
    delta_cores: float  # after - before (negative = improvement)
    combined_err_cores: float
    significant: bool  # |delta| exceeds combined sampling error


def _index(artifact: dict) -> dict[str, dict]:
    return {fn['function']: fn for fn in artifact['functions']}


def compatibility_issues(before: dict, after: dict) -> list[str]:
    """Reasons the two runs are NOT comparable (empty list = comparable).

    Scenario / machine / CPU-config must match; git SHA is allowed (and
    expected) to differ."""
    issues: list[str] = []
    mb, ma = before['manifest'], after['manifest']
    for field in ('scenario', 'machine'):
        if mb.get(field) != ma.get(field):
            issues.append(f"{field} differs: {mb.get(field)!r} vs {ma.get(field)!r}")
    cb, ca = mb.get('config', {}), ma.get('config', {})
    for key in _COMPAT_CONFIG_KEYS:
        if cb.get(key) != ca.get(key):
            issues.append(f"config.{key} differs: {cb.get(key)!r} vs {ca.get(key)!r}")
    return issues


def compare(before: dict, after: dict) -> list[FunctionDelta]:
    """Per-function CPU deltas, ranked by absolute change (largest first)."""
    bi, ai = _index(before), _index(after)
    deltas: list[FunctionDelta] = []
    for function in bi.keys() | ai.keys():
        b = bi.get(function)
        a = ai.get(function)
        before_cores = b['cpu_cores'] if b else 0.0
        after_cores = a['cpu_cores'] if a else 0.0
        before_err = b['err_cores_95'] if b else 0.0
        after_err = a['err_cores_95'] if a else 0.0
        delta = after_cores - before_cores
        combined_err = math.sqrt(before_err**2 + after_err**2)
        deltas.append(
            FunctionDelta(
                function=function,
                before_cores=before_cores,
                after_cores=after_cores,
                delta_cores=delta,
                combined_err_cores=combined_err,
                significant=abs(delta) > combined_err,
            )
        )
    deltas.sort(key=lambda d: abs(d.delta_cores), reverse=True)
    return deltas


def _print_report(before: dict, after: dict, deltas: list[FunctionDelta], top_n: int = 25) -> None:
    mb, ma = before['manifest'], after['manifest']
    print(
        f"\nCompare  {mb['scenario']}  |  BEFORE {mb['version']} {mb['git_sha'][:8]}"
        f"  ->  AFTER {ma['version']} {ma['git_sha'][:8]}"
    )
    tb = mb['total_process_cpu_cores']
    ta = ma['total_process_cpu_cores']
    print(
        f"  total process CPU: {tb * 100:.0f}% -> {ta * 100:.0f}% "
        f"(delta {(ta - tb) * 100:+.0f}%)"
    )
    print(f"\n  {'before':>8}  {'after':>8}  {'delta':>8}   sig  function")
    for d in deltas[:top_n]:
        if d.before_cores < 1e-9 and d.after_cores < 1e-9:
            continue
        mark = '**' if d.significant else '  '
        print(
            f"  {d.before_cores * 100:7.1f}%  {d.after_cores * 100:7.1f}%  "
            f"{d.delta_cores * 100:+7.1f}%  {mark}   {d.function}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare two LVP profile artifacts.')
    parser.add_argument('before', type=Path, help='baseline profile_*.json')
    parser.add_argument('after', type=Path, help='new profile_*.json')
    parser.add_argument(
        '--force', action='store_true', help='compare even if scenario/machine/config differ'
    )
    args = parser.parse_args(argv)
    before = json.loads(args.before.read_text(encoding='utf-8'))
    after = json.loads(args.after.read_text(encoding='utf-8'))

    issues = compatibility_issues(before, after)
    if issues and not args.force:
        print('Refusing to compare -- runs differ in more than the code under test:')
        for issue in issues:
            print(f'  - {issue}')
        print('Re-run with --force to override.')
        return 2
    if issues:
        print('WARNING: comparing despite mismatches (--force):')
        for issue in issues:
            print(f'  - {issue}')

    _print_report(before, after, compare(before, after))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
