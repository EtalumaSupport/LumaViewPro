#!/usr/bin/env python3
"""Repo-wide ruff finding-count ratchet for the pre-commit hook.

The repo carries a known, dispositioned backlog of ruff findings that
is being cleaned up in batches. Until that backlog reaches zero, a
plain ``ruff check`` exit-code gate would block every commit, and no
gate at all lets the count creep upward between cleanup sessions
(which it measurably did). This ratchet is the middle state: commits
that hold or reduce the total finding count pass; commits that
increase it are blocked.

When a commit reduces the count, the baseline file is rewritten to
the new lower number and staged into the same commit, so improvements
lock in without manual bookkeeping. Once the backlog reaches zero,
this tool and the baseline file retire in favor of the plain
exit-code gate -- a zero baseline IS zero-tolerance.

Modes:

    python3 tools/ruff_ratchet.py               # report count vs baseline
    python3 tools/ruff_ratchet.py --pre-commit  # enforce; auto-lower baseline

Skips gracefully (exit 0 with a note) when ruff is unavailable or the
baseline file is absent, so branches that predate the ratchet never
block. The count is taken from the working tree -- the same number a
developer sees running ``ruff check .`` by hand; with partial staging
the gate may count unstaged edits, which is acceptable for a
monotonic ceiling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASELINE_NAME = 'ruff_baseline.txt'


def repo_root() -> Path:
    out = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
    return Path(out).resolve()


def baseline_path(root: Path) -> Path:
    return root / 'tools' / BASELINE_NAME


def parse_baseline(text: str) -> int:
    """Return the recorded count from baseline-file text.

    The file is one integer on the first non-empty, non-comment line;
    ``#`` lines are allowed for human notes.

    Raises:
        ValueError: when no integer line is present.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        return int(stripped)
    raise ValueError(f'no integer count line found in {BASELINE_NAME}')


def live_count(root: Path) -> int | None:
    """Return the current repo-wide ruff finding count, or None if
    ruff is unavailable.

    Uses JSON output so the count is one-diagnostic-per-element exact,
    independent of summary-line wording, and --exit-zero so a nonzero
    finding count is not conflated with ruff itself failing.
    """
    try:
        proc = subprocess.run(
            [sys.executable, '-m', 'ruff', 'check', '.', '--output-format', 'json',
             '--exit-zero'],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        # --exit-zero means any nonzero return is a real ruff failure
        # (bad config, missing module), not a finding count.
        sys.stderr.write(proc.stderr)
        return None
    return len(json.loads(proc.stdout))


def decide(count: int, baseline: int) -> str:
    """Classify the count against the baseline: 'block' | 'lower' | 'ok'."""
    if count > baseline:
        return 'block'
    if count < baseline:
        return 'lower'
    return 'ok'


def rewrite_baseline(path: Path, count: int) -> None:
    """Write the new count, preserving the file's comment header lines."""
    header = [
        line
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip().startswith('#')
    ]
    path.write_text('\n'.join([*header, str(count)]) + '\n', encoding='utf-8')


def run(pre_commit: bool) -> int:
    root = repo_root()
    bpath = baseline_path(root)
    if not bpath.is_file():
        print(f'ruff_ratchet: {bpath.relative_to(root)} absent -- skipping ratchet')
        return 0
    count = live_count(root)
    if count is None:
        print('ruff_ratchet: ruff unavailable -- skipping ratchet')
        return 0
    baseline = parse_baseline(bpath.read_text(encoding='utf-8'))
    verdict = decide(count, baseline)

    if verdict == 'block':
        print(
            f'ruff_ratchet: BLOCKED -- ruff finding count rose to {count} '
            f'(baseline {baseline}).\n'
            f'  Fix the new findings (run: python3 -m ruff check .) or, for an '
            f'intentional finding, add a `# noqa: <RULE> -- <reason>` line.\n'
            f'  The baseline only moves down.',
            file=sys.stderr,
        )
        return 1

    if verdict == 'lower':
        rewrite_baseline(bpath, count)
        print(f'ruff_ratchet: count dropped {baseline} -> {count}; baseline lowered')
        if pre_commit:
            subprocess.run(['git', 'add', str(bpath)], cwd=root, check=True)
        return 0

    print(f'ruff_ratchet: count {count} == baseline; OK')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument(
        '--pre-commit',
        action='store_true',
        help='enforce the ratchet and stage the baseline when it lowers',
    )
    args = parser.parse_args(argv)
    return run(pre_commit=args.pre_commit)


if __name__ == '__main__':
    sys.exit(main())
