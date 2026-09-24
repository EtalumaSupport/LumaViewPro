# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Measure the architecture ratchets across history.

A ratchet tells you today's count beside its pin. It cannot tell you whether
the number is falling because the migration is working, flat because nothing
is migrating, or flat because the instrument cannot see the work. This walks
the trunk and measures each sampled tree, so a trend can be read instead of
guessed.

METHOD, and it is the whole point: the instrument is held fixed and the code
varies. Every tree is measured with the detector functions as they exist in
the WORKING COPY, never with that commit's own guard file. Measuring each
tree with its contemporary guard would vary the instrument and the subject
together, and no two numbers in the series would be comparable. It also lets
a detector added today be run backwards over trees that predate it, which is
usually the interesting question.

The detectors are imported from the guard module rather than restated, so a
series can never drift from what the test suite actually enforces.

A detector that cannot run against an old tree -- a file it reads did not
exist yet -- records None and prints `n/a`. It must never record 0: a missing
file reading as "zero violations" is the plausible-wrong-value failure the
ratchets exist to catch, and it would show up as the migration's best week.

Reading a series:

  A count can fall because logic moved, or because the file holding it was
  deleted, or because the program shrank. The size columns are printed for
  that reason -- a fall alongside a matching fall in lines is not progress.

Usage:
    python3 tools/ratchet_history.py --since 2026-06-22
    python3 tools/ratchet_history.py --since 2026-01-01 --every 14 --json out.json
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.guards import test_architecture_fixes as _guards

# (column label, detector). Each returns a dict whose values are summed.
DETECTORS = (
    ('answers below API', _guards._ui_answerer_call_counts),
    ('private reaches', _guards._ui_private_reach_counts),
    ('except handlers', _guards._ui_except_counts),
    ('orchestration', _guards._gui_orchestration_counts),
    ('_app_ctx reads', _guards._modules_context_read_counts),
    ('.ids[ reads', _guards._modules_widget_read_counts),
    ('ui imports below', _guards._lower_layer_ui_import_counts),
    ('twin answerers', _guards._twin_answerer_names),
)


def _git(*args, cwd=REPO):
    return subprocess.run(
        ['git', *args], cwd=str(cwd), capture_output=True, text=True, check=True
    ).stdout.strip()


def sample_points(since, until, every_days, branch):
    """One commit per interval: the last commit on or before each boundary.

    Returns [(iso date, sha)]. A boundary with no commit at or before it is
    skipped; a boundary whose commit repeats the previous one is dropped, so
    a quiet fortnight yields one point rather than two identical ones.
    """
    start = _dt.date.fromisoformat(since)
    end = _dt.date.fromisoformat(until) if until else _dt.date.today()
    points, seen = [], set()
    day = start
    while day <= end:
        sha = subprocess.run(
            ['git', 'rev-list', '-1', f'--before={day} 23:59:59', branch],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        ).stdout.strip()
        if sha and sha not in seen:
            seen.add(sha)
            points.append((day.isoformat(), sha[:8]))
        day += _dt.timedelta(days=every_days)
    return points


def tree_size(root):
    """Lines and files under ui/ and modules/ -- the denominators."""
    out = {}
    for sub in ('ui', 'modules'):
        files = sorted(pathlib.Path(root, sub).glob('*.py'))
        out[f'{sub}_files'] = len(files)
        out[f'{sub}_lines'] = sum(len(p.read_text(errors='replace').splitlines()) for p in files)
    return out


def measure_tree(worktree):
    """Run every detector against `worktree`, today's instrument throughout.

    The redirection is undone before returning. The detectors read the guard
    module's `REPO_ROOT` at call time, which is what lets one instrument be
    pointed at many trees -- and that global belongs to a module the test
    suite also imports, so leaving it redirected would make every guard in
    the same process measure whichever tree was sampled last.
    """
    row = {}
    previous = _guards.REPO_ROOT
    _guards.REPO_ROOT = str(worktree)
    try:
        for label, fn in DETECTORS:
            try:
                row[label] = sum(fn().values())
            except Exception as exc:
                row[label] = None
                row[f'{label} (why)'] = f'{type(exc).__name__}: {exc}'[:150]
    finally:
        _guards.REPO_ROOT = previous
    with contextlib.suppress(Exception):
        row.update(tree_size(worktree))
    return row


def run(points, branch):
    worktree = pathlib.Path(tempfile.mkdtemp(prefix='ratchet_history_'))
    rows = []
    try:
        _git('worktree', 'add', '-q', '--detach', str(worktree), branch)
        for date, sha in points:
            _git('checkout', '--detach', '-q', sha, cwd=worktree)
            rows.append({'date': date, 'sha': sha, **measure_tree(worktree)})
            print(f'  measured {date} {sha}', file=sys.stderr, flush=True)
    finally:
        with contextlib.suppress(Exception):
            _git('worktree', 'remove', '--force', str(worktree))
        shutil.rmtree(worktree, ignore_errors=True)
    return rows


def render(rows):
    labels = [lab for lab, _ in DETECTORS]
    width = max(len(lab) for lab in labels) + 2
    head = f'{"date":<12}{"sha":<10}' + ''.join(f'{lab:>{width}}' for lab in labels)
    lines = [head, '-' * len(head)]
    for r in rows:
        cells = ''.join(f'{("n/a" if r.get(lab) is None else r[lab])!s:>{width}}' for lab in labels)
        lines.append(f'{r["date"]:<12}{r["sha"]:<10}{cells}')
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        deltas = ''.join(
            f'{("n/a" if first.get(lab) is None or last.get(lab) is None else f"{last[lab] - first[lab]:+d}")!s:>{width}}'
            for lab in labels
        )
        lines += ['-' * len(head), f'{"net":<12}{"":<10}{deltas}']
        for sub in ('ui', 'modules'):
            a, b = first.get(f'{sub}_lines'), last.get(f'{sub}_lines')
            if a and b:
                lines.append(f'{sub}/ lines {a} -> {b}  ({(b / a - 1) * 100:+.0f}%)')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--since', required=True, help='first sample date, YYYY-MM-DD')
    ap.add_argument('--until', default=None, help='last sample date (default: today)')
    ap.add_argument('--every', type=int, default=7, help='days between samples')
    ap.add_argument('--branch', default='dev/4.0.0', help='branch to walk')
    ap.add_argument('--json', default=None, help='also write the rows here')
    args = ap.parse_args(argv)

    points = sample_points(args.since, args.until, args.every, args.branch)
    if not points:
        print(f'no commits on {args.branch} in that range', file=sys.stderr)
        return 1
    print(f'measuring {len(points)} points on {args.branch}', file=sys.stderr)
    rows = run(points, args.branch)
    print(render(rows))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
