# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Prove that every production hunk in a range is pinned by a test.

Before a push, each production hunk between a base and the tip is reverted
alone and the tests that can observe it are run; the hunk is pinned when at
least one of them stops passing. The tests that can observe a range are the
ones that do not pass with the whole range's production diff removed, so the
suite runs once with that diff reverse-applied, the tests it leaves not
passed become the observing set, and only those run against each hunk. A
range no test observes is a pass: a behaviour-preserving refactor is exactly
that, and a behaviour change ships its fail-before test, which the suite with
the change removed fails by construction.

Verdicts per hunk:
    RED        a test that passed at the tip does not pass with the hunk
               reverted alone, or a test file no longer collects
    GREEN      every observing test still passes: the hunk is unpinned
    NEUTRAL    the module's syntax tree, docstrings stripped, is the same
               with and without the hunk; not run, no pin required
    NOT-OBSERVED  a behavioural hunk in a range no test observes

Exit 0 when every behavioural hunk is RED or no test observes the range, 1
when any hunk is GREEN, 2 when the tool refuses to report: an observing test
not passing at the tip, an id pytest does not collect, a revert that changed
nothing, a restore that left the tree dirty, or a run whose outcomes lack an
expected test. A refusal never reads as a verdict.

Usage:
    python3 tools/confirm_pins.py [--checkout DIR] [--base REF]
                                  [--scratch DIR] [--report FILE] [--workers N]

The base defaults to the merge-base of the checkout's branch and its
upstream, so a branch behind its upstream never counts a peer's commits as
its own. Every command runs in a detached scratch worktree at the tip, with
the worktree root as its working directory; the checkout is never touched.
Outcomes are read from pytest's junit file, never from its summary line: a
test that skips after a revert reads as a pass on the summary line and as
RED here. The repository's pytest addopts carry ``-q``, which is what makes
``--co`` print one node id per line; the tool adds none, since a second
``-q`` prints no ids at all. A restore is proven by ``git status``, so the
caches a run writes (``__pycache__/``, ``.pytest_cache/``) must be ignored
by the repository, as they are here; in one that does not ignore them every
restore reads dirty.
"""

from __future__ import annotations

import argparse
import ast
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass

PRODUCTION_PACKAGES = ('modules', 'ui', 'drivers', 'tools')

EXIT_PINNED = 0
EXIT_UNPINNED = 1
EXIT_REFUSED = 2


class RefusedError(Exception):
    """The tool cannot report a verdict; the message is the reason."""


@dataclass(frozen=True)
class Hunk:
    path: str
    header: str
    patch: str


@dataclass(frozen=True)
class Outcome:
    test: str
    result: str


# ---------------------------------------------------------------- the diff


def is_production_path(path: str) -> bool:
    """True for a ``.py`` at the repository root or under a production package."""
    if not path.endswith('.py'):
        return False
    first, sep, _ = path.partition('/')
    return not sep or first in PRODUCTION_PACKAGES


def _path_of(diff_git_line: str) -> str:
    rest = diff_git_line[len('diff --git a/') :].rstrip('\n')
    half = (len(rest) - 3) // 2
    if rest[half : half + 3] != ' b/' or rest[:half] != rest[half + 3 :]:
        raise RefusedError(f'cannot read the file of {diff_git_line.strip()!r}')
    return rest[:half]


def cut_hunks(diff_text: str) -> list[Hunk]:
    """Cut a unified diff into hunks, each carried as a complete one-hunk patch.

    A new or deleted file is one hunk like any other: its file header lines
    (mode, index, the ``/dev/null`` side) travel with it, so ``git apply``
    creates or removes the file from the one-hunk patch.
    """
    hunks: list[Hunk] = []
    file_header: list[str] = []
    path = ''
    body: list[str] | None = None

    def close() -> None:
        if body:
            hunks.append(Hunk(path, body[0].rstrip('\n'), ''.join(file_header + body)))

    for line in diff_text.splitlines(keepends=True):
        if line.startswith('diff --git '):
            close()
            body = None
            file_header = [line]
            path = _path_of(line)
        elif line.startswith('@@'):
            close()
            body = [line]
        elif body is None:
            file_header.append(line)
        else:
            body.append(line)
    close()
    return hunks


def moved_pairs(hunks: Iterable[Hunk]) -> dict[int, int]:
    """Index of each hunk whose removed lines another hunk of the same file adds.

    A function moved within a file is two hunks; the deletion reverted alone
    restores a duplicate definition, so the report names both and the author
    reads them together.
    """
    removed, added = {}, {}
    for n, hunk in enumerate(hunks):
        lines = hunk.patch.splitlines()
        lines = lines[lines.index(hunk.header) + 1 :]
        minus = ''.join(line[1:] + '\n' for line in lines if line.startswith('-'))
        plus = ''.join(line[1:] + '\n' for line in lines if line.startswith('+'))
        if minus.strip():
            removed[n] = (hunk.path, minus)
        if plus.strip():
            added[n] = (hunk.path, plus)
    pairs = {}
    for n, key in removed.items():
        for m, other in added.items():
            if m != n and other == key:
                pairs[n] = m
                pairs[m] = n
    return pairs


# ------------------------------------------------------------- neutrality


def _without_docstrings(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                del node.body[0]
    return ast.dump(tree)


def is_neutral(before: str | None, after: str | None) -> bool:
    """True when the module's syntax tree, docstrings stripped, is unchanged.

    No module on either side (a file added or deleted), or a parse failure
    on either side, is behavioural: the hunk is run.
    """
    if before is None or after is None:
        return False
    try:
        return _without_docstrings(ast.parse(before)) == _without_docstrings(ast.parse(after))
    except SyntaxError:
        return False


# -------------------------------------------------------------- outcomes


def _strip_group(name: str) -> str:
    """Drop the ``@group`` suffix xdist appends under ``--dist loadgroup``.

    The suffix is outside the parameter bracket; an ``@`` inside one is the
    parameter's own.
    """
    depth = 0
    for i, ch in enumerate(name):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        elif ch == '@' and depth == 0:
            return name[:i]
    return name


def node_id(classname: str, name: str, test_files: Collection[str]) -> str:
    """The pytest node id of a junit testcase.

    The dotted classname is split at the longest prefix that is a test file;
    what follows is the class path. An empty classname is a collection error
    whose name is the dotted module: the id is that file.
    """
    if not classname:
        return name.replace('.', '/') + '.py' if name else ''
    parts = classname.split('.')
    for k in range(len(parts), 0, -1):
        file = '/'.join(parts[:k]) + '.py'
        if file in test_files:
            return '::'.join([file, *parts[k:], _strip_group(name)])
    return '::'.join(['/'.join(parts) + '.py', _strip_group(name)])


def read_outcomes(xml_text: str, test_files: Collection[str]) -> list[Outcome]:
    """Every testcase of a junit file as (node id, result).

    Results: ``passed``, ``failed``, ``error``, ``skipped``, ``xfail`` (a
    skip typed ``pytest.xfail``) and ``collection-error`` (an error on a
    testcase with no classname, keyed by its file).
    """
    outcomes = []
    for case in ET.fromstring(xml_text).iter('testcase'):
        classname = case.get('classname', '')
        test = node_id(classname, case.get('name', ''), test_files)
        if case.find('failure') is not None:
            result = 'failed'
        elif case.find('error') is not None:
            result = 'collection-error' if not classname else 'error'
        elif (skipped := case.find('skipped')) is not None:
            result = 'xfail' if skipped.get('type') == 'pytest.xfail' else 'skipped'
        else:
            result = 'passed'
        outcomes.append(Outcome(test, result))
    return outcomes


def _tests_by_file(collection: Collection[str]) -> dict[str, set[str]]:
    by_file: dict[str, set[str]] = defaultdict(set)
    for test in collection:
        by_file[test.split('::', 1)[0]].add(test)
    return by_file


def reported_tests(outcomes: Iterable[Outcome], collection: Collection[str]) -> set[str]:
    """The collected tests a run reported on, directly or through a collection error.

    A file that failed to collect reports on every test the tip collects in
    it. A run that reports on none of the collection ran nothing the tool
    expected: an instrument failure, never a verdict.
    """
    by_file = _tests_by_file(collection)
    tests: set[str] = set()
    for outcome in outcomes:
        if outcome.result == 'collection-error':
            tests |= by_file[outcome.test]
        elif outcome.test in collection:
            tests.add(outcome.test)
    return tests


def observing_tests(outcomes: Iterable[Outcome], collection: Collection[str]) -> set[str]:
    """The tests that did not pass with the range's production diff removed.

    A file that failed to collect contributes every test the tip collects
    in it: a test of a name the range introduced fails exactly that way.
    """
    by_file = _tests_by_file(collection)
    tests: set[str] = set()
    for outcome in outcomes:
        if outcome.result == 'collection-error':
            tests |= by_file[outcome.test]
        elif outcome.result not in ('passed', 'xfail'):
            tests.add(outcome.test)
    return tests


def validate_observing(observing: Collection[str], collection: Collection[str]) -> None:
    """Refuse an observing id pytest does not collect, by name.

    An id the collection lacks makes an xdist run exit 5 with nothing run
    and no name; refusing here is what keeps a later run from reading as a
    verdict.
    """
    unknown = sorted(set(observing) - set(collection))
    if unknown:
        raise RefusedError(
            f'{len(unknown)} observing id(s) pytest does not collect at the tip: {unknown[0]}'
        )


def tip_baseline(outcomes: Iterable[Outcome], expected: Collection[str]) -> tuple[set[str], int]:
    """The observing tests that passed at the tip, and the count that skipped.

    A test that fails, errors or does not collect at the tip is a
    pre-existing red: refused, by name. An expected test absent from the
    outcomes is the instrument's failure, refused the same way.
    """
    results = {o.test: o.result for o in outcomes}
    for outcome in outcomes:
        if outcome.result == 'collection-error':
            raise RefusedError(f'{outcome.test} does not collect at the tip')
    for test in sorted(expected):
        if test not in results:
            raise RefusedError(
                f'INSTRUMENT: {test} is absent from the outcomes of its run at the tip'
            )
        if results[test] in ('failed', 'error'):
            raise RefusedError(
                f'{test} is {results[test]} at the tip; fix it before confirming pins'
            )
    passed = {t for t in expected if results[t] == 'passed'}
    skipped = sum(1 for t in expected if results[t] == 'skipped')
    return passed, skipped


def hunk_verdict(outcomes: Iterable[Outcome], baseline: Collection[str]) -> tuple[str, str]:
    """(verdict, detail) for one hunk reverted alone, against the tip's baseline.

    A collection error is RED and names the file. A baseline test present
    and not passed is RED and names the test; an xfail-typed skip is not a
    change. A baseline test absent from the outcomes is INSTRUMENT. Every
    baseline test present and passed is GREEN.
    """
    outcomes = list(outcomes)
    for outcome in outcomes:
        if outcome.result == 'collection-error':
            return 'RED', f'fails to collect: {outcome.test or "(no file named)"}'
    results = {o.test: o.result for o in outcomes}
    for test in sorted(baseline):
        if test in results and results[test] not in ('passed', 'xfail'):
            return 'RED', f'{test} {results[test]}'
    for test in sorted(baseline):
        if test not in results:
            return 'INSTRUMENT', f'{test} is absent from the outcomes'
    return 'GREEN', f'{len(baseline)} tests passed'


# ------------------------------------------------------------------ runs


# Python keys a module's bytecode cache on the source's size and its mtime in
# whole seconds. Two hunks of one file reverted within the same second often
# leave it the same size, and the second run would then execute the first
# revert's bytecode: an unobserved hunk read RED with its sibling's failure in
# the end-to-end test. The worktree starts with no cache and no run writes one.
_NO_BYTECODE = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}


class Worktree:
    """A detached scratch worktree at the tip; every command runs at its root."""

    def __init__(self, checkout: pathlib.Path, tip: str, scratch: pathlib.Path) -> None:
        self.checkout = checkout
        self.root = scratch / 'worktree'
        self.scratch = scratch
        self.test_files: set[str] = set()
        subprocess.run(['git', 'worktree', 'prune'], cwd=checkout, check=True)
        subprocess.run(
            ['git', 'worktree', 'add', '--detach', str(self.root), tip],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )

    def remove(self) -> None:
        subprocess.run(
            ['git', 'worktree', 'remove', '--force', str(self.root)],
            cwd=self.checkout,
            capture_output=True,
        )

    def git(self, *args: str) -> str:
        done = subprocess.run(
            ['git', *args], cwd=self.root, capture_output=True, text=True, check=True
        )
        return done.stdout

    def apply(self, patch: str, reverse: bool) -> None:
        args = ['git', 'apply', *(['-R'] if reverse else []), '-']
        done = subprocess.run(args, cwd=self.root, input=patch, capture_output=True, text=True)
        if done.returncode:
            way = 'reverse-apply' if reverse else 'apply'
            raise RefusedError(f'git could not {way} the patch: {done.stderr.strip()}')

    def status(self) -> str:
        return self.git('status', '--porcelain')

    def read(self, path: str) -> str | None:
        file = self.root / path
        return file.read_text(encoding='utf-8') if file.exists() else None

    def pytest(self, name: str, workers: str, targets: Collection[str]) -> list[Outcome]:
        """Run pytest on ``targets`` (the whole suite when empty); outcomes from junit."""
        junit = self.scratch / f'{name}.xml'
        log = self.scratch / f'{name}.log'
        junit.unlink(missing_ok=True)
        command = [sys.executable, '-m', 'pytest', '-n', workers, f'--junitxml={junit}', *targets]
        with log.open('w') as out:
            subprocess.run(
                command, cwd=self.root, stdout=out, stderr=subprocess.STDOUT, env=_NO_BYTECODE
            )
        if not junit.exists():
            raise RefusedError(f'pytest wrote no junit file for {name}; its output is in {log}')
        return read_outcomes(junit.read_text(encoding='utf-8'), self.test_files)

    def collect(self) -> set[str]:
        """Every node id pytest collects at the tip; a collection error refuses."""
        log = self.scratch / 'collect.log'
        with log.open('w') as out:
            done = subprocess.run(
                [sys.executable, '-m', 'pytest', '--co'],
                cwd=self.root,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=_NO_BYTECODE,
            )
        if done.returncode:
            raise RefusedError(
                f'pytest --co exited {done.returncode} at the tip; its output is in {log}'
            )
        ids = {
            line.strip()
            for line in log.read_text(encoding='utf-8').splitlines()
            if '::' in line and not line.startswith((' ', '\t'))
        }
        self.test_files = {test.split('::', 1)[0] for test in ids}
        return ids


def resolve_base(checkout: pathlib.Path, base: str | None) -> str:
    if base is None:
        done = subprocess.run(
            ['git', 'merge-base', '@{u}', 'HEAD'], cwd=checkout, capture_output=True, text=True
        )
        if done.returncode:
            raise RefusedError('the branch has no upstream to take a merge-base from; pass --base')
        return done.stdout.strip()
    done = subprocess.run(
        ['git', 'rev-parse', '--verify', f'{base}^{{commit}}'],
        cwd=checkout,
        capture_output=True,
        text=True,
    )
    if done.returncode:
        raise RefusedError(f'--base {base!r} is not a commit of {checkout}')
    return done.stdout.strip()


class Report:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._file = path.open('w', encoding='utf-8')

    def line(self, text: str = '') -> None:
        self._file.write(text + '\n')
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def confirm(
    checkout: pathlib.Path,
    base: str | None,
    scratch: pathlib.Path,
    report: Report,
    workers: str,
) -> int:
    started = time.monotonic()
    tip = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=checkout, capture_output=True, text=True, check=True
    ).stdout.strip()
    base_sha = resolve_base(checkout, base)
    report.line(f'confirm_pins: {checkout}  base {base_sha[:10]}  tip {tip[:10]}')
    worktree = Worktree(checkout, tip, scratch)
    try:
        return _confirm(worktree, base_sha, report, workers, started)
    finally:
        worktree.remove()


def _confirm(worktree: Worktree, base: str, report: Report, workers: str, started: float) -> int:
    names = worktree.git('diff', '--no-renames', '--name-only', f'{base}..HEAD', '--', '*.py')
    files = [path for path in names.splitlines() if is_production_path(path)]
    diff = (
        worktree.git('diff', '--no-renames', '-U1', f'{base}..HEAD', '--', *files) if files else ''
    )
    hunks = cut_hunks(diff)
    report.line(f'production files: {len(files)}  hunks: {len(hunks)}')
    if not hunks:
        report.line('result: PASS, no production hunk in the range (exit 0)')
        return EXIT_PINNED

    collection = worktree.collect()
    report.line(f'tests collected at the tip: {len(collection)}')

    worktree.apply(diff, reverse=True)
    if not worktree.status().strip():
        raise RefusedError('reverse-applying the whole production diff changed nothing')
    t0 = time.monotonic()
    reverted = worktree.pytest('suite_reverted', workers, ())
    suite_seconds = time.monotonic() - t0
    worktree.apply(diff, reverse=False)
    _assert_clean(worktree, 'the whole production diff')
    if not reported_tests(reverted, collection):
        raise RefusedError(
            'INSTRUMENT: the suite with the diff reverse-applied reported none of the'
            ' collected tests; its log is suite_reverted.log in the scratch directory'
        )
    observing = observing_tests(reverted, collection)
    validate_observing(observing, collection)
    report.line(
        f"suite with the range's production diff reverse-applied: {suite_seconds:.1f} s;"
        f' observing tests: {len(observing)}'
    )

    pairs = moved_pairs(hunks)
    if not observing:
        return _report_unobserved(worktree, hunks, pairs, report, workers, started)

    t0 = time.monotonic()
    at_tip = worktree.pytest('observing_at_tip', workers, sorted(observing))
    baseline, skipped = tip_baseline(at_tip, observing)
    report.line(
        f'observing tests at the tip: {time.monotonic() - t0:.1f} s;'
        f' baseline {len(baseline)} passed, {skipped} skipped at both ends'
    )
    report.line('--')

    counts: dict[str, int] = defaultdict(int)
    for n, hunk in enumerate(hunks):
        t0 = time.monotonic()
        verdict, detail = _judge(worktree, hunk, baseline, workers, n)
        counts[verdict] += 1
        note = f'  (moved with hunk {pairs[n] + 1})' if n in pairs else ''
        report.line(
            f'{n + 1:4d}  {hunk.path}  {hunk.header}  {verdict}  {detail}'
            f'  {time.monotonic() - t0:.1f} s{note}'
        )
        if verdict == 'INSTRUMENT':
            raise RefusedError(f'hunk {n + 1} ({hunk.path} {hunk.header}): {detail}')
    report.line('--')
    behavioural = len(hunks) - counts['NEUTRAL']
    report.line(
        f'RED {counts["RED"]}  GREEN {counts["GREEN"]}  NEUTRAL {counts["NEUTRAL"]}'
        f'  (behavioural {behavioural})'
    )
    report.line(f'total {time.monotonic() - started:.1f} s')
    if counts['GREEN']:
        report.line('result: UNPINNED (exit 1)')
        return EXIT_UNPINNED
    report.line('result: PINNED (exit 0)')
    return EXIT_PINNED


def _judge(
    worktree: Worktree, hunk: Hunk, baseline: Collection[str], workers: str, n: int
) -> tuple[str, str]:
    """Revert one hunk alone, decide it, restore, and prove the tree clean."""
    after = worktree.read(hunk.path)
    worktree.apply(hunk.patch, reverse=True)
    if not worktree.status().strip():
        raise RefusedError(f'reverting hunk {n + 1} ({hunk.path} {hunk.header}) changed nothing')
    try:
        if is_neutral(worktree.read(hunk.path), after):
            return 'NEUTRAL', 'docstrings or comments only'
        if not baseline:
            return 'NOT-OBSERVED', 'no test observes the range'
        outcomes = worktree.pytest(f'hunk_{n + 1:03d}', workers, sorted(baseline))
        return hunk_verdict(outcomes, baseline)
    finally:
        worktree.apply(hunk.patch, reverse=False)
        _assert_clean(worktree, f'hunk {n + 1} ({hunk.path} {hunk.header})')


def _report_unobserved(
    worktree: Worktree,
    hunks: list[Hunk],
    pairs: dict[int, int],
    report: Report,
    workers: str,
    started: float,
) -> int:
    report.line(
        'no test observes the range: the suite with its production diff reverse-applied'
        ' passed every test. A behaviour-preserving refactor reads this way, and so does'
        ' a behaviour change without its fail-before test; the commit body says which.'
    )
    report.line('--')
    neutral = 0
    for n, hunk in enumerate(hunks):
        verdict, detail = _judge(worktree, hunk, (), workers, n)
        neutral += verdict == 'NEUTRAL'
        note = f'  (moved with hunk {pairs[n] + 1})' if n in pairs else ''
        report.line(f'{n + 1:4d}  {hunk.path}  {hunk.header}  {verdict}  {detail}{note}')
    report.line('--')
    report.line(f'NOT-OBSERVED {len(hunks) - neutral}  NEUTRAL {neutral}')
    report.line(f'total {time.monotonic() - started:.1f} s')
    report.line('result: PASS, no test observes the range (exit 0)')
    return EXIT_PINNED


def _assert_clean(worktree: Worktree, what: str) -> None:
    dirt = worktree.status().strip()
    if dirt:
        raise RefusedError(f'restoring {what} left the tree dirty:\n{dirt}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    parser.add_argument('--checkout', default='.', help='the checkout whose HEAD is the tip')
    parser.add_argument('--base', help='the range starts here; default: the merge-base with @{u}')
    parser.add_argument('--scratch', help='holds the worktree, the logs and the report')
    parser.add_argument('--report', help='the report file; default: <scratch>/report.txt')
    parser.add_argument('--workers', default='auto', help="pytest-xdist's -n; default auto")
    args = parser.parse_args(argv)
    checkout = pathlib.Path(args.checkout).resolve()
    scratch = pathlib.Path(args.scratch or tempfile.mkdtemp(prefix='confirm_pins_')).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    report = Report(pathlib.Path(args.report or scratch / 'report.txt').resolve())
    print(f'report: {report.path}', flush=True)
    try:
        code = confirm(checkout, args.base, scratch, report, args.workers)
    except RefusedError as refusal:
        report.line(f'refused: {refusal} (exit 2)')
        print(f'refused: {refusal}', file=sys.stderr)
        code = EXIT_REFUSED
    finally:
        report.close()
    print(report.path.read_text(encoding='utf-8').rstrip().rsplit('\n', 1)[-1], flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
