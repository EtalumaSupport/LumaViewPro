# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""FW4.0 release-gate §2.2 — no per-board protocol branches in the V4 path.

The unification thesis:

    LED and motor speak the same wire protocol with shared framing code.
    Host-side, this means V4 command dispatch must use capability probing
    (`has_feature(name)`), never per-board type dispatch.

A drift-prevention AST scan. Forever.

Forbidden patterns in `drivers/` and `modules/`:
    isinstance(<x>, LEDBoard)     # per-board V4 branch
    isinstance(<x>, MotorBoard)
    type(<x>) == LEDBoard
    type(<x>) is  LEDBoard
    (same for MotorBoard)

Allowed (NOT forbidden):
    isinstance(x, NullLEDBoard)   # null-object sentinel; orthogonal to V4
    isinstance(x, SerialBoard)    # base-class check (e.g. in tests)
    isinstance(x, SimulatedLEDBoard) / SimulatedMotorBoard
                                  # test-side fixture detection; V4 dispatch
                                  # already settled by the time a
                                  # simulator reaches production code

Why this test exists:

    Release gate §2.2: "LVP-side has zero per-board protocol branches in
    the V4 path. No isinstance(board, LEDBoard) / isinstance(board,
    MotorBoard) in V4 command-handling code. Capability dispatch by
    has_feature(), not by board class."

    It's cheap to introduce a subtle per-board check during a bugfix;
    this test fails the moment that happens, so the reviewer sees it
    without having to know the rule.

Scope:
    drivers/*.py and modules/*.py only. tests/ is exempt (tests may
    legitimately assert board identity). Test files and the LEDBoard /
    MotorBoard class definitions themselves are fine — those files don't
    call isinstance() with themselves as the second argument.
"""
import ast
import pathlib

import pytest

# Heavy deps mocked by tests/conftest.py at collection time.

# The symbols that must never appear as the second argument of isinstance()
# or on the right-hand side of a type(...) comparison, anywhere in the
# scanned tree.
FORBIDDEN_TYPE_NAMES = frozenset({'LEDBoard', 'MotorBoard'})

# Directories scanned for per-board dispatch. Tests and build scripts are
# exempt. Any new Python source tree that participates in command dispatch
# should be added here as well.
SCAN_DIRS = ('drivers', 'modules')

# Files exempt because their purpose is explicitly to assert board identity
# (e.g., the Lumascope construction-time wiring) rather than per-command
# dispatch. Keep this list minimal; additions require a comment explaining
# why the file is outside the V4 dispatch path.
EXEMPT_FILES: frozenset[str] = frozenset()


def _find_repo_root() -> pathlib.Path:
    """Walk up from this file to find the LumaViewPro root (the directory
    containing `drivers/` and `modules/` and `tests/`)."""
    here = pathlib.Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / 'drivers').is_dir() and (parent / 'modules').is_dir():
            return parent
    raise RuntimeError(f'cannot locate LVP repo root from {here}')


def _iter_scanned_files():
    root = _find_repo_root()
    for subdir in SCAN_DIRS:
        for py in sorted((root / subdir).rglob('*.py')):
            if py.name == '__init__.py':
                continue
            rel = py.relative_to(root).as_posix()
            if rel in EXEMPT_FILES:
                continue
            yield rel, py


def _name_of(node: ast.AST) -> str | None:
    """Return the dotted name of `node` if it's a Name or Attribute chain;
    otherwise None. Handles `LEDBoard`, `drivers.LEDBoard`, etc."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f'{base}.{node.attr}' if base else node.attr
    return None


def _collect_violations(source: str, path: str) -> list[tuple[int, str]]:
    """Parse `source`, return (lineno, message) for every forbidden
    per-board dispatch pattern."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        pytest.fail(f'{path}: syntax error at line {e.lineno}: {e.msg}')

    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # Pattern 1: isinstance(x, LEDBoard) or isinstance(x, MotorBoard)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'isinstance'
                and len(node.args) == 2):
            # Second arg may be a bare name, an attribute (drivers.LEDBoard),
            # or a tuple of such.
            targets = (node.args[1].elts
                       if isinstance(node.args[1], ast.Tuple)
                       else [node.args[1]])
            for t in targets:
                name = _name_of(t)
                if name is None:
                    continue
                leaf = name.rsplit('.', 1)[-1]
                if leaf in FORBIDDEN_TYPE_NAMES:
                    violations.append((
                        node.lineno,
                        f'isinstance(..., {leaf}) — use has_feature() '
                        f'for V4 dispatch',
                    ))

        # Pattern 2: type(x) == LEDBoard / type(x) is LEDBoard (and same
        # for MotorBoard). Also catches LEDBoard == type(x) (swapped form).
        if isinstance(node, ast.Compare):
            left_name = _compare_leaf(node.left)
            right_names = [_compare_leaf(c) for c in node.comparators]
            candidates = [left_name] + right_names
            if any(n == '<type-call>' for n in candidates):
                for other in candidates:
                    if other in FORBIDDEN_TYPE_NAMES:
                        violations.append((
                            node.lineno,
                            f'type(x) comparison with {other} — use '
                            f'has_feature() for V4 dispatch',
                        ))

    return violations


def _compare_leaf(node: ast.AST) -> str | None:
    """For a comparator side, return either the type leaf name or the
    sentinel '<type-call>' if the side is `type(x)`. Everything else returns
    None."""
    if isinstance(node, ast.Call) and _name_of(node.func) == 'type':
        return '<type-call>'
    n = _name_of(node)
    return n.rsplit('.', 1)[-1] if n else None


class TestNoPerBoardBranches:

    def test_no_forbidden_isinstance_or_type_checks(self):
        all_violations: list[str] = []
        scanned = 0

        for rel, py in _iter_scanned_files():
            scanned += 1
            src = py.read_text(encoding='utf-8')
            for lineno, msg in _collect_violations(src, rel):
                all_violations.append(f'{rel}:{lineno}: {msg}')

        assert scanned > 0, 'scanner found zero source files (path misconfig?)'

        if all_violations:
            pytest.fail(
                'FW4.0 §2.2 drift: per-board type dispatch detected.\n'
                'V4 command dispatch must route via has_feature(), not '
                'isinstance(board, LEDBoard) / isinstance(board, MotorBoard). '
                'See docs/FW40_RELEASE_GATE.md §2.2.\n\nOffending sites:\n  '
                + '\n  '.join(all_violations)
            )

    def test_scanner_detects_a_planted_violation(self):
        """Meta-test: the AST walker actually finds violations when
        present. Protects against the scanner silently matching nothing
        (which would turn the gate into a no-op)."""
        planted = (
            "from drivers.ledboard import LEDBoard\n"
            "def f(board):\n"
            "    if isinstance(board, LEDBoard):\n"
            "        return 'LED path'\n"
            "    return 'other'\n"
        )
        vs = _collect_violations(planted, '<planted>')
        assert any('LEDBoard' in msg for _, msg in vs), (
            f'scanner missed a planted isinstance(LEDBoard) violation; got {vs}'
        )

        planted2 = (
            "def g(board):\n"
            "    return type(board) == MotorBoard\n"
        )
        vs2 = _collect_violations(planted2, '<planted2>')
        assert any('MotorBoard' in msg for _, msg in vs2), (
            f'scanner missed a planted type(...) == MotorBoard violation; '
            f'got {vs2}'
        )

    def test_scanner_allows_null_object_sentinel(self):
        """Meta-test: the null-object pattern (isinstance against
        NullLEDBoard) is correctly NOT flagged — that's the is-connected
        idiom, not per-board V4 dispatch."""
        ok = (
            "from drivers.nullboards import NullLEDBoard\n"
            "def h(led):\n"
            "    return not isinstance(led, NullLEDBoard)\n"
        )
        vs = _collect_violations(ok, '<ok>')
        assert vs == [], (
            f'scanner incorrectly flagged NullLEDBoard usage; got {vs}'
        )
