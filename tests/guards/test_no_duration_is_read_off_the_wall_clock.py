# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A duration or a timeout is never measured on the wall clock.

The wall clock steps: an NTP correction, a daylight-saving change on a host
whose clock keeps local time, a user setting the time. A deadline taken off
it ends at once or runs an hour long across a step, and an elapsed time
taken off it gains or loses the step. A timeout reads ``time.monotonic()``,
the timeouts' clock; a duration that is logged or set against frame stamps
reads ``host_clock.steady_ns()``.

This guard fails the build on a ``time.time()`` call that is an operand of
an addition, a subtraction or a comparison in production: a deadline
(``time.time() + timeout``), an elapsed check (``time.time() - start``), a
wait (``while time.time() < deadline``). Its limit: two wall reads kept in
names and subtracted later (``end - start`` over two ``wall_time.now()``
values) take dataflow to see, and this guard does not see them.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, Packages, tracked_files

_PRODUCTION = Packages({'src', 'ui', 'tools'})

# Measured against an instant that has only a wall form, so the wall clock
# is the one to read: uptime is now less the boot instant, and the host
# states its boot (``psutil.boot_time()``) on the wall clock alone.
_WALL_AGAINST_WALL = {
    ('src/_lumascope/modules/common_utils.py', 'host_boot'),
}


def _wall_reads(tree: ast.AST) -> set[str]:
    """What a call is spelled as when it reads ``time.time``: ``<alias>.time``
    for each name the time module is imported as, and each name ``time.time``
    is imported as."""
    modules, functions = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.asname or a.name for a in node.names if a.name == 'time'}
        elif isinstance(node, ast.ImportFrom) and node.module == 'time':
            functions |= {a.asname or a.name for a in node.names if a.name == 'time'}
    return {f'{m}.time' for m in modules} | functions


def _spelling(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f'{func.value.id}.{func.attr}'
    return None


def offences(tree: ast.AST) -> list[tuple[int, str | None]]:
    """``(line, enclosing function)`` of each wall read used in arithmetic or a comparison."""
    reads = _wall_reads(tree)

    def is_wall_read(node: ast.expr) -> bool:
        return isinstance(node, ast.Call) and _spelling(node.func) in reads

    found = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        operands = []
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            operands = [node.left, node.right]
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
            operands = [node.value]
        elif isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
        found.extend((o.lineno, function) for o in operands if is_wall_read(o))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return found


def _production_files() -> list[str]:
    return tracked_files(_PRODUCTION) + tracked_files(pattern='*.py')


def test_every_spelling_of_a_wall_clock_duration_is_found():
    """The guard's own known positives, so its clean result below means none."""
    for source in (
        'import time\ndeadline = time.time() + timeout',
        'import time\nwhile time.time() < deadline: pass',
        'import time\nelapsed = time.time() - start',
        'import time as _time\nup = _time.time() - boot',
        'from time import time\nif deadline > time(): pass',
        'from time import time as now\nt -= now()',
    ):
        assert [line for line, _ in offences(ast.parse(source))] == [2], source
    for clean in (
        'import time\nts_ms = int(time.time() * 1000)',
        'import time\ndeadline = time.monotonic() + timeout',
        'import time\nstamp = time.time()',
        'value.time() - start',
    ):
        assert offences(ast.parse(clean)) == [], clean


def test_no_production_duration_or_timeout_reads_the_wall_clock():
    offenders = []
    for path in _production_files():
        source = (REPO_ROOT / path).read_text(encoding='utf-8')
        for lineno, function in offences(ast.parse(source, filename=path)):
            if (path, function) not in _WALL_AGAINST_WALL:
                offenders.append(f'{path}:{lineno}')
    assert not offenders, (
        'these measure a duration or a timeout on the wall clock, which steps; read '
        'time.monotonic() for a timeout and host_clock.steady_ns() for a duration: '
        + ', '.join(offenders)
    )


def test_every_wall_against_wall_exemption_still_reads_the_wall_clock():
    """An exemption whose read moved or went is removed, never left to cover a new one."""
    held = set()
    for path in {path for path, _ in _WALL_AGAINST_WALL}:
        source = (REPO_ROOT / path).read_text(encoding='utf-8')
        held |= {(path, function) for _, function in offences(ast.parse(source, filename=path))}
    assert held >= _WALL_AGAINST_WALL, sorted(_WALL_AGAINST_WALL - held)
