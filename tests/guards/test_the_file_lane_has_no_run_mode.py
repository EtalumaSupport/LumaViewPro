# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Guard: production code puts only the IO and CAMERA lanes into run mode.

A run's writes are counted by the run's own write batch, not by the file
lane: the lane cannot say which queued writes are a run's, and a run mode
on it let one run's start and end swallow another run's completion. The
lane's run-mode members still exist, because every lane is a
``SequentialIOExecutor`` and shutdown ends run mode on each, so nothing
stops a new call reaching them on the file lane except this guard.

Every production call of a run-mode member must be made on a receiver
known to be the IO or CAMERA lane, or by the executor on itself. A new
receiver -- the file lane, or a name this guard cannot tell apart from
it -- fails until it is added here on purpose.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT

_RUN_MODE_MEMBERS = frozenset(
    {
        'protocol_put',
        'protocol_start',
        'protocol_end',
        'protocol_finish_then_end',
        'clear_protocol_pending',
    }
)

# Receivers that name the IO or CAMERA lane.
_RUN_MODE_LANES = frozenset({'io_executor', '_io_executor', 'camera_executor'})

_EXECUTOR_SOURCE = 'modules/sequential_io_executor.py'

_SOURCES = ('modules', 'ui', 'drivers', 'plugins')


def _receiver_name(node: ast.expr) -> str:
    """The last name in a receiver: ``p._io_executor`` -> ``_io_executor``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def _run_mode_calls():
    paths = [REPO_ROOT / 'lumaviewpro.py']
    for d in _SOURCES:
        if (REPO_ROOT / d).is_dir():
            paths.extend((REPO_ROOT / d).rglob('*.py'))
    for path in paths:
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=rel)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RUN_MODE_MEMBERS
            ):
                yield rel, node.lineno, _receiver_name(node.func.value), node.func.attr


def test_only_the_io_and_camera_lanes_enter_run_mode():
    offenders = [
        f'{rel}:{line}: {receiver}.{member}()'
        for rel, line, receiver, member in _run_mode_calls()
        if receiver not in _RUN_MODE_LANES and not (rel == _EXECUTOR_SOURCE and receiver == 'self')
    ]
    assert not offenders, (
        "A run-mode member is called on a lane that is not IO or CAMERA; a run's "
        'writes are counted by its write batch, never by run mode on the file lane:\n'
        + '\n'.join(offenders)
    )


def test_the_scan_sees_the_run_mode_calls_that_exist():
    """The guard is not vacuous: it finds the run's own IO and CAMERA calls."""
    found = {(rel, receiver, member) for rel, _line, receiver, member in _run_mode_calls()}
    assert ('modules/sequenced_capture_runner.py', 'camera_executor', 'protocol_start') in found
    assert ('modules/sequenced_capture_runner.py', '_io_executor', 'protocol_start') in found
    assert ('modules/protocol_cleanup.py', 'io_executor', 'protocol_end') in found
