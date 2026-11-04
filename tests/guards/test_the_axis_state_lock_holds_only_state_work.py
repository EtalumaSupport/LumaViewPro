# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Nothing slow runs under ``motion._axis_state_lock``: its holds do state work only.

Every motion read and the monitor's state writes take this lock, so a
serial round-trip under it (30-200 ms on the motor bus), a log line, a
listener or a move's ``publish`` stalls every reader behind it. The rule
was enforced at runtime by a 1 ms wall-clock hold warning, which measured
the scheduler rather than the code: a hold of about 1 us read as 12-37 ms
whenever other threads held the interpreter, inside the serial range the
warning existed to catch, and a soak logged 168 of them with no slow call
under the lock. The bodies are read here instead, on every host alike.

The calls below are the ones the holds make today, each a dict, set or
Event operation, a builtin, or a pure helper. A call not in the list fails
this guard by name: move it after the ``with``, or add it here once it is
shown to do state work only.
"""

from __future__ import annotations

import ast
import re

from tests.ast_seams import parse_module

_MOTION = 'modules/lumascope_api/motion.py'

_STATE_WORK = frozenset(
    {
        'MoveFaultError',
        'any',
        'current.settle',  # sets the move's outcome fields; publish runs after the hold
        'dict',
        'float',
        'self._arrival_events[].clear',
        'self._arrival_events[].set',
        'self._axis_state.get',
        'self._axis_state.items',
        'self._axis_state.values',
        'self._current_move.get',
        'self._position_known',  # a set-membership test on the state
        'host_clock.steady_ns',
    }
)


def _is_the_lock(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Attribute)
        and expr.attr == '_axis_state_lock'
        and isinstance(expr.value, ast.Name)
        and expr.value.id == 'self'
    )


def calls_under_the_lock(tree: ast.Module) -> list[tuple[int, str]]:
    """Each call made inside a ``with self._axis_state_lock:`` body, as (line, callee)."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(_is_the_lock(i.context_expr) for i in node.items):
            for stmt in node.body:
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Call):
                        callee = re.sub(r'\[[^\]]*\]', '[]', ast.unparse(inner.func))
                        calls.append((inner.lineno, callee))
    return calls


def test_every_call_under_the_axis_state_lock_is_state_work():
    calls = calls_under_the_lock(parse_module(_MOTION))

    assert calls, f'no hold of _axis_state_lock found in {_MOTION}: the guard reads nothing'
    slow = [f'{_MOTION}:{line} {callee}' for line, callee in calls if callee not in _STATE_WORK]
    assert not slow, 'called under _axis_state_lock, not known to be state work:\n' + '\n'.join(
        slow
    )


def test_a_call_that_is_not_state_work_is_named():
    planted = ast.parse(
        'class M:\n'
        '    def f(self):\n'
        '        with self._axis_state_lock:\n'
        '            self._axis_state.get("Z")\n'
        '            self._driver.move_abs_pos("Z", 0)\n'
    )

    callees = [callee for _, callee in calls_under_the_lock(planted)]

    assert [c for c in callees if c not in _STATE_WORK] == ['self._driver.move_abs_pos']
