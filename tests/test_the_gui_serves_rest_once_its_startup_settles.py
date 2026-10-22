# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI starts REST beside its window once the startup motion settles, never after its close began.

``rest_api.enabled`` set, the window starts the server on its own session
(``rest.server.serve_beside``) after the metrics, as the headless launcher
serves after its own; unset, nothing listens. Everything that follows the
startup motion is refused by the fact the window's close sets first, on the
Kivy thread: the session's own closing is set later, on the close's thread,
so a close in the motion's own tick would otherwise run all of it -- a
server among it -- after the close began.

Scanned rather than executed, as ``test_close_request_gate`` is:
``lumaviewpro.py`` builds a Kivy widget tree at import.
"""

import ast
import os

import pytest

_ENTRY_POINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lumaviewpro.py'
)


def _app_class():
    with open(_ENTRY_POINT, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=_ENTRY_POINT)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'LumaViewProApp':
            return node
    pytest.fail('LumaViewProApp not found in lumaviewpro.py (precondition)')


def _function(scope, name):
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f'{name} not found (precondition)')


def _called(node) -> list[str]:
    """Each call's callee, in source order."""
    calls = sorted(
        (n for n in ast.walk(node) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    return [ast.unparse(n.func) for n in calls]


def test_the_close_sets_its_fact_before_anything_else():
    prepare = _function(_app_class(), '_prepare_the_close')
    first = next(
        s
        for s in prepare.body
        if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)
    )

    assert ast.unparse(first) == 'self._closing_begun = True', ast.unparse(first)


def test_what_follows_the_startup_motion_is_refused_once_the_close_began():
    """Read before the motion's result is raised or anything after it runs."""
    settled = _function(_app_class(), 'startup_motion_settled')
    source = [ast.unparse(s) for s in settled.body]
    refusal = next(i for i, s in enumerate(source) if 'self._closing_begun' in s)
    result = next(i for i, s in enumerate(source) if 'done.result()' in s)

    assert refusal < result, source
    assert source[refusal].startswith('if self._closing_begun') and source[refusal].endswith(
        'return'
    )


def test_the_server_starts_after_the_metrics_from_the_enabled_setting():
    app = _app_class()
    after = _function(app, 'after_the_startup_motion')
    called = _called(after)
    calls = sorted(
        (n for n in ast.walk(after) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    started = [
        i
        for i, call in enumerate(calls)
        if ast.unparse(call.func) == 'run_unasked'
        and ast.unparse(call.args[0]) == 'self._start_the_rest_server'
    ]

    assert len(started) == 1, called
    assert called.index('ctx.session.enable_metrics') < started[0], called
    start = _function(app, '_start_the_rest_server')
    assert "ctx.session.get_setting('rest_api.enabled')" in ast.unparse(start)
    assert 'serve_beside' in _called(start)
