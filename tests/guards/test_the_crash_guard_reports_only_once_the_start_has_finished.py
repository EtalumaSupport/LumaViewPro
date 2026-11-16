# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The main loop's handler reports LVP's own raise only once the start has finished.

``_PluginCrashGuard.started`` is the one switch between the two policies:
before it, a raise no call caught is a start that failed and ends the app;
after it, the raise is reported and the app stays up. It is set once, as
the last statement of ``after_the_startup_motion``, so a raise in any step
of the start still ends the app, and nothing turns it off again. A write
anywhere else, or one moved up the start, fails here.

Scanned rather than executed: ``lumaviewpro.py`` builds a Kivy widget tree
at import, which the suite does not reach.
"""

from __future__ import annotations

import ast

from tests.ast_seams import parse_module


def _writes_of_started(tree: ast.AST) -> list[ast.Assign]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == 'started'
            for target in node.targets
        )
    ]


def test_started_is_set_once_as_the_last_step_of_the_start():
    tree = parse_module('lumaviewpro.py')
    [start_end] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == 'after_the_startup_motion'
    ]

    writes = _writes_of_started(tree)

    assert [ast.unparse(write) for write in writes] == ['crash_guard.started = True']
    assert start_end.body[-1] is writes[0], (
        'the crash guard must start reporting only after every step of the start: '
        'its started flag is the last statement of after_the_startup_motion'
    )
