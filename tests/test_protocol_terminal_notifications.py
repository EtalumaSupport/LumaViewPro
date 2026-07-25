# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Run-terminal error notifications in the protocol run loop must be fatal.

The notification center suppresses non-fatal popups while a protocol is
running, and the suppression flag clears only inside the runner's cleanup
-- which every run-loop abort path calls AFTER emitting its notification.
A run-aborting error posted there without ``fatal=True`` is therefore
guaranteed to be swallowed: the run dies and the user never sees why.
The fatal path is the one popup class permitted mid-run, and the
behavior side of that contract is pinned in test_notification_center.py;
this scan pins the emitting sites.
"""

import ast
from pathlib import Path

RUN_LOOP = Path(__file__).resolve().parent.parent / 'modules' / 'protocol_run_loop.py'

TERMINAL_TITLES = {'Protocol Stopped', 'Protocol Aborted'}


def _terminal_error_calls_missing_fatal():
    tree = ast.parse(RUN_LOOP.read_text(encoding='utf-8'))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'error'):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == 'notifications'):
            continue
        titles = [
            a.value for a in node.args if isinstance(a, ast.Constant) and a.value in TERMINAL_TITLES
        ]
        if not titles:
            continue
        fatal_kw = next((kw for kw in node.keywords if kw.arg == 'fatal'), None)
        is_fatal_true = (
            fatal_kw is not None
            and isinstance(fatal_kw.value, ast.Constant)
            and fatal_kw.value.value is True
        )
        if not is_fatal_true:
            missing.append(f'{titles[0]} at line {node.lineno}')
    return missing


def test_run_terminal_notifications_are_fatal():
    missing = _terminal_error_calls_missing_fatal()
    assert missing == [], (
        'run-terminal notification(s) without fatal=True in '
        'protocol_run_loop.py -- these fire before cleanup clears the '
        'mid-run suppression, so a non-fatal one is silently swallowed '
        f'and the user never learns why the run died: {missing}'
    )
