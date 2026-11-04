# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every run the GUI starts asks for its handlers on the UI thread.

A run's events arrive on the thread that sends them unless its
``RunEvents`` says ``on_ui_thread``, so a REST client's run in the GUI's
process never waits on the Kivy thread. The GUI's handlers draw widgets, so
a ``RunEvents`` built under ``ui/`` with any handler but ``frame_captured``
(delivered in place, on the run's thread, whatever the flag) must say
``on_ui_thread=True``; one that forgets would draw off the Kivy thread.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT as ROOT

# Constructions whose absence would mean the walk lost its way.
AT_LEAST = 4


def _constructions():
    for path in sorted([*(ROOT / 'ui').rglob('*.py'), ROOT / 'lumaviewpro.py']):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'RunEvents':
                yield path.relative_to(ROOT), node


def test_every_gui_run_events_with_a_handler_asks_for_the_ui_thread():
    checked = 0
    forgot = []
    for path, call in _constructions():
        keywords = {k.arg: k.value for k in call.keywords}
        if not set(keywords) - {'frame_captured', 'on_ui_thread'}:
            continue
        checked += 1
        flag = keywords.get('on_ui_thread')
        if not (isinstance(flag, ast.Constant) and flag.value is True):
            forgot.append(f'{path}:{call.lineno}')
    assert checked >= AT_LEAST, f'found {checked} GUI RunEvents with handlers'
    assert forgot == [], f'RunEvents built under ui/ without on_ui_thread=True: {forgot}'
