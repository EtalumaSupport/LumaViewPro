# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""LumaViewPro installs its crash hooks before it runs.

Importing ``lvp_logger`` installs none (a script crashes as Python does), so
the GUI calls ``install_crash_hooks()`` itself; without the call a core
defect leaving ``LumaViewProApp().run()`` -- which ``_PluginCrashGuard``
re-raises for its post-mortem -- would leave no crash record in the log.
"""

from __future__ import annotations

import ast

from tests.ast_seams import parse_module


def _calls(tree: ast.AST, name: str) -> list[int]:
    """The lines calling ``name(...)`` or ``<something>.name(...)``."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            if called == name:
                found.append(node.lineno)
    return found


def test_the_gui_installs_its_crash_hooks_before_it_runs():
    tree = parse_module('lumaviewpro.py')

    installs = _calls(tree, 'install_crash_hooks')
    runs = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'run'
        and isinstance(node.func.value, ast.Call)
        and getattr(node.func.value.func, 'id', None) == 'LumaViewProApp'
    ]

    assert installs, 'lumaviewpro.py never calls install_crash_hooks()'
    assert runs
    assert min(installs) < min(runs)
