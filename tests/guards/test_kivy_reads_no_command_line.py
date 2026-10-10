# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Kivy never reads LumaViewPro's command line.

Kivy parses ``sys.argv`` when it is imported and exits on an argument it does
not know. In the packaged build that exit is silent: the bootloader shows a
dialog for an escaping exception, but nothing for a ``SystemExit``. So an
installed build launched with any argument Kivy does not know vanished. The
GUI sets ``KIVY_NO_ARGS`` before its first Kivy import; the real launch
cannot reach that import without opening a window, so the order is read from
the source, and Kivy's honouring of the variable is run.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

from tests.ast_seams import parse_module


def _sets_no_args(node: ast.AST) -> bool:
    """``os.environ['KIVY_NO_ARGS'] = '1'``."""
    return (
        isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].slice, ast.Constant)
        and node.targets[0].slice.value == 'KIVY_NO_ARGS'
        and isinstance(node.value, ast.Constant)
        and node.value.value == '1'
    )


def _imports_kivy(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split('.')[0] == 'kivy' for alias in node.names)
    return isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'kivy'


def test_the_gui_tells_kivy_to_read_no_command_line_before_it_imports_kivy():
    nodes = list(ast.walk(parse_module('lumaviewpro.py')))
    first_kivy = min(n.lineno for n in nodes if _imports_kivy(n))
    sets = [n.lineno for n in nodes if _sets_no_args(n)]

    assert sets, 'lumaviewpro.py never sets KIVY_NO_ARGS'
    assert min(sets) < first_kivy


def _import_kivy_with_an_unknown_argument(no_args: bool) -> int:
    env = {k: v for k, v in os.environ.items() if k != 'KIVY_NO_ARGS'}
    env.update({'KIVY_NO_CONSOLELOG': '1', 'KIVY_NO_FILELOG': '1'})
    if no_args:
        env['KIVY_NO_ARGS'] = '1'
    return subprocess.run(
        [
            sys.executable,
            '-c',
            "import sys; sys.argv = ['lvp', '--not-a-kivy-option']; import kivy",
        ],
        env=env,
        capture_output=True,
        timeout=60,
    ).returncode


def test_kivy_honours_it_for_an_argument_it_does_not_know():
    assert _import_kivy_with_an_unknown_argument(no_args=True) == 0
    # The known positive: without it, the same argument ends the process.
    assert _import_kivy_with_an_unknown_argument(no_args=False) != 0
