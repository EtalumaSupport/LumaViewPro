# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Kivy starts in the environment the GUI sets: no command line, its WARNING level.

Kivy reads its environment once, when it is imported. Two of its effects
are pinned here, each by a run of a real Kivy import:

- It parses ``sys.argv`` and exits on an argument it does not know. In the
  packaged build that exit is silent: the bootloader shows a dialog for an
  escaping exception, but nothing for a ``SystemExit``, so an installed
  build launched with any argument Kivy does not know vanished.
- It sets its logger's level from the user's ``~/.kivy/config.ini``
  (``info`` by default), over any level set before the import, so its INFO
  startup block reached every support bundle; a PC whose file says
  ``debug`` would send Kivy's debug output there.

The GUI calls ``set_kivy_environment`` before its first Kivy import; the
real launch cannot reach that import without opening a window, so the order
is read from the source. Each child gets a Kivy home of its own, so no run
reads or writes the developer's ``~/.kivy``.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys

from _lumascope.modules.app_environment import KIVY_ENVIRONMENT
from tests.ast_seams import parse_module

# What the child reports: its Kivy logger's level and the records Kivy kept.
_REPORT_THE_LEVEL = (
    'import json, logging; from kivy.config import Config; from kivy.logger import LoggerHistory; '
    "print(json.dumps({'level': logging.getLevelName(logging.getLogger('kivy').level), "
    "'history': [r.getMessage() for r in LoggerHistory.history]}))"
)


def _calls_set_kivy_environment(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'set_kivy_environment'
    )


def _imports_kivy(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split('.')[0] == 'kivy' for alias in node.names)
    return isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'kivy'


def test_the_gui_sets_kivy_s_environment_before_it_imports_kivy():
    nodes = list(ast.walk(parse_module('lumaviewpro.py')))
    first_kivy = min(n.lineno for n in nodes if _imports_kivy(n))
    calls = [n.lineno for n in nodes if _calls_set_kivy_environment(n)]

    assert calls, 'lumaviewpro.py never calls set_kivy_environment'
    assert min(calls) < first_kivy


def _run_kivy(
    kivy_home: pathlib.Path, code: str, *, leave_out: str = ''
) -> subprocess.CompletedProcess:
    """A child importing Kivy in the GUI's environment, less *leave_out*, with its own Kivy home."""
    env = {k: v for k, v in os.environ.items() if k not in KIVY_ENVIRONMENT}
    env.update({k: v for k, v in KIVY_ENVIRONMENT.items() if k != leave_out})
    env['KIVY_HOME'] = str(kivy_home)
    return subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=60
    )


def test_kivy_reads_no_command_line(tmp_path):
    unknown = "import sys; sys.argv = ['lvp', '--not-a-kivy-option']; import kivy"

    assert _run_kivy(tmp_path, unknown).returncode == 0
    # The known positive: without KIVY_NO_ARGS, the same argument ends the process.
    assert _run_kivy(tmp_path, unknown, leave_out='KIVY_NO_ARGS').returncode != 0


def test_kivy_logs_at_warning_whatever_its_config_file_says(tmp_path):
    # A config.ini Kivy wrote itself, so it is current: one written by hand
    # makes Kivy log its upgrade steps while it reads the file.
    assert _run_kivy(tmp_path, 'import kivy').returncode == 0
    config = tmp_path / 'config.ini'
    written = config.read_text()
    assert 'log_level = info' in written
    config.write_text(written.replace('log_level = info', 'log_level = debug'))

    started = json.loads(_run_kivy(tmp_path, _REPORT_THE_LEVEL).stdout)

    assert started == {'level': 'WARNING', 'history': []}
    # The known positive: without the level in the environment, the file's level holds.
    unpinned = json.loads(
        _run_kivy(tmp_path, _REPORT_THE_LEVEL, leave_out='KCFG_KIVY_LOG_LEVEL').stdout
    )
    assert unpinned['level'] == 'DEBUG'
