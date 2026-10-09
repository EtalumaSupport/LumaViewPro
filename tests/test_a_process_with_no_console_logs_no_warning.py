# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Windows process with no console logs nothing about it; one with a console minimizes it.

The installed exe is a windowed build, so it never has a console: a warning
there was logged at every launch and reached the error log, reporting the
expected state as a fault. A run from ``python.exe`` has one, and lvp_logger
minimizes it.

Each runs in a child Python: the suite replaces ``lvp_logger`` with a stand-in
(``tests/conftest.py``), so only a fresh interpreter imports the real one. The
child swaps the logger's file handlers for a list, so nothing reaches this
checkout's logs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_CHILD = """
import ctypes, json, logging, sys
from types import SimpleNamespace
import lvp_logger

records = []
class _Keep(logging.Handler):
    def emit(self, record):
        records.append((record.levelname, record.getMessage()))
for handler in list(lvp_logger.logger.handlers):
    lvp_logger.logger.removeHandler(handler)
lvp_logger.logger.addHandler(_Keep())

shown = []
# a stand-in by design: the Windows console, which no test host but Windows has
ctypes.windll = SimpleNamespace(
    kernel32=SimpleNamespace(GetConsoleWindow=lambda: {console}),
    user32=SimpleNamespace(ShowWindow=lambda window, state: shown.append([window, state])),
)
sys.platform = 'win32'
lvp_logger.minimize_logger_window()
print(json.dumps({{'records': records, 'shown': shown}}))
"""


def _minimize(console: int) -> dict:
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(_CHILD.format(console=console))],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_a_windowed_build_logs_nothing_about_its_missing_console():
    seen = _minimize(console=0)

    assert seen == {'records': [], 'shown': []}


def test_a_console_is_minimized_and_says_so_once():
    seen = _minimize(console=7)

    assert seen['shown'] == [[7, 6]]
    assert seen['records'] == [['INFO', '[Logger  ] Console window minimized']]
