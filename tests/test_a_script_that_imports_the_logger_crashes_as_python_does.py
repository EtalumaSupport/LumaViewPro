# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A script that imports the SDK crashes as any Python program does; the application chooses its own crash policy.

Importing ``lvp_logger`` -- which every SDK module does -- installs no hook,
so a script's uncaught exception, on its main thread or one it started,
prints Python's own traceback on stderr. ``install_crash_hooks()`` is how an
application (LumaViewPro, the REST server) takes LumaViewPro's crash policy:
the crash recorded in its log.

Each runs in a child Python: the suite replaces ``lvp_logger`` with a stand-in
(``tests/conftest.py``), so only a fresh interpreter imports the real one.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _child(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(body)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_an_uncaught_exception_prints_its_traceback_and_exits_1():
    result = _child(
        """
        import lvp_logger
        raise ValueError('kettle boiled dry')
        """
    )

    assert result.returncode == 1
    assert 'ValueError: kettle boiled dry' in result.stderr


def test_a_threads_uncaught_exception_prints_its_traceback():
    result = _child(
        """
        import threading
        import lvp_logger

        def boil():
            raise ValueError('kettle boiled dry')

        worker = threading.Thread(target=boil, name='kettle')
        worker.start()
        worker.join()
        """
    )

    assert result.returncode == 0
    assert 'Exception in thread kettle' in result.stderr
    assert 'ValueError: kettle boiled dry' in result.stderr


def test_an_application_takes_the_crash_policy_by_installing_it():
    # Read as which hooks are set, not by raising through them: a raise
    # would write its crash record into this checkout's own log.
    result = _child(
        """
        import sys, threading
        import lvp_logger
        lvp_logger.install_crash_hooks()
        print(sys.excepthook is lvp_logger.custom_except_hook,
              threading.excepthook is lvp_logger._thread_except_hook)
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['True', 'True']
