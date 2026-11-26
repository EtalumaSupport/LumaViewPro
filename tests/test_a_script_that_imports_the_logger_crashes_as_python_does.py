# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A script that imports the SDK crashes as any Python program does; the application chooses its own crash policy.

Importing ``lvp_logger`` -- which every SDK module does -- installs no hook,
so a script's uncaught exception, on its main thread or one it started,
prints Python's own traceback on stderr. A program that owns its process
(the GUI, the REST server, a tool, a Script) takes LumaViewPro's crash policy
with ``start_program_logging()``: an uncaught exception logged as a crash, an
exception in an object's cleanup logged as an error, and a fatal error in
native code leaving every thread's stack in ``fatal_errors.log``.

Each runs in a child Python: what a script's process does at a crash is the
process's own, and the suite's process is the suite's (``tests/conftest.py``).
A program's child keeps its logs under ``tmp_path``, not this checkout's.
"""

from __future__ import annotations

import shutil
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
        import _lumascope.lvp_logger
        raise ValueError('kettle boiled dry')
        """
    )

    assert result.returncode == 1
    assert 'ValueError: kettle boiled dry' in result.stderr


def test_a_threads_uncaught_exception_prints_its_traceback():
    result = _child(
        """
        import threading
        import _lumascope.lvp_logger

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


def _program(
    tmp_path: Path, body: str, before_logging: str = ''
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run ``body`` after ``start_program_logging()``, its log folder under ``tmp_path``.

    A frozen process with no installer marker keeps its data beside its
    executable, so pointing the executable into ``tmp_path`` moves the log
    folder there; the root carries the shipped ``settings.json``, which the
    logger reads ``debug_mode`` from. ``before_logging`` runs before the
    logging starts, so a handler it installs is the one the logger's own
    handlers find already there.
    """
    (tmp_path / 'data').mkdir()
    shutil.copy(REPO / 'data' / 'settings.json', tmp_path / 'data' / 'settings.json')
    prefix = f"""
        import sys
        sys.frozen = True
        sys._MEIPASS = {str(REPO)!r}
        sys.executable = {str(tmp_path / 'LumaViewPro.exe')!r}
        {textwrap.indent(textwrap.dedent(before_logging), ' ' * 8).strip()}
        import _lumascope.lvp_logger
        _lumascope.lvp_logger.start_program_logging()
        """
    return _child(textwrap.dedent(prefix) + textwrap.dedent(body)), tmp_path / 'logs' / 'LVP_Log'


def test_a_program_that_stops_on_a_crash_says_why_on_its_terminal_and_logs_the_traceback(tmp_path):
    result, log_dir = _program(
        tmp_path,
        """
        def boil():
            raise ValueError('kettle boiled dry')

        boil()
        """,
    )

    assert result.returncode == 1
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert lines, 'the terminal was told nothing'
    assert lines[-1] == 'ValueError: kettle boiled dry'
    assert 'Traceback' not in result.stderr, 'the traceback is the log record, not the terminal'
    log = (log_dir / 'lumaviewpro.log').read_text(encoding='utf-8')
    assert 'CRASH - Uncaught Exception' in log
    assert 'in boil' in log


def test_an_error_in_an_objects_cleanup_is_logged_with_its_traceback(tmp_path):
    # The shape SN 12078's runs died after: a Tk object's __del__ raised where
    # Python can only ignore it, and the record reached the console alone.
    result, log_dir = _program(
        tmp_path,
        """
        class Kettle:
            def __del__(self):
                raise ValueError('kettle boiled dry')

        kettle = Kettle()
        del kettle
        """,
    )

    assert result.returncode == 0, result.stderr
    errors = (log_dir / 'lumaviewpro_errors.log').read_text(encoding='utf-8')
    assert 'Exception ignored' in errors and 'Kettle.__del__' in errors
    assert 'Traceback' in errors and 'ValueError: kettle boiled dry' in errors


def test_a_fatal_error_in_native_code_leaves_every_threads_stack_under_its_process_line(tmp_path):
    # A fatal abort ends the process before any log record can be written:
    # faulthandler's dump is the only record. The SIGABRT stands in for a
    # native library's abort (Tcl's "async handler deleted by the wrong
    # thread"). faulthandler dumps, then hands the signal to the handler it
    # found at enable, which here exits the child itself: a child killed by
    # the signal is a real crash to the host's OS (a macOS crash report and
    # dialog on every suite run), and the dump is written before either.
    result, log_dir = _program(
        tmp_path,
        """
        import os, signal, threading
        print(os.getpid(), flush=True)
        ready, never = threading.Event(), threading.Event()

        def simmer():
            ready.set()
            never.wait()

        threading.Thread(target=simmer, name='kettle', daemon=True).start()
        ready.wait()
        signal.raise_signal(signal.SIGABRT)
        """,
        before_logging="""
        import os, signal
        signal.signal(signal.SIGABRT, lambda signum, frame: os._exit(134))
        """,
    )

    assert result.returncode != 0
    pid = result.stdout.split()[0]
    dump = (log_dir / 'fatal_errors.log').read_text(encoding='utf-8')
    started = [line for line in dump.splitlines() if line.endswith(f'process {pid} started')]
    assert len(started) == 1, dump
    after = dump[dump.index(started[0]) :]
    assert 'Fatal Python error' in after
    assert 'in simmer' in after, "the other thread's stack is not in the dump"
