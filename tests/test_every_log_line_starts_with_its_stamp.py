# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every log LumaViewPro keeps starts each record with one full date and time.

The logs once carried five timestamp forms, and serial.log, camera.log and
api.log carried no date at all: a support bundle's trace lines could only be
placed on a day by counting midnight rollovers. Every record now begins
``2026-10-29 17:10:57.320``, so the lines of any two logs sort together.

The trace logs also name the thread that wrote each line: the instrument's
commands are serialized onto lane workers, and which thread sent a command is
what separates a lane's own write from one made beside it.

Logging starts once per process, and the suite's has declined its files
(``tests/conftest.py``), so the start runs in a child process with its logs
under a temporary data root.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from _lumascope.lvp_logger import LOG_CAPS

REPO = pathlib.Path(__file__).resolve().parents[1]

# Spelled here, not read from _lumascope.modules.log_stamp, so a wrong date format there
# fails this test instead of agreeing with itself.
_STAMP = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) ')

_TRACE_LOGS = {'serial.log', 'camera.log', 'api.log'}

_CHILD = """
import json, logging, os, sys, threading
sys.frozen = True
sys._MEIPASS = {repo!r}
sys.executable = {exe!r}
import _lumascope.lvp_logger
_lumascope.lvp_logger.start_program_logging()
from logging.handlers import RotatingFileHandler

handlers = [v for v in vars(_lumascope.lvp_logger).values() if isinstance(v, RotatingFileHandler)]

def write():
    for handler in handlers:
        record = logging.LogRecord('LVP.probe', logging.WARNING, 'probe.py', 1, 'probe line', None, None)
        handler.emit(record)
        handler.flush()

worker = threading.Thread(target=write, name='IO_WORKER')
worker.start()
worker.join(30)
lines = {{}}
for handler in handlers:
    with open(handler.baseFilename, encoding='utf-8') as log:
        lines[os.path.basename(handler.baseFilename)] = [l for l in log.read().splitlines() if 'probe line' in l]
print(json.dumps(lines))
"""


@pytest.fixture(scope='module')
def probe_lines(tmp_path_factory):
    """The line each of lvp_logger's rotating logs wrote for one probe record."""
    data_root = tmp_path_factory.mktemp('data_root')
    # A frozen process with no installer marker keeps its data beside its
    # executable, so the logs land under data_root, which carries the shipped
    # settings.json an installation does: the logger reads debug_mode from it.
    (data_root / 'data').mkdir()
    shutil.copy(REPO / 'data' / 'settings.json', data_root / 'data' / 'settings.json')
    code = _CHILD.format(exe=str(data_root / 'LumaViewPro.exe'), repo=str(REPO))
    done = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=60, cwd=str(REPO)
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_every_log_starts_its_record_with_the_full_date_and_time(probe_lines):
    now = datetime.datetime.now()
    wrong = {}
    for name, lines in probe_lines.items():
        match = _STAMP.match(lines[0]) if len(lines) == 1 else None
        stamped = match and datetime.datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S.%f')
        if not stamped or abs((now - stamped).total_seconds()) > 60:
            wrong[name] = lines

    assert set(probe_lines) == set(LOG_CAPS)
    assert not wrong, f'a log whose record does not start with the date and time: {wrong}'


def test_a_trace_line_names_its_thread(probe_lines):
    unnamed = {
        name: lines[0]
        for name, lines in probe_lines.items()
        if name in _TRACE_LOGS and '[IO_WORKER] probe line' not in lines[0]
    }

    assert set(probe_lines) >= _TRACE_LOGS
    assert not unnamed, unnamed
