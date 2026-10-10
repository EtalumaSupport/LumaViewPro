# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every process that logs says first what it is.

The GUI, the REST server, a headless script and a test runner all write to
one log, and only the GUI writes the startup banner, so a support bundle
could not tell what any other process was. Importing ``lvp_logger`` now
writes one line before anything else: version, runtime, the folder the
process was launched from, its data root, and its PID.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

_CHILD = """
import os, sys
sys.path.insert(0, os.getcwd())
import lvp_logger
print(os.getpid())
"""


def test_a_process_first_logs_its_version_runtime_roots_and_pid(tmp_path):
    folder = tmp_path / 'launch'
    folder.mkdir()
    for item in ('lvp_logger.py', 'modules', 'lib', 'version.txt'):
        (folder / item).symlink_to(REPO / item)

    pid = subprocess.run(
        [sys.executable, '-c', _CHILD],
        cwd=str(folder),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.split()[-1]

    lines = (folder / 'logs' / 'LVP_Log' / 'lumaviewpro.log').read_text().splitlines()
    assert lines, 'importing the logger wrote nothing that names the process'
    first = lines[0]
    version = (REPO / 'version.txt').read_text(encoding='utf-8-sig').splitlines()[0].strip()
    assert first.endswith(
        f'[Process   ] LumaViewPro {version}, source, from {folder}, data in {folder}, PID {pid}'
    ), first
