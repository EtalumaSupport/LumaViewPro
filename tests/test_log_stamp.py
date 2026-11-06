# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Importing the log stamp does nothing.

The FX2 driver step runs as LocalSystem at uninstall and the profiler runs
under sudo; both take the stamp from ``_lumascope.modules.log_stamp`` and never import
``lvp_logger``, which loads the settings layer and whose start creates the
log folder and attaches its handlers. A stamp module that did either would
write root-owned files into a user's data folder, or log into the live log
it reads.
"""

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

_CHILD = """
import json, logging, os, sys
sys.path.insert(0, {src!r})
before = sorted(os.listdir('.'))
import _lumascope.modules.log_stamp
print(json.dumps([
    sorted(os.listdir('.')) == before,
    logging.getLogger().handlers == [],
    sorted(name for name in sys.modules if name == '_lumascope.lvp_logger' or name.startswith('kivy')),
]))
"""


def test_importing_the_stamp_creates_no_file_and_attaches_no_handler(tmp_path):
    done = subprocess.run(
        [sys.executable, '-I', '-c', _CHILD.format(src=str(REPO / 'src'))],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tmp_path),
    )
    assert done.returncode == 0, done.stderr

    no_new_file, no_handler, loaded = json.loads(done.stdout.strip().splitlines()[-1])

    assert no_new_file
    assert no_handler
    assert loaded == []
    assert list(tmp_path.iterdir()) == []
