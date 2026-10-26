# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every host writes api.log and autofocus.log from the first call of bring-up.

The two files were attached by the GUI alone, after its session was built and
its plugins mounted, and only in engineering mode or a beta: a launch's
bring-up never reached api.log, and the REST server and a script wrote
neither file at all. They are attached when the API is imported, as
camera.log is, so a host has nothing to call and no point after which they
start. Being always on, autofocus.log rotates into one file, so it cannot
pile up.

The check runs in a child Python: the suite replaces ``lvp_logger`` with a
stand-in (``tests/conftest.py``), so only a fresh interpreter imports the
real one. The child imports the API the way the REST server and a script
do, and nothing else.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_CHILD = """
import json, logging
from modules.scope_session import ScopeSession
import lvp_logger

def files(name):
    return sorted(
        h.baseFilename for h in logging.getLogger(name).handlers
        if isinstance(h, logging.FileHandler)
    )

print(json.dumps({
    'api': files('LVP.api'),
    'autofocus': files('LVP.autofocus'),
    'autofocus_rotations': [
        h.backupCount for h in logging.getLogger('LVP.autofocus').handlers
        if getattr(h, 'baseFilename', None) == lvp_logger.AUTOFOCUS_LOG_FILE
    ],
    'api_log': lvp_logger.API_LOG_FILE,
    'autofocus_log': lvp_logger.AUTOFOCUS_LOG_FILE,
}))
"""


def test_importing_the_api_attaches_api_log_and_autofocus_log():
    result = subprocess.run(
        [sys.executable, '-c', _CHILD],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout.splitlines()[-1])

    assert seen['api_log'] in seen['api']
    assert seen['autofocus_log'] in seen['autofocus']
    # Always on, so it rotates into one file and no more (Eric, 2026-10-26).
    assert seen['autofocus_rotations'] == [1]
