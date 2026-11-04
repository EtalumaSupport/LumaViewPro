# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A process that builds a session logs its environment banner, once.

``ScopeSession.create`` is the one bring-up every host makes, the GUI, the
REST server, a script and a probe alike, and it writes the banner, so each
one's log names its host, build and camera SDKs. The banner describes the
process, so a second session in it writes no second banner. The build's own
files are read from the script root, never from the data folder the
session is built on, which on an installed build has no version.txt, and a
host without the UI is not made to import Kivy for its version.

Run in a subprocess: the suite replaces ``lvp_logger`` with a mock, and the
real module is the banner's writer.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

_TWO_SESSIONS = """
import json, logging, sys
import lvp_logger
from modules.app_environment import camera_sdk_probe
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

lines = []
handler = logging.Handler()
handler.emit = lambda record: lines.append(record.getMessage())
lvp_logger.logger.addHandler(handler)
for _ in range(2):
    session = ScopeSession.create(complete_settings(), source_path=sys.argv[1], simulate=True)
    session.shutdown()
print(json.dumps({'lines': lines, 'sdk': camera_sdk_probe(), 'kivy': 'kivy' in sys.modules}))
"""


def test_two_sessions_in_one_process_log_one_banner_from_the_script_root(tmp_path):
    # A data folder as an installed build keeps one: the installation's
    # data, and no version.txt beside it.
    shutil.copytree(REPO / 'data', tmp_path / 'data')
    done = subprocess.run(
        [sys.executable, '-c', _TWO_SESSIONS, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
    )
    assert done.returncode == 0, done.stderr
    heard = json.loads(done.stdout.strip().splitlines()[-1])
    lines = heard['lines']

    pid_lines = [line for line in lines if line.startswith('[LVP Main  ] PID:')]
    assert len(pid_lines) == 1, f'two sessions wrote {len(pid_lines)} banners'
    built = (REPO / 'version.txt').read_text(encoding='utf-8-sig').splitlines()[1].strip()
    assert f'[LVP Main  ] Built:     {built}' in lines, (
        "the banner did not read the build's version.txt from the script root"
    )
    missing = [sdk for sdk in heard['sdk'] if f'[LVP Main  ] {sdk}' not in lines]
    assert not missing, f'the banner left out camera SDK lines: {missing}'
    # A host without the UI writes the banner too, and the banner must not
    # load the UI for it.
    assert not heard['kivy'], 'writing the banner imported Kivy into a headless process'
    assert '[LVP Main  ] Kivy: not loaded by this process' in lines
