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
import _lumascope.lvp_logger
from _lumascope.modules.app_environment import camera_sdk_probe
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

lines = []
handler = logging.Handler()
handler.emit = lambda record: lines.append(record.getMessage())
_lumascope.lvp_logger.logger.addHandler(handler)
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


# Every level is open before the drivers import: a record a level check drops
# is invisible to a handler, and that is how the registry's import-time lines
# went uncounted.
_DRIVERS = """
import json, logging, sys
import _lumascope.lvp_logger
for name in ('', 'LVP', 'lvp_logger'):
    logging.getLogger(name).setLevel(logging.DEBUG)
heard = []
handler = logging.Handler()
handler.emit = lambda record: heard.append((record.levelno, record.getMessage()))
logging.getLogger().addHandler(handler)
import _lumascope.modules.lumascope_api._lumascope
from _lumascope.drivers import registry
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
at_import = list(heard)
ScopeSession.create(complete_settings(), source_path=sys.argv[1], simulate=True).shutdown()
fx2 = sys.modules.get('_lumascope.drivers.fx2driver')
print(json.dumps({
    'at_import': at_import,
    'banner': heard[len(at_import):],
    'registered': {
        'motor': registry.motor_registry.registered_names(),
        'led': registry.led_registry.registered_names(),
        'camera': registry.camera_registry.registered_names(),
    },
    'fx2_gate_open': None if fx2 is None else fx2._FX2_GATE.available,
}))
"""


def _is_driver_line(message):
    return message.startswith(('[registry]', '[FX2 Driver]')) or message.startswith(
        ('[SCOPE API ] FX2', '[SCOPE API ] IDS')
    )


def test_the_banner_states_what_the_driver_imports_found_and_the_import_writes_none(tmp_path):
    # The drivers decide their availability as they import, where nothing has
    # started logging; the banner is where a support bundle gets it.
    shutil.copytree(REPO / 'data', tmp_path / 'data')
    done = subprocess.run(
        [sys.executable, '-c', _DRIVERS, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
    )
    assert done.returncode == 0, done.stderr
    heard = json.loads(done.stdout.strip().splitlines()[-1])

    written_at_import = [m for _level, m in heard['at_import'] if _is_driver_line(m)]
    assert not written_at_import, f'the driver imports wrote: {written_at_import}'

    banner = [(level, m) for level, m in heard['banner'] if _is_driver_line(m)]
    for kind, names in heard['registered'].items():
        line = [m for level, m in banner if m.startswith(f'[registry] {kind}: registered ')]
        assert len(line) == 1, f'the banner states the {kind} registry {len(line)} times'
        missing = [name for name in names if f'{name!r} -> ' not in line[0]]
        assert not missing, f'the {kind} line leaves out {missing}: {line[0]}'
    if heard['fx2_gate_open'] is False:
        declined = [m for _level, m in banner if "declined 'fx2' (" in m]
        assert len(declined) == 2, f'the camera and LED registries each decline the FX2: {banner}'
    fx2_lines = [m for _level, m in banner if m.startswith(('[FX2 Driver]', '[SCOPE API ] FX2'))]
    assert fx2_lines, 'the banner does not state the FX2 gate'
    if 'ids' not in heard['registered']['camera']:
        assert [m for _level, m in banner if m.startswith('[SCOPE API ] IDS')], (
            'the IDS driver did not register and the banner does not say why'
        )
