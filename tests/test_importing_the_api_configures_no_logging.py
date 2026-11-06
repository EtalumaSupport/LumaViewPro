# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Importing the API configures no logging; a process's logging starts once, at one call.

``import lumascope`` attached LumaViewPro's files to the root logger at
DEBUG, so an integrator's ``logging.basicConfig()`` did nothing and their
own records were written to ``lumaviewpro.log``. Importing any part of the
API now opens no log file, attaches no handler, sets no level and writes no
record. A session's first call (``load_user_settings`` or ``create``)
starts the log folder on LumaViewPro's own loggers, so a script leaves the
support bundle and keeps its root logger; ``start_logging(log_files=False)``
first turns the files off. A LumaViewPro program owns its process and
starts its logging on root itself. A start asking for anything other than
what the process has is refused, and changes nothing.

Each case is a fresh interpreter: logging is process-wide, and the suite's
process has started its own, with no files (``tests/conftest.py``).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

# Every record any logger attempts is counted, whatever its level: a counter
# past the level check never sees a DEBUG line written before anything sets
# a level, and that is how the registry's import-time lines went unseen.
_IMPORT = """
import json, logging, os, sys
sys.path.insert(0, 'src')
seen = []
_handle, _enabled = logging.Logger.handle, logging.Logger.isEnabledFor
def _counting(self, record):
    seen.append(record.name)
    return _handle(self, record)
logging.Logger.handle = _counting
logging.Logger.isEnabledFor = lambda self, level: True
root_level = logging.getLogger().level
{importing}
logging.Logger.handle, logging.Logger.isEnabledFor = _handle, _enabled
from _lumascope.modules.path_utils import get_log_dir
state = {{
    'ours': [n for n in seen if n.split('.')[0] in ('LVP', 'lvp_logger')],
    'root_handlers': len(logging.getLogger().handlers),
    'root_level_kept': logging.getLogger().level == root_level,
    'log_folder_made': get_log_dir().exists(),
}}
heard = []
logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].emit = lambda record: heard.append(record.getMessage())
logging.getLogger('myapp').info('my record')
state['heard'] = heard
print(json.dumps(state))
"""


def _run(code, cwd):
    done = subprocess.run(
        [sys.executable, '-c', code], cwd=str(cwd), capture_output=True, text=True, timeout=180
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_importing_the_api_writes_no_record_and_leaves_the_root_logger_alone(tmp_path):
    for case, importing in enumerate(
        ('import lumascope', 'from _lumascope.modules.image_mode import ImageMode')
    ):
        # An installation's folder, so its data root, where the log folder
        # would be made, starts with none.
        folder = _installation(tmp_path / str(case))
        state = _run(_IMPORT.format(importing=importing), folder)

        assert state['ours'] == [], f'{importing} wrote {state["ours"]}'
        assert state['root_handlers'] == 0, f'{importing} attached a handler to root'
        assert state['root_level_kept'], f'{importing} changed the root level'
        assert not state['log_folder_made'], f'{importing} made the log folder'
        assert state['heard'] == ['my record'], f'after {importing} basicConfig() did not work'


def _installation(tmp_path):
    """A folder an installation launches from: its data copied, so a write stays here."""
    folder = tmp_path / 'launch'
    folder.mkdir(parents=True)
    for item in ('src', 'version.txt'):
        (folder / item).symlink_to(REPO / item)
    shutil.copytree(REPO / 'data', folder / 'data')
    # A stored file a release behind: the load merges the key it lacks, and
    # says so in the log.
    stored = json.loads((REPO / 'data' / 'settings.json').read_text(encoding='utf-8'))
    stored.pop('debug_mode')
    (folder / 'data' / 'current.json').write_text(json.dumps(stored), encoding='utf-8')
    return folder


_SCRIPT = """
import json, logging, sys
sys.path.insert(0, 'src')
heard = []
logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].emit = lambda record: heard.append(record.getMessage())
import lumascope
import _lumascope.lvp_logger
from lumascope import ScopeSession
{before}
ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True).shutdown()
from _lumascope.drivers import firmware_updater
from _lumascope.modules import tech_support_report
firmware_updater.logger.info('a firmware updater line')
tech_support_report.logger.info('a support report line')
logging.getLogger('myapp').info('my record after the session')
for owner in (logging.getLogger('LVP'), logging.getLogger('lvp_logger')):
    for handler in owner.handlers:
        handler.flush()
from _lumascope.modules.path_utils import get_log_dir
print(json.dumps({{
    'heard': heard,
    'root_handlers': len(logging.getLogger().handlers),
    'log_dir': str(get_log_dir()),
}}))
"""


def test_a_scripts_first_session_call_starts_the_bundle_on_lumascopes_own_loggers(tmp_path):
    folder = _installation(tmp_path)

    state = _run(_SCRIPT.format(before=''), folder)

    log = (pathlib.Path(state['log_dir']) / 'lumaviewpro.log').read_text(encoding='utf-8')
    lines = log.splitlines()
    assert '[Process   ] LumaViewPro ' in lines[0], lines[0]
    assert any('[Settings ] Merged ' in line and "'debug_mode'" in line for line in lines), (
        "the settings load's lines are not in the bundle"
    )
    assert any('[LVP Main  ] PID:' in line for line in lines), 'the banner is not in the bundle'
    # Root, and so basicConfig(), stays the script's.
    assert state['root_handlers'] == 1
    assert 'my record after the session' in state['heard']
    assert 'my record after the session' not in log
    # LumaViewPro's records also reach the script, as any library's do, and
    # each is written to the bundle once.
    assert 'a support report line' in state['heard']
    assert log.count('a firmware updater line') == 1
    assert log.count('a support report line') == 1


def test_a_script_that_declines_the_files_opens_none_and_still_hears_lumascope(tmp_path):
    folder = _installation(tmp_path)

    state = _run(
        _SCRIPT.format(before='_lumascope.lvp_logger.start_logging(log_files=False)'), folder
    )

    assert not pathlib.Path(state['log_dir']).exists(), 'a declined process made the log folder'
    assert 'a support report line' in state['heard']
    debug_line = [m for m in state['heard'] if m.startswith('[LVP Main  ] Debug:')]
    assert debug_line == [
        '[LVP Main  ] Debug:     not read -- this process started no log files '
        '(start_logging(log_files=False)); each session states its own level'
    ]


_STARTS = """
import json, logging, os, sys
sys.path.insert(0, 'src')
from _lumascope import lvp_logger
from _lumascope.modules.exceptions import LoggingStartRefusedError
from _lumascope.modules.path_utils import get_log_dir
refused = []
for start in {starts!r}:
    try:
        getattr(lvp_logger, start[0])(**start[1])
    except LoggingStartRefusedError as e:
        refused.append([e.reason, e.has, e.asked])
print(json.dumps({{
    'refused': refused,
    'started': lvp_logger.logging_started(),
    'root_handlers': [type(h).__name__ for h in logging.getLogger().handlers],
    'files': sorted(os.listdir(get_log_dir())) if get_log_dir().exists() else None,
}}))
"""


def _starts(tmp_path, *starts):
    return _run(_STARTS.format(starts=list(starts)), _installation(tmp_path))


_PROGRAM = ('start_program_logging', {})
_SESSION = ('start_session_logging', {})
_FILES = ('start_logging', {})
_DECLINE = ('start_logging', {'log_files': False})


def test_a_program_takes_root_with_the_files_the_gui_opens_and_a_second_start_adds_none(tmp_path):
    state = _starts(tmp_path, _PROGRAM, _PROGRAM, _SESSION)

    assert state['refused'] == []
    assert state['started'] == 'root'
    assert state['root_handlers'] == ['RotatingFileHandler', 'RotatingFileHandler']
    # The files opened at the start; the REST, protocol, autofocus and api
    # logs open at their first write.
    assert state['files'] == [
        'camera.log',
        'gui_interactions.log',
        'lumaviewpro.log',
        'lumaviewpro_errors.log',
        'metrics.log',
        'serial.log',
    ]


def test_a_start_asking_for_another_logging_than_the_process_has_is_refused_and_changes_nothing(
    tmp_path,
):
    files_after_decline = _starts(tmp_path / 'a', _DECLINE, _FILES, _PROGRAM, _SESSION)
    assert files_after_decline['refused'] == [
        ['logging_started_otherwise', 'declined', 'lumascope'],
        ['logging_started_otherwise', 'declined', 'root'],
    ]
    assert files_after_decline['started'] == 'declined'
    assert files_after_decline['files'] is None, 'a refused start opened files'

    after_a_session = _starts(tmp_path / 'b', _SESSION, _DECLINE, _PROGRAM, _FILES)
    assert after_a_session['refused'] == [
        ['logging_started_otherwise', 'lumascope', 'declined'],
        ['logging_started_otherwise', 'lumascope', 'root'],
    ]
    assert after_a_session['started'] == 'lumascope'
    assert after_a_session['root_handlers'] == [], 'a refused program start took root'
    assert 'lumaviewpro.log' in after_a_session['files'], 'a refused decline closed the files'
