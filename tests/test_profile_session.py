# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the profiling harness (tools/profiling/profile_session.py).

Regression: a bare ``py-spy`` invocation fails under sudo (needed to attach on
macOS) because sudo sanitizes PATH. The harness must resolve py-spy's absolute
path from the interpreter's own bin dir so the same command works elevated.

The build an artifact names is the profiled process's, read from its own launch
banner in the log it holds open, never the profiler's checkout.
"""

import contextlib
import json
import pathlib
import subprocess
import sys
import time

import pytest

from tools.profiling import profile_session
from tools.profiling.profile_session import _pyspy_path


class TestPyspyPath:
    def test_resolves_next_to_interpreter(self, tmp_path):
        # py-spy is installed alongside the interpreter; that path wins even when
        # PATH is empty (the sudo case).
        (tmp_path / 'py-spy').write_text('')
        assert _pyspy_path(interpreter_dir=tmp_path) == str(tmp_path / 'py-spy')

    def test_falls_back_to_path(self, tmp_path, monkeypatch):
        # Not next to the interpreter -> use PATH.
        monkeypatch.setattr(
            'tools.profiling.profile_session.shutil.which', lambda name: '/usr/local/bin/py-spy'
        )
        assert _pyspy_path(interpreter_dir=tmp_path) == '/usr/local/bin/py-spy'

    def test_raises_when_missing_everywhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr('tools.profiling.profile_session.shutil.which', lambda name: None)
        with pytest.raises(FileNotFoundError, match='py-spy not found'):
            _pyspy_path(interpreter_dir=tmp_path)


# --- the build the profiler names is the profiled process's own --------------

REPO = pathlib.Path(__file__).resolve().parents[1]

# A child LumaViewPro process: it imports the real lvp_logger from a folder of
# links, so its log is that folder's (as the simulator recipe's launch is), then
# does what its mode says and waits, holding the log open, until stdin closes.
_CHILD = """
import sys
sys.path.insert(0, sys.argv[1])
import lvp_logger
mode = sys.argv[2]
if mode in ('banner', 'rotated'):
    lvp_logger.log_environment_banner(sys.argv[1], lvp_logger.version, [])
if mode == 'rotated':
    lvp_logger.file_handler.doRollover()
print('ready', flush=True)
sys.stdin.readline()
"""


def _linked_folder(tmp_path):
    folder = tmp_path / 'launch'
    folder.mkdir()
    for item in ('lvp_logger.py', 'modules', 'lib', 'version.txt'):
        (folder / item).symlink_to(REPO / item)
    return folder


@contextlib.contextmanager
def _lumaviewpro(folder, mode):
    child = subprocess.Popen(
        [sys.executable, '-c', _CHILD, str(folder), mode],
        cwd=str(folder),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == 'ready'
        yield child.pid
    finally:
        child.stdin.close()
        child.wait(timeout=30)


def _main_log(folder):
    return folder / 'logs' / 'LVP_Log' / 'lumaviewpro.log'


def _foreign_banner(pid, when):
    stamp = time.strftime('%m/%d/%Y %H:%M:%S.000', time.localtime(when))
    lines = [('Version', 'other'), ('PID', str(pid)), ('Git', 'other')]
    return ''.join(
        f'[INFO] [MainThread] {stamp} - lvp_logger.py - [LVP Main  ] {label}: {value}\n'
        for label, value in lines
    )


def test_a_profile_names_the_build_its_process_launched_with(tmp_path, monkeypatch):
    folder = _linked_folder(tmp_path)
    monkeypatch.setattr(
        'tools.profiling.profile_session._run_pyspy',
        lambda pid, duration_s, rate_hz, raw_path: raw_path.write_text(''),
    )

    with _lumaviewpro(folder, 'banner') as pid:
        artifact = json.loads(
            profile_session.profile(pid, 1, 50, 'probe', tmp_path / 'out', None).read_text()
        )

    build = artifact['manifest']['build']
    assert set(build) == {'version', 'built', 'commit_guid', 'build_id', 'runtime', 'pid', 'git'}
    assert build['pid'] == str(pid)
    assert build['version'] == (REPO / 'version.txt').read_text().splitlines()[0].strip()
    assert artifact['manifest']['build_identity_source'] == str(_main_log(folder).resolve())


def test_a_banner_rotated_into_a_backup_is_found_there(tmp_path):
    folder = _linked_folder(tmp_path)

    with _lumaviewpro(folder, 'rotated') as pid:
        found = profile_session._build_of(pid)

    assert found['build']['pid'] == str(pid)
    assert pathlib.Path(found['build_identity_source']).name == 'lumaviewpro.1.log'


def test_a_process_that_wrote_no_banner_has_none_though_another_did(tmp_path):
    folder = _linked_folder(tmp_path)

    with _lumaviewpro(folder, 'headless') as pid:
        with _main_log(folder).open('a') as log:
            log.write(_foreign_banner(1, time.time()))
        found = profile_session._build_of(pid)

    assert found['build'] is None
    assert found['build_identity_source'].startswith(f'no banner for process {pid}')


def test_a_banner_older_than_the_process_is_not_its_own(tmp_path):
    folder = _linked_folder(tmp_path)

    with _lumaviewpro(folder, 'headless') as pid:
        with _main_log(folder).open('a') as log:
            log.write(_foreign_banner(pid, time.time() - 3600))
        found = profile_session._build_of(pid)

    assert found['build'] is None


def test_a_process_holding_no_log_says_so():
    with subprocess.Popen(
        [sys.executable, '-c', 'import sys; print("ready", flush=True); sys.stdin.readline()'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        child.stdout.readline()
        found = profile_session._build_of(child.pid)
        child.stdin.close()

    assert found == {
        'build': None,
        'build_identity_source': f'no LumaViewPro log open in process {child.pid}',
    }
