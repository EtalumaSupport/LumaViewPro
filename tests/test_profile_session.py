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
from tools.profiling.aggregate import parse_folded
from tools.profiling.profile_session import (
    AUSTIN,
    NO_PYTHON_FRAME,
    PY_SPY,
    _attach_hint,
    _recorded,
    _refused_attach,
    _sampler,
    _sampler_command,
    _tool_path,
    austin_to_folded,
)


class TestPyspyPath:
    def test_resolves_next_to_interpreter(self, tmp_path):
        # py-spy is installed alongside the interpreter; that path wins even when
        # PATH is empty (the sudo case).
        (tmp_path / 'py-spy').write_text('')
        assert _tool_path('py-spy', interpreter_dir=tmp_path) == str(tmp_path / 'py-spy')

    def test_falls_back_to_path(self, tmp_path, monkeypatch):
        # Not next to the interpreter -> use PATH.
        monkeypatch.setattr(
            'tools.profiling.profile_session.shutil.which', lambda name: '/usr/local/bin/py-spy'
        )
        assert _tool_path('py-spy', interpreter_dir=tmp_path) == '/usr/local/bin/py-spy'

    def test_raises_when_missing_everywhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr('tools.profiling.profile_session.shutil.which', lambda name: None)
        with pytest.raises(FileNotFoundError, match='py-spy not found'):
            _tool_path('py-spy', interpreter_dir=tmp_path)


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


def _evidence(pid, folder):
    """What the profiler had to go on, read while the process lives, for a miss's message.

    These tests failed once in a push suite (both on one xdist worker) and in
    no run since, alone or under load; a miss names the process's start, every
    banner line on disk and what the profiler parsed from it.
    """
    import psutil

    lines = [f'process {pid} started {psutil.Process(pid).create_time():.3f}']
    for path in sorted(_main_log(folder).parent.glob('lumaviewpro*')):
        lines.append(f'{path.name}: {path.stat().st_size} bytes')
        lines.append(
            f'  parsed: {[(when, b.get("pid")) for when, b in profile_session._banners(path)]}'
        )
        text = path.read_text(errors='replace').splitlines()
        lines += [f'  {line}' for line in text if '[LVP Main  ]' in line][:10]
    return '\n'.join(lines)


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
        'tools.profiling.profile_session._run_sampler',
        lambda sampler, pid, duration_s, rate_hz, raw_path: raw_path.write_text(''),
    )

    with _lumaviewpro(folder, 'banner') as pid:
        artifact = json.loads(
            profile_session.profile(pid, 1, 50, 'probe', tmp_path / 'out', None).read_text()
        )
        evidence = _evidence(pid, folder)

    assert artifact['manifest']['sampler'] == _sampler()
    build = artifact['manifest']['build']
    assert build is not None, f'{artifact["manifest"]["build_identity_source"]}\n{evidence}'
    assert set(build) == {'version', 'built', 'commit_guid', 'build_id', 'runtime', 'pid', 'git'}
    assert build['pid'] == str(pid)
    assert build['version'] == (REPO / 'version.txt').read_text().splitlines()[0].strip()
    assert artifact['manifest']['build_identity_source'] == str(_main_log(folder).resolve())


def test_a_banner_rotated_into_a_backup_is_found_there(tmp_path):
    folder = _linked_folder(tmp_path)

    with _lumaviewpro(folder, 'rotated') as pid:
        found = profile_session._build_of(pid)
        evidence = _evidence(pid, folder)

    assert found['build'] is not None, f'{found["build_identity_source"]}\n{evidence}'
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


# --- the sampler, by host ------------------------------------------------------


def test_the_mac_samples_with_austin_on_cpu_only_and_every_other_host_with_py_spy():
    assert _sampler('darwin') == AUSTIN
    assert _sampler('win32') == PY_SPY
    assert _sampler('linux') == PY_SPY


def test_austin_records_on_cpu_stacks_at_the_requested_rate(monkeypatch):
    monkeypatch.setattr('tools.profiling.profile_session._tool_path', lambda name: name)

    command = _sampler_command(AUSTIN, 123, 15, 50, pathlib.Path('/out/stacks.mojo'))

    assert command == [
        'austin',
        '-c',
        '-i',
        '20000us',
        '-x',
        '15',
        '-p',
        '123',
        '-o',
        '/out/stacks.mojo',
    ]


def test_austin_samples_become_one_folded_sample_each_with_py_spy_shaped_frames():
    # mojo2austin's text: one line per sample, frames file:qualname:line, the
    # sample's microseconds last; a sample on a thread with no Python frame too.
    austin_text = (
        '# austin: 4.0.0\n'
        '# mode: cpu\n'
        'P7;T0:84;<frozen runpy>:_run_module_as_main:199;/w/_synthetic_workload.py:_hot_a:67 24987\n'
        'P7;T0:61;/py/threading.py:Thread._bootstrap:1032;/py/threading.py:Thread.run:1012 20011\n'
        'P7;T0:61 19870\n'
    )

    folded = austin_to_folded(austin_text)
    self_counts, total, skipped = parse_folded(folded)

    assert folded.splitlines()[0] == (
        '_run_module_as_main (<frozen runpy>:199);_hot_a (/w/_synthetic_workload.py:67) 1'
    )
    assert self_counts == {
        '_hot_a (/w/_synthetic_workload.py:67)': 1,
        'Thread.run (/py/threading.py:1012)': 1,
        NO_PYTHON_FRAME: 1,
    }
    assert (total, skipped) == (3, 0)


def test_a_refused_attach_on_the_mac_names_its_cause_by_whether_it_ran_as_root():
    assert 'sudo' in _attach_hint('darwin', is_root=False)
    as_root = _attach_hint('darwin', is_root=True)
    assert 'hardened' in as_root and 'Homebrew' in as_root
    assert _attach_hint('win32', is_root=False) == ''


def test_austin_ending_its_window_is_a_recording_and_only_its_eperm_is_a_refused_attach():
    # austin ends its -x window by emulating Ctrl-C and exits 254 (-SIGINT); a
    # window cut short by the target's exit returns an error, not a recording.
    assert _recorded(AUSTIN, 254)
    assert not _recorded(AUSTIN, 0)
    assert not _recorded(AUSTIN, 2)
    assert _refused_attach(AUSTIN, 2, '')
    assert not _refused_attach(AUSTIN, 1, '')
    assert not _recorded(PY_SPY, 254)
    assert _refused_attach(PY_SPY, 1, 'This program requires root on OSX.')
