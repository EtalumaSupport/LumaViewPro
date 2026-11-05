# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The session a host builds turns the profile tracer on from its settings.

src/_lumascope/lib/profile_trace reads no setting: ``ScopeSession.create`` opens a run when
its settings hold ``profile_trace_enabled``, into ``profile_trace_output_dir``
or else the data root's ``logs/profile``, and records the commit the host
reads. It turns the tracer on only when it is off and never turns it off, so
a second session in one process keeps the first's run in one directory.
"""

import json
import sys
from pathlib import Path

import pytest

from _lumascope.lib import profile_trace
from _lumascope.modules import path_utils
from _lumascope.modules.scope_session import ScopeSession
from tests.ast_seams import REPO_ROOT
from tests.settings_fixtures import complete_settings


@pytest.fixture(autouse=True)
def _tracer_off():
    profile_trace.disable()
    yield
    profile_trace.disable()


def _create(**settings):
    session = ScopeSession.create(
        complete_settings(**settings), simulate=True, source_path=str(REPO_ROOT)
    )
    session.shutdown()


def _runs(base: Path) -> list[Path]:
    return sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []


def test_a_session_whose_settings_say_on_opens_a_run_naming_the_hosts_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules['_lumascope.lvp_logger'], 'git_revision', lambda: 'c0ffee')

    _create(profile_trace_enabled=True, profile_trace_output_dir=str(tmp_path))

    [run] = _runs(tmp_path)
    assert json.loads((run / 'run_info.json').read_text())['git_sha'] == 'c0ffee'


def test_a_session_whose_settings_say_off_opens_none(tmp_path):
    _create(profile_trace_enabled=False, profile_trace_output_dir=str(tmp_path))

    assert _runs(tmp_path) == []
    assert profile_trace.ENABLE_PROFILE_TRACE is False


def test_with_no_folder_named_the_run_goes_under_the_data_roots_logs(tmp_path, monkeypatch):
    # The data root is the process's (no argument), not the session's
    # source_path: the logs and the traces sit in one tree.
    real = path_utils.get_source_root
    monkeypatch.setattr(
        path_utils,
        'get_source_root',
        lambda source_path=None: tmp_path if source_path is None else real(source_path),
    )

    _create(profile_trace_enabled=True, profile_trace_output_dir=None)

    assert len(_runs(tmp_path / 'logs' / 'profile')) == 1


def test_a_second_session_leaves_the_open_run_where_it_is(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    _create(profile_trace_enabled=True, profile_trace_output_dir=str(first))

    _create(profile_trace_enabled=True, profile_trace_output_dir=str(second))
    _create(profile_trace_enabled=False, profile_trace_output_dir=str(second))

    assert len(_runs(first)) == 1
    assert _runs(second) == []
    assert profile_trace.ENABLE_PROFILE_TRACE is True
