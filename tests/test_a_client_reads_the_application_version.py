# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A client reads which LumaViewPro it is talking to, and how it was launched.

A script or a REST client that reports a problem, or depends on a member
added in one release, has to know the application's version, and only
the session can answer it: ``session.app_version`` is ``version.txt``'s
first line, through its one reader, or None where the build has none.
``session.app_runtime`` says whether that is a source run, a packaged
bundle or an installed build, from ``path_utils.app_runtime``.
"""

import pytest

from modules import path_utils


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


def test_the_version_is_version_txts_first_line(session):
    first_line = (
        (path_utils.get_script_root() / 'version.txt')
        .read_text(encoding='utf-8-sig')
        .splitlines()[0]
        .strip()
    )

    assert first_line
    assert session.app_version == first_line


def test_a_build_with_no_version_answers_none(session, monkeypatch):
    monkeypatch.setattr(path_utils, 'read_version', lambda: ('', ''))

    assert session.app_version is None


def test_a_source_run_says_so(session):
    assert session.app_runtime is path_utils.AppRuntime.SOURCE


def test_the_runtime_is_the_processs_not_a_copy(session, tmp_path, monkeypatch):
    # The session asks afresh: a packaged build's process answers bundle.
    monkeypatch.setattr('sys.frozen', True, raising=False)
    monkeypatch.setattr('sys.executable', str(tmp_path / 'LumaViewPro.exe'))

    assert session.app_runtime is path_utils.AppRuntime.BUNDLE
