# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A path the session hands out is a ``pathlib.Path``, never a string.

A wire client is given a path as its live-folder name and its host path,
and a path rule can only reach a value typed as a path: five were typed
``str`` -- the session's data folder, the settings file set aside and the
name it is retired under, a run's composite and its autofocus data -- so
nothing could tell them from any other text. Each is now a ``Path`` at its
store. The run's two are pinned where a run produces them
(``test_composite_run_e2e``, ``test_run_outcome_reports_autofocus_data``).
"""

from __future__ import annotations

import pathlib

import pytest

from modules import settings_init


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


def test_the_data_folder_is_a_path(session):
    assert isinstance(session.source_path, pathlib.Path)
    assert session.source_path == session.scope.source_path


def test_the_settings_file_set_aside_and_its_retired_name_are_paths(session, monkeypatch, tmp_path):
    current = tmp_path / 'current.json'
    current.write_text('{not json', encoding='utf-8')
    monkeypatch.setattr(settings_init, 'rejected_current_json', (str(current), 'unparseable'))

    set_aside = session.bring_up_record().settings_set_aside
    assert set_aside.path == current
    assert isinstance(set_aside.path, pathlib.Path)

    retired = session.retire_rejected_settings()
    assert isinstance(retired, pathlib.Path)
    assert retired.is_file() and retired.parent == tmp_path
