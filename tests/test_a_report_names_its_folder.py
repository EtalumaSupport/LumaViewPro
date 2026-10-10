# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report or a logs zip is written into the folder its caller names.

``make_support_report`` and ``make_logs_zip`` wrote to the Desktop when no
folder was given -- a folder a REST client cannot name, on a machine it
may not be sitting at -- and the default sat in two places (the report's
own and the command line's). The folder is now required; the Desktop is
one public path helper, ``path_utils.desktop_folder``, that the GUI's two
calls and the command-line report pass.
"""

from __future__ import annotations

import pathlib
import types

import pytest

from modules import app_context, path_utils


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


@pytest.mark.parametrize('member', ['make_support_report', 'make_logs_zip'])
def test_a_report_with_no_folder_is_a_type_error(session, member):
    with pytest.raises(TypeError, match='output_dir'):
        getattr(session, member)()


def test_the_desktop_is_platformdirs_own(monkeypatch, tmp_path):
    import platformdirs

    monkeypatch.setattr(platformdirs, 'user_desktop_dir', lambda: str(tmp_path))
    assert path_utils.desktop_folder() == tmp_path


def test_a_machine_with_no_desktop_folder_gives_the_home_folder(monkeypatch, tmp_path):
    import platformdirs

    monkeypatch.setattr(platformdirs, 'user_desktop_dir', lambda: str(tmp_path / 'absent'))
    assert path_utils.desktop_folder() == pathlib.Path.home()


@pytest.mark.parametrize('starts', ['_start_support_report', 'zip_logs_only'])
def test_the_guis_two_calls_name_the_desktop(monkeypatch, tmp_path, starts):
    import ui.microscope_settings as panel

    monkeypatch.setattr(path_utils, 'desktop_folder', lambda: tmp_path)
    asked = {}
    session = types.SimpleNamespace(
        make_support_report=lambda **kwargs: asked.update(kwargs),
        make_logs_zip=lambda **kwargs: asked.update(kwargs),
    )
    monkeypatch.setattr(app_context, 'ctx', types.SimpleNamespace(session=session))
    host = types.SimpleNamespace(
        _make_zip=lambda title, label, make, budget_of=None: make(lambda *progress: None)
    )

    getattr(panel.MicroscopeSettings, starts)(host)

    assert asked['output_dir'] == tmp_path
