# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An L2 caller assembles a sequenced capture config with no GUI.

Assembling this config used to be reachable only two ways, and neither
was open to a caller outside the app: through config_ui_getters, which
reads the running Kivy widget tree, or by importing config_helpers
directly -- reaching past the session layer into a module, which the
layer architecture does not allow.

The session now answers for it, alongside the layer, stim and auto-gain
config wrappers it already carried.
"""

import pytest

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create_headless(settings=complete_settings(live_folder=str(tmp_path)))
    try:
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def test_the_session_assembles_the_config(session):
    config = session.get_sequenced_capture_config()

    assert config['labware_id']
    assert config['objective_id']
    assert config['frame_dimensions']['width'] > 0
    assert config['binning_size'] >= 1


def test_the_caller_states_tiling_and_zstacking(session):
    """Neither has a settings home -- they do not survive a restart -- so a
    caller with no widgets says what it wants and is believed."""
    config = session.get_sequenced_capture_config(tiling='2x2', use_zstacking=True)

    assert config['tiling'] == '2x2'
    assert config['use_zstacking'] is True


def test_assembling_it_never_touches_the_app_context(session, monkeypatch):
    """The point of the entry point: nothing reaches for the running app.

    The GUI lane answers this question through modules.app_context, whose
    ctx carries the live widget tree. A caller that is not the app has no
    such tree, so any reach for it is the config becoming unassemblable
    outside LVP.exe. Asserting no Kivy import would not catch this -- the
    widget reads go through the context object and import no toolkit.
    """

    class NoAppHere:
        def __getattr__(self, name):
            raise AssertionError(f'assembling the config reached the app context for {name!r}')

    import modules.app_context as app_context

    monkeypatch.setattr(app_context, 'ctx', NoAppHere())

    config = session.get_sequenced_capture_config(tiling='2x2')

    assert config['tiling'] == '2x2'
