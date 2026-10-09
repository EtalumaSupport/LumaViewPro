"""An L2 caller can deliberately persist configuration.

save_settings used to live on a Kivy widget and read the app context, so
a headless caller could read and write layer config but had no way to put
it on disk. It lives on the session now; these pin the move and the
hardware-presence gate that came with it.
"""

import json
import os
import shutil

import pytest

import modules.settings_init as settings_init
from modules.exceptions import SettingsSaveRefusedError
from modules.scope_session import ScopeSession
from tests.installation_fixtures import copy_installation_files

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_TEMPLATE = os.path.join(REPO_ROOT, 'data', 'settings.json')


@pytest.fixture
def session(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    shutil.copy(SHIPPED_TEMPLATE, data / 'settings.json')
    shutil.copy(SHIPPED_TEMPLATE, data / 'current.json')
    # The factory builds the session's helpers from this root and refuses
    # to configure the scope without them.
    copy_installation_files(data)
    monkeypatch.setattr(settings_init, 'settings', None)
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)
    return ScopeSession.create(
        ScopeSession.load_user_settings(str(tmp_path)), source_path=str(tmp_path), simulate=True
    )


def _disconnect(session, monkeypatch):
    # The real scope with no hardware found at its bring-up, so the save
    # still reads the live runtime state it records the turret slot from.
    monkeypatch.setattr(type(session.scope), 'no_hardware', property(lambda self: True))


def test_a_deliberate_save_reaches_disk(session, tmp_path):
    session.settings['live_folder'] = '/data/run7'
    session.save_settings(force=True)

    with open(tmp_path / 'data' / 'current.json') as f:
        assert json.load(f)['live_folder'] == '/data/run7'


def test_no_hardware_this_session_skips_the_write(session, tmp_path, monkeypatch):
    """The sliders would be at their defaults; those are not the user's values.

    The skip is announced, not silent: a caller that cannot tell a
    refusal from a success reports success on a write that never
    happened, which is how a whole session's changes get lost.
    """
    _disconnect(session, monkeypatch)
    before = (tmp_path / 'data' / 'current.json').read_text()

    session.settings['live_folder'] = '/data/should_not_persist'
    with pytest.raises(SettingsSaveRefusedError) as excinfo:
        session.save_settings()

    assert excinfo.value.reason == 'no_hardware'
    assert (tmp_path / 'data' / 'current.json').read_text() == before


def test_force_overrides_the_hardware_gate(session, tmp_path, monkeypatch):
    """An API write has no slider behind it to misread."""
    _disconnect(session, monkeypatch)

    session.settings['live_folder'] = '/data/deliberate'
    session.save_settings(force=True)

    with open(tmp_path / 'data' / 'current.json') as f:
        assert json.load(f)['live_folder'] == '/data/deliberate'


def test_the_plugins_receive_what_was_written(session, monkeypatch):
    """A session's plugins are told what was written; a session without plugins tells no one."""
    session.save_settings(force=True)  # no plugins: nothing to tell, nothing raised

    from modules.plugins import PluginRegistry

    seen = []
    session.plugins = PluginRegistry()
    monkeypatch.setattr(
        session.plugins, 'settings_saved', lambda host, settings: seen.append((host, settings))
    )
    session.settings['live_folder'] = '/data/hooked'
    session.save_settings(force=True)

    assert len(seen) == 1
    assert seen[0][0] is session
    assert seen[0][1]['live_folder'] == '/data/hooked'
