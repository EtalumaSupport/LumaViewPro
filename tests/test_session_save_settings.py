"""An L2 caller can deliberately persist configuration.

save_settings used to live on a Kivy widget and read the app context, so
a headless caller could read and write layer config but had no way to put
it on disk. It lives on the session now; these pin the move and the
hardware-presence gate that came with it.
"""

import json
import os
import shutil
from types import SimpleNamespace

import pytest

import modules.settings_init as settings_init
from modules.exceptions import SettingsSaveRefusedError
from modules.scope_session import ScopeSession

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
    for name in ('objectives.json', 'labware.json'):
        shutil.copy(os.path.join(os.path.dirname(SHIPPED_TEMPLATE), name), data / name)
    monkeypatch.setattr(settings_init, 'settings', None)
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)
    return ScopeSession.create_headless(source_path=str(tmp_path))


def _disconnect(session, monkeypatch):
    monkeypatch.setattr(
        session,
        'scope',
        SimpleNamespace(camera_connected=False, motor_connected=False, led_connected=False),
    )


def test_a_deliberate_save_reaches_disk(session, tmp_path):
    session.settings['live_folder'] = '/data/run7'
    session.save_settings(force=True)

    with open(tmp_path / 'data' / 'current.json') as f:
        assert json.load(f)['live_folder'] == '/data/run7'


def test_a_relative_path_resolves_against_the_session_source(session, tmp_path):
    """Not the working directory -- an installed build cannot write beside itself."""
    session.save_settings(force=True)
    assert (tmp_path / 'data' / 'current.json').exists()


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


def test_running_on_the_template_declines_even_when_forced(session, tmp_path, monkeypatch):
    """current.json is the user's only copy; do not overwrite what we could not read."""
    monkeypatch.setattr(
        settings_init, 'rejected_current_json', (str(tmp_path / 'data' / 'current.json'), 'garbled')
    )
    before = (tmp_path / 'data' / 'current.json').read_text()

    session.settings['live_folder'] = '/data/template_values'
    with pytest.raises(SettingsSaveRefusedError) as excinfo:
        session.save_settings(force=True)

    assert excinfo.value.reason == 'settings_provisional'
    assert (tmp_path / 'data' / 'current.json').read_text() == before


def test_the_saved_hook_receives_what_was_written(session):
    """The host's plugin notifier is a callback, so a headless session has none."""
    seen = []
    session._settings_saved_hook = seen.append

    session.settings['live_folder'] = '/data/hooked'
    session.save_settings(force=True)

    assert len(seen) == 1
    assert seen[0]['live_folder'] == '/data/hooked'
