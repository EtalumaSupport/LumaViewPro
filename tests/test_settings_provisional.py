# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An unreadable current.json must not cost the user their configuration.

current.json is the only copy of what a user actually had set. When it
cannot be read the app comes up on the shipped template -- and that
template is then sitting in the same dict that gets written back to
current.json on a 300 s timer and again at exit. Left alone, a corrupt
file therefore becomes a destroyed configuration within five minutes of
the next session with hardware attached.

So the file is not touched until a human says so. Until then the app runs
on defaults and refuses to save over it. The rename that follows a
"start over" choice keeps the original bytes: support can often read a
configuration out of a file the app could not.

The direction of the popup's two buttons is load-bearing rather than
cosmetic. show_confirmation_popup treats a programmatic or lifecycle
dismiss as CANCEL, so cancel has to be the branch that changes nothing;
if the destructive branch sat there, a dialog closed by anything other
than a person would rename the user's file.
"""

import json
import os

import pytest

from modules import settings_init


@pytest.fixture
def appdata(tmp_path):
    """A settings directory whose current.json is recognisably the user's."""
    data = tmp_path / 'data'
    data.mkdir()
    repo_template = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'settings.json',
    )
    with open(repo_template) as f:
        template = json.load(f)
    with open(data / 'settings.json', 'w') as f:
        json.dump(template, f, indent=4)

    user = dict(template)
    user['live_folder'] = '/tmp/the-users-own-folder'
    with open(data / 'current.json', 'w') as f:
        json.dump(user, f, indent=4)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Put back every process global these tests disturb.

    load_lvp_settings writes settings_init.settings, and unrelated tests
    read it -- a simulated scope built after this file ran would otherwise
    be configured from a temp directory's copy.
    """
    saved_settings = settings_init.settings
    saved_flag = settings_init.rejected_current_json
    settings_init.rejected_current_json = None
    yield
    settings_init.settings = saved_settings
    settings_init.rejected_current_json = saved_flag


def _break(appdata, how):
    current = appdata / 'data' / 'current.json'
    if how == 'unparseable':
        with open(current, 'w') as f:
            f.write('{"live_folder": "/tmp/the-users-own-folder",,,')
    elif how == 'missing_required':
        with open(current) as f:
            loaded = json.load(f)
        del loaded['microscope']
        with open(current, 'w') as f:
            json.dump(loaded, f)
    elif how == 'toplevel_list':
        with open(current, 'w') as f:
            json.dump(['not', 'a', 'settings', 'object'], f)
    return current


class TestTheFileSurvives:
    @pytest.mark.parametrize('how', ['unparseable', 'missing_required', 'toplevel_list'])
    def test_load_leaves_the_users_file_untouched(self, appdata, how, caplog):
        current = _break(appdata, how)
        before = current.read_bytes()

        settings_init.load_lvp_settings(__import__('logging').getLogger('t'), str(appdata))

        assert current.read_bytes() == before
        assert settings_init.settings_are_provisional()

    @pytest.mark.parametrize('how', ['unparseable', 'missing_required', 'toplevel_list'])
    def test_app_comes_up_on_the_template(self, appdata, how):
        _break(appdata, how)
        settings_init.load_lvp_settings(__import__('logging').getLogger('t'), str(appdata))
        # The user's marker is gone -- that is the whole hazard, and why
        # saving is refused until they are told.
        assert settings_init.settings['live_folder'] != '/tmp/the-users-own-folder'

    def test_a_healthy_file_is_not_provisional(self, appdata):
        settings_init.load_lvp_settings(__import__('logging').getLogger('t'), str(appdata))
        assert not settings_init.settings_are_provisional()
        assert settings_init.settings['live_folder'] == '/tmp/the-users-own-folder'

    def test_a_second_load_does_not_inherit_the_verdict(self, appdata):
        log = __import__('logging').getLogger('t')
        _break(appdata, 'unparseable')
        settings_init.load_lvp_settings(log, str(appdata))
        assert settings_init.settings_are_provisional()

        # Repair it and load again: the flag must clear, or the app would
        # refuse to save forever after one bad startup.
        with open(appdata / 'data' / 'settings.json') as f:
            healthy = f.read()
        with open(appdata / 'data' / 'current.json', 'w') as f:
            f.write(healthy)
        settings_init.load_lvp_settings(log, str(appdata))
        assert not settings_init.settings_are_provisional()


class TestRetiring:
    def test_retire_renames_and_preserves_the_bytes(self, appdata):
        current = _break(appdata, 'unparseable')
        original = current.read_bytes()
        settings_init.load_lvp_settings(__import__('logging').getLogger('t'), str(appdata))

        retired = settings_init.retire_rejected_current_json()

        assert retired is not None
        assert not os.path.exists(current)
        with open(retired, 'rb') as f:
            assert f.read() == original
        assert not settings_init.settings_are_provisional()

    def test_retire_is_a_no_op_when_nothing_was_rejected(self, appdata):
        settings_init.load_lvp_settings(__import__('logging').getLogger('t'), str(appdata))
        assert settings_init.retire_rejected_current_json() is None


class TestTheSaveGuard:
    """The destination test, which is what keeps the guard honest."""

    @pytest.mark.parametrize(
        'path',
        [
            './data/current.json',
            'data/current.json',
            '/an/absolute/path/to/current.json',
            './data/current',  # save_settings appends .json
            'CURRENT.JSON',
        ],
    )
    def test_recognises_the_live_configuration(self, path):
        assert settings_init.targets_current_json(path)

    @pytest.mark.parametrize(
        'path',
        ['./data/settings.json', '/tmp/somewhere/else.json', 'protocol.json'],
    )
    def test_leaves_other_destinations_alone(self, path):
        # A caller writing somewhere else is not writing the user's live
        # configuration and must not be blocked.
        assert not settings_init.targets_current_json(path)
