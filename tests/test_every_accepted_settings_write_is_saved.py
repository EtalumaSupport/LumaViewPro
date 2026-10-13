# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every accepted settings write is saved, by the Session, to the file its host named.

Before, a write reached disk only when the GUI saved -- every five minutes
and at its close -- so a write over REST or from a script was lost with
the process unless the caller saved it, and a crash lost up to five
minutes of the GUI's. Now the Session saves a second after the first
unsaved write and once at its close, whole or not at all, to the one file
``create(settings_file=...)`` named; a session that names none saves
nothing, and one whose writes cannot be kept says so once.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time

import pytest

import modules.settings_init as settings_init
from modules.scope_session import ScopeSession
from modules.settings_saver import SettingsSaver
from tests.installation_fixtures import copy_installation_files
from tests.settings_fixtures import complete_settings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestTheSaver:
    def test_writes_inside_the_window_share_one_save_made_a_window_after_the_first(self):
        saved = threading.Event()
        saves = []

        def save():
            saves.append(time.monotonic())
            saved.set()

        saver = SettingsSaver(save, window_s=0.2)
        saver.start()
        try:
            first = time.monotonic()
            for _ in range(3):
                saver.wake()
            assert saved.wait(5.0)
        finally:
            saver.stop()

        assert len(saves) == 1
        assert saves[0] - first >= 0.2

    def test_a_stop_inside_the_window_saves_nothing_itself(self):
        saves = []
        saver = SettingsSaver(lambda: saves.append(1), window_s=5.0)
        saver.start()
        saver.wake()

        saver.stop()

        # The Session's close makes the one save at stop; the saver leaves it.
        assert saves == []


@pytest.fixture
def heard():
    return []


@pytest.fixture
def saving_session(tmp_path, heard):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')),
        simulate=True,
        settings_file=str(tmp_path / 'current.json'),
        outcome_listener=heard.append,
    )
    yield session
    session.shutdown()


def _change_the_frame_rate(session) -> int:
    value = 20 if session.get_setting('video.max_fps') == 30 else 30
    session.update_settings('video.max_fps', value)
    return value


def test_an_accepted_write_is_in_the_named_file_after_the_close(saving_session, tmp_path):
    value = _change_the_frame_rate(saving_session)

    saving_session.shutdown()

    with open(tmp_path / 'current.json') as f:
        assert json.load(f)['video']['max_fps'] == value


def test_the_close_saves_with_no_write_so_a_first_run_leaves_its_file(saving_session, tmp_path):
    saving_session.shutdown()

    with open(tmp_path / 'current.json') as f:
        assert json.load(f)['microscope'] == saving_session.get_setting('microscope')


def test_a_save_the_drive_refuses_leaves_the_previous_file_whole_and_is_reported(
    saving_session, tmp_path, heard, monkeypatch
):
    target = tmp_path / 'current.json'
    target.write_text('{"previous": true}')

    def refuse(src, dst):
        raise OSError(28, 'No space left on device')

    # a stand-in by design: a drive that refuses the replace is a fault the
    # simulator cannot yet produce (its file lane stalls, it does not refuse).
    monkeypatch.setattr(os, 'replace', refuse)
    _change_the_frame_rate(saving_session)
    saving_session.shutdown()
    monkeypatch.undo()

    assert target.read_text() == '{"previous": true}'
    assert sorted(os.listdir(tmp_path)) == ['current.json', 'live']
    failed = [n for n in heard if n.reason == 'settings_file_not_written']
    assert failed and 'No space left on device' in failed[0].message


def test_a_session_naming_no_file_writes_none_not_even_the_one_it_was_read_from(
    tmp_path, monkeypatch
):
    data = tmp_path / 'data'
    data.mkdir()
    shutil.copy(os.path.join(REPO_ROOT, 'data', 'settings.json'), data / 'settings.json')
    shutil.copy(os.path.join(REPO_ROOT, 'data', 'settings.json'), data / 'current.json')
    copy_installation_files(data)
    monkeypatch.setattr(settings_init, 'settings', None)
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)
    before = (data / 'current.json').read_text()
    settings = ScopeSession.load_user_settings(str(tmp_path))
    # The shipped template's firmware tier costs seconds and is not the claim.
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, source_path=str(tmp_path), simulate=True)

    _change_the_frame_rate(session)
    session.shutdown()

    assert (data / 'current.json').read_text() == before


def test_a_provisional_session_keeps_its_writes_unsaved_and_says_so_once(
    tmp_path, heard, monkeypatch
):
    monkeypatch.setattr(
        settings_init, 'rejected_current_json', (str(tmp_path / 'current.json'), 'unreadable')
    )
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')),
        simulate=True,
        settings_file=str(tmp_path / 'current.json'),
        outcome_listener=heard.append,
    )
    _change_the_frame_rate(session)
    _change_the_frame_rate(session)
    session.shutdown()

    assert not (tmp_path / 'current.json').exists()
    told = [n for n in heard if n.title == 'Settings not saved']
    assert [n.reason for n in told] == ['settings_provisional']


def test_a_scope_with_no_hardware_keeps_its_writes_unsaved_and_says_so_once(
    saving_session, tmp_path, heard, monkeypatch
):
    # a stand-in by design: a real scope that found no hardware at bring-up
    # is a state the simulator cannot yet produce (a simulated scope always
    # finds its simulated parts).
    monkeypatch.setattr(type(saving_session.scope), 'no_hardware', property(lambda self: True))
    _change_the_frame_rate(saving_session)
    _change_the_frame_rate(saving_session)
    saving_session.shutdown()

    assert not (tmp_path / 'current.json').exists()
    told = [n for n in heard if n.title == 'Settings not saved']
    assert [n.reason for n in told] == ['no_hardware']
