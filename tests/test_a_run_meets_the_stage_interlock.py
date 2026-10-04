# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run is refused while the stage's lid is open, and one that meets the lid ends at once.

Every run commands X and Y at every plate step, and a stage that guards
motion with its lid refuses each of those moves. A run admitted with the lid
open would fail its first step; one that met the lid mid-way was retried
every period to the strike ceiling and ended telling the user to check the
USB cable. Now ``prepare()`` refuses the run naming the lid, after telling
an unhomed scope to home, and a step's move refused for the lid ends the run
at once with the refusal's own words.

Driven through the Session on the simulated scope, its board standing in for
one with a lid.
"""

import threading

import pytest

from drivers.exceptions import MotionInterlockError
from modules.exceptions import ProtocolRunRefusedError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import _images, _run, _settings

LID_WORDS = "The microscope's lid is open. Close it to move or home the stage."


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    yield session
    session.shutdown()


@pytest.fixture
def shown(monkeypatch):
    """Every notification the user is shown, as (title, message)."""
    from modules.notification_center import notifications

    seen = []
    for level in ('info', 'warning', 'error', 'critical'):
        original = getattr(notifications, level)

        def _record(category, title, message, *args, _original=original, **kwargs):
            seen.append((title, message))
            return _original(category, title, message, *args, **kwargs)

        monkeypatch.setattr(notifications, level, _record)
    return seen


def _lid(session, monkeypatch, *, open_now):
    reads = []

    def interlocks():
        reads.append(threading.current_thread().name)
        return frozenset({'lid_open'}) if open_now else frozenset()

    monkeypatch.setattr(session.scope._motion_driver, 'interlocks', interlocks)
    return reads


class TestARunIsRefusedWhileTheLidIsOpen:
    def test_a_homed_scope_is_refused_naming_the_lid(self, session, monkeypatch, tmp_path):
        home_sim_scope(session.scope)
        _lid(session, monkeypatch, open_now=True)

        with pytest.raises(ProtocolRunRefusedError) as refused:
            _run(session, tmp_path)

        assert (refused.value.reason, refused.value.title) == ('lid_open', 'Lid Open')
        assert 'lid is open' in refused.value.message
        assert _images(tmp_path) == []

    def test_an_unhomed_scope_is_told_to_home_without_reading_the_lid(
        self, session, monkeypatch, tmp_path
    ):
        reads = _lid(session, monkeypatch, open_now=True)

        with pytest.raises(ProtocolRunRefusedError) as refused:
            _run(session, tmp_path)

        assert refused.value.reason == 'position_unknown'
        assert reads == []

    def test_a_closed_lid_runs(self, session, monkeypatch, tmp_path):
        home_sim_scope(session.scope)
        reads = _lid(session, monkeypatch, open_now=False)

        outcome = _run(session, tmp_path)

        assert (outcome.status, outcome.reason) == ('completed', 'completed')
        assert reads, 'the run was admitted without reading the lid'


class TestARunThatMeetsTheLidEndsAtOnce:
    def test_ends_failed_naming_the_lid_on_the_first_refusal_not_retried(
        self, session, monkeypatch, tmp_path, shown
    ):
        home_sim_scope(session.scope)
        board = session.scope._motion_driver
        drive = board.move_abs_pos
        refused = []

        def lid_opened(axis, pos, **kwargs):
            if axis in ('X', 'Y'):
                refused.append(axis)
                raise MotionInterlockError('lid_open', moved=False, stopped=True)
            return drive(axis, pos, **kwargs)

        monkeypatch.setattr(board, 'move_abs_pos', lid_opened)

        outcome = _run(session, tmp_path, scans=3)

        assert (outcome.status, outcome.reason) == ('failed', 'interlock')
        assert outcome.title == 'Protocol Aborted -- Lid Open'
        assert LID_WORDS in outcome.message
        assert len(refused) == 1, f'the refused move was retried: {refused}'
        assert [title for title, _ in shown if title == outcome.title] == [outcome.title]
        assert not any('USB' in message for _, message in shown), 'an open lid is not a cable fault'
