# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run is refused while a stage interlock is open, and one that meets the lid ends at once.

Every run commands X and Y at every plate step, and a stage that guards
motion with its lid refuses each of those moves. A run admitted with the lid
open would fail its first step; one that met the lid mid-way was retried
every period to the strike ceiling and ended telling the user to check the
USB cable. Now ``prepare()`` refuses the run naming the lid, after telling
an unhomed scope to home, and a step's move refused for the lid ends the run
at once with the refusal's own words. A lid opened while a step's move is
under way ends the run the same way: the stage stopped short of the step,
and no one pressed Stop.

Driven through the Session on the simulated scope, its board standing in for
one with a lid.
"""

import threading

import pytest

from _lumascope.drivers.exceptions import MotionInterlockError
from _lumascope.modules.exceptions import ProtocolRunRefusedError
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import _images, _run, _settings

LID_WORDS = "The microscope's lid is open. Close it to move or home the stage."


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    yield session
    session.shutdown()


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

        assert (refused.value.reason, refused.value.title) == ('lid_open_before_motion', 'Lid Open')
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


def test_a_run_on_an_unpowered_stage_is_refused_naming_its_power(tmp_path):
    """On the simulated LS720, its production TMCM-6110 driver: a homed stage
    whose power is then lost refuses the run before anything moves, in the
    interlock's own words, as an open lid does."""
    session = ScopeSession.create(
        complete_settings(**_settings(tmp_path, microscope='LS720')), simulate=True
    )
    try:
        session.scope.motion.home()
        session.scope.simulation.interlock(powered=False)

        with pytest.raises(ProtocolRunRefusedError) as refused:
            _run(session, tmp_path)

        assert (refused.value.reason, refused.value.title) == (
            'stage_unpowered_before_motion',
            'No Stage Power',
        )
        assert refused.value.message == (
            "Cannot start the run. The stage has no power. Check the stage's power supply, "
            'then home.'
        )
        assert _images(tmp_path) == []
    finally:
        session.shutdown()


class TestARunThatMeetsTheLidEndsAtOnce:
    def test_ends_failed_naming_the_lid_on_the_first_refusal_not_retried(
        self, session, monkeypatch, tmp_path, centre_posts
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

        shown = [(n.title, n.message) for n in centre_posts if n.shown]
        assert (outcome.status, outcome.reason) == ('failed', 'interlock')
        assert outcome.title == 'Protocol Failed -- Stage Interlock'
        assert LID_WORDS in outcome.message
        assert len(refused) == 1, f'the refused move was retried: {refused}'
        assert [title for title, _ in shown if title == outcome.title] == [outcome.title]
        assert not any('USB' in message for _, message in shown), 'an open lid is not a cable fault'

    def test_a_lid_opened_mid_move_ends_it_failed_naming_the_lid_with_nothing_captured(
        self, tmp_path
    ):
        """On the simulated LS720, its production TMCM-6110 driver: the lid
        opens at the first arrival poll of X or Y once the step has started
        both. Opened before Y's start, Y would be refused before it moved,
        the case the test above pins, and the run would never meet a move
        under way."""
        session = ScopeSession.create(
            complete_settings(**_settings(tmp_path, microscope='LS720')), simulate=True
        )
        try:
            session.scope.motion.home()
            driver = session.scope._motion_driver
            polled = driver.target_status
            drive = driver.move_abs_pos
            started = set()

            def recorded_start(axis, pos):
                drive(axis, pos)
                started.add(axis)

            def lid_opens_while_x_and_y_move(axis):
                if axis in ('X', 'Y') and {'X', 'Y'} <= started:
                    session.scope.simulation.interlock(lid_open=True)
                return polled(axis)

            driver.move_abs_pos = recorded_start
            driver.target_status = lid_opens_while_x_and_y_move

            outcome = _run(session, tmp_path)

            assert (outcome.status, outcome.reason) == ('failed', 'interlock')
            assert 'lid was opened' in outcome.message
            assert _images(tmp_path) == [], 'the step captured where the lid stopped the stage'
        finally:
            session.shutdown()
