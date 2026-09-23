# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run needs every axis position the scope has, and never saves one it lost.

Every run moves every axis the scope has, and each move on an axis whose
position is not known is refused. A run on an un-homed scope used to be
admitted, fail every scan, and end after three strikes telling the user to
check the USB cable. A position lost mid-run took the same path; and when
the lost axis was the Z an autofocus sweep failed to restore, the step
captured anyway, so the run ended "completed" with an image saved at a
position nobody knew.

Driven through the Session on the simulated scope. The faults are the
board's own: a target write that gets no answer, which the motion API
turns into a lost axis position exactly as it does on hardware.
"""

import pathlib
import threading
import time

import pytest

import modules.protocol_run_loop as protocol_run_loop
from modules.exceptions import ProtocolRunRefusedError
from modules.lumascope_api import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

COMPLETION_TIMEOUT = 60


def _settings(tmp_path, **extra):
    settings = {
        'stage_offset': {'x': 0.0, 'y': 0.0},
        'live_folder': str(tmp_path),
    }
    for layer in ('BF', 'PC', 'DF', 'Red', 'Green', 'Blue', 'Lumi'):
        settings[layer] = {'autofocus': False}
    settings.update(extra)
    return settings


@pytest.fixture
def session_factory(tmp_path):
    sessions = []

    def _make(**extra):
        session = ScopeSession.create(
            complete_settings(**_settings(tmp_path, **extra)), simulate=True
        )
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown()


@pytest.fixture
def notices(monkeypatch):
    """Every notification the user is shown, as (title, message)."""
    from modules.notification_center import notifications

    shown = []
    for level in ('info', 'warning', 'error', 'critical'):
        original = getattr(notifications, level)

        def _record(category, title, message, *args, _original=original, **kwargs):
            shown.append((title, message))
            return _original(category, title, message, *args, **kwargs)

        monkeypatch.setattr(notifications, level, _record)
    return shown


def _drop_board_writes(session, prefix, which):
    """Make the board leave the ``which``-th (1-based) write starting with ``prefix`` unanswered."""
    driver = session.scope._motion_driver
    original = driver.exchange_command
    seen = {'n': 0}

    def _exchange(command, *args, **kwargs):
        if isinstance(command, str) and command.startswith(prefix):
            seen['n'] += 1
            if seen['n'] == which:
                return None
        return original(command, *args, **kwargs)

    driver.exchange_command = _exchange
    return seen


def _record_moves(session):
    driver = session.scope._motion_driver
    original = driver.exchange_command
    moves = []

    def _exchange(command, *args, **kwargs):
        if isinstance(command, str) and command.startswith('TARGET_W'):
            moves.append(command)
        return original(command, *args, **kwargs)

    driver.exchange_command = _exchange
    return moves


def _run(session, tmp_path, protocol=None, scans=1):
    runner = session.create_protocol_runner()
    done = threading.Event()
    kwargs = {
        'protocol': protocol or _make_single_step_protocol(),
        'sequence_name': 'position',
        'parent_dir': str(tmp_path),
        'image_capture_config': runner.build_image_capture_config(image_mode='8bit'),
        'callbacks': {'run_complete': lambda **kw: done.set(), 'files_complete': lambda **kw: None},
    }
    if scans == 1:
        runner.run_single_scan(**kwargs)
    else:
        runner.run_protocol(**kwargs)
    assert done.wait(COMPLETION_TIMEOUT), 'the run never ended'
    outcome = runner.wait_for_completion(timeout=COMPLETION_TIMEOUT)
    assert outcome is not None
    return outcome


def _images(tmp_path):
    return [p for p in pathlib.Path(tmp_path).rglob('*') if p.suffix.lower() in ('.tif', '.tiff')]


class TestTheQuestion:
    def test_every_axis_of_an_unhomed_scope_is_unknown(self, session_factory):
        motion = session_factory().scope.motion
        assert motion.axes_without_position() == {
            'X': AxisState.UNKNOWN,
            'Y': AxisState.UNKNOWN,
            'Z': AxisState.UNKNOWN,
        }

    def test_a_homed_scope_has_nothing_unknown(self, session_factory):
        session = session_factory()
        home_sim_scope(session.scope)
        assert session.scope.motion.axes_without_position() == {}

    def test_a_scope_with_no_axes_has_nothing_unknown(self, session_factory):
        motion = session_factory().scope.motion
        motion._init_axes([], [])
        assert motion.axes_without_position() == {}


class TestARunCannotStartWithoutEveryPosition:
    def test_an_unhomed_scope_is_refused_naming_its_axes_and_nothing_moves(
        self, session_factory, tmp_path, notices
    ):
        session = session_factory()
        moves = _record_moves(session)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _run(session, tmp_path)

        assert excinfo.value.reason == 'position_unknown'
        assert 'the X, Y and Z positions are unknown' in excinfo.value.message
        assert 'Home the scope' in excinfo.value.message
        assert moves == [], 'a refused run must not move the stage'
        assert [title for title, _ in notices] == [excinfo.value.title], 'the refusal is shown once'

    def test_an_unknown_turret_slot_is_refused_naming_t(self, session_factory, tmp_path):
        session = session_factory(
            microscope='LS850T',
            turret_objectives={'1': '10x Oly', '2': None, '3': None, '4': None},
        )
        home_sim_scope(session.scope)
        # A timed-out turret move leaves the slot unknown this way.
        session.scope.motion._set_axis_state('T', AxisState.UNKNOWN)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _run(session, tmp_path)

        assert excinfo.value.reason == 'position_unknown'
        assert 'the T position is unknown' in excinfo.value.message

    def test_a_home_in_progress_says_to_wait(self, session_factory, tmp_path):
        session = session_factory()
        home_sim_scope(session.scope)
        session.scope.motion._set_axis_state('Z', AxisState.HOMING)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _run(session, tmp_path)

        assert 'Z is still homing' in excinfo.value.message
        assert 'Wait for the home to finish' in excinfo.value.message

    def test_a_z_home_over_an_unhomed_stage_names_each_axis(self, session_factory, tmp_path):
        # The GUI's Z-only home leaves X and Y unknown while Z homes.
        session = session_factory()
        session.scope.motion._set_axis_state('Z', AxisState.HOMING)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _run(session, tmp_path)

        assert 'Z is still homing; the X and Y positions are unknown' in excinfo.value.message
        assert 'Home the scope' in excinfo.value.message

    def test_a_homed_scope_runs(self, session_factory, tmp_path):
        session = session_factory()
        home_sim_scope(session.scope)

        outcome = _run(session, tmp_path)

        assert (outcome.status, outcome.reason) == ('completed', 'completed')
        assert len(_images(tmp_path)) == 1


class TestALostPositionEndsTheRunOnce:
    def test_a_move_the_board_never_answered_ends_the_run_naming_the_axis(
        self, session_factory, tmp_path, notices
    ):
        session = session_factory()
        home_sim_scope(session.scope)
        _drop_board_writes(session, 'TARGET_WX', which=1)

        outcome = _run(session, tmp_path)

        assert (outcome.status, outcome.reason) == ('failed', 'position_lost')
        assert 'the X position is unknown' in outcome.message
        assert 'Home the scope' in outcome.message
        endings = [title for title, _ in notices if title == outcome.title]
        assert endings == [outcome.title], 'the ending is shown once'
        assert not any('USB' in message for _, message in notices), (
            'a lost position is not a cable fault'
        )

    @pytest.mark.parametrize('which', [3, 6])
    def test_a_z_lost_during_autofocus_saves_no_image(
        self, session_factory, tmp_path, notices, which
    ):
        session = session_factory(BF={'autofocus': True})
        home_sim_scope(session.scope)
        dropped = _drop_board_writes(session, 'TARGET_WZ', which=which)
        protocol = _build_real_protocol(
            [dict(_make_single_step_protocol().step(idx=0), Auto_Focus=True)]
        )

        outcome = _run(session, tmp_path, protocol=protocol)

        assert dropped['n'] >= which, 'the sweep never reached the dropped write'
        assert (outcome.status, outcome.reason) == ('failed', 'position_lost')
        assert 'the Z position is unknown' in outcome.message
        assert _images(tmp_path) == [], 'an image was saved at an unknown Z'
        restore = [message for title, message in notices if title == 'Z Position Not Restored']
        assert restore and 'home the scope' in restore[0], (
            'the restore notice must send the user to home, not to a move the API refuses'
        )

    def test_a_position_lost_between_scans_ends_the_run_before_the_period(
        self, session_factory, tmp_path
    ):
        session = session_factory()
        home_sim_scope(session.scope)
        # The first X write is scan 1's own move; the second is the return to
        # the first step between scans.
        _drop_board_writes(session, 'TARGET_WX', which=2)
        period_min = 1.0
        protocol = _build_real_protocol(
            [dict(_make_single_step_protocol().step(idx=0))],
            period_min=period_min,
            duration_hrs=3 * period_min / 60,
        )

        started = time.monotonic()
        outcome = _run(session, tmp_path, protocol=protocol, scans=3)

        assert (outcome.status, outcome.reason) == ('failed', 'position_lost')
        assert time.monotonic() - started < period_min * 60 / 2, (
            'the run waited out the period before ending on a lost position'
        )


def test_a_scan_that_captured_is_not_a_strike(session_factory, tmp_path, monkeypatch):
    # Every between-scan return fails with every position known: each scan
    # itself captures, so the run completes rather than tripping the strike
    # ceiling that counts only scans that failed.
    session = session_factory()
    home_sim_scope(session.scope)
    original = protocol_run_loop.ProtocolRunLoop._return_to_first_step_between_scans

    def _failing_return(self):
        if self._inter_scan_wait_follows():
            raise RuntimeError('return move refused')
        return original(self)

    monkeypatch.setattr(
        protocol_run_loop.ProtocolRunLoop, '_return_to_first_step_between_scans', _failing_return
    )
    period_min = 0.02
    scans = 4
    protocol = _build_real_protocol(
        [dict(_make_single_step_protocol().step(idx=0))],
        period_min=period_min,
        duration_hrs=scans * period_min / 60,
    )

    outcome = _run(session, tmp_path, protocol=protocol, scans=scans)

    assert (outcome.status, outcome.reason) == ('completed', 'completed')
    assert len(_images(tmp_path)) == scans
