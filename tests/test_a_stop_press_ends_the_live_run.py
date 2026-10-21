# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pressing STOP ends the live run, whatever the run is doing.

``stop_motion`` is the one STOP every client presses: the GUI's red button,
REST, the SDK. A run whose stage it halts ends through the run's own Stop,
recorded ``stage_stopped`` and told once, and captures nothing more. Before,
the STOP reached only the moves on the wire: a run whose move it halted
captured where the stage stopped and ended "The run finished normally.", and
one it caught between moves went on to every remaining step.
"""

from __future__ import annotations

import datetime
import threading

import pytest

from modules.exceptions import MotorStopFailedError
from modules.run_events import RunEvents
from modules.scope_session import ScopeSession
from modules.sequenced_capture_runner import SequencedCaptureRunner
from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

_HOLD_WAIT_S = 10.0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def _steps(*xs):
    base = _make_single_step_protocol().step_row(idx=0)
    return [
        {**base, 'Name': f'S{i}', 'Label': f'S{i}', 'Well': f'A{i + 1}', 'X': x, 'Y': 20.0}
        for i, x in enumerate(xs)
    ]


def _told_stopped(centre_posts):
    return [n for n in centre_posts if n.shown and n.title == 'Protocol Stopped']


def test_stop_pressed_mid_move_ends_the_run_and_tells_it_once(session, tmp_path, centre_posts):
    held = session.scope.simulation.hold_axis('X')
    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0)), parent_dir=str(tmp_path)
    )
    assert held.wait(_HOLD_WAIT_S), "the run's move never got part of the way"

    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 0
    assert len(_told_stopped(centre_posts)) == 1


def test_stop_pressed_with_nothing_moving_ends_the_run_before_its_next_step(
    session, tmp_path, centre_posts
):
    """A STOP pressed as the first step's frame lands, with no axis moving."""
    pressed = []

    def on_frame(*_args):
        if not pressed:
            pressed.append(True)
            session.scope.motion.stop_motion()

    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(40.0, 60.0, 80.0)),
        parent_dir=str(tmp_path),
        events=RunEvents(frame_captured=on_frame),
    )
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert pressed
    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 1
    assert len(_told_stopped(centre_posts)) == 1


def test_stop_pressed_between_scans_ends_the_protocol(session, tmp_path, centre_posts):
    """A protocol waiting for its next scan, nothing moving, is a live run too."""
    first_scan_ended = threading.Event()
    protocol = _build_real_protocol(_steps(60.0))
    protocol.modify_time_params(
        period=datetime.timedelta(seconds=60), duration=datetime.timedelta(seconds=180)
    )
    run = session.create_protocol_runner().start_protocol(
        protocol,
        parent_dir=str(tmp_path),
        events=RunEvents(scan_ended=lambda *_args: first_scan_ended.set()),
    )
    assert first_scan_ended.wait(COMPLETION_TIMEOUT), 'the first scan never ended'

    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 1
    assert len(_told_stopped(centre_posts)) == 1


@pytest.fixture
def turret_session(tmp_path):
    s = ScopeSession.create(
        complete_settings(
            **_settings(
                tmp_path,
                microscope='LS850T',
                simulator_tier='fast',
                turret_objectives=dict(TEST_TURRET_OBJECTIVES),
            )
        ),
        simulate=True,
    )
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


class _HoldAfterFrame:
    """Hold ``axis``'s next move once the run's ``nth`` frame lands, and note
    where ``axis`` stood at each frame. Armed on the run's own thread, which
    waits for the event, so the move held is the first one after that frame."""

    def __init__(self, session, axis, nth):
        self._session, self._axis, self._nth = session, axis, nth
        self.at_frames = []
        self._armed = threading.Event()
        self._held = None

    def __call__(self, *_args):
        self.at_frames.append(self._session.scope.motion.get_current_position(axis=self._axis))
        if len(self.at_frames) == self._nth:
            self._held = self._session.scope.simulation.hold_axis(self._axis)
            self._armed.set()

    def wait(self):
        return self._armed.wait(COMPLETION_TIMEOUT) and self._held.wait(_HOLD_WAIT_S)


def _every_minute_for_three(protocol):
    protocol.modify_time_params(
        period=datetime.timedelta(seconds=60), duration=datetime.timedelta(seconds=180)
    )
    return protocol


def test_stop_pressed_during_a_steps_turret_change_ends_the_run_with_z_parked(
    turret_session, tmp_path, centre_posts
):
    """The turret leg: Z parked, T turning. The change ends with Z left parked
    and the step's X and Y never driven."""
    session = turret_session
    held = session.scope.simulation.hold_axis('T')
    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol([{**_steps(60.0)[0], 'Objective': '4x Oly'}]),
        parent_dir=str(tmp_path),
    )
    assert held.wait(_HOLD_WAIT_S), "the run's turret move never got part of the way"

    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 0
    assert len(_told_stopped(centre_posts)) == 1
    position = session.scope.motion.get_current_position
    assert position(axis='Z') == 0.0, 'Z was driven back up after the STOP'
    assert (position(axis='X'), position(axis='Y')) == (0.0, 0.0), 'the step was driven to'
    assert session.scope.motion.get_turret_slot() is None


def test_stop_pressed_during_grease_redistribution_ends_the_protocol(
    session, tmp_path, centre_posts, monkeypatch
):
    """The grease routine drives Z to 0 and back after every hundredth
    autofocus, between scans; a STOP during it leaves Z where it stopped."""
    # The hundredth autofocus is this run's one, rather than a hundred
    # sweeps run to get there.
    real_reset = SequencedCaptureRunner._reset_vars

    def _reset_one_short(self):
        real_reset(self)
        self._autofocus_count = 99

    monkeypatch.setattr(SequencedCaptureRunner, '_reset_vars', _reset_one_short)
    # After the frame, the first Z move is the grease's move to 0.
    grease = _HoldAfterFrame(session, 'Z', nth=1)
    run = session.create_protocol_runner().start_protocol(
        _every_minute_for_three(_build_real_protocol([{**_steps(60.0)[0], 'Auto_Focus': True}])),
        parent_dir=str(tmp_path),
        events=RunEvents(frame_captured=grease),
    )
    assert grease.wait(), "the grease's move to 0 never got part of the way"

    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 1
    assert len(_told_stopped(centre_posts)) == 1
    z = session.scope.motion.get_current_position(axis='Z')
    assert 0.0 < z < grease.at_frames[0], f'Z was driven on after the STOP: {z}'


def test_stop_pressed_during_the_return_between_scans_ends_the_protocol(
    session, tmp_path, centre_posts
):
    """Between scans the stage returns to the first step; a STOP during
    that return leaves X where it stopped."""
    # After the last step's frame, the first X move is the return.
    back = _HoldAfterFrame(session, 'X', nth=2)
    run = session.create_protocol_runner().start_protocol(
        _every_minute_for_three(_build_real_protocol(_steps(40.0, 80.0))),
        parent_dir=str(tmp_path),
        events=RunEvents(frame_captured=back),
    )
    assert back.wait(), 'the return to the first step never got part of the way'

    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 2
    assert len(_told_stopped(centre_posts)) == 1
    first, last = back.at_frames
    x = session.scope.motion.get_current_position(axis='X')
    assert min(first, last) < x < max(first, last), f'X was driven on after the STOP: {x}'


def test_a_fault_that_ended_the_run_first_keeps_its_ending(session, tmp_path):
    held = session.scope.simulation.hold_axis('X')
    runner = session.create_protocol_runner()
    run = runner.start_single_scan(
        protocol=_build_real_protocol(_steps(60.0)), parent_dir=str(tmp_path)
    )
    assert held.wait(_HOLD_WAIT_S), "the run's move never got part of the way"
    session.sequenced_capture_runner.abort_run_fatal('camera_failure', 'The camera failed.')

    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('failed', 'camera_failure')


def test_a_stop_the_board_fails_still_ends_the_run(session, tmp_path, monkeypatch):
    """The presser hears the failed STOP; the run ends all the same, in words
    that say only that STOP was pressed, which is true either way."""
    from tests.test_a_failed_motor_stop_is_raised import _stop_fails

    held = session.scope.simulation.hold_axis('X')
    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0)), parent_dir=str(tmp_path)
    )
    assert held.wait(_HOLD_WAIT_S), "the run's move never got part of the way"
    # a stand-in by design: a board that does not answer the STOP, as the
    # failed-stop tests make it; the simulator cannot fail its STOP yet.
    _stop_fails(session.scope, monkeypatch)

    with pytest.raises(MotorStopFailedError):
        session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')


def test_a_run_started_after_stop_runs(session, tmp_path):
    session.scope.motion.stop_motion()
    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0)), parent_dir=str(tmp_path)
    )
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('completed', 'completed')


def start_an_autofocus_held_mid_sweep(session):
    """Start a standalone autofocus and return its run once a sweep move is held.

    The run's first Z move is its go-to-step, to the Z it already stands
    at; that move is let go, and a hold armed after it holds a move of the
    sweep itself.
    """
    # Mid-travel, so the sweep's window lies inside Z's travel.
    session.scope.motion.move_absolute('Z', 5000.0)
    to_the_step = session.scope.simulation.hold_axis('Z')
    run = session.create_protocol_runner().start_autofocus('BF')
    assert to_the_step.wait(_HOLD_WAIT_S), "the autofocus run's go-to-step never started"
    session.scope.simulation.clear()
    in_the_sweep = session.scope.simulation.hold_axis('Z')
    assert in_the_sweep.wait(_HOLD_WAIT_S), 'no move of the sweep got part of the way'
    return run


def test_stop_pressed_during_an_autofocus_run_ends_it_and_tells_it_once(session, centre_posts):
    """A standalone autofocus is a run: a STOP while it sweeps ends it.
    Before, it went on and ended 'completed'. Where the sweep leaves Z is
    the sweep's own (``test_a_stopped_autofocus_leaves_z.py``)."""
    run = start_an_autofocus_held_mid_sweep(session)

    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert len(_told_stopped(centre_posts)) == 1
