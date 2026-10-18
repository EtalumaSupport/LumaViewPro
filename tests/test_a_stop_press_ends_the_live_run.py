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

from modules.run_events import RunEvents
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
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
    base = _make_single_step_protocol().step(idx=0)
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


def test_a_run_started_after_stop_runs(session, tmp_path):
    session.scope.motion.stop_motion()
    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0)), parent_dir=str(tmp_path)
    )
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('completed', 'completed')


def test_stop_pressed_during_an_autofocus_run_ends_it_and_tells_it_once(session, centre_posts):
    """A standalone autofocus is a run: a STOP while it moves ends it.
    Before, it went on and ended 'completed'. Where the sweep leaves Z is
    the sweep's own (``test_a_stopped_autofocus_leaves_z.py``)."""
    # Mid-travel, so the sweep's window lies inside Z's travel.
    session.scope.motion.move_absolute('Z', 5000.0)
    held = session.scope.simulation.hold_axis('Z')
    run = session.create_protocol_runner().start_autofocus('BF')
    assert held.wait(_HOLD_WAIT_S), "the autofocus run's Z move never got part of the way"

    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert len(_told_stopped(centre_posts)) == 1
