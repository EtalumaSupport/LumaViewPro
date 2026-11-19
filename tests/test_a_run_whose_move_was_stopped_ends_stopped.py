# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run whose move a STOP cut ends stopped, whoever sent the STOP.

The run judges the moves it makes before a step commits. A STOP it never
heard -- a client cancelling its own move, a press whose tell found no run
yet -- still ends the run as a person's Stop does: ``aborted``,
``stage_stopped``, nothing captured where the stage happened to stop. Before,
the run read only whether anything moved: a move stopped short stands still
with its position known, so the step captured there and the run ended
``completed``.
"""

from __future__ import annotations

import datetime
import threading

import pytest

from _lumascope.modules.run_events import RunEvents

from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
from tests.test_a_stop_press_ends_the_live_run import (  # noqa: F401 -- fixtures
    _HOLD_WAIT_S,
    _steps,
    _told_stopped,
    session,
    turret_session,
)
from tests.test_run_refusal_contract import _build_real_protocol
from tests.test_zstack_group_autofocus import focusing_session  # noqa: F401 -- fixture


def _from_a_client(fn):
    """Run ``fn`` on a thread that acts for no activity, as a REST or SDK client's call does."""
    failure = []

    def body():
        try:
            fn()
        except BaseException as ex:  # handed back to the test, which raises it
            failure.append(ex)

    t = threading.Thread(target=body, name='a client')
    t.start()
    t.join(_HOLD_WAIT_S)
    assert not t.is_alive(), 'the client call never returned'
    if failure:
        raise failure[0]


def test_a_clients_stop_of_its_own_move_that_cuts_the_runs_move_ends_the_run(
    session, tmp_path, monkeypatch, centre_posts
):
    """A client's Z move still travels when a run takes the scope; the client
    stops it while the run's X is in flight, and the board-wide STOP cuts X.
    Before: ``completed``, 2 captures, X short of step 0."""
    motion = session.scope.motion
    session.scope.simulation.hold_axis('Z')
    theirs = motion.start_move_z(motion.get_current_position(axis='Z') + 1000.0)
    x_held = session.scope.simulation.hold_axis('X')
    stopped = threading.Event()
    real_start_move_z = motion.start_move_z

    # A timing seam, not a stand-in: the run's Z start supersedes the client's
    # Z, so the stop lands between the run's XY start and its Z start, the
    # window in which the client's move is still its own.
    def start_move_z(*args, **kwargs):
        if not stopped.is_set() and threading.current_thread().name != 'a client':
            stopped.set()
            x_held.result(_HOLD_WAIT_S)
            _from_a_client(theirs.stop)
        return real_start_move_z(*args, **kwargs)

    monkeypatch.setattr(motion, 'start_move_z', start_move_z)
    run = session.protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0, 80.0)), parent_dir=str(tmp_path)
    )
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert stopped.is_set()
    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 0
    assert len(_told_stopped(centre_posts)) == 1


def test_a_press_whose_tell_found_no_run_ends_the_run_its_stop_cut(session, tmp_path, centre_posts):
    """A press tells its listeners before it sends the STOP; a run that starts
    in between is never told, and the STOP cuts its X. Before: ``completed``,
    2 captures."""
    motion = session.scope.motion
    told = threading.Event()
    send = threading.Event()

    def presser_loses_the_cpu():
        if threading.current_thread().name == 'the presser':
            told.set()
            send.wait(_HOLD_WAIT_S)

    motion.add_stop_press_listener(presser_loses_the_cpu)
    presser = threading.Thread(target=motion.stop_motion, name='the presser')
    presser.start()
    assert told.wait(_HOLD_WAIT_S)
    x_held = session.scope.simulation.hold_axis('X')
    run = session.protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0, 80.0)), parent_dir=str(tmp_path)
    )
    x_held.result(_HOLD_WAIT_S)
    send.set()
    presser.join(_HOLD_WAIT_S)
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 0
    assert len(_told_stopped(centre_posts)) == 1


def test_a_clients_stop_that_withholds_the_runs_moves_ends_the_run(
    session, tmp_path, monkeypatch, centre_posts
):
    """A client stops its own move after a run was asked: the stop withholds
    every move asked before it, the run's included. Before: ``completed``
    with every drive withheld."""
    motion = session.scope.motion
    held = session.scope.simulation.hold_axis('X')
    theirs = motion.start_move_xy(100000.0, 50000.0)
    held.result(_HOLD_WAIT_S)
    stopped = threading.Event()
    real_start_move_xy = motion.start_move_xy

    # A timing seam, not a stand-in: the run's first XY start supersedes the
    # client's move, so it waits until the client's stop has landed.
    def start_move_xy(*args, **kwargs):
        stopped.wait(_HOLD_WAIT_S)
        return real_start_move_xy(*args, **kwargs)

    monkeypatch.setattr(motion, 'start_move_xy', start_move_xy)
    run = session.protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0, 80.0)), parent_dir=str(tmp_path)
    )

    _from_a_client(theirs.stop)
    stopped.set()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 0
    assert len(_told_stopped(centre_posts)) == 1


def test_a_clients_stop_that_withholds_the_runs_turret_change_ends_the_run(
    turret_session, tmp_path, centre_posts
):
    """The turret leg: the run's turret change, asked before a client's stop
    of its own turret move, is withheld and raises. Before: counted a
    transient scan failure, ``failed``, ``consecutive_scan_failures``."""
    session = turret_session
    motion = session.scope.motion
    held = session.scope.simulation.hold_axis('T')
    theirs = motion.start_move_turret(3 if motion.turret_status.slot != 3 else 2)
    held.result(_HOLD_WAIT_S)
    run = session.protocol_runner().start_single_scan(
        protocol=_build_real_protocol([{**_steps(60.0)[0], 'Objective': '4x Oly'}]),
        parent_dir=str(tmp_path),
    )

    _from_a_client(theirs.stop)
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert outcome.captures.captured == 0
    assert len(_told_stopped(centre_posts)) == 1


@pytest.mark.slow
def test_a_stall_whose_stop_cuts_the_runs_turret_change_keeps_its_lost_position(
    turret_session, tmp_path
):
    """A stall is a fault, not a person's Stop. The between-scan return's X
    stalls while the next scan's turret turns; the motion monitor gives X up
    and sends the STOP that ends the turret change ``stopped``. The run ends
    on the lost position, in its words, never ``stage_stopped``. Slow: the
    monitor gives an axis up only after its stall window."""
    session = turret_session
    frames = []

    def hold_the_return_and_the_next_turn(*_args):
        frames.append(True)
        if len(frames) == 2:
            session.scope.simulation.hold_axis('X')
            session.scope.simulation.hold_axis('T')

    protocol = _build_real_protocol(
        [
            {**_steps(40.0)[0], 'Objective': '4x Oly'},
            {**_steps(40.0, 80.0)[1], 'Objective': '10x Oly'},
        ]
    )
    protocol.modify_time_params(
        period=datetime.timedelta(seconds=1), duration=datetime.timedelta(seconds=2)
    )
    run = session.protocol_runner().start_protocol(
        protocol,
        parent_dir=str(tmp_path),
        events=RunEvents(frame_captured=hold_the_return_and_the_next_turn),
    )
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert len(frames) == 2
    assert (outcome.status, outcome.reason) == ('failed', 'position_lost')
    assert set(outcome.axes_without_position) == {'X'}, outcome.axes_without_position


def _hold_the_sweeps_nth_z_start(session, monkeypatch, nth):
    """Count the Z moves the sweep starts, and hold the ``nth`` (1-based) as it goes out.

    Returns the sweep's Z targets so far, and an event set once the ``nth``
    is held, carrying the hold's operation. ``nth`` 0 holds nothing.
    """
    motion = session.scope.motion
    real_start_move_z = motion.start_move_z
    targets = []
    armed = threading.Event()
    armed.hold = None

    # A timing seam, not a stand-in: the moves are the sweep's own, through
    # the public member, and the hold is the simulator's.
    def start_move_z(position, *args, **kwargs):
        if threading.current_thread().name == 'autofocus_thread':
            targets.append(position)
            if len(targets) == nth:
                armed.hold = session.scope.simulation.hold_axis('Z')
                armed.set()
        return real_start_move_z(position, *args, **kwargs)

    monkeypatch.setattr(motion, 'start_move_z', start_move_z)
    return targets, armed


def test_a_stop_that_cuts_the_autofocus_final_move_ends_the_run_and_is_told_once(
    focusing_session, monkeypatch, centre_posts
):
    """The sweep's last move, to the focus it found, is cut by a press whose
    tell found no run. The run ends ``stage_stopped`` and tells it once, as
    the run's ending, not also as the sweep's "Move Stopped". Before: the run
    ended ``completed`` with Z a few microns off the focus it reported."""
    session = focusing_session
    motion = session.scope.motion
    motion.move_z(5000.0)
    targets, _ = _hold_the_sweeps_nth_z_start(session, monkeypatch, nth=0)
    first = session.protocol_runner().start_autofocus('BF').result(timeout=COMPLETION_TIMEOUT)
    assert (first.status, first.reason) == ('completed', 'completed')
    sweep_length = len(targets)
    monkeypatch.undo()
    centre_posts.clear()

    motion.move_z(5000.0)
    targets, armed = _hold_the_sweeps_nth_z_start(session, monkeypatch, nth=sweep_length)
    told = threading.Event()
    send = threading.Event()

    def presser_loses_the_cpu():
        if threading.current_thread().name == 'the presser':
            told.set()
            send.wait(_HOLD_WAIT_S)

    motion.add_stop_press_listener(presser_loses_the_cpu)
    presser = threading.Thread(target=motion.stop_motion, name='the presser')
    presser.start()
    assert told.wait(_HOLD_WAIT_S)
    run = session.protocol_runner().start_autofocus('BF')
    try:
        assert armed.wait(COMPLETION_TIMEOUT), (
            f'the sweep made {len(targets)} Z moves, not {sweep_length}'
        )
        armed.hold.result(_HOLD_WAIT_S)
    finally:
        send.set()
        presser.join(_HOLD_WAIT_S)
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert len(targets) == sweep_length
    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert [n.title for n in centre_posts if n.shown] == ['Protocol Stopped']
