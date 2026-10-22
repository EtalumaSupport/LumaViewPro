# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture asked during a several-move command returns no frame from between its moves.

A turret change parks Z, turns the turret and puts Z back; going to a step
turns the turret and then moves X, Y and Z; a click on the stage moves X
and then Y. Frame validity waits out each move, and between two of a
command's moves every axis can be at rest with nothing pending, so a
capture asked then returned a frame of neither where the command began
nor where it ends: Z parked at 0, the new objective at the previous step's
X, the new X at the old Y. ``motion.one_command`` holds the command open,
and no motion change settles while it is.

Each test asks captures back to back while the command runs. A frame from
between the moves is one that arrived after the command's first move was
recorded and no later than its last: once the last move is recorded, its
own change holds the capture until it arrives. The frame's arrival number
is in its record's judgments, the changes' in what ``invalidate`` returned;
neither is the gate's verdict, and neither depends on which frame the
sensor's pipeline delivered, which is the stale-frame plan's step 8.
"""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from modules.exceptions import MoveStoppedError
from modules.frame_validity import FrameValidity

# How late each drive is sent: the gap between two moves on a loaded host.
# On an idle one the gap is about one frame, and the capture's own recheck
# of the changes made during its grab usually catches it.
DRIVE_DELAY_S = 0.25
_WAIT_S = 5.0


def _load_the_host(monkeypatch, motion) -> None:
    """Send every drive DRIVE_DELAY_S late from here on.

    A stand-in by design: the simulator cannot load the host, and a loaded
    host is what widens the gap between a command's moves from a frame to
    many. The drive itself is the production one.
    """
    send = motion._send_drive

    def late(*args, **kwargs):
        time.sleep(DRIVE_DELAY_S)
        return send(*args, **kwargs)

    monkeypatch.setattr(motion, '_send_drive', late)


def _between(scope, monkeypatch, command) -> tuple[int, list[int]]:
    """Run *command* while capturing back to back; the motion changes it
    recorded, and the arrival number of each frame from between its moves.

    The changes are read as ``invalidate`` records them: a stand-in by
    design, an observer that passes each call through unchanged.
    """
    fv = scope.frame_validity
    recorded = []
    invalidate = fv.invalidate

    def observed(source, *, began_ns):
        change = invalidate(source, began_ns=began_ns)
        if source in FrameValidity.MOTION_SOURCES:
            recorded.append(change.at_seq)
        return change

    monkeypatch.setattr(fv, 'invalidate', observed)
    arrived = []
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        running = pool.submit(command)
        while not running.done():
            scope.imaging.capture_and_wait()
            arrived.append(scope.imaging.last_capture_info['frame_record'].judgments[0].frame_seq)
        running.result()
    first, last = min(recorded), max(recorded)
    return len(recorded), [seq for seq in arrived if first < seq <= last]


@pytest.fixture
def scope(sim_turreted_session):
    scope = sim_turreted_session.scope
    scope.motion.move_turret(1)
    scope.motion.move_xy(x_um=20_000.0)
    scope.motion.move_xy(y_um=20_000.0)
    scope.motion.move_z(3_000.0)
    return scope


def test_a_turret_change_returns_no_frame_from_between_its_moves(scope, monkeypatch):
    _load_the_host(monkeypatch, scope.motion)

    moves, between = _between(scope, monkeypatch, lambda: scope.motion.move_turret(2))

    assert moves == 3  # the park, the turret, the restore
    assert between == []


def test_going_to_a_step_on_another_objective_returns_no_frame_from_between_its_moves(
    sim_turreted_session, scope, monkeypatch
):
    session = sim_turreted_session
    session.set_layer_acquire('BF', 'image')
    protocol = session.create_empty_protocol()
    session.add_step(protocol)
    scope.motion.move_turret(2)
    scope.motion.move_xy(x_um=23_000.0)
    scope.motion.move_z(2_500.0)
    session.add_step(protocol)
    session.go_to_step(protocol, 0)
    _load_the_host(monkeypatch, scope.motion)

    moves, between = _between(scope, monkeypatch, lambda: session.go_to_step(protocol, 1))

    assert moves == 5  # the park, the turret, X, Y and Z
    assert between == []


def _click(motion, x_um, y_um) -> None:
    """X then Y as one command, as the stage map's click moves them."""
    with motion.one_command(('X', 'Y'), then='move the stage'):
        moves = (
            motion.start_move_xy(x_um=x_um),
            motion.start_move_xy(y_um=y_um),
        )
    for move in moves:
        move.result(timeout=_WAIT_S)


def test_a_two_axis_command_returns_no_frame_from_between_its_moves(scope, monkeypatch):
    _load_the_host(monkeypatch, scope.motion)

    moves, between = _between(scope, monkeypatch, lambda: _click(scope.motion, 20_050.0, 20_050.0))

    assert moves == 2
    assert between == []


def test_a_command_gives_an_in_force_time_only_to_the_changes_it_held(scope):
    """A turret change's turret change, held while Z was put back, is given
    the close's time; the X and Y move before it keeps its own."""
    fv = scope.frame_validity
    xy_before = fv.changes()['xy_move'].in_force_ns
    assert xy_before is not None

    scope.motion.move_turret(2)

    turret = fv.changes()['turret']
    assert fv.changes()['xy_move'].in_force_ns == xy_before
    assert turret.in_force_ns is not None and turret.in_force_ns >= turret.written_ns


def test_a_stop_during_a_turret_changes_park_closes_the_command(scope):
    parked = scope.simulation.hold_axis('Z')
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        change = pool.submit(scope.motion.move_turret, 2)
        assert parked.wait(_WAIT_S), 'the park never got part of the way'
        scope.motion.stop_motion()
        with pytest.raises(MoveStoppedError):
            change.result(timeout=_WAIT_S)
    scope.simulation.clear()

    # Held open, the capture would drain until its wall bound and fail.
    scope.imaging.capture_and_wait()

    assert scope.imaging.frames_until_valid() == 0


def _slot_when_the_turret_came_into_force(scope, monkeypatch, command) -> int | None:
    """Run *command*; the slot the turret answered at the moment its change
    was given its in-force time.

    The stamps are read as ``motion_settled`` is called: a stand-in by
    design, an observer that passes each call through unchanged.
    """
    fv = scope.frame_validity
    slot_at = {}
    motion_settled = fv.motion_settled

    def observed(source, at_ns, *, moving_seen_ns):
        if source == 'turret':
            slot_at[at_ns] = scope.motion.get_turret_slot()
        motion_settled(source, at_ns, moving_seen_ns=moving_seen_ns)

    monkeypatch.setattr(fv, 'motion_settled', observed)
    command()
    return slot_at[fv.changes()['turret'].in_force_ns]


@pytest.mark.parametrize(
    ('command', 'slot'),
    [
        (lambda motion: motion.move_turret(2), 2),
        (lambda motion: motion.home('T'), 1),
        (lambda motion: motion.home(), 1),
    ],
    ids=['move_turret', 'turret home', 'whole-scope home'],
)
def test_a_turret_change_is_in_force_only_once_the_turret_answers_its_slot(
    scope, monkeypatch, command, slot
):
    """A frame counts for a turret change once the change is in force, and
    the objective a frame is made under is the slot's, so the in-force time
    comes after the slot is recorded: before it, a frame that counts can be
    made with no objective in the light path. Each command that turns the
    turret records its slot inside the command that stamps it; each is its
    own code path, so each is its own case."""
    scope.motion.move_turret(3)

    answered = _slot_when_the_turret_came_into_force(
        scope, monkeypatch, lambda: command(scope.motion)
    )

    assert answered == slot
