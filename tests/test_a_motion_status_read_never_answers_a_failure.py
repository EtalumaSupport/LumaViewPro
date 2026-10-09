# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motion status read that cannot answer refuses or raises; it never answers a value.

``get_target_status`` answered False with no motor board and on a failed
board read, and True for a turret the scope lacks; ``get_limit_switch_status``
answered ``(0, 0)`` -- both switches clear -- for an axis or board the scope
lacks. Over REST each was a 200 a client could not tell from a real answer.
The motion monitor took the failed read as "not arrived", so a board that
stopped answering STATUS_R ended the move ``'stalled'``, telling the person
to check for an obstruction, with a warning on every poll. The simulator
answered a failed read as False and 0, so it hid the failure the board's
driver raises.

Both reads now ask the presence question every motion command asks, and a
failed read raises the driver's HardwareError. The monitor gives a move
whose arrival was never read up as ``'status_unread'``, warning once.
"""

from __future__ import annotations

import logging
import time

import pytest

from drivers.exceptions import HardwareError
from modules.exceptions import HardwareCommandRefusedError, MissingPart, MoveNotCompletedError
from modules.lumascope_api._constants import AxisState
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session


@pytest.fixture
def make_session(tmp_path):
    sessions = []

    def _make(model, **kwargs):
        session = _session(tmp_path, model, **kwargs)
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown()


def _refused(call) -> HardwareCommandRefusedError:
    with pytest.raises(HardwareCommandRefusedError) as exc:
        call()
    return exc.value


def _status_reads(motion, axis):
    return {
        'get_target_status': lambda: motion.get_target_status(axis),
        'get_limit_switch_status': lambda: motion.get_limit_switch_status(axis),
    }


# --- Hardware the scope does not have ---------------------------------------------


@pytest.mark.parametrize('member', ['get_target_status', 'get_limit_switch_status'])
def test_a_turret_the_scope_lacks_is_refused(make_session, member):
    scope = make_session('LS850').scope

    refusal = _refused(_status_reads(scope.motion, 'T')[member])

    assert (refusal.reason, refusal.missing) == ('axis_absent', MissingPart.TURRET)


@pytest.mark.parametrize('member', ['get_target_status', 'get_limit_switch_status'])
def test_a_manual_scope_has_no_motors_to_read(make_session, member):
    scope = make_session('LS620').scope

    refusal = _refused(_status_reads(scope.motion, 'Z')[member])

    assert (refusal.reason, refusal.missing) == ('axis_absent', MissingPart.MOTORS)


@pytest.mark.parametrize('member', ['get_target_status', 'get_limit_switch_status'])
def test_a_pulled_cable_is_refused(make_session, monkeypatch, member):
    scope = make_session('LS850').scope
    monkeypatch.setattr(scope._motion_driver, 'is_connected', lambda: False)

    refusal = _refused(_status_reads(scope.motion, 'Z')[member])

    assert (refusal.reason, refusal.missing) == ('not_connected', MissingPart.MOTOR_CONTROLLER)


def test_every_switch_with_a_pulled_cable_is_refused(make_session, monkeypatch):
    scope = make_session('LS850').scope
    monkeypatch.setattr(scope._motion_driver, 'is_connected', lambda: False)

    refusal = _refused(scope.motion.get_limit_switch_status_all_axes)

    assert refusal.reason == 'not_connected'


# --- A board that does not answer -----------------------------------------------------


def test_a_failed_arrival_read_raises(make_session):
    scope = make_session('LS850').scope
    scope._motion_driver._fail_on.add('STATUS_RZ')

    with pytest.raises(HardwareError):
        scope.motion.get_target_status('Z')


def test_a_failed_switch_read_is_the_documented_unread(make_session):
    scope = make_session('LS850').scope
    scope._motion_driver._fail_on.add('STATUS_RZ')

    assert scope.motion.get_limit_switch_status('Z') == (-1, -1)


def test_a_present_axis_answers_its_read(make_session):
    scope = make_session('LS850').scope

    assert scope.motion.get_target_status('Z') is True
    assert scope.motion.get_limit_switch_status('Z') in {(0, 0), (0, 1), (1, 0), (1, 1)}


# --- The monitor -------------------------------------------------------------------


def test_a_move_whose_arrival_is_never_read_ends_status_unread(make_session, centre_posts, caplog):
    from modules.notification_center import Severity

    scope = make_session('LS850').scope
    motion = scope.motion
    # Give up on the next poll rather than after the production bound.
    motion._MOTION_SETTLE_TIMEOUT_S = 0.0
    scope._motion_driver._fail_on.add('STATUS_RZ')

    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        move = motion.start_move_relative('Z', 50.0)
        # The wait's bound is the same zero: let the monitor end the move
        # first, then the wait raises the object the monitor reported.
        deadline = time.monotonic() + 5.0
        while motion.is_moving() and time.monotonic() < deadline:
            time.sleep(0.02)
        with pytest.raises(MoveNotCompletedError) as exc:
            move.wait()

    assert exc.value.reason == 'status_unread'
    assert exc.value.title == 'Motor Position Unknown'
    assert 'did not report whether it arrived' in str(exc.value)
    assert motion._axis_state['Z'] == AxisState.UNKNOWN
    errors = [n for n in centre_posts if n.severity == Severity.ERROR]
    assert [n.title for n in errors] == ['Motor Position Unknown']
    warned = [r for r in caplog.records if 'Z arrival read failed' in r.getMessage()]
    assert len(warned) == 1, [r.getMessage() for r in warned]


def test_a_home_whose_switch_read_fails_still_homes(make_session):
    session = make_session('LS850', homed=False)
    scope = session.scope
    for axis in scope.capabilities.axes:
        scope._motion_driver._fail_on.add(f'STATUS_R{axis}')

    scope.motion.home()

    assert all(scope.motion.position_is_known(axis) for axis in scope.capabilities.axes)


def test_a_home_whose_controller_goes_as_it_finishes_leaves_no_axis_homing(
    make_session, monkeypatch
):
    # The switch log line after the home's mechanics reads the driver: a
    # refusal raised there would skip the home's state writes and leave
    # every axis HOMING.
    session = make_session('LS850', homed=False)
    scope = session.scope
    driver = scope._motion_driver
    real_home = driver.home

    def home_then_cable_pulled():
        result = real_home()
        monkeypatch.setattr(driver, 'is_connected', lambda: False)
        return result

    monkeypatch.setattr(driver, 'home', home_then_cable_pulled)

    scope.motion.home()

    states = {ax: scope.motion._axis_state[ax] for ax in scope.capabilities.axes}
    assert AxisState.HOMING not in states.values(), states
