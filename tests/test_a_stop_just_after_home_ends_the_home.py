# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Stop pressed just after Home ends that home, on every motor board.

The API's home reads the limit switches before it asks the driver to
home, and the driver forgets a stop that came before its home began: the
LS720's 6110 clears its abort at the start of its home, the EL-0940's
firmware takes no STOP before its HOME. A Stop in that window was lost and
the whole home ran -- about 90 s on an LS720. The API asks its own stop
generation immediately before the driver's home, so the home ends stopped
with nothing moved.

Once the board is homing, a Stop ends the home only on a board that can
take one mid-home. The LS720's host runs its home a command at a time and
each reads the stop; the EL-0940's driver holds the board's exchange until
the home replies, so a STOP sent then waited out the whole home, stopped
nothing, and the home reported success. There the Stop is refused.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

from modules.exceptions import HardwareCommandRefusedError, HomingStoppedError
from modules.notification_center import OutcomeKind, outcome_of
from modules.scope_session import ScopeSession
from tests.scope_fakes import (
    bind_settings_like_a_session,
    build_scope,
    home_sim_scope,
    record_turret_answer,
)
from tests.settings_fixtures import complete_settings


@pytest.fixture(params=['LS720', 'LS850', 'LS850T'])
def homed(request):
    s = build_scope(simulate=True, sim_model=request.param, source_path='.', register_atexit=False)
    record_turret_answer(s)
    bind_settings_like_a_session(s, objective_id='10x Oly', stage_offset={'x': 5500.0, 'y': 4000.0})
    s.motion.home()
    s.motion.move_absolute('X', 30_000)
    s.motion.move_absolute('Z', 2_000)
    yield s
    s.disconnect()


def _press_stop_during_the_apis_switch_read(scope, monkeypatch):
    """The person's Stop lands while the API reads the switches before the home."""
    driver = scope._motion_driver
    read = driver.limit_switch_status
    pressed = []

    def read_then_stop(*args, **kwargs):
        answer = read(*args, **kwargs)
        if not pressed:
            pressed.append(True)
            scope.motion.stop_motion()
        return answer

    monkeypatch.setattr(driver, 'limit_switch_status', read_then_stop)


@pytest.mark.parametrize('axis', ['ALL', 'Z', 'T'])
def test_a_stop_before_the_driver_homes_ends_the_home_with_nothing_moved(homed, monkeypatch, axis):
    if axis != 'ALL' and axis not in homed.capabilities.axes:
        pytest.skip(f'this scope has no {axis}')
    before = homed.motion.get_current_positions()
    _press_stop_during_the_apis_switch_read(homed, monkeypatch)

    with pytest.raises(HomingStoppedError) as ended:
        homed.motion.home(axis)

    assert (ended.value.reason, ended.value.title) == ('stopped', 'Homing Stopped')
    assert outcome_of(ended.value).kind == OutcomeKind.NOTICE
    assert ended.value.axes == ()
    assert 'before it began' in str(ended.value)
    assert homed.motion.axes_without_position() == {}
    assert homed.motion.get_current_positions() == before


# Whether the model's motor board can take a STOP while it homes.
_STOPS_A_HOME = {'LS720': True, 'LS850': False, 'LS850T': False}


def _announce_the_board_homing(scope, monkeypatch, axis):
    """An Event set once the board has begun the home, before it ends."""
    driver = scope._motion_driver
    homing = threading.Event()
    if scope.capabilities.model == 'LS720':
        # The first step of the TMCM-6110's home after it clears the abort
        # a stop sets: a stop before it would be forgotten by design.
        name = '_set_switch_polarities'
    elif axis == 'T':
        # The turret home's first drive is its Z park.
        name = 'move_abs_pos'
    else:
        name = {'ALL': 'home', 'Z': 'zhome'}[axis]
    real = getattr(driver, name)

    def announced(*args, **kwargs):
        homing.set()
        return real(*args, **kwargs)

    monkeypatch.setattr(driver, name, announced)
    return homing


@pytest.mark.parametrize('axis', ['ALL', 'Z', 'T'])
def test_a_stop_while_the_board_homes_ends_it_only_where_the_board_can_take_one(
    homed, monkeypatch, axis
):
    """Through the started home's ``stop()``, which is ``stop_motion``."""
    if axis != 'ALL' and axis not in homed.capabilities.axes:
        pytest.skip(f'this scope has no {axis}')
    homing = _announce_the_board_homing(homed, monkeypatch, axis)

    home = homed.motion.start_home(axis)
    assert homing.wait(30)
    if _STOPS_A_HOME[homed.capabilities.model]:
        home.stop()
        with pytest.raises(HomingStoppedError) as ended:
            home.result(60)
        assert (ended.value.reason, ended.value.title) == ('stopped', 'Homing Stopped')
        assert outcome_of(ended.value).kind == OutcomeKind.NOTICE
        assert ended.value.axes
    else:
        with pytest.raises(HardwareCommandRefusedError) as refused:
            home.stop()
        assert (refused.value.reason, refused.value.member) == ('home_not_stoppable', 'stop_motion')
        assert home.result(60) is None
        assert homed.motion.axes_without_position() == {}


def test_the_el0940_driver_refuses_a_stop_while_it_homes(monkeypatch):
    """The real ``MotorBoard`` driver, on the simulated EL-0940 firmware."""
    scope = build_scope(
        simulate=True, sim_model='LS850', sim_tier='firmware', register_atexit=False
    )
    try:
        homing = _announce_the_board_homing(scope, monkeypatch, 'Z')
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            home = pool.submit(scope.motion.home, 'Z')
            assert homing.wait(30)
            with pytest.raises(HardwareCommandRefusedError) as refused:
                scope.motion.stop_motion()
            assert (refused.value.reason, refused.value.member) == (
                'home_not_stoppable',
                'stop_motion',
            )
            assert home.result(60) is None
    finally:
        scope.disconnect()


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _stop_as_the_scope_is_let_go(session, started):
    """Ask ``started[0].stop()`` from the run-state listener the home's
    release calls: the home is over and the scope free, but its operation
    is not yet done. Returns the commands the board was sent from then on."""
    driver = session.scope._motion_driver
    sent = []
    real = driver.exchange_command
    asked = []

    def stop_when_free():
        if started and not asked and session.exclusive_activity is None:
            asked.append(True)
            sent.clear()
            started[0].stop()

    def recorded(command, *args, **kwargs):
        sent.append(command)
        return real(command, *args, **kwargs)

    driver.exchange_command = recorded
    session.add_run_state_listener(stop_when_free)
    return sent, asked


def test_a_homes_stop_once_it_has_let_go_of_the_scope_sends_nothing(session):
    """A home's ``stop()`` stops the board only while the home drives: one
    asked once it has let go of the scope, before its operation is done,
    would halt whatever took the scope next."""
    started = []
    sent, asked = _stop_as_the_scope_is_let_go(session, started)
    started.append(session.scope.motion.start_home('Z'))

    assert started[0].result(60) is None
    assert asked, 'the listener never saw the scope let go'
    assert 'STOP' not in sent


def test_the_startup_motions_stop_once_it_has_let_go_of_the_scope_sends_nothing(session):
    started = []
    sent, asked = _stop_as_the_scope_is_let_go(session, started)
    started.append(session.start_application_session())

    assert started[0].result(60) is None
    assert asked, 'the listener never saw the scope let go'
    assert 'STOP' not in sent
