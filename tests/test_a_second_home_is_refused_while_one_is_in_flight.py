# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A home asked while a home is in flight is refused when it is asked.

The Home button queued the whole home on the io lane, so a second press
waited behind the first and then ran a second full home: 70 s, then 38 s
more on an LS720, with a run started meanwhile refused "still homing". Two
API callers asking ``home()`` at once did the same. A home now holds the
session's activity claim from the moment it is asked until its body ends:
another home, by any caller, on any axis, is refused ``'home_in_flight'``
at once and nothing is queued. ``start_home`` asks the same home without
waiting, so the button's second press reaches the API while the first is
in flight. The guard is the Session's, so the scope is built through one.
"""

from __future__ import annotations

import threading

import pytest

from modules.exceptions import HardwareCommandRefusedError, HomingFailedError
from tests.scope_fakes import build_scope, record_turret_answer
from tests.settings_fixtures import complete_settings

HOME_S = 60.0


@pytest.fixture(params=['LS720', 'LS850', 'LS850T'])
def scope(request, tmp_path):
    from modules.scope_session import ScopeSession

    # Not homed first: the home under test is the scope's first.
    session = ScopeSession.create(
        complete_settings(
            microscope=request.param,
            live_folder=str(tmp_path),
            objective_id='10x Oly',
            stage_offset={'x': 5500.0, 'y': 4000.0},
        ),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        yield record_turret_answer(session.scope)
    finally:
        session.shutdown()


class _HeldHome:
    """The board's home, held inside the driver until released, and counted."""

    def __init__(self, scope, monkeypatch):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.count = 0
        real = scope._motion_driver.home

        def held(*args, **kwargs):
            self.count += 1
            self.entered.set()
            assert self.release.wait(HOME_S), 'the held home was never released'
            return real(*args, **kwargs)

        monkeypatch.setattr(scope._motion_driver, 'home', held)


def _refused_in_flight(ask):
    with pytest.raises(HardwareCommandRefusedError) as refused:
        ask()
    assert refused.value.reason == 'home_in_flight'
    assert str(refused.value) == 'The microscope is already homing. Wait for the home to finish.'


class TestOneHomeAtATime:
    def test_a_second_caller_is_refused_and_the_board_homes_once(self, scope, monkeypatch):
        board = _HeldHome(scope, monkeypatch)
        first = threading.Thread(target=scope.motion.home)
        first.start()
        assert board.entered.wait(HOME_S)

        _refused_in_flight(scope.motion.home)
        _refused_in_flight(lambda: scope.motion.home('Z'))
        _refused_in_flight(scope.motion.start_home)

        board.release.set()
        first.join(HOME_S)
        assert not first.is_alive()
        assert board.count == 1
        assert scope.motion.axes_without_position() == {}

    # Two simulated homes, 6-8 s on the LS850 and LS850T.
    @pytest.mark.slow
    def test_a_started_home_refuses_the_next_press_until_it_settles(self, scope, monkeypatch):
        board = _HeldHome(scope, monkeypatch)
        started = scope.motion.start_home()
        assert board.entered.wait(HOME_S)

        _refused_in_flight(scope.motion.start_home)

        board.release.set()
        assert started.result(timeout=HOME_S) is None
        assert board.count == 1

        scope.motion.start_home().result(timeout=HOME_S)
        assert board.count == 2


class TestTheClaimEndsWithItsHome:
    def test_a_failed_home_lets_the_next_one_run(self, scope, monkeypatch):
        real = scope._motion_driver.home
        monkeypatch.setattr(scope._motion_driver, 'home', lambda: False)
        with pytest.raises(HomingFailedError):
            scope.motion.home()
        with pytest.raises(HomingFailedError):
            scope.motion.start_home().result(timeout=HOME_S)

        monkeypatch.setattr(scope._motion_driver, 'home', real)
        scope.motion.home()
        assert scope.motion.axes_without_position() == {}


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def _from_another_thread(call):
    outcome = []

    def ask():
        try:
            call()
            outcome.append(None)
        except Exception as e:
            outcome.append(e)

    t = threading.Thread(target=ask)
    t.start()
    t.join(HOME_S)
    return outcome[0]


class TestAHomeTheLaneRefusesHoldsNothing:
    @pytest.mark.parametrize('ask', ['home', 'start_home'])
    def test_refused_while_a_diagnostic_holds_the_scope_then_homes(self, sim_session, ask):
        motion = sim_session.scope.motion
        call = motion.home if ask == 'home' else lambda: motion.start_home().result(HOME_S)
        with sim_session.diagnostic_claim():
            refused = _from_another_thread(call)
        assert isinstance(refused, HardwareCommandRefusedError), refused
        assert refused.reason == 'exclusive_activity_running'

        assert _from_another_thread(call) is None


class TestTheLaneNamesTheHome:
    """The lane names a task by its action in its log lines (a slow task, a
    refusal); the claimed body keeps the name of the home it runs."""

    @pytest.mark.parametrize('axis', ['ALL', 'Z', 'T'])
    def test_the_body_carries_its_homes_name(self, axis):
        s = build_scope(simulate=True, sim_model='LS850T', source_path='.', register_atexit=False)
        try:
            impl, _ = s.motion._home_body(axis)
            body, release_if_unrun, _taking = s.motion.claim_home(impl)
            release_if_unrun()
            assert body.__name__ == impl.__name__
        finally:
            s.disconnect()
