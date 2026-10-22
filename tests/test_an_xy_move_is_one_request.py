# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An X and Y move is one request: the caller says what it wants, the API decides how.

``move_xy`` / ``start_move_xy`` move X and Y to a place, ``move_xy_by`` /
``start_move_xy_by`` by a distance. Every axis given is refused before
either drives, so a refusal moves neither: as two calls, the first axis
moved and the second was refused, leaving the stage at a place nobody
asked for. An axis given as None is not read, refused or written, so an X
move leaves Y as it is, travelling or unknown. With neither given the
request is done at once.

Each test drives the simulated scope through the public members.
"""

from __future__ import annotations

import pytest

from drivers.tmcm6110 import Tmcm6110Board
from modules.exceptions import (
    ArgumentRefusedError,
    AxisStateUnknownError,
    HardwareCommandRefusedError,
    MoveFaultError,
    PositionOutOfRangeError,
)
from modules.lumascope_api import AxisState
from tests.scope_fakes import build_scope, home_sim_scope

_START = {'X': 60_000.0, 'Y': 40_000.0}


@pytest.fixture
def scope():
    scope = home_sim_scope(build_scope(simulate=True, sim_model='LS850', register_atexit=False))
    scope.motion.move_xy(_START['X'], _START['Y'])
    yield scope
    scope.disconnect()


def _targets(scope):
    return {axis: scope.motion.get_target_position(axis) for axis in ('X', 'Y')}


def _lose_y(scope, monkeypatch):
    """Y's move halts and never arrives, and the monitor gives Y up as stalled: Y is UNKNOWN."""
    monkeypatch.setattr(scope.motion, '_MOVE_STALL_WINDOW_S', 0.1)
    held = scope.simulation.hold_axis('Y')
    y = scope.motion.start_move_xy(y_um=_START['Y'] + 2_000.0)
    assert held.wait(10.0)
    with pytest.raises(MoveFaultError) as lost:
        y.result()
    assert lost.value.reason == 'stalled'
    scope.simulation.clear()
    assert scope.motion.get_axis_state('Y') == AxisState.UNKNOWN


class TestARefusalMovesNeither:
    def test_a_place_with_y_past_travel_leaves_x_where_it_was(self, scope):
        y_max = scope.motion.get_axis_limits('Y').max_um

        with pytest.raises(PositionOutOfRangeError) as refused:
            scope.motion.start_move_xy(_START['X'] - 500.0, y_max + 1_000.0)

        assert (refused.value.reason, refused.value.axis) == ('position_out_of_range', 'Y')
        assert _targets(scope) == _START
        assert scope.motion.get_axis_state('X') == AxisState.IDLE

    def test_a_distance_taking_y_past_travel_leaves_x_where_it_was(self, scope):
        y_max = scope.motion.get_axis_limits('Y').max_um

        with pytest.raises(PositionOutOfRangeError) as refused:
            scope.motion.start_move_xy_by(100.0, y_max)

        assert (refused.value.reason, refused.value.axis) == ('position_out_of_range', 'Y')
        assert _targets(scope) == _START

    def test_a_y_that_does_not_know_its_position_leaves_x_where_it_was(self, scope, monkeypatch):
        _lose_y(scope, monkeypatch)

        with pytest.raises(AxisStateUnknownError) as refused:
            scope.motion.start_move_xy(_START['X'] - 500.0, _START['Y'])

        assert refused.value.reason == 'position_unknown'
        assert scope.motion.get_target_position('X') == _START['X']

    def test_a_y_that_is_no_number_is_refused_by_its_name(self, scope):
        with pytest.raises(ArgumentRefusedError) as refused:
            scope.motion.start_move_xy(_START['X'] - 500.0, float('nan'))

        assert (refused.value.reason, refused.value.argument) == ('not_a_number', 'y_um')
        assert _targets(scope) == _START


class TestAnAxisNotAskedForIsLeftAsItIs:
    def test_an_x_move_leaves_a_travelling_y_to_arrive_as_its_own(self, scope):
        held = scope.simulation.hold_axis('Y')
        y = scope.motion.start_move_xy(y_um=_START['Y'] + 2_000.0)
        assert held.wait(10.0)

        scope.motion.move_xy(x_um=_START['X'] - 500.0)

        assert scope.motion.get_axis_state('Y') == AxisState.MOVING
        scope.simulation.clear()
        y.result()
        assert _targets(scope) == {'X': _START['X'] - 500.0, 'Y': _START['Y'] + 2_000.0}

    def test_an_x_move_moves_x_while_y_does_not_know_its_position(self, scope, monkeypatch):
        _lose_y(scope, monkeypatch)

        scope.motion.move_xy(x_um=_START['X'] - 500.0)

        assert scope.motion.get_target_position('X') == _START['X'] - 500.0
        assert scope.motion.get_axis_state('Y') == AxisState.UNKNOWN

    def test_a_distance_for_y_alone_leaves_x(self, scope):
        scope.motion.move_xy_by(dy_um=-200.0)

        assert _targets(scope) == {'X': _START['X'], 'Y': _START['Y'] - 200.0}

    def test_neither_given_is_done_at_once_and_moves_nothing(self, scope):
        moved = scope.motion.start_move_xy()

        assert moved.done() and moved.result() is None
        assert _targets(scope) == _START


class TestTheLidOnAClassic:
    """The LS720's driver reads its lid before each X or Y command."""

    @pytest.fixture
    def classic(self):
        scope = build_scope(
            simulate=True, sim_model='LS720', source_path='.', register_atexit=False
        )
        assert isinstance(scope._motion_driver, Tmcm6110Board)
        scope.motion.home()
        yield scope
        scope.disconnect()

    def test_a_lid_open_before_the_request_drives_nothing(self, classic):
        before = {axis: classic._motion_driver.current_pos(axis) for axis in ('X', 'Y')}
        classic.simulation.interlock(lid_open=True)

        with pytest.raises(HardwareCommandRefusedError) as refused:
            classic.motion.move_xy(1_000.0, 2_000.0)

        assert refused.value.reason == 'lid_open'
        assert {a: classic._motion_driver.current_pos(a) for a in ('X', 'Y')} == before
        assert classic.motion.get_axis_state('X') == AxisState.IDLE
        assert classic.motion.get_axis_state('Y') == AxisState.IDLE

    def test_a_lid_opened_after_x_is_commanded_stops_x_and_drives_no_y(self, classic, monkeypatch):
        driver = classic._motion_driver
        y_before = driver.current_pos('Y')
        real_write = driver.move_abs_pos

        # A stand-in by design: the simulator opens its lid when asked, and
        # this times the opening to the one exchange between X's command and
        # Y's lid read.
        def write_then_open_after_x(axis, pos):
            real_write(axis, pos)
            if axis == 'X':
                classic.simulation.interlock(lid_open=True)

        monkeypatch.setattr(driver, 'move_abs_pos', write_then_open_after_x)
        with pytest.raises(HardwareCommandRefusedError) as refused:
            classic.motion.start_move_xy(20_000.0, 2_000.0)

        assert refused.value.reason == 'lid_open'
        classic.motion.wait_until_finished_moving(timeout_s=10.0)
        assert classic.motion.get_axis_state('X') == AxisState.IDLE
        assert driver.current_pos('X') != pytest.approx(20_000.0, abs=1.0), 'X was not stopped'
        assert classic.motion.get_axis_state('Y') == AxisState.IDLE
        assert driver.current_pos('Y') == y_before


class TestOverRest:
    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient

        from modules.scope_session import ScopeSession
        from rest.app import build_app
        from tests.settings_fixtures import complete_settings

        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            home_sim_scope(session.scope)
            with TestClient(build_app(session)) as client:
                yield client, session.scope.motion
        finally:
            session.shutdown()

    def test_a_move_of_both_axes_moves_both(self, client):
        client, motion = client

        answer = client.post('/api/v1/scope/motion/move_xy', json={'x_um': 61_000, 'y_um': 41_000})

        assert answer.status_code == 200, answer.text
        assert {a: motion.get_target_position(a) for a in ('X', 'Y')} == {
            'X': 61_000.0,
            'Y': 41_000.0,
        }

    def test_an_empty_body_moves_nothing(self, client):
        client, motion = client
        before = {a: motion.get_target_position(a) for a in ('X', 'Y')}

        answer = client.post('/api/v1/scope/motion/move_xy', json={})

        assert answer.status_code == 200, answer.text
        assert {a: motion.get_target_position(a) for a in ('X', 'Y')} == before

    @pytest.mark.parametrize(
        'member', ['move_absolute', 'move_relative', 'start_move_absolute', 'start_move_relative']
    )
    def test_the_single_axis_routes_are_gone(self, client, member):
        client, _ = client

        answer = client.post(f'/api/v1/scope/motion/{member}', json={'axis': 'Z', 'position': 1})

        assert answer.status_code == 404, answer.text
