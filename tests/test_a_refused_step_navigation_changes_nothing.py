# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A navigation the scope cannot perform leaves the protocol alone.

`go_to_step` used to commit the step pointer as its second act, before
anything had asked whether the scope could reach the glass that step
names. When it could not, the old code logged, raised a dialog, and then
moved X, Y and Z anyway -- to a step it could not put the right objective
in front of.

The pointer is the part that matters, and it is why it is written only
once the Session has gone to the step. `modify_step_ex` and
`insert_step_ex` address a step BY that pointer and fill it from the LIVE
stage, so a pointer left on a step the user never arrived at means the
next edit writes the previous step's coordinates into it. The protocol
file is then wrong, on disk, with nothing in it saying so.

The rule is the API's, so the click goes to a simulated turret scope's
Session and is refused there: the turret's assignment decides, exactly as
it does for a run. The Session's own move is
``tests/test_going_to_a_step_is_the_sessions_move.py``.
"""

from __future__ import annotations

import datetime
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import ui.step_navigation as step_navigation
import ui.ui_helpers as ui_helpers
from modules.protocol import Protocol
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import ENQUEUED
from tests.installation_fixtures import shipped_grids
from tests.pool_fakes import run_task_now
from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
from tests.stock_channels import LED1


ON_TURRET = '10x Oly'
NOT_ON_TURRET = '60x w/collar'

# Every test here clicks on a turret scope that carries the test objectives.
pytestmark = pytest.mark.parametrize(
    'sim_session',
    [{'microscope': 'LS850T', 'turret_objectives': dict(TEST_TURRET_OBJECTIVES)}],
    indirect=True,
)


def _step(x: float, objective: str) -> dict:
    return {
        'Name': 'step',
        'X': x,
        'Y': 20.0,
        'Z': 4000.0,
        'Auto_Focus': False,
        'Color': LED1,
        'False_Color': False,
        'Illumination': 50.0,
        'Gain': 0.0,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': 1,
        'Objective': objective,
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': True,
        'Tile Group ID': 0,
        'Z-Stack Group ID': 0,
        'Acquire': 'image',
        'Video Config': {'duration': 1.0, 'fps': 5},
        'Stim_Config': {},
        'Step Index': 0,
        'Auto_Named': False,
        'Label': '',
        'Channel Group ID': 0,
        'Turret Slot': -1,
    }


def _protocol(objective: str) -> Protocol:
    """Two steps apart in X, the second naming *objective*; the first the turret carries."""
    return Protocol(
        tiling_grids=shipped_grids(),
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame([_step(20.0, ON_TURRET), _step(40.0, objective)]),
            'period': datetime.timedelta(minutes=1.0),
            'duration': datetime.timedelta(hours=1.0),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


@pytest.fixture
def nav_env(gui_ctx, monkeypatch):
    """The real go_to_step over the GUI's context on a homed simulated turret scope.

    The panel and the layer widgets are the test's own: the Kivy tree never
    runs here. The click's task runs on the worker pool at once, and every
    go-to the Session is asked for is recorded with what it raised, then
    made.
    """
    session = gui_ctx.session
    home_sim_scope(session.scope)
    pool = MagicMock()

    def _put(task):
        run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _put
    gui_ctx.worker_pool = pool
    protocol_settings = MagicMock()
    protocol_settings.curr_step = 3
    layer_obj = MagicMock()
    gui_ctx.motion_settings = SimpleNamespace(
        ids={'protocol_settings_id': protocol_settings, 'verticalcontrol_id': MagicMock()},
        update_xy_stage_control_gui=MagicMock(),
    )
    gui_ctx.image_settings = SimpleNamespace(
        layer_lookup=MagicMock(return_value=layer_obj),
        ids={'toggle_imagesettings': SimpleNamespace(state='down')},
        place_settings=MagicMock(),
    )
    # No Kivy app runs, so no lock record locks the controls.
    monkeypatch.setattr(sys.modules['kivy.app'].App, 'get_running_app', lambda: None)
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn: fn(0))
    monkeypatch.setattr(step_navigation, '_schedule_ui', lambda fn: fn(0))
    went = []
    real_go = ScopeSession.start_go_to_step

    def _start_go_to_step(self, protocol, step_index):
        try:
            op = real_go(self, protocol, step_index)
        except Exception as e:
            went.append((step_index, e))
            raise
        went.append((step_index, None))
        return op

    monkeypatch.setattr(ScopeSession, 'start_go_to_step', _start_go_to_step)
    return SimpleNamespace(
        session=session,
        protocol_settings=protocol_settings,
        went=went,
        layer_obj=layer_obj,
    )


def _navigate(env, objective: str, *, step_index: int = 1, include_move: bool = True) -> Protocol:
    protocol = _protocol(objective)
    # The panel shows this protocol: a completed move lands only on the
    # protocol and step list it was sent for.
    env.protocol_settings._protocol = protocol
    step_navigation.go_to_step(protocol, step_index=step_index, include_move=include_move)
    return protocol


def _targets(env) -> dict:
    return env.session.scope.motion.get_target_positions()


class TestARefusedNavigationIsANoOp:
    def test_the_step_pointer_does_not_move(self, nav_env):
        """The whole point: an edit addressed by this pointer would corrupt."""
        _navigate(nav_env, NOT_ON_TURRET)

        [(_, refused)] = nav_env.went
        assert refused.reason == 'turret_objectives_unassigned'
        assert nav_env.protocol_settings.curr_step == 3, (
            'the pointer moved to a step the scope never navigated to; '
            'modify_step_ex would now fill that step from the live stage'
        )

    def test_the_layer_is_not_applied(self, nav_env):
        """A refused navigation shows nothing of the step it did not reach."""
        before = _targets(nav_env)

        _navigate(nav_env, NOT_ON_TURRET)

        assert nav_env.layer_obj.set_step_state.call_count == 0
        assert _targets(nav_env) == before

    def test_a_refused_navigation_does_not_propagate_to_its_caller(self, nav_env):
        """The reporter told the user; a step button has nothing left to do."""
        _navigate(nav_env, NOT_ON_TURRET)


class TestAnAdmissibleNavigationStillWorks:
    def test_the_pointer_moves(self, nav_env):
        _navigate(nav_env, ON_TURRET)

        assert nav_env.protocol_settings.curr_step == 1

    def test_the_session_goes_to_the_step(self, nav_env):
        protocol = _navigate(nav_env, ON_TURRET)

        assert nav_env.went == [(1, None)]
        step = nav_env.session.scope.protocols.step_targets(protocol, 1)
        assert (_targets(nav_env)['X'], _targets(nav_env)['Y']) == pytest.approx(
            (step.x_um, step.y_um)
        )

    def test_an_index_before_the_first_step_is_the_sessions_to_refuse(self, nav_env):
        """A typed 0 arrives as -1; the panel clears its pointer only for a
        protocol with no steps, never for a number the protocol lacks."""
        _navigate(nav_env, ON_TURRET, step_index=-1)

        [(index, refused)] = nav_env.went
        assert (index, refused.reason) == (-1, 'no_such_step')
        assert nav_env.protocol_settings.curr_step == 3

    def test_the_gui_makes_no_apply_of_its_own_once_the_session_has_gone(self, nav_env):
        """The Session's go-to puts the step's layer on the camera itself."""
        _navigate(nav_env, ON_TURRET)

        nav_env.layer_obj.apply_settings.assert_not_called()


class TestARunNavigationOnlyDisplays:
    def test_a_run_navigation_asks_the_session_for_no_move(self, nav_env):
        """The run navigates with include_move=False; it moved the scope itself."""
        before = _targets(nav_env)

        _navigate(nav_env, ON_TURRET, step_index=0, include_move=False)

        assert nav_env.went == []
        assert _targets(nav_env) == before
        assert nav_env.protocol_settings.curr_step == 0
