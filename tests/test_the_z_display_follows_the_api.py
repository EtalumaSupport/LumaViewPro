# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Z display follows the API, run or not; the engine names no display hook.

The Z slider was driven by a widget writer handed to the session
(af_ui_update_func) and called by the autofocus runner and the step runner,
while every Z move_position a run sent was dropped by update_gui's run
gate. An autofocus that gave up restored the stage with no call, so the
slider kept the last sample's Z while the stage was back at the start. The
turret display followed a run only through move_position('T').

Now the listener bridge draws every Z position change, the slider at the
API's Z target and the text at the polled Z, and every turret change, as the
motion API announces them; the engine calls no display.
"""

import ast
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.lumascope_api import AxisPosition, AxisState, TurretStatus
from tests.ast_seams import iter_package_modules
from tests.scope_fakes import home_sim_scope
from ui.listener_bridge import UIListenerBridge
from ui.vertical_control import VerticalControl


class _Box:
    def __init__(self):
        self.focus = False
        self.text = ''


class _Slider:
    def __init__(self):
        self.value = 0.0
        self.user_interacting = False


@pytest.fixture
def control(gui_ctx):
    """The Z panel over a simulated session; its Z target is where Z was last sent."""
    if gui_ctx.scope.capabilities.has_focus:
        home_sim_scope(gui_ctx.scope)
    ctrl = VerticalControl.__new__(VerticalControl)
    ctrl.ids = {'z_position_id': _Box(), 'obj_position': _Slider()}
    ctrl.motion = gui_ctx.scope.motion
    return ctrl


def test_the_slider_shows_the_target_and_the_text_the_polled_z(control):
    control.motion.move_z(2500.0)

    control.show_z(1800.0)

    assert control.ids['obj_position'].value == 2500.0
    assert control.ids['z_position_id'].text == '1800.00'


def test_a_held_slider_is_left_under_the_users_drag(control):
    control.motion.move_z(2500.0)
    control.ids['obj_position'].value = 900.0
    control.ids['obj_position'].user_interacting = True

    control.show_z(1800.0)

    assert control.ids['obj_position'].value == 900.0
    assert control.ids['z_position_id'].text == '1800.00'


@pytest.mark.parametrize('sim_session', [{'microscope': 'LS620'}], indirect=True)
def test_a_scope_with_no_z_leaves_the_slider_where_it_is(control):
    """An LS620 has no focus motor, so the API holds no Z target."""
    control.ids['obj_position'].value = 900.0

    control.show_z(1800.0)

    assert control.ids['obj_position'].value == 900.0


def test_the_display_shows_the_apis_z_as_it_answers_it(control):
    """No floor: a negative target or reading is shown as the API gives it,
    never as a plausible 0. A move past the configured travel is how Z gets
    there."""
    control.motion.move_z(-3.5, ignore_limits=True)

    control.show_z(-2.0)

    assert control.ids['obj_position'].value == -3.5
    assert control.ids['z_position_id'].text == '-2.00'


def test_a_finished_move_redraws_the_z_target(control, monkeypatch):
    """A move's completion redraws the panel at the API's Z target, run or not:
    the display reads no run state."""
    control.motion.move_z(2500.0)
    monkeypatch.setattr(
        'ui.vertical_control.Clock.schedule_once', lambda fn, dt: fn(dt), raising=False
    )

    control.update_gui()

    assert control.ids['obj_position'].value == 2500.0
    assert control.ids['z_position_id'].text == '2500.00'


@pytest.fixture
def bridge():
    z_ctrl = MagicMock()
    ctx = SimpleNamespace(
        motion_settings=SimpleNamespace(
            ids={'verticalcontrol_id': z_ctrl},
            update_xy_stage_control_gui=MagicMock(),
        )
    )
    # The position handler reaches no scope member.
    b = UIListenerBridge(scope=SimpleNamespace(), ctx=ctx, stage=MagicMock())
    return b, z_ctrl, ctx


def test_a_z_change_draws_the_z_display_with_the_polled_position(bridge):
    b, z_ctrl, _ = bridge

    b._on_position_change('Z', AxisPosition(AxisState.MOVING, 1234.5))

    z_ctrl.show_z.assert_called_once_with(1234.5)


def test_a_homing_z_draws_the_z_display_with_no_position(bridge):
    b, z_ctrl, _ = bridge

    b._on_position_change('Z', AxisPosition(AxisState.HOMING, None))

    z_ctrl.show_z.assert_called_once_with(None)


def test_a_turret_change_draws_the_turret_without_asking_the_objective(bridge):
    b, z_ctrl, _ = bridge

    b._on_turret_change(TurretStatus(AxisState.IDLE, 2))

    z_ctrl.show_turret_state.assert_called_once_with(prompt=False)


def test_an_xy_change_draws_the_stage_and_not_the_z_display(bridge):
    b, z_ctrl, ctx = bridge

    b._on_position_change('X', AxisPosition(AxisState.IDLE, 10.0))

    ctx.motion_settings.update_xy_stage_control_gui.assert_called_once_with()
    z_ctrl.show_z.assert_not_called()


def test_the_engine_names_no_display_hook():
    retired = {'move_position', 'ui_update_func', 'z_ui_update_func', 'af_ui_update_func'}
    found = []
    for rel_path, tree in iter_package_modules(('modules',)):
        for node in ast.walk(tree):
            name = (
                node.arg
                if isinstance(node, (ast.arg, ast.keyword))
                else node.attr
                if isinstance(node, ast.Attribute)
                else node.value
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                else None
            )
            if name in retired:
                found.append(f'{rel_path}:{node.lineno} {name}')
    assert found == [], found
