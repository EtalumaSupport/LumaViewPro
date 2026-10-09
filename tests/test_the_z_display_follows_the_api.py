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

from modules import app_context as _app_ctx
from modules.lumascope_api import AxisPosition, AxisState
from tests.ast_seams import iter_package_modules
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
def control(monkeypatch):
    target = {'Z': 2500.0}
    motion = SimpleNamespace(get_target_position=lambda axis: target[axis])
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(lumaview=SimpleNamespace(scope=SimpleNamespace(motion=motion))),
    )
    ctrl = VerticalControl.__new__(VerticalControl)
    ctrl.ids = {'z_position_id': _Box(), 'obj_position': _Slider()}
    ctrl.target = target
    return ctrl


def test_the_slider_shows_the_target_and_the_text_the_polled_z(control):
    control.show_z(1800.0)

    assert control.ids['obj_position'].value == 2500.0
    assert control.ids['z_position_id'].text == '1800.00'


def test_a_held_slider_is_left_under_the_users_drag(control):
    control.ids['obj_position'].value = 900.0
    control.ids['obj_position'].user_interacting = True

    control.show_z(1800.0)

    assert control.ids['obj_position'].value == 900.0
    assert control.ids['z_position_id'].text == '1800.00'


def test_no_target_leaves_the_slider_where_it_is(control):
    control.target['Z'] = None
    control.ids['obj_position'].value = 900.0

    control.show_z(1800.0)

    assert control.ids['obj_position'].value == 900.0


def test_a_run_does_not_silence_the_z_display(control, monkeypatch):
    # The run gate dropped every engine Z update; the display now has none.
    _app_ctx.ctx.session = SimpleNamespace(run_in_progress=True)
    shown = []
    monkeypatch.setattr(control, '_show_z_target', lambda: shown.append('target'))
    monkeypatch.setattr(
        'ui.vertical_control.Clock.schedule_once', lambda fn, dt: fn(dt), raising=False
    )
    monkeypatch.setattr('ui.vertical_control.run_reported', lambda _o, fn, _n: fn())

    control.update_gui()

    assert shown == ['target']


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

    b._on_position_change('T', AxisPosition(AxisState.IDLE, 2.0))

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
