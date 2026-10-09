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

The rule is the API's, so the Session stand here refuses through the real
one: the scope's turret configuration decides, exactly as it does for a
run. The Session's own move is
``tests/test_going_to_a_step_is_the_sessions_move.py``.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from modules.lumascope_api.protocols import ProtocolsAPI
from modules.protocol import StepNotFoundError
from tests.gesture_fakes import inline_submit_move


ON_TURRET = '10x Oly'
NOT_ON_TURRET = '20x Oly'

LAYER_SETTINGS = {
    'autofocus': False,
    'false_color': False,
    'illumination_ma': 350.0,
    'gain_db': 0.0,
    'auto_gain': False,
    'exposure_ms': 10.0,
    'sum': 1,
    'acquire': 'image',
    'focus': 0.0,
}


def _make_step(objective: str) -> dict:
    return {
        'Name': 'step',
        'X': 1.0,
        'Y': 2.0,
        'Z': 3.0,
        'Auto_Focus': False,
        'Color': 'Green',
        'False_Color': False,
        'Illumination': 350.0,
        'Gain': 0.0,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': 1,
        'Acquire': True,
        'Objective': objective,
    }


@pytest.fixture
def nav_env(monkeypatch):
    """The real go_to_step over a fake ctx, carrying the REAL rule.

    `scope.protocols` is a genuine ProtocolsAPI bound to this fake scope,
    so the admissibility decision under test is production's, not a
    stand-in that can drift from it.
    """
    layer_obj = SimpleNamespace(
        ids={'enable_led_btn': SimpleNamespace(state='normal')},
        apply_settings=MagicMock(),
        set_step_state=MagicMock(),
    )
    protocol_settings = MagicMock()
    protocol_settings.curr_step = 3

    carried: dict = {1: ON_TURRET, 2: None, 3: None, 4: None}

    scope = SimpleNamespace(
        capabilities=SimpleNamespace(has_turret=True, axes=('X', 'Y', 'Z', 'T')),
        runtime_state=SimpleNamespace(get_turret_config=lambda: carried),
        motion=SimpleNamespace(
            move_absolute=MagicMock(),
            move_turret=MagicMock(),
            # Every axis knows its position unless a test says otherwise:
            # these tests are about the objective rule.
            refuse_unknown_positions=MagicMock(),
        ),
        motor_connected=True,
        imaging=SimpleNamespace(active_cached=False),
        illumination=SimpleNamespace(
            color2ch=MagicMock(return_value=3),
            apply_transition=MagicMock(),
        ),
    )
    scope.protocols = ProtocolsAPI(scope)
    scope.motion.get_turret_position_for_objective_id = (
        lambda objective_id, persisted_position=None: next(
            (pos for pos, obj in carried.items() if obj == objective_id), None
        )
    )

    ctx = SimpleNamespace(
        settings={
            'protocol_led_on': True,
            'stage_offset': {'x': 0, 'y': 0},
            'turret_position': None,
            'Green': dict(LAYER_SETTINGS),
        },
        settings_lock=threading.Lock(),
        coordinate_transformer=SimpleNamespace(
            plate_to_stage=MagicMock(return_value=(1.0, 2.0)),
        ),
        motion_settings=SimpleNamespace(
            ids={
                'protocol_settings_id': protocol_settings,
                'verticalcontrol_id': SimpleNamespace(update_turret_gui=MagicMock()),
            },
            update_xy_stage_control_gui=MagicMock(),
        ),
        image_settings=SimpleNamespace(
            layer_lookup=MagicMock(return_value=layer_obj),
            ids={'toggle_imagesettings': SimpleNamespace(state='down')},
            set_expanded_layer=MagicMock(),
            toggle_settings=MagicMock(),
        ),
        scope=scope,
        protocol_running=SimpleNamespace(is_set=MagicMock(return_value=False)),
        session=SimpleNamespace(
            is_protocol_running=False,
            run_lockout=False,
            run_in_progress=False,
            # The Session's member, refusing through the real rule and
            # recording the step it was asked to go to.
            start_go_to_step=MagicMock(
                side_effect=lambda protocol, step_idx: (
                    scope.protocols.refuse_unaddressable_objectives(
                        [protocol.step(idx=step_idx)['Objective']]
                    )
                )
            ),
        ),
        stage=SimpleNamespace(draw_labware=MagicMock()),
        io_executor=object(),
    )
    monkeypatch.setattr('modules.app_context.ctx', ctx)

    ui_helpers = MagicMock()
    # The move runs at once, so what it would drive is seen.
    ui_helpers.submit_move.side_effect = inline_submit_move
    monkeypatch.setitem(sys.modules, 'ui.ui_helpers', ui_helpers)
    monkeypatch.setitem(sys.modules, 'ui.layer_control', MagicMock())
    monkeypatch.setattr('ui.step_navigation._schedule_ui', lambda fn, t: fn(0))
    monkeypatch.setattr(
        'modules.config_ui_getters.get_selected_labware',
        lambda: ('labware', MagicMock()),
    )
    return SimpleNamespace(
        ctx=ctx,
        carried=carried,
        scope=scope,
        protocol_settings=protocol_settings,
        session_go_to_step=ctx.session.start_go_to_step,
        layer_obj=layer_obj,
    )


def _navigate(objective: str, *, step_idx: int = 0, include_move: bool = True):
    import ui.step_navigation as step_navigation

    protocol = SimpleNamespace(
        num_steps=MagicMock(return_value=2),
        step=MagicMock(return_value=_make_step(objective)),
        step_list_revision=0,
    )
    # The panel shows this protocol: a completed move lands only on the
    # protocol and step list it was sent for.
    import modules.app_context as _app_ctx

    _app_ctx.ctx.motion_settings.ids['protocol_settings_id']._protocol = protocol
    step_navigation.go_to_step(
        protocol,
        step_idx=step_idx,
        include_move=include_move,
    )


class TestARefusedNavigationIsANoOp:
    def test_the_step_pointer_does_not_move(self, nav_env):
        """The whole point: an edit addressed by this pointer would corrupt."""
        _navigate(NOT_ON_TURRET)

        assert nav_env.protocol_settings.curr_step == 3, (
            'the pointer moved to a step the scope never navigated to; '
            'modify_step_ex would now fill that step from the live stage'
        )

    def test_the_layer_is_not_applied(self, nav_env):
        """A refused navigation shows nothing of the step it did not reach."""
        _navigate(NOT_ON_TURRET)

        assert nav_env.layer_obj.apply_settings.call_count == 0
        assert nav_env.layer_obj.set_step_state.call_count == 0

    def test_a_refused_navigation_does_not_propagate_to_its_caller(self, nav_env):
        """The reporter told the user; a step button has nothing left to do."""
        _navigate(NOT_ON_TURRET)


class TestAnAdmissibleNavigationStillWorks:
    def test_the_pointer_moves(self, nav_env):
        _navigate(ON_TURRET, step_idx=1)

        assert nav_env.protocol_settings.curr_step == 1

    def test_the_session_is_asked_to_go_to_the_step(self, nav_env):
        _navigate(ON_TURRET, step_idx=1)

        assert nav_env.session_go_to_step.call_count == 1
        assert nav_env.session_go_to_step.call_args.args[1] == 1

    def test_an_index_before_the_first_step_is_the_sessions_to_refuse(self, nav_env):
        """A typed 0 arrives as -1; the panel clears its pointer only for a
        protocol with no steps, never for a number the protocol lacks."""
        nav_env.session_go_to_step.side_effect = StepNotFoundError(index=-1, num_steps=2)

        _navigate(ON_TURRET, step_idx=-1)

        assert nav_env.session_go_to_step.call_args.args[1] == -1
        assert nav_env.protocol_settings.curr_step == 3

    def test_the_layer_is_applied_once_the_session_has_gone(self, nav_env):
        _navigate(ON_TURRET, step_idx=1)

        assert nav_env.layer_obj.apply_settings.call_args_list == [call(update_led=False)]


class TestARunNavigationOnlyDisplays:
    def test_a_run_navigation_asks_the_session_for_no_move(self, nav_env):
        """The run navigates with include_move=False; it moved the scope itself."""
        import ui.step_navigation as step_navigation

        protocol = SimpleNamespace(
            num_steps=MagicMock(return_value=2),
            step=MagicMock(return_value=_make_step(ON_TURRET)),
            step_list_revision=0,
        )
        import modules.app_context as _app_ctx

        _app_ctx.ctx.motion_settings.ids['protocol_settings_id']._protocol = protocol
        step_navigation.go_to_step(protocol, step_idx=0, include_move=False)

        nav_env.session_go_to_step.assert_not_called()
        assert nav_env.protocol_settings.curr_step == 0
