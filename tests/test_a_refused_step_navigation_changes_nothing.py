# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A navigation the scope cannot perform leaves the protocol alone.

`go_to_step` used to commit the step pointer as its second act, before
anything had asked whether the scope could reach the glass that step
names. When it could not, the old code logged, raised a dialog, and then
moved X, Y and Z anyway -- to a step it could not put the right objective
in front of.

The pointer is the part that matters, and it is why this refuses ABOVE
the write rather than beside the move. `modify_step_ex` and
`insert_step_ex` address a step BY that pointer and fill it from the LIVE
stage, so a pointer left on a step the user never arrived at means the
next edit writes the previous step's coordinates into it. The protocol
file is then wrong, on disk, with nothing in it saying so.

The rule is the API's, so these tests drive the real one: the scope's
turret configuration decides, exactly as it does for a run.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.lumascope_api.protocols import ProtocolsAPI


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
        motion=SimpleNamespace(),
        motor_connected=True,
        imaging=SimpleNamespace(active_cached=False),
        illumination=SimpleNamespace(
            color2ch=MagicMock(return_value=3),
            apply_transition_async=MagicMock(),
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
        session=SimpleNamespace(is_protocol_running=False, run_lockout=False),
        sequenced_capture_runner=SimpleNamespace(run_in_progress=lambda: False),
        stage=SimpleNamespace(draw_labware=MagicMock()),
    )
    monkeypatch.setattr('modules.app_context.ctx', ctx)

    ui_helpers = MagicMock()
    # Every axis knows its position unless a test says otherwise: these
    # tests are about the objective rule.
    ui_helpers.unknown_position_refused.return_value = False
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
        move_absolute=ui_helpers.move_absolute,
        unknown_position_refused=ui_helpers.unknown_position_refused,
    )


def _navigate(objective: str, *, step_idx: int = 0, called_from_protocol: bool = False):
    import ui.step_navigation as step_navigation

    protocol = SimpleNamespace(
        num_steps=MagicMock(return_value=2),
        step=MagicMock(return_value=_make_step(objective)),
    )
    step_navigation.go_to_step(
        protocol,
        step_idx=step_idx,
        include_move=True,
        called_from_protocol=called_from_protocol,
    )


class TestARefusedNavigationIsANoOp:
    def test_the_step_pointer_does_not_move(self, nav_env):
        """The whole point: an edit addressed by this pointer would corrupt."""
        _navigate(NOT_ON_TURRET)

        assert nav_env.protocol_settings.curr_step == 3, (
            'the pointer moved to a step the scope never navigated to; '
            'modify_step_ex would now fill that step from the live stage'
        )

    def test_no_axis_moves(self, nav_env):
        _navigate(NOT_ON_TURRET)

        assert nav_env.move_absolute.call_count == 0, (
            f'a refused navigation moved the stage: {nav_env.move_absolute.call_args_list}'
        )

    def test_the_layer_settings_are_untouched(self, nav_env):
        """A refused manual navigation must not load the step into the layer."""
        before = dict(nav_env.ctx.settings['Green'])

        _navigate(NOT_ON_TURRET)

        assert nav_env.ctx.settings['Green'] == before

    def test_the_refusal_carries_the_admissibility_reason(self, nav_env):
        """Refused for the reason the run would give, not a navigation one."""
        with pytest.raises(ProtocolRunRefusedError) as refusal:
            nav_env.scope.protocols.refuse_unaddressable_objectives([NOT_ON_TURRET])

        assert refusal.value.reason == 'turret_objectives_unassigned'

    def test_a_refused_navigation_does_not_propagate_to_its_caller(self, nav_env):
        """The API told the user; a step button has nothing left to do.

        go_to_step is also the run's navigation callback, so a raise here
        would end a run over a refusal the engine already answered at
        prepare().
        """
        _navigate(NOT_ON_TURRET, called_from_protocol=True)


class TestAnAdmissibleNavigationStillWorks:
    def test_the_pointer_moves(self, nav_env):
        _navigate(ON_TURRET, step_idx=1)

        assert nav_env.protocol_settings.curr_step == 1

    def test_the_stage_moves(self, nav_env):
        _navigate(ON_TURRET, step_idx=1)

        axes = [
            c.kwargs.get('axis', c.args[0] if c.args else None)
            for c in nav_env.move_absolute.call_args_list
        ]
        assert 'X' in axes and 'Y' in axes and 'Z' in axes, f'axes moved: {axes}'

    def test_the_turret_moves_to_the_slot_that_carries_the_glass(self, nav_env):
        _navigate(ON_TURRET, step_idx=1)

        t_moves = [
            c
            for c in nav_env.move_absolute.call_args_list
            if c.kwargs.get('axis') == 'T' or (c.args and c.args[0] == 'T')
        ]
        assert len(t_moves) == 1, f'expected one T move, got {t_moves}'
        assert t_moves[0].kwargs['position'] == 1


class TestTheSlotLookupCannotDisagreeWithTheRule:
    def test_a_missing_slot_after_the_rule_passed_is_a_defect_not_a_move(self, nav_env):
        """The rule and the lookup read the same store, so this cannot happen.

        When it does, the two have disagreed and the honest answer is to
        stop. The old code logged, raised a dialog, and then moved X, Y
        and Z without the objective -- a capture through whatever glass
        happened to be in the path, named for the glass the step asked
        for.
        """
        nav_env.scope.motion.get_turret_position_for_objective_id = (
            lambda objective_id, persisted_position=None: None
        )

        with pytest.raises(RuntimeError) as defect:
            _navigate(ON_TURRET, step_idx=1)

        assert 'slot' in str(defect.value).lower()
        assert nav_env.move_absolute.call_count == 0, (
            'the stage moved despite the turret having nowhere to go'
        )


class TestAnUnhomedNavigationIsANoOp:
    """An axis that does not know its position refuses the navigation once,
    before the pointer moves -- the same no-op as an unaddressable objective."""

    def test_the_positions_are_asked_once_for_every_axis_before_anything_moves(self, nav_env):
        nav_env.unknown_position_refused.return_value = True

        _navigate(ON_TURRET)

        nav_env.unknown_position_refused.assert_called_once_with(
            ('X', 'Y', 'Z', 'T'), recording=False, then='go to the step'
        )
        assert nav_env.protocol_settings.curr_step == 3
        assert nav_env.move_absolute.call_count == 0

    def test_a_run_navigation_does_not_ask(self, nav_env):
        """The run navigates with include_move=False; prepare() settled positions."""
        import ui.step_navigation as step_navigation

        protocol = SimpleNamespace(
            num_steps=MagicMock(return_value=2),
            step=MagicMock(return_value=_make_step(ON_TURRET)),
        )
        step_navigation.go_to_step(
            protocol, step_idx=0, include_move=False, called_from_protocol=True
        )

        nav_env.unknown_position_refused.assert_not_called()
