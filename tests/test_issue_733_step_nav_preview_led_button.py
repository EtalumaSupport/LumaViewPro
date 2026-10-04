# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issue #733: 'LED On When Stepping' cannot be turned off.

Bug class: UI widget state used as an LED command channel. The manual-nav
preview branch of ``ui.step_navigation.go_to_step`` force-wrote
``enable_led_btn.state = 'down'`` before calling ``apply_settings``, whose
``update_led=True`` default re-derives hardware intent from that widget --
so every manual navigation (next/prev/delete-step/right-click) re-lit the
channel even after the user toggled the LED button off. A second writer in
``go_to_step_update_ui`` forced 'down' whenever ``protocol_led_on`` was set,
leaving a stale 'down' for any later ``apply_settings(update_led=True)`` to
re-read.

Contract under test: outside a protocol run, the listener bridge is the SOLE
writer of ``enable_led_btn``; the manual-nav preview lights the channel only
through the illumination authority's MANUAL_STEP transition; ``apply_settings``
is called with ``update_led=False`` so it cannot re-derive LED intent from
the widget.
"""

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.lumascope_api.protocols import ProtocolsAPI

from tests.gesture_fakes import inline_submit_move


GREEN_LAYER_SETTINGS = {
    'autofocus': False,
    'false_color': True,
    'illumination_ma': 0.0,
    'gain_db': 0.0,
    'auto_gain': False,
    'exposure_ms': 10.0,
    'sum': 1,
    'acquire': True,
    'focus': 0.0,
    'video_config': {'duration': 5},
    'stim_config': {
        'enabled': False,
        'illumination_ma': 50,
        'frequency': 1,
        'pulse_width': 10,
        'pulse_count': 1,
    },
}


def _make_step():
    return {
        'X': 10.0,
        'Y': 20.0,
        'Z': 3.0,
        'Color': 'Green',
        'Auto_Focus': False,
        'False_Color': True,
        'Illumination': 350.0,
        'Gain': 0.0,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': 1,
        'Acquire': True,
        'Objective': 'obj1',
    }


@pytest.fixture
def stepnav_env(monkeypatch):
    """Fake AppContext driving the REAL go_to_step, mocked at the ctx boundary."""
    layer_obj = SimpleNamespace(
        ids={'enable_led_btn': SimpleNamespace(state='normal')},
        apply_settings=MagicMock(),
        set_step_state=MagicMock(),
    )
    protocol_settings = MagicMock()
    # A real ProtocolsAPI, because go_to_step asks it whether this scope can
    # put the step's glass in the light path before it moves anything. A
    # stand-in here would answer for production's rule without being it.
    scope = SimpleNamespace(
        motion=SimpleNamespace(),
        capabilities=SimpleNamespace(has_turret=False, axes=('X', 'Y', 'Z')),
        # With no turret the rule admits only the selected objective, so the
        # stand has the step's glass selected.
        runtime_state=SimpleNamespace(get_current_objective_id=lambda: 'obj1'),
        motor_connected=False,
        imaging=SimpleNamespace(active_cached=False),
        illumination=SimpleNamespace(
            color2ch=MagicMock(return_value=3),
            apply_transition=MagicMock(),
        ),
    )
    scope.protocols = ProtocolsAPI(scope)
    ctx = SimpleNamespace(
        settings={
            'protocol_led_on': True,
            'stage_offset': {'x': 0, 'y': 0},
            'Green': dict(GREEN_LAYER_SETTINGS),
        },
        settings_lock=threading.Lock(),
        coordinate_transformer=SimpleNamespace(
            plate_to_stage=MagicMock(return_value=(1.0, 2.0)),
        ),
        motion_settings=SimpleNamespace(
            ids={'protocol_settings_id': protocol_settings},
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
            # The Session's member: the moves, the layer write and the LED
            # preview are its (tests/test_going_to_a_step_is_the_sessions_move.py).
            go_to_step=MagicMock(),
        ),
        stage=SimpleNamespace(draw_labware=MagicMock()),
        io_executor=object(),
    )
    monkeypatch.setattr('modules.app_context.ctx', ctx)
    # ui.ui_helpers and ui.layer_control pull kivy submodules the conftest
    # kivy mock cannot provide; go_to_step defers both imports, so
    # module-boundary stubs suffice. The move runs at once, so what follows
    # it is seen.
    ui_helpers = MagicMock()
    ui_helpers.submit_move.side_effect = inline_submit_move
    monkeypatch.setitem(sys.modules, 'ui.ui_helpers', ui_helpers)
    monkeypatch.setitem(sys.modules, 'ui.layer_control', MagicMock())
    # Run scheduled UI callbacks inline so the closures under test execute.
    monkeypatch.setattr('ui.step_navigation._schedule_ui', lambda fn, t: fn(0))
    monkeypatch.setattr(
        'modules.config_ui_getters.get_selected_labware',
        lambda: ('labware', MagicMock()),
    )
    return SimpleNamespace(ctx=ctx, layer_obj=layer_obj)


def _run_manual_nav(env):
    import ui.step_navigation as step_navigation

    protocol = SimpleNamespace(
        num_steps=MagicMock(return_value=1),
        step=MagicMock(return_value=_make_step()),
        step_list_revision=0,
    )
    # The panel shows this protocol: a completed move lands only on the
    # protocol and step list it was sent for.
    import modules.app_context as _app_ctx

    _app_ctx.ctx.motion_settings.ids['protocol_settings_id']._protocol = protocol
    step_navigation.go_to_step(
        protocol,
        step_idx=0,
        include_move=True,
    )


class TestStepNavPreviewRespectsLedEnable:
    def test_led_button_not_forced_down_by_manual_nav(self, stepnav_env):
        """The button must stay 'normal': outside a run the bridge is the
        sole writer. Locks BOTH forced writers (the preview closure and
        go_to_step_update_ui) because _schedule_ui runs both inline."""
        _run_manual_nav(stepnav_env)
        assert stepnav_env.layer_obj.ids['enable_led_btn'].state == 'normal'

    def test_the_gui_issues_no_led_command_of_its_own(self, stepnav_env):
        """The preview is the Session's one LED command, inside its go_to_step;
        the GUI asks the Session once and lights nothing itself."""
        _run_manual_nav(stepnav_env)
        assert stepnav_env.ctx.scope.illumination.apply_transition.call_count == 0
        assert stepnav_env.ctx.session.go_to_step.call_count == 1
        # The move rides the IO lane; with no motor board no axis is redrawn.
        move = sys.modules['ui.ui_helpers'].submit_move
        assert move.call_count == 1
        assert move.call_args.kwargs['axes'] == ()

    def test_apply_settings_cannot_rederive_led_from_widget(self, stepnav_env):
        """apply_settings must receive update_led=False once the Session has
        gone, so the widget can never act as an LED command channel."""
        _run_manual_nav(stepnav_env)
        assert stepnav_env.layer_obj.apply_settings.call_count == 1
        assert stepnav_env.layer_obj.apply_settings.call_args.kwargs['update_led'] is False


def test_a_step_click_on_a_scope_with_no_motor_board_warns_nothing(stepnav_env, caplog):
    """A manual scope (no motor board) is a shipped model, not a fault: its
    step click goes to the step without moving, and logs no warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger='LVP.ui.step_navigation'):
        _run_manual_nav(stepnav_env)

    assert stepnav_env.ctx.session.go_to_step.call_count == 1
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
