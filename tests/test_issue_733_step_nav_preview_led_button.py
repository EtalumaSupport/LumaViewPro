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

from modules.lumascope_api.illumination import LedTransition


GREEN_LAYER_SETTINGS = {
    'autofocus': False,
    'false_color': True,
    'ill_ma': 0.0,
    'gain_db': 0.0,
    'auto_gain': False,
    'exp_ms': 10.0,
    'sum': 1,
    'acquire': True,
    'focus': 0.0,
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
        scope=SimpleNamespace(
            motion=SimpleNamespace(has_turret=MagicMock(return_value=False)),
            motor_connected=False,
            imaging=SimpleNamespace(active_cached=False),
            illumination=SimpleNamespace(
                color2ch=MagicMock(return_value=3),
                apply_transition_async=MagicMock(),
            ),
        ),
        protocol_running=SimpleNamespace(is_set=MagicMock(return_value=False)),
        sequenced_capture_runner=SimpleNamespace(run_in_progress=lambda: False),
        stage=SimpleNamespace(draw_labware=MagicMock()),
    )
    monkeypatch.setattr('modules.app_context.ctx', ctx)
    # ui.ui_helpers and ui.layer_control pull kivy submodules the conftest
    # kivy mock cannot provide; go_to_step defers both imports and this
    # test's path never calls into them, so module-boundary stubs suffice.
    monkeypatch.setitem(sys.modules, 'ui.ui_helpers', MagicMock())
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
    )
    step_navigation.go_to_step(
        protocol,
        step_idx=0,
        include_move=True,
        called_from_protocol=False,
    )


class TestStepNavPreviewRespectsLedEnable:
    def test_led_button_not_forced_down_by_manual_nav(self, stepnav_env):
        """The button must stay 'normal': outside a run the bridge is the
        sole writer. Locks BOTH forced writers (the preview closure and
        go_to_step_update_ui) because _schedule_ui runs both inline."""
        _run_manual_nav(stepnav_env)
        assert stepnav_env.layer_obj.ids['enable_led_btn'].state == 'normal'

    def test_preview_still_lights_via_authority_transition(self, stepnav_env):
        """Removing the widget write must NOT remove the preview: the
        MANUAL_STEP transition is the one LED command."""
        _run_manual_nav(stepnav_env)
        apply_async = stepnav_env.ctx.scope.illumination.apply_transition_async
        assert apply_async.call_count == 1
        transition, led_ctx = apply_async.call_args.args
        assert transition is LedTransition.MANUAL_STEP
        assert led_ctx.preview_on is True

    def test_apply_settings_cannot_rederive_led_from_widget(self, stepnav_env):
        """apply_settings must receive update_led=False in the preview
        branch, so the widget can never act as an LED command channel."""
        _run_manual_nav(stepnav_env)
        assert stepnav_env.layer_obj.apply_settings.call_count == 1
        assert stepnav_env.layer_obj.apply_settings.call_args.kwargs['update_led'] is False
