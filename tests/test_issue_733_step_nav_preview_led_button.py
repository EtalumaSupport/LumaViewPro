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
writer of ``enable_led_btn``, and the step's LED preview is the Session's
alone (``ScopeSession.start_go_to_step``): the GUI lights nothing itself.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ui.step_navigation as step_navigation
import ui.ui_helpers as ui_helpers
from modules.acquire_mode import AcquireMode
from modules.lumascope_api.illumination import LedState
from modules.sequential_io_executor import ENQUEUED
from tests.pool_fakes import run_task_now
from tests.scope_fakes import home_sim_scope


@pytest.fixture
def stepnav_env(gui_ctx, monkeypatch):
    """The real go_to_step over the GUI's context on a simulated session.

    Homed when the scope has a motor board: a step click asks every axis
    whether it knows its position. The panel and the layer widgets are the
    test's own (the Kivy tree never runs here), the layer's LED button
    starting 'normal'. The click's task runs on the worker pool at once.
    """
    session = gui_ctx.session
    if session.scope.motor_connected:
        home_sim_scope(session.scope)
    pool = MagicMock()

    def _put(task):
        run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _put
    gui_ctx.worker_pool = pool
    layer_obj = SimpleNamespace(
        ids={'enable_led_btn': SimpleNamespace(state='normal')},
        set_step_state=MagicMock(),
        apply_settings=MagicMock(),
    )
    gui_ctx.motion_settings = SimpleNamespace(
        ids={'protocol_settings_id': MagicMock(), 'verticalcontrol_id': MagicMock()},
        update_xy_stage_control_gui=MagicMock(),
    )
    gui_ctx.image_settings = SimpleNamespace(
        layer_lookup=MagicMock(return_value=layer_obj),
        ids={'toggle_imagesettings': SimpleNamespace(state='down')},
        place_settings=MagicMock(),
    )
    # No Kivy app runs, so no lock record locks the controls.
    monkeypatch.setattr(sys.modules['kivy.app'].App, 'get_running_app', lambda: None)
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    monkeypatch.setattr(step_navigation, '_schedule_ui', lambda fn, t: fn(0))
    return SimpleNamespace(session=session, ctx=gui_ctx, layer_obj=layer_obj)


def _run_manual_nav(env):
    """Click the one step of a protocol the Session built where the scope stands, in BF; return it."""
    session = env.session
    session.set_channel_acquire('BF', AcquireMode.IMAGE)
    protocol = session.scope.protocols.create_protocol(
        empty_config=session.get_sequenced_capture_config()
    )
    session.add_step(protocol, before_step_index=0)
    # The panel shows this protocol: a completed move lands only on the
    # protocol and step list it was sent for.
    env.ctx.motion_settings.ids['protocol_settings_id']._protocol = protocol
    step_navigation.go_to_step(protocol, step_index=0, include_move=True)
    return protocol


def _shown(env) -> str:
    """The layer whose step state the click showed first: the step's own.

    One layer widget stands for every layer, so the step's stimulation
    settings for the other layers land on it too.
    """
    return env.layer_obj.set_step_state.call_args_list[0].args[0]['Color']


def _lit(env) -> list:
    return [r for r in env.session.scope.illumination.get_led_states() if r.state is LedState.LIT]


class TestStepNavPreviewRespectsLedEnable:
    def test_led_button_not_forced_down_by_manual_nav(self, stepnav_env):
        """The button must stay 'normal': outside a run the bridge is the
        sole writer. Locks BOTH forced writers (the preview closure and
        go_to_step_update_ui) because _schedule_ui runs both inline."""
        _run_manual_nav(stepnav_env)

        assert _shown(stepnav_env) == 'BF'
        assert stepnav_env.layer_obj.ids['enable_led_btn'].state == 'normal'

    def test_with_led_on_when_stepping_off_the_gui_lights_nothing(self, stepnav_env):
        """The preview is the Session's one LED command, inside its go-to:
        with 'LED On When Stepping' off it puts every channel dark, and the
        GUI lights nothing after it."""
        stepnav_env.session.update_settings('protocol_led_on', False)

        _run_manual_nav(stepnav_env)

        assert _shown(stepnav_env) == 'BF'
        assert _lit(stepnav_env) == []

    def test_every_axis_the_scope_has_is_redrawn(self, stepnav_env):
        _run_manual_nav(stepnav_env)

        stepnav_env.ctx.motion_settings.update_xy_stage_control_gui.assert_called_once()
        vertical = stepnav_env.ctx.motion_settings.ids['verticalcontrol_id']
        vertical.update_gui.assert_called_once()


@pytest.mark.parametrize('sim_session', [{'microscope': 'LS560'}], indirect=True)
def test_a_step_click_on_a_scope_with_no_motor_board_warns_nothing(stepnav_env, caplog):
    """A manual scope (no motor board) is a shipped model, not a fault: its
    step click goes to the step without moving, and logs no warning."""
    assert not stepnav_env.session.scope.motor_connected

    with caplog.at_level(logging.WARNING, logger='LVP.ui.step_navigation'):
        _run_manual_nav(stepnav_env)

    assert _shown(stepnav_env) == 'BF'
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
