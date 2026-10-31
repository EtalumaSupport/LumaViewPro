# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each GUI hardware write hands its API member to the lane that member dispatches to.

The gating wave moved every LED, motion and camera write the GUI makes onto
``submit_reported(..., lane=...)``: the call runs on its device's lane, where
the member runs inline, and its outcome is reported once. A handler that
quietly stopped submitting, submitted to the wrong lane, or submitted a
different member would still render; these call the real handlers over a
stand-in context and read what reached the boundary.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
from tests.stock_channels import LED1


class _Boundary:
    """Records each submit_reported / run_reported and runs its call at once."""

    def __init__(self):
        self.submits = []

    def submit_reported(self, call, redraw, label, *, stop=False, lane=None):
        self.submits.append(SimpleNamespace(call=call, redraw=redraw, label=label, lane=lane))
        call()

    def run_reported(self, call, redraw, label):
        self.submits.append(SimpleNamespace(call=call, redraw=redraw, label=label, lane=None))
        call()


@pytest.fixture
def boundary():
    return _Boundary()


@pytest.fixture
def ctx(monkeypatch):
    # a stand-in by design: the lane dispatch is the subject; each member is interchangeable
    scope = SimpleNamespace(
        illumination=MagicMock(),
        camera=MagicMock(),
        imaging=MagicMock(),
        motion=MagicMock(),
        led_connected=True,
        motor_connected=True,
    )
    context = SimpleNamespace(
        scope=scope,
        lumaview=SimpleNamespace(scope=scope),
        io_executor=object(),
        camera_executor=object(),
        settings={LED1: {'illumination_ma': 120.0, 'focus': 4321.0}},
        session=SimpleNamespace(
            controls_locked=False,
            light_shown_channel=MagicMock(),
            show_channel=MagicMock(),
        ),
        ui_listener_bridge=SimpleNamespace(reconcile_led_buttons=MagicMock()),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    return context


# ---------------------------------------------------------------------------
# LED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('state, lit', [('down', True), ('normal', False)])
def test_an_led_toggle_lights_the_shown_channel_through_the_session(
    ctx, boundary, monkeypatch, state, lit
):
    """The button is the shown channel's: its press asks the Session, on the
    worker pool since the Session's member dispatches to the IO lane itself."""
    import ui.layer_control as layer_control

    monkeypatch.setattr(layer_control, 'submit_reported', boundary.submit_reported)
    monkeypatch.setattr(layer_control.gui_logger, 'toggle', MagicMock())
    widget = SimpleNamespace(layer=LED1, ids={'enable_led_btn': SimpleNamespace(state=state)})

    layer_control.LayerControl.led_toggle(widget)

    (submit,) = boundary.submits
    assert submit.lane is None
    ctx.session.light_shown_channel.assert_called_once_with(lit)
    ctx.scope.illumination.led_on.assert_not_called()
    ctx.scope.illumination.led_off.assert_not_called()
    # The toggle shows what the API reports lit once the command has landed.
    assert submit.redraw is ctx.ui_listener_bridge.reconcile_led_buttons


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def test_the_layer_apply_shows_the_layer_through_the_session(ctx, boundary, monkeypatch):
    """A settings change shows the layer: show_channel puts its LED and camera
    settings on the hardware, so the apply sends no device command itself."""
    import ui.layer_control as layer_control

    monkeypatch.setattr(layer_control, 'submit_reported', boundary.submit_reported)
    widget = MagicMock()
    widget.layer = LED1
    widget._initializing = False

    layer_control.LayerControl.apply_settings(widget)

    (submit,) = boundary.submits
    assert submit.lane is None
    ctx.session.show_channel.assert_called_once_with(LED1)
    assert ctx.scope.illumination.mock_calls == []
    assert ctx.scope.camera.mock_calls == []
    assert ctx.scope.imaging.mock_calls == []


@pytest.mark.parametrize(
    'handler, widget_id, member, expected_args',
    [
        (
            'update_high_conversion_gain',
            'high_conversion_gain',
            'set_high_conversion_gain',
            (True,),
        ),
        (
            'update_line_noise_reduction',
            'line_noise_reduction',
            'set_line_noise_reduction',
            (True,),
        ),
    ],
)
def test_an_advanced_camera_switch_writes_its_member_on_the_camera_lane(
    ctx, boundary, monkeypatch, handler, widget_id, member, expected_args
):
    import ui.advanced_settings as advanced_settings

    monkeypatch.setattr(advanced_settings, 'submit_reported', boundary.submit_reported)
    monkeypatch.setattr(advanced_settings.gui_logger, 'select', MagicMock())
    # The camera API member commands the camera and stores what it took.
    setattr(ctx.scope.camera, member, MagicMock(return_value=True))
    widget = SimpleNamespace(
        ids={widget_id: SimpleNamespace(active=True)}, _show_camera_modes=MagicMock()
    )

    getattr(advanced_settings.AdvancedSettings, handler)(widget)

    (submit,) = boundary.submits
    assert submit.lane is ctx.camera_executor
    getattr(ctx.scope.camera, member).assert_called_once_with(*expected_args)


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


def test_the_acceleration_limit_is_written_on_the_io_lane(ctx, boundary, monkeypatch):
    import ui.advanced_settings as advanced_settings

    monkeypatch.setattr(advanced_settings, 'submit_reported', boundary.submit_reported)
    widget = SimpleNamespace(
        _pending_acceleration_pct=40, _show_stored_acceleration_limit=lambda: None
    )
    # The Session member commands the motors and stores what they took.
    ctx.session.set_acceleration_limit = MagicMock()

    advanced_settings.AdvancedSettings._dispatch_acceleration_to_motor(widget)

    (submit,) = boundary.submits
    assert submit.lane is ctx.io_executor
    # The redraw shows the stored limit, so a refused entry does not stay in the box.
    assert submit.redraw is widget._show_stored_acceleration_limit
    ctx.session.set_acceleration_limit.assert_called_once_with(40)


def test_goto_focus_moves_z_to_the_layers_saved_focus(ctx, monkeypatch):
    import ui.layer_control as layer_control
    import ui.ui_helpers as ui_helpers

    from modules.scope_session import ScopeSession

    moved = MagicMock()
    monkeypatch.setattr(ui_helpers, 'move_z', moved)
    monkeypatch.setattr(layer_control.gui_logger, 'button', MagicMock())
    # The saved focus is the Session's answer, read from the settings it holds.
    ctx.session.saved_focus_z_um = lambda layer: ScopeSession.saved_focus_z_um(ctx, layer)

    layer_control.LayerControl.goto_focus(SimpleNamespace(layer=LED1))

    moved.assert_called_once_with(4321.0)


def test_goto_focus_on_a_channel_never_saved_is_refused_and_moves_nothing(ctx, monkeypatch):
    import ui.layer_control as layer_control
    import ui.ui_helpers as ui_helpers
    from modules.notification_center import notifications
    from modules.scope_session import ScopeSession

    moved = MagicMock()
    reported = []
    monkeypatch.setattr(ui_helpers, 'move_z', moved)
    monkeypatch.setattr(layer_control.gui_logger, 'button', MagicMock())
    monkeypatch.setattr(
        notifications,
        'report_outcome',
        lambda exc, **kw: reported.append((type(exc).__name__, kw['category'])),
    )
    ctx.settings[LED1]['focus'] = None
    ctx.session.saved_focus_z_um = lambda layer: ScopeSession.saved_focus_z_um(ctx, layer)

    layer_control.LayerControl.goto_focus(SimpleNamespace(layer=LED1))

    moved.assert_not_called()
    assert reported == [('FocusNotSavedError', f'UI:GOTO_FOCUS_{LED1}')]


def test_home_stage_homes_every_axis_through_the_reporter(ctx, boundary, monkeypatch):
    import ui.motion_settings as motion_settings

    homed = MagicMock()
    monkeypatch.setattr(motion_settings, 'run_reported', boundary.run_reported)
    monkeypatch.setattr(motion_settings, 'move_home', homed)
    monkeypatch.setattr(motion_settings.gui_logger, 'button', MagicMock())

    # The debounce wraps the handler; the body is what a press runs.
    motion_settings.XYStageControl.home.__wrapped__(SimpleNamespace())

    assert [s.label for s in boundary.submits] == ['HOME_XY']
    homed.assert_called_once_with(axis='ALL')


# ---------------------------------------------------------------------------
# Protocol panel
# ---------------------------------------------------------------------------


def test_the_panels_step_navigation_is_a_persons_move(monkeypatch):
    import ui.protocol_settings as protocol_settings

    navigated = MagicMock()
    monkeypatch.setattr(protocol_settings, 'go_to_step', navigated)
    panel = SimpleNamespace(_protocol='protocol')

    protocol_settings.ProtocolSettings.go_to_step(panel, step_index=2)

    navigated.assert_called_once()
    assert navigated.call_args.kwargs['include_move'] is True
    assert navigated.call_args.kwargs['step_index'] == 2


def test_the_fire_and_forget_led_and_motion_members_are_gone():
    """A GUI write reaches its lane through submit_reported and the blocking
    member, so a failure is reported once; the *_async members queued the
    work and reported it nowhere the caller could see (#792)."""
    from modules.lumascope_api.illumination import IlluminationAPI
    from modules.lumascope_api.motion import MotionAPI

    survivors = [
        f'{api.__name__}.{name}'
        for api in (IlluminationAPI, MotionAPI)
        for name in dir(api)
        if name.endswith('_async')
    ]
    assert survivors == []
