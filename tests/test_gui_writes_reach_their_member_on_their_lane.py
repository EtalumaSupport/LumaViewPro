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
    scope = SimpleNamespace(
        illumination=MagicMock(),
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
        settings={'Green': {'illumination_ma': 120.0, 'focus': 4321.0}},
        session=SimpleNamespace(controls_locked=False, recover_file_writer=MagicMock()),
        ui_listener_bridge=SimpleNamespace(reconcile_led_buttons=MagicMock()),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    return context


# ---------------------------------------------------------------------------
# LED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('state, member', [('down', 'led_on'), ('normal', 'led_off')])
def test_an_led_toggle_writes_through_the_illumination_api_on_the_io_lane(
    ctx, boundary, monkeypatch, state, member
):
    import ui.layer_control as layer_control

    monkeypatch.setattr(layer_control, 'submit_reported', boundary.submit_reported)
    monkeypatch.setattr(layer_control.gui_logger, 'toggle', MagicMock())
    widget = SimpleNamespace(
        layer='Green',
        _initializing=False,
        ids={'enable_led_btn': SimpleNamespace(state=state)},
        apply_settings=MagicMock(),
    )

    layer_control.LayerControl.update_led_state(widget, apply_settings=False)

    (submit,) = boundary.submits
    assert submit.lane is ctx.io_executor
    getattr(ctx.scope.illumination, member).assert_called_once()
    assert getattr(ctx.scope.illumination, member).call_args.args[0] == 'Green'
    # The toggle shows what the API reports lit once the command has landed.
    assert submit.redraw is ctx.ui_listener_bridge.reconcile_led_buttons


def test_a_camera_pause_darkens_the_leds_on_the_io_lane_and_resume_restores_them(
    ctx, boundary, monkeypatch
):
    import ui.main_display as main_display

    monkeypatch.setattr(main_display, 'submit_reported', boundary.submit_reported)
    ctx.scope.illumination.save_led_state.return_value = 'snapshot'
    display = SimpleNamespace(scope=ctx.scope, _pause_led_snapshot=None)
    viewer = SimpleNamespace(play=True, pause=MagicMock(), resume=MagicMock())

    main_display.MainDisplay._toggle_play(display, viewer)
    main_display.MainDisplay._toggle_play(display, viewer)

    assert [s.label for s in boundary.submits] == ['CAM_PAUSE_LEDS', 'CAM_RESUME_LEDS']
    assert all(s.lane is ctx.io_executor for s in boundary.submits)
    ctx.scope.illumination.leds_off.assert_called_once_with()
    ctx.scope.illumination.restore_led_state.assert_called_once_with('snapshot')


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def test_the_layer_apply_writes_the_camera_on_the_camera_lane(boundary, monkeypatch):
    import ui.layer_control as layer_control

    monkeypatch.setattr(layer_control, 'submit_reported', boundary.submit_reported)
    context = MagicMock()
    context.settings = {'Green': {'exposure_ms': 2.0, 'gain_db': 1.0}, 'protocol_led_on': True}
    context.session.run_lockout = False
    context.scope.imaging.is_focusing = False
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    widget = MagicMock()
    widget.layer = 'Green'
    widget._initializing = False
    widget.effective_auto_gain.return_value = False
    import modules.config_ui_getters as getters

    monkeypatch.setattr(getters, 'get_auto_gain_settings', lambda: {'target_brightness': 0.3})
    monkeypatch.setattr(getters, 'get_ag_ae_max_exposure_ms', lambda layer: 50.0)
    monkeypatch.setattr(getters, 'get_ag_ae_min_exposure_ms', lambda layer: 0.1)

    layer_control.LayerControl.apply_settings(widget, update_led=False)

    camera = [s for s in boundary.submits if s.label == 'CAMERA_SETTINGS_Green']
    assert len(camera) == 1, [s.label for s in boundary.submits]
    assert camera[0].lane is context.camera_executor
    context.lumaview.scope.imaging.apply_layer_camera_settings.assert_called_once_with(
        layer='Green',
        gain_db=1.0,
        exposure_ms=2.0,
        auto_gain=False,
        auto_gain_settings={
            'target_brightness': 0.3,
            'max_exposure_ms': 50.0,
            'min_exposure_ms': 0.1,
        },
    )


@pytest.mark.parametrize(
    'handler, widget_id, member, expected_args',
    [
        (
            'update_high_conversion_gain',
            'high_conversion_gain',
            'set_conversion_gain_mode',
            ('High',),
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
    ctx.settings['camera'] = {}
    widget = SimpleNamespace(ids={widget_id: SimpleNamespace(active=True)})

    getattr(advanced_settings.AdvancedSettings, handler)(widget)

    (submit,) = boundary.submits
    assert submit.lane is ctx.camera_executor
    getattr(ctx.scope.imaging, member).assert_called_once_with(*expected_args)


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


def test_the_acceleration_limit_is_written_on_the_io_lane(ctx, boundary, monkeypatch):
    import ui.advanced_settings as advanced_settings

    monkeypatch.setattr(advanced_settings, 'submit_reported', boundary.submit_reported)
    widget = SimpleNamespace(_pending_acceleration_pct=40)

    advanced_settings.AdvancedSettings._dispatch_acceleration_to_motor(widget)

    (submit,) = boundary.submits
    assert submit.lane is ctx.io_executor
    ctx.scope.motion.set_acceleration_limit.assert_called_once_with(val_pct=40)


def test_goto_focus_moves_z_to_the_layers_saved_focus(ctx, monkeypatch):
    import ui.layer_control as layer_control
    import ui.ui_helpers as ui_helpers

    moved = MagicMock()
    monkeypatch.setattr(ui_helpers, 'move_absolute', moved)
    monkeypatch.setattr(layer_control.gui_logger, 'button', MagicMock())

    layer_control.LayerControl.goto_focus(SimpleNamespace(layer='Green'))

    moved.assert_called_once_with('Z', 4321.0)


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

    protocol_settings.ProtocolSettings.go_to_step(panel, step_idx=2)

    navigated.assert_called_once()
    assert navigated.call_args.kwargs['include_move'] is True
    assert navigated.call_args.kwargs['step_idx'] == 2


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
