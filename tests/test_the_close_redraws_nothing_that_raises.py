# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Closing the scope raises nothing in the window's position redraws.

The close's disconnect() shuts the scope's lanes, then writes every axis
UNKNOWN, which tells the position listeners; the window's bridge schedules
the XY readout and the stage map for the next Kivy tick. Both read
get_axis_limits only to clamp a position the API already keeps inside
travel, and get_axis_limits refuses scope_disconnected once the lanes are
shut: the refusal was raised on the Kivy clock, the loop ended, and on_stop
never ran (the LS850T, 2026-10-15 and 2026-10-16).

Driven here as the bench met it: a simulated session, the real listener
bridge, the real XY readout and stage-map reads, a dispatcher that queues
what the bridge schedules and is drained on this thread as Kivy's clock
drains it, and the close on a 'session-close' thread held at the motor
board's own disconnect -- where the bench was when its redraw ran -- until
one tick has drained.
"""

import queue
import threading
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.stage as stage_module
from modules.kivy_utils import UiDispatcher
from modules.scope_session import ScopeSession
from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
from tests.settings_fixtures import complete_settings
from ui.listener_bridge import UIListenerBridge
from ui.motion_settings import XYStageControl
from ui.stage import Stage


class _DrawnError(Exception):
    """The stage map's reads are done and it went on to draw."""


def _refuse_to_draw():
    raise _DrawnError


@pytest.fixture
def window(tmp_path, monkeypatch):
    session = ScopeSession.create(
        complete_settings(
            microscope='LS850T',
            turret_objectives=dict(TEST_TURRET_OBJECTIVES),
            live_folder=str(tmp_path),
        ),
        simulate=True,
    )
    home_sim_scope(session.scope)
    scheduled = queue.Queue()
    ScopeSession.set_ui_dispatcher(
        UiDispatcher(schedule=lambda f, t=0: scheduled.put(f), thread=threading.current_thread())
    )
    xy = XYStageControl.__new__(XYStageControl)
    stage = Stage.__new__(Stage)
    stage._protocol_step_redraw = False
    stage._prev_x_target = stage._prev_y_target = None
    stage._prev_x_current = stage._prev_y_current = None
    stage.draw_labware = lambda *a, **k: stage.draw_labware_io_calculations()
    monkeypatch.setattr(stage_module, 'get_selected_labware', _refuse_to_draw)
    ctx = SimpleNamespace(
        lumaview=SimpleNamespace(scope=session.scope),
        scope=session.scope,
        session=session,
        settings={'stage_offset': {'x': 0, 'y': 0}},
        ready=True,
        motion_settings=SimpleNamespace(
            update_xy_stage_control_gui=xy.get_xy_targets,
            # No vertical control: the Z and turret handlers return early.
            ids={},
        ),
        image_settings=SimpleNamespace(draw_shown_channel=lambda: None),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    bridge = UIListenerBridge(scope=session.scope, ctx=ctx, stage=stage)
    bridge.register_all()
    try:
        yield SimpleNamespace(session=session, scheduled=scheduled, bridge=bridge)
    finally:
        ScopeSession.set_ui_dispatcher(None)
        session.shutdown()


def _drain(scheduled, raised):
    while True:
        try:
            callback = scheduled.get_nowait()
        except queue.Empty:
            return
        try:
            callback(0)
        except _DrawnError:
            pass
        except Exception as e:
            raised.append(f'{type(e).__name__}: {e}')


def test_the_redraws_the_close_schedules_raise_nothing(window):
    session, scheduled = window.session, window.scheduled
    _drain(scheduled, [])
    board = session.scope._motion_driver
    disconnect_board = board.disconnect
    at_the_board = threading.Event()
    drained = threading.Event()

    def held(*args, **kwargs):
        at_the_board.set()
        drained.wait(10)
        return disconnect_board(*args, **kwargs)

    board.disconnect = held
    closing = threading.Thread(target=session.shutdown, name='session-close')
    closing.start()
    raised = []
    assert at_the_board.wait(10)
    _drain(scheduled, raised)
    drained.set()
    closing.join(30)
    _drain(scheduled, raised)

    assert not closing.is_alive()
    assert raised == []


def test_once_the_bridge_unregisters_the_scopes_events_schedule_nothing(window):
    """The host's close drops the bridge before the scope goes away, so a
    move, an LED and a shown-channel change reach no window redraw; asked
    twice (a close that raised, then on_stop), it raises nothing."""
    session, scheduled = window.session, window.scheduled

    def tell(x_um, ma, layer):
        # Started, not waited on: this thread is the dispatcher's UI thread,
        # and the move's start already tells the position listeners.
        session.scope.motion.start_move_xy(x_um=x_um)
        session.scope.illumination.led_on('BF', ma)
        session.show_channel(layer)

    tell(10000.0, 10.0, 'Blue')
    assert not scheduled.empty(), 'registered, the events schedule redraws'
    window.bridge.unregister_all()
    window.bridge.unregister_all()
    _drain(scheduled, [])

    tell(20000.0, 20.0, 'BF')

    assert scheduled.empty()
