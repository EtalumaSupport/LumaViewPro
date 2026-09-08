# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: a scope swap must rebind every scope-holding session object.

Swapping in a new Lumascope and rewiring only the sequenced-capture
runner and the autofocus runner leaves the UI listener bridge
registered on the DISCARDED scope, so position, LED, and
camera-setting events from the live scope reach no listener: stage
redraw, LED button state, and gain/exposure text silently stop
updating until app restart. session.scope likewise keeps pointing at
the discarded scope -- start_application_session() reads it -- and
the manual-recording controller's scope handle goes stale the same
way (a later recording would drive the discarded scope).

One stranded-reference cluster, three members: the bridge, the
session, the recording controller. The bridge exposes rebind(scope);
the session exposes set_scope(scope) which rewires itself and the
recording controller. Whoever swaps a scope calls both; these are
the seams the auto-reconnect item inherits.
"""

from unittest.mock import MagicMock

from tests.scope_fakes import spec_scope


def _run_dispatch_inline(func, dt):
    func(0)


class TestBridgeRebind:
    def test_rebind_moves_listeners_to_new_scope(self):
        from modules.ui_listener_bridge import UIListenerBridge

        old_scope = spec_scope()
        new_scope = spec_scope()
        bridge = UIListenerBridge(
            scope=old_scope,
            ctx=MagicMock(),
            stage=MagicMock(),
            ui_dispatcher=_run_dispatch_inline,
        )
        bridge.register_all()
        old_scope.motion.add_position_listener.assert_called_once()
        registered_position = old_scope.motion.add_position_listener.call_args.args[0]
        registered_led = old_scope.illumination.add_led_listener.call_args.args[0]
        registered_camera = old_scope.imaging.add_camera_listener.call_args.args[0]

        bridge.rebind(new_scope)

        # The old scope keeps no listener registrations (it is about to
        # be discarded; a lingering registration pins it in memory and
        # double-fires if it is ever reused).
        old_scope.motion.remove_position_listener.assert_called_once_with(registered_position)
        old_scope.illumination.remove_led_listener.assert_called_once_with(registered_led)
        old_scope.imaging.remove_camera_listener.assert_called_once_with(registered_camera)

        # The new scope carries the same three listeners.
        new_scope.motion.add_position_listener.assert_called_once_with(registered_position)
        new_scope.illumination.add_led_listener.assert_called_once_with(registered_led)
        new_scope.imaging.add_camera_listener.assert_called_once_with(registered_camera)

    def test_rebind_reads_driver_truth_from_new_scope(self):
        """After rebind, a listener event reads driver state from the
        NEW scope, not the construction-time one."""
        from modules.ui_listener_bridge import UIListenerBridge

        old_scope = spec_scope()
        new_scope = spec_scope()
        ctx = MagicMock()
        ctx.ready = True
        bridge = UIListenerBridge(
            scope=old_scope,
            ctx=ctx,
            stage=MagicMock(),
            ui_dispatcher=_run_dispatch_inline,
        )
        bridge.register_all()
        bridge.rebind(new_scope)

        # Pre-seed the lazily-imported widget class so the LED write does
        # not import Kivy inside this headless test.
        bridge._LayerControl = MagicMock()
        bridge._on_led_state_changed('BF', True, 10.0, 'test')
        assert not old_scope.illumination.get_led_state.called, (
            'a rebound bridge must not read LED state from the discarded scope'
        )
        assert new_scope.illumination.get_led_state.called


class TestSessionSetScope:
    def test_set_scope_rewires_session_and_recording_controller(self):
        from modules.scope_session import ScopeSession

        old_scope = spec_scope()
        new_scope = spec_scope()
        session = ScopeSession(
            settings={},
            scope=old_scope,
            io_executor=MagicMock(),
            camera_executor=MagicMock(),
        )

        session.set_scope(new_scope)

        assert session.scope is new_scope
        assert session.manual_recording._scope is new_scope, (
            'the recording controller is part of the same stranded-reference '
            'cluster: left on the old handle, a later recording '
            'drives the discarded scope'
        )
