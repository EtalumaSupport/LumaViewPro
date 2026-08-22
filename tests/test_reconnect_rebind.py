# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: reconnect must rebind every scope-holding session object.

reconnect() builds a NEW Lumascope and rewires the sequenced-capture
runner and the autofocus runner -- but nothing else. The UI listener
bridge keeps its listeners registered on the DISCARDED scope, so from
the first reconnect onward position, LED, and camera-setting events
from the live scope reach no listener: stage redraw, LED button
state, and gain/exposure text silently stop updating until app
restart. session.scope likewise keeps pointing at the discarded
scope -- and reconnect() itself calls
session.start_application_session(), which reads it -- and the
manual-recording controller's scope handle goes stale the same way
(a post-reconnect recording would drive the discarded scope).

One stranded-reference cluster, three members: the bridge, the
session, the recording controller. The bridge exposes rebind(scope);
the session exposes set_scope(scope) which rewires itself and the
recording controller; reconnect() calls both.
"""

from unittest.mock import MagicMock

import tests.ast_seams as ast_seams
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
            'cluster: left on the old handle, a post-reconnect recording '
            'drives the discarded scope'
        )


class TestReconnectWiring:
    """reconnect() calls the two rebind seams. AST pin (not a string
    pin) so a refactor of the surrounding code cannot silently drop
    the calls."""

    def _method_calls_in(self, rel_path, func_name, class_name):
        import ast

        node = ast_seams.find_def(rel_path, func_name, class_name=class_name)
        assert node is not None, f'{rel_path}: {class_name}.{func_name} not found'
        calls = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                parts = []
                cur = sub.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                calls.add('.'.join(reversed(parts)))
        return calls

    def _attribute_chains_in(self, rel_path, func_name, class_name):
        """Every dotted attribute chain in the method, with its line."""
        import ast

        node = ast_seams.find_def(rel_path, func_name, class_name=class_name)
        assert node is not None, f'{rel_path}: {class_name}.{func_name} not found'
        chains = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute):
                parts = []
                cur = sub
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                chains.append(('.'.join(reversed(parts)), sub.lineno))
        return chains

    def _call_linenos_in(self, rel_path, func_name, class_name):
        import ast

        node = ast_seams.find_def(rel_path, func_name, class_name=class_name)
        assert node is not None
        linenos = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                parts = []
                cur = sub.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                name = '.'.join(reversed(parts))
                linenos.setdefault(name, sub.lineno)
        return linenos

    def test_reconnect_rebinds_bridge_and_session(self):
        calls = self._method_calls_in(
            'ui/microscope_settings.py', 'reconnect', 'MicroscopeSettings'
        )
        assert 'ctx.ui_listener_bridge.rebind' in calls, (
            'reconnect() must rebind the UI listener bridge onto the new '
            'scope; without it every push-listener UI update is dead after '
            'the first reconnect'
        )
        assert 'ctx.session.set_scope' in calls, (
            'reconnect() must hand the new scope to the session; without it '
            'session.scope (read by start_application_session in this very '
            'method) and the recording controller drive the discarded scope'
        )

    def test_reconnect_services_the_scope_before_homing(self):
        """set_scope carries the whole scope bring-up (executor
        registration included); start_application_session dispatches
        ALL-axis homing through those executors. Reordered, the homing
        would hit an unserviced scope and run inline on the Kivy
        thread -- the exact harm the bring-up move fixed."""
        linenos = self._call_linenos_in(
            'ui/microscope_settings.py', 'reconnect', 'MicroscopeSettings'
        )
        assert 'ctx.session.set_scope' in linenos
        assert 'ctx.session.start_application_session' in linenos
        assert (
            linenos['ctx.session.set_scope'] < linenos['ctx.session.start_application_session']
        ), 'reconnect() must call set_scope BEFORE start_application_session'

    def test_reconnect_checks_activity_before_teardown(self):
        """The set_scope guard fires only AFTER disconnect() has already
        torn the camera down under a live activity -- too late to
        protect it. reconnect() must read the activity facts
        (exclusive_activity / manual_recording.is_busy) before the
        first disconnect call."""
        chains = self._attribute_chains_in(
            'ui/microscope_settings.py', 'reconnect', 'MicroscopeSettings'
        )
        linenos = self._call_linenos_in(
            'ui/microscope_settings.py', 'reconnect', 'MicroscopeSettings'
        )
        disconnect_line = min(
            line for name, line in linenos.items() if name.endswith('.disconnect')
        )
        guard_reads = [
            line
            for name, line in chains
            if name.endswith('.exclusive_activity') or name.endswith('.is_busy')
        ]
        assert guard_reads and min(guard_reads) < disconnect_line, (
            'reconnect() must refuse (read exclusive_activity / '
            'manual_recording.is_busy) BEFORE tearing down the old scope'
        )

    def test_reconnect_refreshes_ctx_scope_registry_field(self):
        import ast

        node = ast_seams.find_def(
            'ui/microscope_settings.py', 'reconnect', class_name='MicroscopeSettings'
        )
        assert node is not None
        assigns_ctx_scope = any(
            isinstance(sub, ast.Assign)
            and any(
                isinstance(t, ast.Attribute)
                and t.attr == 'scope'
                and isinstance(t.value, ast.Name)
                and t.value.id == 'ctx'
                for t in sub.targets
            )
            for sub in ast.walk(node)
        )
        assert assigns_ctx_scope, (
            'reconnect() must refresh ctx.scope; the display render path, '
            'the stall watchdog, composite capture, and plugins all read '
            'the registry field, and with no refresh they operate on the '
            'discarded scope until app restart'
        )
