# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""LVP-A-6 -- Lumascope state-change -> UI update bridge.

Lumascope publishes state-change events (position, LED, camera-setting)
via ``add_position_listener``, ``add_led_listener``, and
``add_camera_listener``. The handlers translate those events into UI
updates: stage redraw on motion, LED button state on LED change, gain
/ exposure text on camera-setting change. Three listener implementations,
each with closure state for coalescing rapid events.

Pre-LVP-A-6 these lived inline in ``lumaviewpro.py:on_start`` (~110
lines of nested closures). Lifting them into one module:

- Lets future entry points (REST API status mirror, headless metrics
  consumer, CLI tools that mirror state to a TUI) wire the same
  listeners through one call.
- Makes the coalescing pattern (``_pending_*`` dict per listener)
  reusable instead of three slightly-different copies.
- Stays Rule-15 clean: the bridge takes a ``ui_dispatcher`` callable
  (``Clock.schedule_once`` for the Kivy app, anything matching the
  ``(callable, dt)`` signature elsewhere) so it doesn't import Kivy.

Usage:

    from modules.ui_listener_bridge import UIListenerBridge
    bridge = UIListenerBridge(
        scope=lumaview.scope,
        ctx=ctx,
        stage=stage,
        ui_dispatcher=Clock.schedule_once,
    )
    bridge.register_all()
"""

from __future__ import annotations

from lvp_logger import logger
import modules.common_utils as common_utils


class UIListenerBridge:
    """Wires Lumascope's three push-listener events to UI updates.

    The bridge owns the per-listener coalescing state (each listener
    deduplicates rapid back-to-back events, scheduling at most one UI
    update per Kivy frame). It does NOT own widget references -- those
    are looked up via ``ctx`` so a widget rebuild (LS850 <-> LS620 scope
    swap) doesn't leave the bridge holding stale handles.
    """

    def __init__(self, *, scope, ctx, stage, ui_dispatcher):
        """Initialize the bridge.

        Args:
            scope: ``Lumascope`` API instance -- listener-add methods
                are called on this.
            ctx: ``AppContext`` -- UI widget lookups (motion_settings,
                image_settings) and runtime state (ready, settings,
                session) read from here so a widget rebuild doesn't
                strand the bridge.
            stage: Stage widget -- the position listener calls
                ``stage.draw_labware()`` on XY motion.
            ui_dispatcher: Callable matching
                ``Clock.schedule_once(func, dt)`` -- used to marshal
                listener callbacks (which fire on the worker thread
                that caused the change) onto the UI thread. Passed
                instead of imported (the bridge stays GUI-agnostic).
        """
        self._scope = scope
        self._ctx = ctx
        self._stage = stage
        self._ui_dispatch = ui_dispatcher

        # Per-listener coalescing state -- populated lazily on first
        # event for each LED color so the bridge construction stays
        # cheap.
        self._pending_led_updates: dict[str, bool] = {}

        # LayerControl is imported lazily inside the LED listener to
        # avoid a UI-import at module-load time (the bridge module
        # stays GUI-agnostic: no GUI imports).
        self._LayerControl = None

    # ------------------ Listener implementations ------------------

    def _on_position_change(self, axis, target, state):
        """Position listener -- XY motion redraws stage; Z motion updates Z text.

        Fires from the IO worker thread (or whichever thread mutated
        position cache). Marshals to UI via ``ui_dispatcher``.
        """
        ctx = self._ctx
        if axis in ('X', 'Y'):
            self._ui_dispatch(lambda dt: ctx.motion_settings.update_xy_stage_control_gui(), 0)
            self._ui_dispatch(lambda dt: self._stage.draw_labware(), 0)
        elif axis == 'Z':
            z_ctrl = ctx.motion_settings.ids.get('verticalcontrol_id')
            if z_ctrl:
                self._ui_dispatch(lambda dt: z_ctrl._update_z_text(target), 0)

    def _on_led_state_changed(self, color, enabled, mA, owner):
        """LED listener -- coalesces rapid stim pulses to one UI update per color per Kivy frame.

        Replaces all manual ``update_led_toggle_ui()`` calls.
        """
        if color in self._pending_led_updates:
            return  # Already scheduled, will pick up latest state
        self._pending_led_updates[color] = True

        def _update_led_ui(dt, c=color):
            self._pending_led_updates.pop(c, None)
            self._write_led_button_from_driver(color=c)

        self._ui_dispatch(_update_led_ui, 0)

    def _write_led_button_from_driver(self, color: str) -> None:
        """Write one channel's enable toggle from CURRENT driver truth.

        Reads the driver state (not event args, which may be stale) and
        writes 'down'/'normal' with the LED-command suppression flag held,
        so reflecting driver truth cannot itself drive an LED.
        """
        ctx = self._ctx
        if not ctx.ready:
            return
        try:
            layer_obj = ctx.image_settings.layer_lookup(layer=color)
        except Exception:
            return
        # Lazy import keeps the executor layer GUI-agnostic (no GUI
        # module imported at bridge construction time).
        if self._LayerControl is None:
            from ui.layer_control import LayerControl

            self._LayerControl = LayerControl
        state = self._scope.illumination.get_led_state(color=color)
        target = 'down' if state.get('enabled', False) else 'normal'
        if layer_obj.ids['enable_led_btn'].state != target:
            self._LayerControl._suppressing_led_log = True
            try:
                layer_obj.ids['enable_led_btn'].state = target
            finally:
                self._LayerControl._suppressing_led_log = False

    def reconcile_led_buttons(self) -> None:
        """Level-based reconcile of EVERY channel's enable toggle to driver truth.

        The LED listener above is edge-triggered: a widget left stale by a
        writer whose expected LED event never fired (e.g. a run-indicator
        write for a step a Stop cancelled, restored by an all-dark diff that
        emits no events) is never corrected by events alone. Call this at
        run completion, AFTER the hardware restore has settled, so the
        read is the run's true end state.
        """

        def _reconcile(dt):
            for color in common_utils.get_layers_with_led():
                self._write_led_button_from_driver(color=color)

        self._ui_dispatch(_reconcile, 0)

    def _on_camera_setting_changed(self, param, value):
        """Camera listener -- fires on set_gain_db / set_exposure_ms.

        Updates the OPEN tab's text fields with what the camera is
        actually running at (after AF, auto-gain, REST API, etc.).
        Never writes back into the slider -- that was the root cause of
        the handler-recursion feedback loop in #617.

        During a protocol run the engine cycles gain/exposure across
        channels; the listener no-ops to avoid showing channel-N's
        values in the UI for channel-M's open tab.
        """
        ctx = self._ctx

        def _update_camera_ui(dt, p=param, v=value):
            if not ctx.ready:
                return
            if ctx.session.is_protocol_running:
                return
            opened_layer = common_utils.get_opened_layer(ctx.image_settings)
            if not opened_layer:
                return
            try:
                layer_obj = ctx.image_settings.layer_lookup(layer=opened_layer)
            except Exception:
                return
            if not layer_obj:
                return
            # Respect an _initializing flag set by another code path
            # (e.g. layer switch via set_step_state).
            if layer_obj._initializing:
                return

            settings = ctx.settings
            if p == 'gain':
                rounded = round(v, 1)
                # Only update if this layer's configured value matches
                # what the camera reports. If another layer changed
                # the camera (composite, AF restore), don't display
                # its value in this layer's text field. (#610)
                expected = settings[opened_layer]['gain_db']
                if abs(rounded - expected) > 0.5:
                    return
                text = str(rounded)
                if layer_obj.ids['gain_text'].text != text:
                    layer_obj.ids['gain_text'].text = text
            elif p == 'exposure':
                rounded = round(v, 2)
                expected = settings[opened_layer]['exp_ms']
                if abs(rounded - expected) > 0.5:
                    return
                text = str(rounded)
                if layer_obj.ids['exp_text'].text != text:
                    layer_obj.ids['exp_text'].text = text

        self._ui_dispatch(_update_camera_ui, 0)

    # ------------------ Lifecycle ------------------

    def register_all(self):
        """Register every listener on the underlying Lumascope.

        Idempotent? The underlying add_*_listener methods do not
        de-duplicate, so calling this twice would double-fire each
        listener. Call once at application startup.
        """
        self._scope.motion.add_position_listener(self._on_position_change)
        self._scope.illumination.add_led_listener(self._on_led_state_changed)
        self._scope.imaging.add_camera_listener(self._on_camera_setting_changed)
        logger.info('[UIListenerBridge] registered position + LED + camera listeners')

    def rebind(self, scope) -> None:
        """Move every listener registration onto a NEW scope.

        A reconnect discards the old Lumascope and builds a fresh one;
        listeners left registered on the discarded scope never fire
        again (stage redraw, LED buttons, and gain/exposure text all go
        silent until app restart), and they pin the dead scope in
        memory. Unregister from the old scope first so a later reuse of
        it cannot double-fire.
        """
        old = self._scope
        old.motion.remove_position_listener(self._on_position_change)
        old.illumination.remove_led_listener(self._on_led_state_changed)
        old.imaging.remove_camera_listener(self._on_camera_setting_changed)
        self._scope = scope
        scope.motion.add_position_listener(self._on_position_change)
        scope.illumination.add_led_listener(self._on_led_state_changed)
        scope.imaging.add_camera_listener(self._on_camera_setting_changed)
        logger.info('[UIListenerBridge] rebound position + LED + camera listeners to new scope')
