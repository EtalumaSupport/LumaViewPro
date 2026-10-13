# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The GUI's subscriber to the scope's state-change events.

Lumascope publishes position, LED and camera-setting changes to
registered listeners (``add_position_listener``, ``add_led_listener``,
``add_camera_listener``), and the Session the channel it shows
(``add_shown_channel_listener``). This module holds the GUI's handlers for
position, LED and shown channel: stage redraw on motion, LED button state on
LED change, the open drawer on a change of the shown channel.
Camera-setting changes have no GUI handler: a layer's gain and exposure
boxes show the stored setting, and what the camera applied (a quantizing
camera answers 1000 ms with 1000.0057) is the API's to answer and the
saved frame's to record, never the box's to show.

It belongs in ``ui/`` because every handler ends in a widget write.
The layers below publish the events and hold the truth; they do not
import this one.

Two things earn a class here rather than a set of closures per listener at the
registration site. The coalescing state (a ``_pending_*`` map per
listener, so a burst of events costs at most one UI update per frame)
is one implementation instead of slightly different copies. Each
handler reaches the UI thread through the process's one UI dispatcher
(``kivy_utils.schedule_ui``) rather than an imported ``Clock``, so a test,
which sets none, drives the handlers synchronously and asserts what they
wrote.

Usage:

    from ui.listener_bridge import UIListenerBridge
    bridge = UIListenerBridge(
        scope=lumaview.scope,
        ctx=ctx,
        stage=stage,
    )
    bridge.register_all()
"""

from __future__ import annotations

from lvp_logger import logger
import modules.common_utils as common_utils
from modules import gui_logger
from modules.kivy_utils import schedule_ui
from modules.lumascope_api.illumination import LedState


class UIListenerBridge:
    """Wires Lumascope's position and LED push-listener events to UI updates.

    The bridge owns the per-listener coalescing state (each listener
    deduplicates rapid back-to-back events, scheduling at most one UI
    update per Kivy frame). It does NOT own widget references -- those
    are looked up via ``ctx`` so a widget rebuild (LS850 <-> LS620 scope
    swap) doesn't leave the bridge holding stale handles.
    """

    def __init__(self, *, scope, ctx, stage):
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
        """
        self._scope = scope
        self._ctx = ctx
        self._stage = stage

        # Per-listener coalescing state -- populated lazily on first
        # event for each LED color so the bridge construction stays
        # cheap.
        self._pending_led_updates: dict[str, bool] = {}
        self._shown_channel_pending = False

    # ------------------ Listener implementations ------------------

    def _on_position_change(self, axis, at):
        """Position listener -- XY motion redraws the stage; Z the Z slider and text; T the turret.

        Fires from the IO worker thread (or whichever thread mutated the
        position cache), for every move a run, an autofocus or a person
        makes. ``at`` is the axis's ``AxisPosition``: its polled position,
        None while it homes or is unknown. Marshals to UI via
        ``schedule_ui``.
        """
        ctx = self._ctx
        if axis in ('X', 'Y'):
            schedule_ui(lambda dt: ctx.motion_settings.update_xy_stage_control_gui(), 0)
            schedule_ui(lambda dt: self._stage.draw_labware(), 0)
            return
        z_ctrl = ctx.motion_settings.ids.get('verticalcontrol_id')
        if not z_ctrl:
            return
        if axis == 'Z':
            schedule_ui(lambda dt: z_ctrl.show_z(at.position), 0)
        elif axis == 'T':
            # Display only: a person's own turret press asks the objective
            # question where it ends, never on each event of the move.
            schedule_ui(lambda dt: z_ctrl.show_turret_state(prompt=False), 0)

    def _on_led_state_changed(self, address, name, state, illumination_ma):
        """LED listener -- coalesces rapid stim pulses to one UI update per address per Kivy frame.

        Replaces all manual ``update_led_toggle_ui()`` calls.
        """
        if address in self._pending_led_updates:
            return  # Already scheduled, will pick up latest state
        self._pending_led_updates[address] = True

        def _update_led_ui(dt, a=address):
            self._pending_led_updates.pop(a, None)
            if not self._ctx.ready:
                return
            # Re-read: the event's fields may be stale by now. None with
            # no LED board installed, after disconnect.
            reading = self._scope.illumination.get_led_state(channel=a)
            if reading is not None and reading.name is not None:
                self._write_led_button(reading.name, lit=reading.state is LedState.LIT)

        schedule_ui(_update_led_ui, 0)

    def _on_shown_channel_changed(self, layer, lit):
        """Shown-channel listener -- the drawers draw the Session's shown channel.

        In a run too: a run's steps move the shown channel, and the drawer
        follows. Coalesced to one draw per Kivy frame, which re-reads the
        store, so a run's rapid steps draw the newest.
        """
        if self._shown_channel_pending:
            return
        self._shown_channel_pending = True

        def _draw(dt):
            self._shown_channel_pending = False
            if not self._ctx.ready:
                return
            self._ctx.image_settings.draw_shown_channel()

        schedule_ui(_draw, 0)

    def _write_led_button(self, color: str, *, lit: bool) -> None:
        """Write one layer's enable toggle as the API reads its LED.

        Writes 'down'/'normal'. A state write is not a press -- the button
        drives an LED only from on_release -- so reflecting the API's
        reading cannot itself drive an LED.
        """
        ctx = self._ctx
        if not ctx.ready:
            return
        layer_obj = ctx.image_settings.layer_lookup(layer=color)
        target = 'down' if lit else 'normal'
        gui_logger.display(f'LED_BUTTON_{color}', 'lit' if target == 'down' else 'off')
        if layer_obj.ids['enable_led_btn'].state != target:
            layer_obj.ids['enable_led_btn'].state = target

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
            # None with no LED board installed: nothing is lit.
            lit = {
                r.name
                for r in self._scope.illumination.get_led_states() or ()
                if r.state is LedState.LIT
            }
            for color in common_utils.get_layers_with_led():
                self._write_led_button(color, lit=color in lit)

        schedule_ui(_reconcile, 0)

    # ------------------ Lifecycle ------------------

    def register_all(self):
        """Register every listener on the underlying Lumascope.

        Idempotent? The underlying add_*_listener methods do not
        de-duplicate, so calling this twice would double-fire each
        listener. Call once at application startup.
        """
        self._scope.motion.add_position_listener(self._on_position_change)
        self._scope.illumination.add_led_listener(self._on_led_state_changed)
        self._ctx.session.add_shown_channel_listener(self._on_shown_channel_changed)
        logger.info('[UIListenerBridge] registered position, LED and shown-channel listeners')
