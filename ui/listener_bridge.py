# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The GUI's subscriber to the scope's state-change events.

Lumascope publishes position, turret, LED, gain and exposure changes to
registered listeners (``add_position_listener``, ``add_turret_listener``,
``add_led_listener``, ``add_gain_listener``, ``add_exposure_listener``), and
the Session the channel it shows (``add_shown_channel_listener``) and each
change to the settings store (``add_settings_listener``). This module holds
the GUI's handlers for position, turret, LED, shown channel and settings:
stage redraw on motion, the turret buttons on a turret change, LED
button state on LED change, the open drawer on a change of the shown
channel, and the redraw of every panel showing a setting that changed,
whoever wrote it -- the window, or a REST client served beside it.
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

from collections.abc import Callable

from lvp_logger import logger
import modules.common_utils as common_utils
from modules import gui_logger
from modules.kivy_utils import schedule_ui
from modules.lumascope_api.illumination import LedState


def _microscope(ctx):
    return ctx.motion_settings.ids['microscope_settings_id']


def _redraw_layer(ctx, path: str) -> None:
    ctx.image_settings.layer_lookup(layer=path.split('.')[0]).sync_widgets_from_settings()


def _redraw_image_mode(ctx, path: str) -> None:
    _microscope(ctx)._redraw_image_mode()


def _redraw_framing(ctx, path: str) -> None:
    _microscope(ctx)._redraw_framing()


def _redraw_capture(ctx, path: str) -> None:
    _microscope(ctx).show_capture_settings()


def _redraw_stimulation(ctx, path: str) -> None:
    _microscope(ctx).apply_stimulation_support()


def _redraw_model(ctx, path: str) -> None:
    _microscope(ctx).show_scope_model()


def _redraw_turret(ctx, path: str) -> None:
    ctx.motion_settings.ids['verticalcontrol_id'].show_turret_state(prompt=False)


def _redraw_labware(ctx, path: str) -> None:
    ctx.motion_settings.ids['protocol_settings_id'].show_labware()


def _redraw_zstack(ctx, path: str) -> None:
    ctx.motion_settings.ids['verticalcontrol_id'].ids['zstack_id'].show_settings()


def _redraw_step_locations(ctx, path: str) -> None:
    ctx.stage.show_protocol_steps(enable=ctx.settings['show_step_locations'])


def _apply_live_view_fps(ctx, path: str) -> None:
    # The running display and the window's copy of the cap follow the store.
    ctx.live_view_fps = ctx.settings['live_view_fps']
    ctx.scope_display_thread.set_fps(ctx.live_view_fps)


def _redraw_advanced(ctx, path: str) -> None:
    popup = _microscope(ctx).advanced_settings_shown()
    if popup is not None:
        popup.show_settings()


# Every setting the window shows, by path prefix, and the redraw that shows
# it; a prefix may be shown in two places. ``<layer>`` is any layer's own
# settings. A changed path runs every redraw a row routes it to, each once
# per frame. tests/guards/test_every_setting_shown_is_redrawn_when_it_changes.py
# fails the build on a template leaf neither here nor in NOT_SHOWN.
SHOWN_SETTINGS: tuple[tuple[str, Callable[[object, str], None]], ...] = (
    ('<layer>', _redraw_layer),
    ('image_mode', _redraw_image_mode),
    ('binning', _redraw_framing),
    ('frame', _redraw_framing),
    ('image_output_format', _redraw_capture),
    ('jpg_quality', _redraw_capture),
    ('video_as_frames', _redraw_capture),
    ('scale_bar', _redraw_capture),
    ('show_tooltips', _redraw_capture),
    ('stimulation_enabled', _redraw_stimulation),
    ('stimulation_enabled', _redraw_advanced),
    ('microscope', _redraw_model),
    ('microscope', _redraw_advanced),
    ('objective_id', _redraw_turret),
    ('objective_confirmed', _redraw_turret),
    ('turret_objectives', _redraw_turret),
    ('protocol.labware', _redraw_labware),
    ('zstack', _redraw_zstack),
    ('show_step_locations', _redraw_step_locations),
    ('show_step_locations', _redraw_advanced),
    ('live_view_fps', _apply_live_view_fps),
    ('live_view_fps', _redraw_advanced),
    ('motion', _redraw_advanced),
    ('video', _redraw_advanced),
    ('separate_folder_per_channel', _redraw_advanced),
    ('protocol_led_on', _redraw_advanced),
    ('keep_led_between_steps', _redraw_advanced),
    ('tiling_overlap_percent', _redraw_advanced),
    ('camera', _redraw_advanced),
)

# The settings the window does not show: read where they are used, set only
# by the installation, derived by the store, or kept for a run, a capture
# or the next start.
NOT_SHOWN: frozenset[str] = frozenset(
    {
        'simulator_tier',
        'stage_offset',
        'live_folder',
        'bookmark',
        'channel_defaults',
        'protocol.period',
        'protocol.duration',
        'protocol.filepath',
        'protocol.autogain',
        'logging',
        'mode',
        'debug_mode',
        'cprofile_enabled',
        'focus_score_enabled',
        'profile_trace_enabled',
        'profile_trace_output_dir',
        'tracemalloc_enabled',
        'memory_profile_enabled',
        'memory_profile_interval_s',
        'preview_host_downscale',
        'fx2_debug_wire_enabled',
        'profiling',
        'ag_ae_max_exposure_ms',
        'lvp_lock_port',
        'rest_api',
        'step_channel_order',
    }
)


def _routed(path: str) -> list[tuple[Callable[[object, str], None], str | None]]:
    """The redraws *path* routes to, each with the layer it names (None for a panel's own)."""
    first = path.split('.')[0]
    keyed = f'<layer>{path[len(first) :]}' if first in common_utils.get_layers() else path
    return [
        (redraw, first if prefix == '<layer>' else None)
        for prefix, redraw in SHOWN_SETTINGS
        if keyed == prefix or keyed.startswith(f'{prefix}.')
    ]


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
        # The settings changed since the last frame's redraw, None while no
        # redraw is scheduled.
        self._settings_pending: list[str] | None = None

    # ------------------ Listener implementations ------------------

    def _on_position_change(self, axis, at):
        """Position listener -- XY motion redraws the stage; Z the Z slider and text.

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
            schedule_ui(lambda dt: z_ctrl.show_z(at.position_um), 0)

    def _on_turret_change(self, at):
        """Turret listener -- the turret buttons follow every turret change, whoever made it.

        A REST client's turret move included. ``at`` is the ``TurretStatus``;
        the display reads the slot from the API when it redraws. Marshals to
        UI via ``schedule_ui``.
        """
        z_ctrl = self._ctx.motion_settings.ids.get('verticalcontrol_id')
        if not z_ctrl:
            return
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
            # no LED board installed, after disconnect. Read by address,
            # since the event's address may have no name.
            readings = self._scope.illumination.get_led_states() or ()
            reading = next((r for r in readings if r.address == a), None)
            if reading is not None and reading.name is not None:
                self._write_led_button(reading.name, lit=reading.state == LedState.LIT)

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

    def _on_settings_changed(self, paths):
        """Settings listener -- every panel showing a changed setting redraws it from the store.

        Whoever wrote it: the window's own writes are redrawn too, so it
        shows what was stored (a gain that snapped). The paths of a frame
        are gathered and each redraw they route to runs once, on the UI
        thread, reading the store as it is then.
        """
        if self._settings_pending is not None:
            self._settings_pending.extend(paths)
            return
        self._settings_pending = list(paths)

        def _draw(dt):
            changed, self._settings_pending = self._settings_pending, None
            if not self._ctx.ready:
                return
            drawn = set()
            for path in changed:
                for redraw, layer in _routed(path):
                    key = (redraw, layer)
                    if key not in drawn:
                        drawn.add(key)
                        redraw(self._ctx, path)

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
                if r.state == LedState.LIT
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
        self._scope.motion.add_turret_listener(self._on_turret_change)
        self._scope.illumination.add_led_listener(self._on_led_state_changed)
        self._ctx.session.add_shown_channel_listener(self._on_shown_channel_changed)
        self._ctx.session.add_settings_listener(self._on_settings_changed)
        logger.info(
            '[UIListenerBridge] registered position, turret, LED, shown-channel and settings listeners'
        )

    def unregister_all(self):
        """Remove every listener register_all added; a second call removes nothing.

        The window's close calls it before the scope is torn down, so the
        teardown's own position, turret and LED events schedule no redraw of a scope
        that is going.
        """
        self._scope.motion.remove_position_listener(self._on_position_change)
        self._scope.motion.remove_turret_listener(self._on_turret_change)
        self._scope.illumination.remove_led_listener(self._on_led_state_changed)
        self._ctx.session.remove_shown_channel_listener(self._on_shown_channel_changed)
        self._ctx.session.remove_settings_listener(self._on_settings_changed)
        logger.info(
            '[UIListenerBridge] removed position, turret, LED, shown-channel and settings listeners'
        )
