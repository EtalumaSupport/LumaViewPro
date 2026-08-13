# Copyright Etaluma, Inc.
"""
UI helper functions -- manipulate Kivy widgets, window titles, LED buttons.

Moved from modules/ui_helpers.py to ui/ because this is GUI code (imports
Kivy Window, ScrollView). A compatibility shim at modules/ui_helpers.py
re-exports everything for existing callers.
"""

import logging
import typing

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.scrollview import ScrollView
from modules.kivy_utils import schedule_ui as _schedule_ui

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
from modules.exceptions import ProtocolRunRefusedError
from modules.sequential_io_executor import IOTask

logger = logging.getLogger('LVP.modules.ui_helpers')


def publish_protocol_running(running: bool) -> None:
    """Mirror the protocol-running state onto the App ``protocol_running``
    property so kv ``disabled:`` bindings react.

    Written on the Kivy main thread (safe to call from any thread). The
    ctx.protocol_running Event stays the authoritative store; this only
    publishes a UI reflection of it.
    """
    app = App.get_running_app()
    if app is None:
        return
    Clock.schedule_once(lambda dt: setattr(app, 'protocol_running', running), 0)


def run_committed_start(
    commit_fn: typing.Callable[[], None],
    start_fn: typing.Callable[[], None],
) -> None:
    """Commit the running-UI state, then start -- restoring the
    pre-commit state when start() still refuses.

    start() can refuse AFTER a successful prepare() (a rival activity
    claim, an already-running race); without the restore, the committed
    state (the protocol_running Event, its kv mirror, the stage motion
    lock) strands set with no run live and nothing scheduled to clear
    it. The snapshot reads the EVENT, never the mirror property -- the
    mirror publishes through Clock.schedule_once and can be one frame
    stale; restore republishes the mirror from the snapshotted Event
    value, and a rival's held Event stays held. Pre-commit refusals
    raise out of prepare() before commit_fn runs and never reach here.
    """
    ctx = _app_ctx.ctx
    event_was_set = ctx.protocol_running.is_set()
    motion_was_enabled = ctx.stage.motion_capability()
    commit_fn()
    try:
        start_fn()
    except ProtocolRunRefusedError:
        if event_was_set:
            ctx.protocol_running.set()
        else:
            ctx.protocol_running.clear()
        publish_protocol_running(event_was_set)
        ctx.stage.set_motion_capability(motion_was_enabled)
        raise


def run_with_refusal_boundary(
    start_fn: typing.Callable[[], None],
    on_refused: typing.Callable[[], None],
) -> None:
    """The single UI boundary for the runner's typed run refusal.

    A refused run is a designed outcome, not a failure to propagate: the
    runner's refusal funnel has already logged it and notified the user
    exactly once, and no running-state was committed (commit_ui_state
    runs only after a successful prepare). What remains is per-starter:
    undo the pre-gate button cosmetics via on_refused. Every UI starter
    (scan, protocol, autofocus scan, z-stack) routes its prepare/start
    sequence through this one handler so refusal handling cannot drift
    between them.
    """
    try:
        start_fn()
    except ProtocolRunRefusedError:
        on_refused()


# ============================================================================
# Saved-folder helper
# ============================================================================


def set_last_save_folder(dir):
    if dir is None:
        return

    ctx = _app_ctx.ctx
    ctx.last_save_folder = dir


# ============================================================================
# Protocol nav helpers
# ============================================================================


def focus_log(positions, values):
    ctx = _app_ctx.ctx
    ctx.focus_round = config_helpers.focus_log(positions, values, ctx.focus_round, ctx.source_path)


def update_autofocus_selection_after_protocol():
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers():
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        layer_obj.init_autofocus()


def find_nearest_step(x, y, protocol):
    return config_helpers.find_nearest_step(x, y, protocol)


# ============================================================================
# LED / Illumination Helpers
# ============================================================================

# _handle_ui_for_leds_off and _handle_ui_for_led removed --
# LED observer handles UI sync. See Phase 1 commit 96defe3.


def scope_leds_off(no_callback: bool = False):
    """Turn off all LEDs. UI sync is handled by the LED observer."""
    ctx = _app_ctx.ctx
    if ctx.protocol_running.is_set():
        return

    # LED observer handles UI button sync -- no manual callback needed.
    # The no_callback parameter is kept for API compatibility but is now
    # effectively always True (observer replaces the callback).
    ctx.scope.illumination.leds_off_async()


# ============================================================================
# Protocol Step Navigation Helpers
# ============================================================================


def _update_step_number_callback(step_num: int):
    ctx = _app_ctx.ctx
    protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
    protocol_settings.curr_step = step_num - 1
    _schedule_ui(lambda dt: protocol_settings.update_step_ui(), 0)


# ============================================================================
# Motion Helpers
# ============================================================================


def _handle_ui_update_for_axis(axis: str, vertical_control: bool = False):
    ctx = _app_ctx.ctx
    axis = axis.upper()
    if axis == 'Z':
        ctx.motion_settings.ids['verticalcontrol_id'].update_gui(vertical_control=vertical_control)
    elif axis in ('X', 'Y', 'XY'):
        ctx.motion_settings.update_xy_stage_control_gui()


def _handle_autofocus_ui(pos: float):
    ctx = _app_ctx.ctx
    ctx.motion_settings.ids['verticalcontrol_id'].update_autofocus_gui(pos=pos)


def _user_motion_locked(axis: str) -> bool:
    """True while an exclusive activity locks the control surface.

    kv ``disabled:`` reaches widgets, but bound input observers (the
    viewer's right-click-to-center, scroll-to-focus) fire before any
    widget's disabled state is consulted -- so the user-gesture motion
    funnel enforces the lock itself, once, for every gesture path.
    Headless callers run without a Kivy app and are never locked here.
    """
    from kivy.app import App

    app = App.get_running_app()
    locked = getattr(app, 'controls_locked', False) if app is not None else False
    # The lock engages only on the App property's explicit True: the
    # real BooleanProperty always yields a bool, and anything else means
    # there is no real app (headless / mocked hosts are never locked).
    if locked is not True:
        return False
    logger.info(f'[UI] {axis} move blocked: controls locked (protocol run or recording active)')
    return True


# Wrapper to move and update the UI position. `protocol=False` (UI
# thread) dispatches via the API's async path. `protocol=True` runs on
# protocol_thread -- a DIFFERENT thread from the io_executor worker --
# so the move is queued through io_executor.protocol_put and awaited,
# keeping it ordered behind the step's leds_off/led_on on the single
# worker. A direct call would race them and leave the prior step's LED
# lit through the move. Awaiting is deadlock-free: the caller is
# protocol_thread, not the worker.
def move_absolute_position(
    axis: str,
    pos: float,
    wait_until_complete: bool = False,
    overshoot_enabled: bool = True,
    protocol: bool = False,
    vertical_control: bool = False,
    restore_z: bool = True,
):
    ctx = _app_ctx.ctx

    if not protocol and _user_motion_locked(axis):
        return

    if axis == 'T':
        # Turret moves go through the GUI widget which manages homing and objective settings
        if not protocol:
            ctx.io_executor.put(
                IOTask(
                    action=ctx.motion_settings.ids['verticalcontrol_id'].turret_select,
                    kwargs={'selected_position': pos},
                    callback=_handle_ui_update_for_axis,
                    cb_kwargs={'axis': axis, 'vertical_control': vertical_control},
                )
            )
        else:
            ctx.motion_settings.ids['verticalcontrol_id'].turret_select(
                selected_position=pos, protocol=True, restore_z=restore_z
            )
    else:
        if not protocol:
            ctx.scope.motion.move_absolute_async(
                axis,
                pos,
                wait_until_complete=wait_until_complete,
                overshoot_enabled=overshoot_enabled,
                callback=_handle_ui_update_for_axis,
                cb_kwargs={'axis': axis},
            )
        else:
            fut = ctx.io_executor.protocol_put(
                IOTask(
                    action=ctx.scope.motion.move_absolute_position,
                    kwargs={
                        'axis': axis,
                        'pos': pos,
                        'wait_until_complete': wait_until_complete,
                        'overshoot_enabled': overshoot_enabled,
                    },
                ),
                return_future=True,
            )
            if fut:
                fut.result(timeout=60)

        _schedule_ui(lambda dt: _handle_ui_update_for_axis(axis=axis), 0)


def move_relative_position(
    axis: str, um: float, wait_until_complete: bool = False, overshoot_enabled: bool = True
):
    if _user_motion_locked(axis):
        return
    ctx = _app_ctx.ctx
    ctx.scope.motion.move_relative_async(
        axis,
        um,
        wait_until_complete=wait_until_complete,
        overshoot_enabled=overshoot_enabled,
        callback=_handle_ui_update_for_axis,
        cb_kwargs={'axis': axis},
    )


def move_home(axis: str):
    if _user_motion_locked(axis):
        return
    ctx = _app_ctx.ctx
    axis = axis.upper()
    set_title_event_text('Homing, please wait...')
    ctx.scope.motion.move_home_async(axis, callback=move_home_cb, cb_args=(axis))


# ============================================================================
# Window Title Helpers
# ============================================================================
#
# Single-owner title bar:
# - shader.py::_update_status_bar is the ONLY caller of Window.set_title().
# - Other callers set the event-suffix via set_title_event_text() -- the next
#   status-bar tick (~5 Hz) composes the final title with FPS + MB/s + suffix.
# - This eliminates: (a) the FPS getting clobbered by event messages,
#   (b) the LumaViewPro / Lumaview Pro spelling oscillation between tickers,
#   (c) the ordering race where event messages briefly hide live FPS.
# Canonical product spelling is `LumaViewPro` (matches the repo name).

_title_event_text = None


def get_title_event_text():
    return _title_event_text


def set_title_event_text(text):
    """Set the suffix shown after the FPS/MB/s portion of the window title.
    Pass None or '' to clear. Safe to call from any thread (single attribute
    write on a module-level CPython str/None -- atomic under GIL)."""
    global _title_event_text
    _title_event_text = text or None


# Should only be called from main thread
def set_recording_title(elapsed_sec=None, total_sec=None):
    if elapsed_sec is None:
        set_title_event_text('Recording Video...')
    elif total_sec:
        set_title_event_text(f'Recording Video... {int(elapsed_sec)}s / {int(total_sec)}s')
    else:
        set_title_event_text(f'Recording Video... {int(elapsed_sec)}s')


# Should only be called from main thread
def set_writing_title(progress=None):
    if progress is None:
        set_title_event_text('Writing Video...')
    else:
        set_title_event_text(f'Writing Video... {int(progress)}%')


def reset_title():
    set_title_event_text(None)


def move_home_cb(axis):
    _handle_ui_update_for_axis(axis=axis)
    set_title_event_text(None)


# ============================================================================
# Histogram / Contrast Helpers
# ============================================================================


def live_histo_off():
    ctx = _app_ctx.ctx
    if ctx.live_histo_setting and ctx.scope_display.use_live_image_histogram_equalization:
        ctx.scope_display.use_live_image_histogram_equalization = False
        logger.info('[LVP Main  ] Live Histogram Equalization] False')


def live_histo_reverse():
    ctx = _app_ctx.ctx
    if ctx.live_histo_setting and not ctx.scope_display.use_live_image_histogram_equalization:
        ctx.scope_display.use_live_image_histogram_equalization = True
        logger.info('[LVP Main  ] Live Histogram Equalization] True')


# ============================================================================
# UI State Helpers
# ============================================================================


def reset_acquire_ui():
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers():
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        layer_obj._initializing = True
        try:
            if ctx.settings[layer]['acquire'] == 'image':
                layer_obj.ids['acquire_image'].active = True
            elif ctx.settings[layer]['acquire'] == 'video':
                layer_obj.ids['acquire_video'].active = True
            else:
                layer_obj.ids['acquire_none'].active = True
        finally:
            layer_obj._initializing = False


def reset_stim_ui():
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers():
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        if 'stim_config' in ctx.settings[layer] and ctx.settings[layer]['stim_config'] is not None:
            with ctx.settings_lock:
                ctx.settings[layer]['stim_config']['enabled'] = False
            layer_obj._initializing = True
            try:
                layer_obj.ids['stim_disable_btn'].active = True
            finally:
                layer_obj._initializing = False
            layer_obj.update_stim_controls_visibility()


# ============================================================================
# ScrollView Memory Cleanup
# ============================================================================


def cleanup_scrollview_viewport(scrollview):
    """
    Clean up ScrollView viewport textures to prevent memory accumulation.
    This is called after accordion collapse events to release viewport resources.
    """
    try:
        if not isinstance(scrollview, ScrollView):
            return

        # Clear viewport canvas
        if (
            hasattr(scrollview, '_viewport')
            and scrollview._viewport
            and hasattr(scrollview._viewport, 'canvas')
        ):
            scrollview._viewport.canvas.ask_update()

        # Clear effect textures (primary source of memory accumulation)
        for effect in [scrollview.effect_x, scrollview.effect_y]:
            if effect and hasattr(effect, '_texture'):
                effect._texture = None

        # Clear viewport texture reference
        if hasattr(scrollview, '_viewport_texture'):
            scrollview._viewport_texture = None

        logger.debug('[LVP Main  ] ScrollView viewport cleanup completed')
    except Exception as e:
        logger.warning(f'[LVP Main  ] ScrollView cleanup error: {e}')
