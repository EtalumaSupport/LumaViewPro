# Copyright Etaluma, Inc.
"""
UI helper functions -- manipulate Kivy widgets, window titles, LED buttons.

Moved from modules/ui_helpers.py to ui/ because this is GUI code (imports
Kivy Window, ScrollView). A compatibility shim at modules/ui_helpers.py
re-exports everything for existing callers.
"""

import logging
import typing

from kivy.uix.scrollview import ScrollView
from modules.kivy_utils import schedule_ui as _schedule_ui

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
from modules.exceptions import (
    ObjectiveUnknownError,
    ProtocolRunRefusedError,
    RunAlreadyEndedError,
)

logger = logging.getLogger('LVP.modules.ui_helpers')


def run_with_refusal_boundary(
    start_fn: typing.Callable[[], None],
    on_refused: typing.Callable[[], None],
) -> None:
    """The single UI boundary for the runner's typed run refusal.

    A refused run is a designed outcome, not a failure to propagate: the
    runner's refusal funnel has already logged it, and no running-state
    was committed (commit_ui_state runs only after a successful
    prepare). What remains is per-starter: undo the pre-gate button
    cosmetics via on_refused. Every UI starter (scan, protocol,
    autofocus scan, z-stack) routes its prepare/start sequence through
    this one handler so refusal handling cannot drift between them.

    The funnel also DELIVERS the user notification: a refusal answers a
    button press, so it is posted solicited and reaches the user during
    a run of any kind. A starter therefore adds no popup of its own --
    a second one would say what the engine already said, and would say
    it only to whoever is looking at this GUI.

    One answer arrives without the funnel: an unknown objective. The API
    raises it as its own typed error while it assembles the run, before
    any run exists to refuse, so nothing has logged or shown it yet, and
    it is shown here.
    """
    try:
        start_fn()
    except ProtocolRunRefusedError:
        on_refused()
    except ObjectiveUnknownError as e:
        show_objective_unknown_refusal('Run', e)
        on_refused()


def show_objective_unknown_refusal(action: str, error: ObjectiveUnknownError) -> None:
    """Show the API's unknown-objective answer as a refusal of *action*.

    The API decided and wrote the sentence (home the turret, assign the
    slot); a click that meets it is refused, not failed, so it is a
    warning, never an ERROR with a traceback. Posted the way the run
    funnel posts a refusal -- solicited, under the one refusal key -- so
    it reaches the user during a run and a second press replaces the
    dialog rather than stacking one.
    """
    from modules.notification_center import REFUSAL_OPERATION_KEY, notifications

    logger.warning(f'[UI] {action} refused ({error.reason}): {error}')
    notifications.warning(
        'Protocol',
        'Objective Unknown',
        str(error),
        solicited=True,
        operation_key=REFUSAL_OPERATION_KEY,
    )


def reset_with_refusal_boundary(runner, run) -> bool:
    """Stop *run*, the handle this control's start returned, and say whether anything is left.

    The teardown half of the boundary above. Whether *run* may be stopped
    is the engine's decision: it stops the live run by its handle and
    refuses a handle naming any other, having already logged (and, when
    another run is live, notified) once. What the widget needs back is not
    the exception but the outcome -- a refused stop while another run is
    live left that run running, so the caller must not go on to restyle
    its button as though a stop were under way.

    Returns True when the run was torn down or no run is live, False when
    another run is live. Without this the refusal reached a starter's
    blanket handler, which renders str(e) -- the joined `reason: message`
    debugging form, in a dialog, at a user.
    """
    try:
        runner.reset(run)
    except RunAlreadyEndedError:
        return True
    except ProtocolRunRefusedError:
        return False
    return True


# ============================================================================
# Saved-folder helper
# ============================================================================


def live_display_callbacks() -> dict:
    """The run callbacks that feed the live display, for every GUI run starter.

    One key today: the hold that keeps a just-saved protocol frame on screen.
    Late-bound on purpose -- the display is resolved when the writer calls,
    not when the starter builds its dict -- so a run started before the
    display exists degrades inside the writer's own guard, exactly as the
    writer's former direct read did, in one place rather than at each
    starter.
    """
    return {
        'hold_protocol_saved_image': lambda image, significant_bits: (
            _app_ctx.ctx.scope_display.hold_protocol_saved_image(image, significant_bits)
        ),
    }


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


def sync_layer_widgets_from_settings():
    """Every layer's panel shows its stored settings again.

    The run-end callback: a protocol run displays each step in the panel
    without writing the user's settings, so at run end the widgets are
    the last step's and the settings are the user's; this puts the panel
    back on the settings. One call per run, every layer.
    """
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers():
        ctx.image_settings.layer_lookup(layer=layer).sync_widgets_from_settings()


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
    if ctx.session.run_lockout:
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
    elif axis == 'T':
        # A run's turret move: show the slot and objective the API reports.
        ctx.motion_settings.ids['verticalcontrol_id'].show_turret_state(prompt=False)
    elif axis == 'ALL':
        # A full home moves every axis the scope has, the turret included.
        ctx.motion_settings.ids['verticalcontrol_id'].update_gui(vertical_control=vertical_control)
        ctx.motion_settings.update_xy_stage_control_gui()
        if ctx.scope.capabilities.has_turret:
            ctx.motion_settings.ids['verticalcontrol_id'].show_turret_state()


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


def move_absolute(
    axis: str,
    position: float,
    wait_until_complete: bool = False,
    overshoot_enabled: bool = True,
    frame: str = 'stage',
):
    """Move an axis for a person's gesture, keeping the gesture lock in one place.

    A turret slot goes to the turret widget, which asks the API to move and
    then shows where the API says the turret is.

    ``frame='plate'`` hands the API the number a user typed, in plate mm,
    instead of converting first. The conversion and its bound then happen
    inside the submitted task, which is what keeps a refusal on the worker
    thread: raised in a Kivy handler's own frame it would reach the crash
    guard rather than a notification.
    """
    ctx = _app_ctx.ctx

    if _user_motion_locked(axis):
        return

    if axis == 'T':
        ctx.motion_settings.ids['verticalcontrol_id'].turret_select(position)
        return

    ctx.scope.motion.move_absolute_async(
        axis,
        position,
        wait_until_complete=wait_until_complete,
        overshoot_enabled=overshoot_enabled,
        callback=_handle_ui_update_for_axis,
        cb_kwargs={'axis': axis},
        frame=frame,
    )
    _schedule_ui(lambda dt: _handle_ui_update_for_axis(axis=axis), 0)


def unknown_position_refused(axes: typing.Iterable[str], *, recording: bool, then: str) -> bool:
    """The single UI boundary for the motion API's unknown-position refusal.

    A gesture that moves or saves several axes asks the API once, before
    it does anything, whether the scope knows where those axes are; the
    API decides, logs and notifies. What remains for the gesture is only
    to stop, so every gesture asks through here and none carries its own
    handling of the refusal, the way every run starter routes its refusal
    through ``run_with_refusal_boundary``.

    Args:
        axes: The axes the gesture needs.
        recording: True when the gesture saves the position, False when
            it moves.
        then: What the user does once the scope knows its position,
            ending the refusal the API shows.

    Returns:
        bool: True when the API refused (already logged and shown); the
            caller stops. False when the gesture may go ahead.
    """
    from modules.exceptions import AxisStateUnknownError

    try:
        _app_ctx.ctx.scope.motion.refuse_unknown_positions(axes, recording=recording, then=then)
    except AxisStateUnknownError:
        return True
    return False


def show_jog_refusal(label: str, error: Exception) -> None:
    """Display a jog the API refused, and why (e.g. home the turret)."""
    from modules.notification_center import notifications

    logger.warning(f'[Motion] {label} refused: {error}')
    notifications.warning('Motion', 'Jog refused', str(error))


def move_relative(
    axis: str, distance: float, wait_until_complete: bool = False, overshoot_enabled: bool = True
):
    if _user_motion_locked(axis):
        return
    ctx = _app_ctx.ctx
    ctx.scope.motion.move_relative_async(
        axis,
        distance,
        wait_until_complete=wait_until_complete,
        overshoot_enabled=overshoot_enabled,
        callback=_handle_ui_update_for_axis,
        cb_kwargs={'axis': axis},
    )


def move_home(axis: str, wait: bool = False):
    """Home an axis. Returns whether it succeeded when ``wait`` is set.

    The UI buttons leave ``wait`` off: they run on the UI thread, and
    blocking it for the length of a home would freeze the window. The
    startup orchestration passes it, because it has to know whether the
    reference frame is good before it drives anything else.
    """
    if _user_motion_locked(axis):
        return False
    ctx = _app_ctx.ctx
    axis = axis.upper()
    set_title_event_text('Homing, please wait...')
    if not wait:
        ctx.scope.motion.move_home_async(axis, callback=move_home_cb, cb_args=(axis))
        return None
    try:
        return ctx.scope.motion.move_home_and_wait(axis)
    finally:
        move_home_cb(axis)


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
