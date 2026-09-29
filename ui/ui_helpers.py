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

if typing.TYPE_CHECKING:
    from modules.sequential_io_executor import SequentialIOExecutor

logger = logging.getLogger('LVP.modules.ui_helpers')


def run_reported(
    call: typing.Callable[[], object],
    redraw: typing.Callable[[], None] | None,
    label: str,
) -> None:
    """Run an API call a person asked for, here, and report its outcome; then redraw.

    The one place in the GUI where an API call's exception is caught. The
    exception is the API's answer, and the reporter shows it as its type says
    -- a refusal as a warning under its own title, a fault as an error -- so
    no widget writes its own popup or decides what an outcome means. The
    redraw then shows the scope's state read back from the API, whatever the
    call did; a widget changes nothing ahead of that answer.

    For members that do not wait on a lane: the call runs on this thread,
    before the next thing this thread does, so a continuation that reads what
    the call applied sees it. A member that waits on a lane raises here, by
    name, instead of freezing the window; it goes through submit_reported.

    Args:
        call: The API call, closed over any value a widget holds. Its return
            is ignored: nothing in the GUI branches on it.
        redraw: Shows the API's state, or None when the call's own callback
            already does.
        label: The gesture's interaction-log label; the outcome's category.
    """
    from modules.sequential_io_executor import inline_outcome

    with inline_outcome():
        _reported(call, label)
    _reported(redraw, label)


def submit_reported(
    call: typing.Callable[[], object],
    redraw: typing.Callable[[], None] | None,
    label: str,
    *,
    stop: bool = False,
    lane: 'SequentialIOExecutor | None' = None,
) -> None:
    """Run an API call that may block off the GUI thread and report its outcome; then redraw.

    run_reported's twin for members that wait on a lane (hardware, a lane's
    answer). The redraw is scheduled back onto the GUI thread afterwards,
    whatever the outcome. The call reads no widget and touches nothing in
    the GUI: any value a widget holds is read before this is called and
    closed over.

    Where the call runs:
        lane: The one lane every member in *call* dispatches to (the camera
            lane for camera members, the IO lane for motion and LED ones).
            The call runs on that lane's worker, where its members run
            inline, so a person's actions on one device run in the order
            they were made while the other device's lane keeps working -- a
            gain change does not wait behind a home. Only a call whose
            members all dispatch to this one lane may name it: a member
            that dispatches elsewhere raises on the lane worker rather than
            wait on another lane.
        None: the GUI's worker pool, for a call that spans lanes (a run's
            start or Stop). One worker, so actions run in the order they
            were made, and a Stop submitted at high priority goes first.

    The redraw runs exactly once per submit, whatever the outcome: the call
    returned or raised, the lane refused the task at submit, while it was
    queued or as it left the queue, or the executor was not taking work. A
    widget that shows a request as pending is cleared only by its redraw,
    so a lost redraw leaves it dead. An executor that is not taking work
    (closing down, or fenced by a run) never runs the task and answers
    nothing, so that one outcome is redrawn from here; every other outcome
    reaches the task's callback once. The executor's own narration is the
    record of a refused or dropped call, under the gesture's label.

    Two executor paths run no callback, so their redraw is lost: a queue
    drained by ``clear_pending`` (the lane closing down), and a task whose
    worker was abandoned by wedge recovery while stuck inside it.

    ``stop`` is for a Stop: it goes ahead of every queued request, so a
    person stopping a run is never kept waiting behind work they asked for
    before it.
    """
    from modules.sequential_io_executor import PRIORITY_HIGH, PRIORITY_MED, IOTask

    def _redraw():
        _schedule_ui(lambda dt: _reported(redraw, label))

    def _off_the_gui_thread():
        _reported(call, label)

    # The executor names a refused task by its action, so the action carries
    # the gesture's label rather than this wrapper's name.
    _off_the_gui_thread.__name__ = _off_the_gui_thread.__qualname__ = f'UI:{label}'

    executor = lane if lane is not None else _app_ctx.ctx.worker_pool
    priority = PRIORITY_HIGH if stop else PRIORITY_MED
    queued = executor.put(IOTask(action=_off_the_gui_thread, callback=_redraw, priority=priority))
    if queued is None:
        _redraw()


def _reported(fn: typing.Callable[[], object] | None, label: str) -> None:
    """Run *fn* and hand whatever it raises to the one reporter, as a person's request.

    The reporting core both boundary forms share: the only place in the GUI
    that catches an API call's exception. A redraw goes through it too, so a
    widget that fails to draw is reported as a fault rather than exiting the
    app from a clock callback.
    """
    if fn is None:
        return
    from modules.notification_center import notifications

    try:
        fn()
    except Exception as e:
        notifications.report_outcome(e, solicited=True, category=f'UI:{label}')


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
    inside the submitted call, on the IO lane, where the reporter shows a
    refusal; the axis boxes redraw once the command has landed.
    """
    ctx = _app_ctx.ctx

    if _user_motion_locked(axis):
        return

    if axis == 'T':
        ctx.motion_settings.ids['verticalcontrol_id'].turret_select(position)
        return

    submit_reported(
        lambda: ctx.scope.motion.move_absolute(
            axis,
            position,
            wait_until_complete=wait_until_complete,
            overshoot_enabled=overshoot_enabled,
            frame=frame,
        ),
        lambda: _handle_ui_update_for_axis(axis=axis),
        f'MOVE_{axis}',
        lane=ctx.io_executor,
    )


def move_relative(
    axis: str, distance: float, wait_until_complete: bool = False, overshoot_enabled: bool = True
):
    if _user_motion_locked(axis):
        return
    ctx = _app_ctx.ctx
    submit_reported(
        lambda: ctx.scope.motion.move_relative(
            axis,
            distance,
            wait_until_complete=wait_until_complete,
            overshoot_enabled=overshoot_enabled,
        ),
        lambda: _handle_ui_update_for_axis(axis=axis),
        f'JOG_{axis}',
        lane=ctx.io_executor,
    )


def submit_gesture(
    label: str,
    *,
    axes: typing.Iterable[str],
    then: str,
    moves: typing.Callable[[], None],
    on_moved: typing.Callable[[], None] | None = None,
) -> None:
    """Run a person's several-axis gesture as one task on the IO lane.

    The lane asks the motion API once whether every axis the gesture needs
    knows where it is, then runs *moves*, whose members run inline there. A
    home or a stop can no longer land between the question and the moves,
    and a refusal is shown once by the reporter instead of once per axis.
    A move refused part way leaves the axes that already moved where they
    went.

    Submits and returns; nothing here waits on the lane, so a caller that
    is itself running inline on a lane may start a gesture.

    Args:
        label: The gesture, as the reporter and the executor name it.
        axes: The axes the gesture moves. They are asked about, and redrawn
            once the task has ended whatever its outcome. Empty when the
            gesture moves nothing (the scope has no motor board).
        then: What the user does once the scope knows its position, ending
            the refusal the API shows.
        moves: The API calls, all on the IO lane.
        on_moved: GUI work that belongs to a gesture that happened, run on
            the GUI thread after the redraw and only when *moves* returned.
    """
    ctx = _app_ctx.ctx
    axes = tuple(axes)
    if _user_motion_locked(label):
        return
    motion = ctx.scope.motion
    # Written on the lane, read by the redraw, which submit_reported runs
    # once after the task has ended: the one thing the redraw needs to know
    # about the outcome the reporter has already shown.
    moved = False

    def call() -> None:
        nonlocal moved
        if axes:
            motion.refuse_unknown_positions(axes, recording=False, then=then)
        moves()
        moved = True

    def redraw() -> None:
        _redraw_gesture_axes(axes)
        if moved and on_moved is not None:
            on_moved()

    submit_reported(call, redraw, label, lane=ctx.io_executor)


def _redraw_gesture_axes(axes: tuple[str, ...]) -> None:
    ctx = _app_ctx.ctx
    vertical_control = ctx.motion_settings.ids['verticalcontrol_id']
    if 'X' in axes or 'Y' in axes:
        ctx.motion_settings.update_xy_stage_control_gui()
    if 'Z' in axes:
        vertical_control.update_gui()
    if 'T' in axes:
        # A person's turret move: after a failed one the objective question
        # is asked again.
        vertical_control.show_turret_state()


def move_home(axis: str):
    """Home an axis from a Home button, without blocking the UI thread.

    The home runs on the io lane; blocking the UI thread for the length
    of a home would freeze the window.
    """
    if _user_motion_locked(axis):
        return
    ctx = _app_ctx.ctx
    axis = axis.upper()
    set_title_event_text('Homing, please wait...')
    submit_reported(
        lambda: ctx.scope.motion.home(axis),
        lambda: move_home_cb(axis),
        f'HOME_{axis}',
        lane=ctx.io_executor,
    )


def startup_home(axis: str) -> None:
    """The startup home: waits for the home and raises as it does.

    The window title says the scope is homing for its length, and the
    axis is redrawn however the home ended.
    """
    set_title_event_text('Homing, please wait...')
    try:
        _app_ctx.ctx.scope.motion.home(axis)
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


_DRAIN_TITLE = 'Writing protocol scan files to disk...'


def draw_shared_run_displays() -> None:
    """Draw the displays every run shares -- equalization, title, LED toggles.

    Their only writer. Each run control redraws on every run-state edge,
    including the edge where ANOTHER run takes the scope, so a control
    that wrote these while drawing its own idle state would undo the live
    run's display. These are drawn from what the session says holds the
    scope, never from any one control's run.

    While anything holds the scope its own writers title the window, so
    the title is left to them; the LED toggles are reconciled only once
    nothing holds the scope, after the run's hardware restore has settled
    and with no diagnostic lighting the LEDs underneath them.
    """
    ctx = _app_ctx.ctx
    session = ctx.session
    if session.run_lockout:
        live_histo_off()
    else:
        live_histo_reverse()
    if session.exclusive_activity is not None:
        return
    if session.protocol_files_draining:
        set_title_event_text(_DRAIN_TITLE)
    else:
        reset_title()
    ctx.ui_listener_bridge.reconcile_led_buttons()


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
