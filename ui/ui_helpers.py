# Copyright Etaluma, Inc.
"""
UI helper functions -- manipulate Kivy widgets, window titles, LED buttons.

Moved from modules/ui_helpers.py to ui/ because this is GUI code (imports
Kivy Window, ScrollView). A compatibility shim at modules/ui_helpers.py
re-exports everything for existing callers.
"""

import logging
import typing

from kivy.properties import StringProperty
from kivy.uix.accordion import AccordionItem
from kivy.uix.scrollview import ScrollView
from modules import gui_logger
from modules.kivy_utils import schedule_ui as _schedule_ui

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
import modules.image_utils as image_utils

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

    A call that raises is also recorded against the frame it ran in, for
    refused_in_this_input: a button whose own touch committed a refused
    edit does not act on the value the person just tried to change.

    Args:
        call: The API call, closed over any value a widget holds. Its return
            is ignored: nothing in the GUI branches on it.
        redraw: Shows the API's state, or None when the call's own callback
            already does.
        label: The gesture's interaction-log label; the outcome's category.
    """
    from modules.sequential_io_executor import inline_outcome

    global _unanswered_frame
    with inline_outcome():
        if not _reported(call, label):
            _unanswered_frame = _input_frame()
    _reported(redraw, label)


# The frame in which run_reported last reported a call that raised.
_unanswered_frame: int | None = None


def _input_frame() -> int:
    from kivy.clock import Clock

    return Clock.frames


def refused_in_this_input() -> bool:
    """Whether a request made in the input now being handled was not taken.

    Kivy commits a focused field on the touch that leaves it, before that
    touch's button handler runs, and both happen in one frame. A button
    pressed straight off an edit the API refused would otherwise act on the
    value the person just tried to change, so a handler that acts on such a
    value asks this first and does nothing when it is true.
    """
    return _unanswered_frame is not None and _unanswered_frame == _input_frame()


def submit_reported(
    call: typing.Callable[[], object],
    redraw: typing.Callable[[], None] | None,
    label: str,
    *,
    stop: bool = False,
    lane: 'SequentialIOExecutor | None' = None,
    budget_of: typing.Callable[..., object] | None = None,
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
            wait on another lane. Or an executor that is not a lane, for a
            call that waits on several lanes for minutes (the support
            report on ``diagnostics_executor``), so it holds neither a
            device lane nor the worker pool a Stop goes through.
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

    ``budget_of`` is the API member *call* runs, when that member declares
    how long it may legitimately take (``@slow_task_budget``): the task
    carries the member's budget, so a member that runs for minutes is not
    logged as a slow task each time it succeeds. The cost is the member's;
    the GUI only says which member it called.
    """
    from modules.sequential_io_executor import (
        PRIORITY_HIGH,
        PRIORITY_MED,
        SLOW_TASK_BUDGET_ATTR,
        IOTask,
    )

    def _redraw():
        _schedule_ui(lambda dt: _reported(redraw, label))

    def _off_the_gui_thread():
        _reported(call, label)

    # The executor names a refused task by its action, so the action carries
    # the gesture's label rather than this wrapper's name.
    _off_the_gui_thread.__name__ = _off_the_gui_thread.__qualname__ = f'UI:{label}'
    budget = getattr(budget_of, SLOW_TASK_BUDGET_ATTR, None)
    if budget is not None:
        setattr(_off_the_gui_thread, SLOW_TASK_BUDGET_ATTR, budget)

    executor = lane if lane is not None else _app_ctx.ctx.worker_pool
    priority = PRIORITY_HIGH if stop else PRIORITY_MED
    queued = executor.put(IOTask(action=_off_the_gui_thread, callback=_redraw, priority=priority))
    if queued is None:
        _redraw()


def run_unasked(call: typing.Callable[[], object], label: str) -> bool:
    """Run for an edge no person asked for; a raise is reported, not raised.

    A run-state edge redraws on the clock, a step-list change redraws the
    step editor, the application's close stops a live run. A raise out of a
    clock callback or a shutdown step reaches the main loop and closes the
    application, and nothing waits on these calls, so a fault stops here --
    reported as unasked, which an unattended run's mute and the repeat
    window apply to -- and the next edge runs again.

    True when *call* returned, False when it raised and was reported: a
    step that must follow whether or not the call got through runs on
    False, without deciding what the outcome meant.
    """
    return _contained(call, label, solicited=False)


def typed_number(
    text: str, cast: typing.Callable[[str], float], put_back: typing.Callable[[], None]
) -> float | None:
    """The number a typed box committed, or None with the box put back.

    The kv float and int filters admit entries that are not numbers -- '',
    '.', '-', '-.' -- and a box that commits on focus loss hands them over
    as typed. Such an entry is not a request: nothing moves, no setting
    changes, and *put_back* shows the box what the API or the settings hold
    again. The caller then records what the box went back to under its own
    ``<NAME>_APPLIED``, written at the call site so the interaction census
    can read the name. One parse for every typed-number box, so they cannot
    each choose their own failure.
    """
    try:
        return cast(text)
    except ValueError:
        put_back()
        return None


def _reported(fn: typing.Callable[[], object] | None, label: str) -> bool:
    """Run *fn* and hand whatever it raises to the one reporter, as a person's request.

    The reporting core both boundary forms share. A redraw goes through it
    too, so a widget that fails to draw is reported as a fault rather than
    exiting the app from a clock callback. True when *fn* returned.
    """
    return _contained(fn, label, solicited=True)


def _contained(fn: typing.Callable[[], object] | None, label: str, *, solicited: bool) -> bool:
    """The only place in the GUI that catches an outcome: run *fn*, report what it raises.

    True when *fn* returned (or there was none), False when it raised.
    """
    if fn is None:
        return True
    from modules.notification_center import notifications

    try:
        fn()
    except Exception as e:
        notifications.report_outcome(e, solicited=solicited, category=f'UI:{label}')
        return False
    return True


# ============================================================================
# Saved-folder helper
# ============================================================================


def show_captured_frame(image, frames_summed: int, frame_significant_bits: int) -> None:
    """Every GUI run's ``frame_captured``: hold the frame a run just captured on screen.

    Rendered as a JPG save renders it -- a sum against one frame's white,
    brighter -- on the run's thread, where the engine delivers the event, so
    the display only uploads the result. The display is looked up when the
    frame arrives, not when the run starts.
    """
    rendered = image_utils.convert_sum_to_8bit(image, frames_summed, frame_significant_bits)
    _app_ctx.ctx.scope_display.hold_protocol_saved_image(rendered, 8)


def restore_display_after_run(*_ended) -> None:
    """Put the display back on the user's settings once a run has ended.

    A run shows each step in the layer panel and in the shader's false
    colour without writing the user's settings; this puts every layer's
    widgets back on the settings and the shader back on the open drawer's
    layer, or BF when none is open. Takes and ignores ``run_ended``'s values.
    """
    sync_layer_widgets_from_settings()
    ctx = _app_ctx.ctx
    layer_name = common_utils.get_opened_layer(ctx.image_settings)
    if layer_name is not None:
        ctx.image_settings.layer_lookup(layer=layer_name).update_shader(dt=0)
        return
    ctx.viewer.update_shader(false_color='BF')


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
    elif axis == 'ALL':
        # A full home moves every axis the scope has, the turret included.
        ctx.motion_settings.ids['verticalcontrol_id'].update_gui(vertical_control=vertical_control)
        ctx.motion_settings.update_xy_stage_control_gui()
        if ctx.scope.capabilities.has_turret:
            ctx.motion_settings.ids['verticalcontrol_id'].show_turret_state()


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
    overshoot_enabled: bool = True,
    frame: str = 'stage',
):
    """Start an axis moving for a person's gesture, keeping the gesture lock in one place.

    Started, not waited: a gesture returns at once, and a move that fails
    on the way is the motion monitor's to report.

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
        lambda: ctx.scope.motion.start_move_absolute(
            axis,
            position,
            overshoot_enabled=overshoot_enabled,
            frame=frame,
        ),
        lambda: _handle_ui_update_for_axis(axis=axis),
        f'MOVE_{axis}',
        lane=ctx.io_executor,
    )


def move_relative(axis: str, distance: float, overshoot_enabled: bool = True):
    """Start an axis jogging for a person's gesture; started, not waited, as ``move_absolute``."""
    if _user_motion_locked(axis):
        return
    ctx = _app_ctx.ctx
    submit_reported(
        lambda: ctx.scope.motion.start_move_relative(
            axis,
            distance,
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
    axes = tuple(axes)

    def call() -> None:
        if axes:
            _app_ctx.ctx.scope.motion.refuse_unknown_positions(axes, recording=False, then=then)
        moves()

    submit_move(label, axes=axes, call=call, on_moved=on_moved)


def submit_move(
    label: str,
    *,
    axes: typing.Iterable[str],
    call: typing.Callable[[], None],
    on_moved: typing.Callable[[], None] | None = None,
) -> None:
    """Run a person's move on the IO lane, redraw its axes, then its GUI work.

    The lane half of ``submit_gesture``, for a move an API member composes
    itself (asking about its axes included): the control-surface lock is
    enforced here, *call* runs as one task on the IO lane, the axes are
    redrawn once the task has ended whatever its outcome, and *on_moved*
    runs on the GUI thread only when *call* returned. A gesture's moves are
    started, never waited on, so the task holds the lane only for the
    commands.
    """
    ctx = _app_ctx.ctx
    axes = tuple(axes)
    if _user_motion_locked(label):
        return
    # Written on the lane, read by the redraw, which submit_reported runs
    # once after the task has ended: the one thing the redraw needs to know
    # about the outcome the reporter has already shown.
    moved = False

    def moving() -> None:
        nonlocal moved
        call()
        moved = True

    def redraw() -> None:
        _redraw_gesture_axes(axes)
        if moved and on_moved is not None:
            on_moved()

    submit_reported(moving, redraw, label, lane=ctx.io_executor)


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

    The API starts the home and answers at once: a home asked while one is
    in flight is refused then, and shown, rather than queued behind it. The
    home's outcome is reported, and the axis redrawn, when it settles.
    """
    if _user_motion_locked(axis):
        return
    ctx = _app_ctx.ctx
    axis = axis.upper()

    def start() -> None:
        ctx.scope.motion.start_home(axis).add_done_callback(
            lambda done: _schedule_ui(lambda _dt: _home_settled(done, axis))
        )
        # Only a home that started says so; the settle clears it.
        set_title_event_text('Homing, please wait...')

    run_reported(start, None, f'HOME_{axis}')


def _home_settled(done, axis: str) -> None:
    """Read the settled home on the Kivy thread: its failure to the reporter, then the redraw."""
    run_reported(done.result, lambda: move_home_cb(axis), f'HOME_{axis}')


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


# The suffix the last video progress wrote, so its 'ended' clears only its
# own text: the slot also holds homing, compositing and file-drain text.
_video_progress_text = None


def show_video_progress(progress) -> None:
    """Every GUI run's ``video_progress``: say in the title what a video step is doing.

    On the UI thread. ``'ended'`` clears the suffix only while it still holds
    the text this wrote; anything another writer has put there since stays.
    """
    global _video_progress_text
    if progress.phase == 'ended':
        if get_title_event_text() == _video_progress_text:
            set_title_event_text(None)
        _video_progress_text = None
        return
    if progress.phase == 'recording':
        text = f'Recording Video... {int(progress.elapsed_s)}s / {int(progress.total_s)}s'
    else:
        text = f'Writing Video... {int(progress.percent)}%'
    _video_progress_text = text
    set_title_event_text(text)


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


def homing_banner_shown(session) -> bool:
    """Whether the middle of the window shows the homing banner: while a home holds the scope.

    A home only: a run shows its own progress, and a banner over the live
    view would hide the images it captures.
    """
    return session.exclusive_activity == 'home'


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
    # Display only: a loaded protocol's stimulation was turned off by the
    # Session (``ScopeSession.apply_layer_settings``), which writes it.
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers():
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        if 'stim_config' in ctx.settings[layer] and ctx.settings[layer]['stim_config'] is not None:
            layer_obj._initializing = True
            try:
                layer_obj.ids['stim_disable_btn'].active = True
            finally:
                layer_obj._initializing = False
            layer_obj.update_stim_controls_visibility()


# ============================================================================
# ScrollView Memory Cleanup
# ============================================================================


class LoggedAccordionItem(AccordionItem):
    """An accordion item that records a person opening it.

    Kivy expands a collapsed item in one place, its touch handler; the app's
    own expands (start-up, a step's layer, a model swap, a resort) write
    ``collapse`` directly and never pass through it. So the record is written
    here, before the expand and whatever it causes, and only for a person.
    ``on_collapse`` cannot carry it: it fires for both and does not say which.

    Every item sets both names in the kv; the GUI-logging census refuses an
    item that does not (tests/gui_logging_census.py).
    """

    log_group = StringProperty('')
    log_item = StringProperty('')

    def on_touch_down(self, touch):
        if self.collapse and not self.disabled and self.collide_point(*touch.pos):
            gui_logger.select(self.log_group, self.log_item)
        return super().on_touch_down(touch)


def resort_accordion(accordion, items: typing.Sequence[tuple[object | None, bool]]) -> None:
    """Put an accordion's items back in canonical order, the one way both panels do it.

    A live scope-model switch re-adds an item that was hidden, and add_widget
    lands it wherever it lands; after a few switches the order is wrong. This
    takes every child out and adds back, in *items* order, each widget whose
    flag says it is shown, then any child *items* does not name (a plugin's
    tab, registered after the kv build) below them, in the order they had.

    Kivy renders children[0] last, at the bottom, and add_widget with no
    index prepends, so walking *items* forward puts the first at the top.
    Items are matched by ``uid``: ``ids.get`` returns a WeakProxy whose
    Python id is not the widget's. An item's open or closed state lives on
    the widget, so taking it out and adding it back keeps it.

    Args:
        accordion: The Accordion whose children are reordered.
        items: (widget, shown) pairs in canonical order, top first. A widget
            that is None, or not shown, is left out.
    """
    tracked = {widget.uid for widget, _shown in items if widget is not None}
    untracked = [w for w in reversed(accordion.children) if w.uid not in tracked]
    for widget in list(accordion.children):
        accordion.remove_widget(widget)
    for widget, shown in items:
        if widget is None or not shown:
            continue
        # A shown item still attached elsewhere is detached first: add_widget
        # refuses a widget that has a parent.
        if widget.parent is not None:
            widget.parent.remove_widget(widget)
        accordion.add_widget(widget)
    for widget in untracked:
        accordion.add_widget(widget, 0)


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
