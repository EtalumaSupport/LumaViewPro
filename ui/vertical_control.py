# Copyright Etaluma, Inc.
import logging

from kivy.clock import Clock
from kivy.properties import BooleanProperty
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from modules.debounce import debounce
from modules.run_outcome import PendingRunOutcome
from ui.protocol_settings import require_file_writes_idle
from ui.ui_helpers import (
    _handle_ui_update_for_axis,
    live_display_callbacks,
    move_absolute,
    move_home,
    move_relative,
    run_reported,
    submit_reported,
)

logger = logging.getLogger('LVP.ui.vertical_control')

AF_SAFETY_TIMEOUT_S = 15  # Seconds before AF is considered stuck and force-reset


# ============================================================================
# VerticalControl -- Z-Axis, Objectives, Turret, and Autofocus
# ============================================================================


class VerticalControl(BoxLayout):
    # True while this button's own request is on its way to the engine; the
    # Autofocus button is disabled until that request's redraw.
    autofocus_pending = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] VerticalControl.__init__()')

        # boolean describing whether the scope is currently in the process of autofocus
        self.is_autofocus = False
        self.record_autofocus_to_file = False
        self._next_pos = None
        self._af_safety_event = None
        # The handle this button's last start returned: what its Stop and
        # its stuck-AF bound name. The engine answers whether it is live.
        self._autofocus_run: PendingRunOutcome | None = None
        # The run the stuck-AF bound was last armed for: one bound per run.
        self._af_safety_run: PendingRunOutcome | None = None

        self.queue_slider_position_trigger = Clock.create_trigger(
            lambda dt: self.queue_slider_position(), 0.1
        )

    def update_gui(self, vertical_control=False):
        ctx = _app_ctx.ctx
        if ctx.sequenced_capture_runner.run_in_progress():
            return
        if not vertical_control:
            # The target is a cache read -- no lane, no serial I/O -- shown on
            # the GUI thread because a move's completion reaches this from
            # the lane that made the move.
            Clock.schedule_once(lambda dt: run_reported(None, self._show_z_target, 'Z_TARGET'), 0)
        else:
            Clock.schedule_once(lambda dt: self.update_text_only(), 0)

    def _write_z_text(self, pos):
        """Write a Z read-back into its box, unless the user is typing in it.

        The box commits on focus loss (`on_focus: if not self.focus:
        root.set_position_text(self.text)`), so a value written underneath a
        part-typed entry is not merely displayed -- it is committed as a Z
        move when the user clicks away. Every read-back write goes through
        here so the guard cannot be present at three sites and missing at
        the fourth.
        """
        box = self.ids['z_position_id']
        if box.focus:
            return
        new_text = format(max(0, pos), '.2f')
        # Cache text to prevent redundant ScrollView updates
        if box.text != new_text:
            box.text = new_text

    def update_autofocus_gui(self, pos=None):
        if pos is None:
            return

        self.ids['obj_position'].value = max(0, pos)
        self._write_z_text(pos)

    def update_text_only(self):
        self._write_z_text(self.ids['obj_position'].value)

    def _show_z_target(self):
        """Show the Z target the API holds; nothing when it holds none."""
        pos = _app_ctx.ctx.lumaview.scope.motion.get_target_position('Z')
        if pos is not None:
            self._update_z_position(pos)

    def _update_z_position(self, pos):
        """Update Z slider and text -- must be called on main thread.

        Only updates text field when user is not typing (focus check),
        matching XY behavior. Without this, the text shows current
        position during motion then snaps to target -- confusing.
        """
        self.ids['obj_position'].value = max(0, pos)
        self._write_z_text(pos)

    def _update_z_text(self, pos):
        """Update Z text only -- must be called on main thread."""
        self._write_z_text(pos)

    def _z_jog(self, direction: int, coarse: bool, overshoot_enabled: bool = False):
        """Shared Z-axis jog handler.

        Args:
            direction: +1 for up, -1 for down.
            coarse: True for coarse step, False for fine step.
            overshoot_enabled: Enable backlash compensation overshoot.
        """
        ctx = _app_ctx.ctx
        if ctx.session.controls_locked:
            return
        label = f'Z_{"COARSE" if coarse else "FINE"}_{"UP" if direction > 0 else "DOWN"}'
        gui_logger.button(label)
        logger.info(f'[LVP Main  ] VerticalControl._z_jog({label})')
        run_reported(
            lambda: move_relative(
                'Z',
                direction * ctx.scope.motion.jog_step('Z', coarse),
                overshoot_enabled=overshoot_enabled,
            ),
            redraw=None,
            label=label,
        )

    @debounce(0.2)
    def coarse_up(self, overshoot_enabled: bool = False):
        self._z_jog(+1, coarse=True, overshoot_enabled=overshoot_enabled)

    @debounce(0.2)
    def fine_up(self, overshoot_enabled: bool = False):
        self._z_jog(+1, coarse=False, overshoot_enabled=overshoot_enabled)

    @debounce(0.2)
    def fine_down(self, overshoot_enabled: bool = False):
        self._z_jog(-1, coarse=False, overshoot_enabled=overshoot_enabled)

    @debounce(0.2)
    def coarse_down(self, overshoot_enabled: bool = False):
        self._z_jog(-1, coarse=True, overshoot_enabled=overshoot_enabled)

    def _queue_z_move(self, pos):
        """Parse a committed Z value and queue the move; None when refused.

        The slider and the text box share the move but not the record. The
        slider reports the value it resolved to; the box reports what the user
        typed, before anything parsed it. A drag and a keystroke have to stay
        distinguishable in the bundle, so each caller writes its own line and
        only the move lives here.
        """
        ctx = _app_ctx.ctx
        if ctx.session.controls_locked:
            return None

        logger.info('[LVP Main  ] VerticalControl.set_position()')
        try:
            self._next_pos = float(pos)
        except Exception:
            return None
        self.queue_slider_position_trigger()
        return self._next_pos

    def set_position(self, pos):
        """The SLIDER's commit -- the kv binds this to its on_release."""
        resolved = self._queue_z_move(pos)
        if resolved is not None:
            gui_logger.slider('Z_POSITION', resolved)

    def set_position_text(self, text):
        """The BOX's commit -- what was typed is recorded before the move.

        Separate from ``set_position`` so a typed commit does not report
        itself as a drag; the record name is shared because it is the same
        setting, and the verb is what tells them apart.
        """
        gui_logger.text_input('Z_POSITION', text)
        self._queue_z_move(text)

    def queue_slider_position(self):
        move_absolute('Z', self._next_pos)
        self._next_pos = None

    def set_bookmark(self):
        gui_logger.button('SET_Z_BOOKMARK')
        logger.info('[LVP Main  ] VerticalControl.set_bookmark()')
        run_reported(self.ex_set_bookmark, None, 'SET_Z_BOOKMARK')

    def ex_set_bookmark(self):
        ctx = _app_ctx.ctx
        ctx.scope.motion.refuse_unknown_positions(('Z',), recording=True, then='save the bookmark')
        height = ctx.lumaview.scope.motion.get_current_position('Z')  # Get current z height in um
        with ctx.settings_lock:
            ctx.settings['bookmark']['z'] = height

    def set_all_bookmarks(self):
        gui_logger.button('SET_ALL_BOOKMARKS')
        logger.info('[LVP Main  ] VerticalControl.set_all_bookmarks()')
        run_reported(self.ex_set_all_bookmarks, None, 'SET_ALL_BOOKMARKS')

    def ex_set_all_bookmarks(self):
        ctx = _app_ctx.ctx
        # This one also writes every layer's focus from the Z.
        ctx.scope.motion.refuse_unknown_positions(('Z',), recording=True, then='save the bookmarks')
        height = ctx.lumaview.scope.motion.get_current_position('Z')  # Get current z height in um
        with ctx.settings_lock:
            settings = ctx.settings
            settings['bookmark']['z'] = height
            settings['BF']['focus'] = height
            settings['PC']['focus'] = height
            settings['DF']['focus'] = height
            settings['Blue']['focus'] = height
            settings['Green']['focus'] = height
            settings['Red']['focus'] = height
            settings['Lumi']['focus'] = height

    def goto_bookmark(self):
        gui_logger.button('GOTO_Z_BOOKMARK')
        ctx = _app_ctx.ctx
        if ctx.session.controls_locked:
            return
        logger.info('[LVP Main  ] VerticalControl.goto_bookmark()')
        with ctx.settings_lock:
            pos = ctx.settings['bookmark']['z']
        move_absolute('Z', pos)

    @debounce(1.0)
    def home(self):
        gui_logger.button('HOME_Z')
        ctx = _app_ctx.ctx
        if ctx.session.controls_locked:
            return
        logger.info('[LVP Main  ] VerticalControl.home()')
        run_reported(lambda: move_home(axis='Z'), None, 'HOME_Z')

    def load_objectives(self):
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.load_objectives()')
        spinner = self.ids['objective_spinner2']
        spinner.values = ctx.objective_helper.get_objectives_list()

    def pick_objective(self, objective_id):
        """A person picked an objective from the spinner; the Session decides.

        On a turreted scope the pick says what is installed in the slot in
        the light path, so the Session assigns it there; with no turret it
        selects it. A refusal -- the slot is unknown, a run holds the scope
        -- is shown, and either way the display shows the API's answer, so
        a refused pick never stays on screen as if it had been taken.
        """
        gui_logger.select('OBJECTIVE', objective_id)
        run_reported(
            lambda: _app_ctx.ctx.session.select_objective(objective_id),
            redraw=lambda: self.show_turret_state(prompt=False),
            label='OBJECTIVE',
        )

    def show_turret_state(self, prompt=True):
        """Show what the API says: the turret slot in the light path, each
        slot's assignment, and the active objective.

        Every turret action ends here, a failed one included -- a press,
        a home, a run's step, startup, an objective answer -- because only
        the API knows where the turret is. The button down is the slot the
        API reports and none is down while that slot is unknown; the
        spinner shows the active objective, or 'Unknown'.

        Args:
            prompt: Ask the objective question when the objective is
                unknown. The Session decides whether a question is owed,
                and owes none while an activity holds the scope: a prompt
                must never interrupt an unattended run, and a recording
                would refuse the answer.
        """
        ctx = _app_ctx.ctx
        slot = ctx.scope.motion.get_turret_slot()
        catalogue = ctx.objective_helper.get_objectives_list()
        for position, assigned in ctx.scope.runtime_state.get_turret_config().items():
            button = self.ids[f'turret_pos_{position}_btn']
            button.state = 'down' if position == slot else 'normal'
            if assigned is None:
                button.text = f'< {position} >'
            elif assigned in catalogue:
                magnification = ctx.session.get_objective_info(objective_id=assigned)[
                    'magnification'
                ]
                button.text = f'{magnification}x'
            else:
                # Shown as assigned, because it is: the active objective
                # there is unknown, and the spinner says so.
                button.text = assigned

        objective_id = ctx.scope.runtime_state.get_current_objective_id()
        self.ids['objective_spinner2'].text = objective_id or 'Unknown'
        ctx.motion_settings.ids['microscope_settings_id'].refresh_fov_labels()
        if objective_id is None and prompt:
            # Scheduled, never opened from here: the startup home's display
            # runs before the event loop, and a popup opened then is painted
            # under the app root -- open, and invisible. On the Clock it
            # waits for the loop and folds into the startup question.
            Clock.schedule_once(lambda dt: self.prompt_if_objective_unknown(), 0)

    def run_autofocus_from_ui(self):
        """Autofocus on the open layer, or stop the autofocus this button started.

        The run is ProtocolRunner.run_autofocus, the one a script or REST
        calls: everything about it -- the objective, the capture, every
        refusal -- is the member's, and the boundary shows a refusal once.
        This button states only what a running GUI knows: the open drawer,
        its own trigger (the attended one), and the live engineering flag,
        which is also whether the sweep's characterization data is saved.

        Whether the press means Stop is the engine's answer -- is the run
        this button started still live -- never the toggle's, which Kivy has
        already flipped. The button changes nothing ahead of the engine's
        answer; draw_autofocus_button shows it.
        """
        gui_logger.button('AUTOFOCUS')
        logger.info('[LVP Main  ] VerticalControl.run_autofocus_from_ui()')
        ctx = _app_ctx.ctx
        run = self._autofocus_run
        if ctx.sequenced_capture_runner.is_live_run(run):
            self._stop_autofocus(run)
            return

        # The post-run file drain outlives the run by design: writes keep
        # landing after the run itself has ended, so it needs a gate of its
        # own here rather than riding on the run's. The gate helper owns the
        # stalled-writer recovery popup.
        if not require_file_writes_idle('start autofocus'):
            self.draw_autofocus_button()
            return

        member = ctx.session.create_protocol_runner()
        layer = common_utils.get_opened_layer(ctx.image_settings)
        engineering_mode = ctx.engineering_mode
        callbacks = {
            **live_display_callbacks(),
            'move_position': _handle_ui_update_for_axis,
            'run_complete': self._autofocus_run_complete,
        }

        def _start():
            self._autofocus_run = member.run_autofocus(
                layer=layer,
                save_characterization_data=engineering_mode,
                callbacks=callbacks,
                run_trigger_source='autofocus',
                engineering_mode=engineering_mode,
            )

        self._submit_autofocus_request(_start)

    def _stop_autofocus(self, run: PendingRunOutcome | None) -> None:
        runner = _app_ctx.ctx.sequenced_capture_runner
        self._submit_autofocus_request(lambda: runner.reset(run), stop=True)

    def _submit_autofocus_request(self, call, stop: bool = False) -> None:
        # The button is disabled until this request's own redraw, so a
        # second press cannot race the first one to the pool.
        self.autofocus_pending = True
        submit_reported(call, self._autofocus_request_done, 'AUTOFOCUS', stop=stop)

    def _autofocus_request_done(self):
        self.autofocus_pending = False
        self.draw_autofocus_button()

    def draw_autofocus_button(self):
        """Show the autofocus this button started, as the engine reports it.

        The only code that styles the button: after each of its own
        requests and on every run-state edge -- including the run's return
        to idle. It draws this button's own run and nothing else; a
        protocol's autofocus steps are not this button's to show.
        """
        runner = _app_ctx.ctx.sequenced_capture_runner
        run = self._autofocus_run
        button = self.ids['autofocus_id']
        if not runner.is_live_run(run):
            button.state = 'normal'
            button.text = 'Autofocus'
            return
        button.state = 'down'
        button.text = 'Stopping...' if runner.is_stopping(run) else 'Focusing...'
        # Armed by the run it bounds, the first time that run is seen live,
        # and once per run: a timer armed before the run existed outlived
        # every exit that started nothing and reached forward to abort the
        # next autofocus that did.
        if self._af_safety_run is not run:
            self._af_safety_run = run
            self._schedule_af_safety_timer(run)

    def _unschedule_af_safety_timer(self):
        if self._af_safety_event is not None:
            Clock.unschedule(self._af_safety_event)
            self._af_safety_event = None

    def _schedule_af_safety_timer(self, run: PendingRunOutcome) -> None:
        """Arm the stuck-AF bound for *run*: a standalone AF that stops
        progressing is force-aborted rather than holding the lockout until
        the user notices. The run pipeline bounds stalled MOTION, not a
        stalled AF algorithm, so the bound lives with this starter.
        """
        ctx = _app_ctx.ctx
        self._unschedule_af_safety_timer()

        def _af_safety(dt):
            # Keyed on the run this bound was armed for, never on whatever
            # this button holds when it fires, and never on the AF thread
            # being busy: a timer that outlived its run must stay out of
            # reach of the next autofocus, and a rival run's AF step is
            # never this button's to stop.
            if ctx.sequenced_capture_runner.is_live_run(run):
                logger.warning('[AF Safety] Autofocus appeared stuck. Forced abort.')
                self._stop_autofocus(run)

        self._af_safety_event = Clock.schedule_once(_af_safety, AF_SAFETY_TIMEOUT_S)

    def _autofocus_run_complete(self, **kwargs):
        ctx = _app_ctx.ctx
        self._unschedule_af_safety_timer()

        # Defensive abort -- if the AF thread is somehow still in flight
        # at the completion path, this is a no-op; if not, it unwinds.
        # Ahead of the store write below, which is allowed to raise: the
        # unwind must not be skippable by a failure in the focus update.
        try:
            if ctx.autofocus_thread is not None:
                ctx.autofocus_thread.abort()
        except Exception:
            logger.debug('[AF] defensive AF-thread abort at completion failed', exc_info=True)

        # Ask the autofocus what it found. Sampling the stage instead
        # reads an in-transit coordinate: the pre-AF restore is issued
        # without waiting, so at this point the stage may still be
        # travelling, and that coordinate was committed to the layer and
        # persisted. No result means no answer to store -- an autofocus
        # that found nothing must leave the layer's focus alone.
        layer = common_utils.get_opened_layer(ctx.image_settings)
        focus_z = (
            ctx.autofocus_runner.best_focus_position() if ctx.autofocus_runner is not None else None
        )
        if layer is not None and focus_z is not None:
            with ctx.settings_lock:
                ctx.settings[layer]['focus'] = focus_z
            logger.info(f'[AF] Updated {layer} focus to {focus_z:.2f}um')

        # AF restored the camera from committed settings; an uncommitted
        # text edit (typed, no Enter) would keep showing a value the
        # hardware no longer has. Re-point the widgets at the truth. Not
        # conditional on a result: the camera restore happens on every
        # terminal path, and these widgets show illumination, gain and
        # exposure rather than focus.
        if layer is not None:
            try:
                layer_obj = ctx.image_settings.layer_lookup(layer=layer)
                layer_obj.sync_widgets_from_settings()
            except Exception as e:
                logger.warning(f'[AF] Widget sync after AF failed: {e}')

    def reset_turret_objective(self):
        """Clear the assignment of the slot in the light path.

        The Session clears the slot the API reports and refuses while that
        slot is unknown; the refusal is shown, and the display follows the
        API either way.
        """
        gui_logger.button('RESET_TURRET_OBJECTIVE')
        run_reported(
            _app_ctx.ctx.session.clear_current_turret_objective,
            redraw=lambda: self.show_turret_state(prompt=False),
            label='RESET_TURRET_OBJECTIVE',
        )

        # No prompt follows, deliberately. The press IS the user saying
        # this slot is empty, and the objective prompt has no cancel
        # path -- so asking here forced an objective back into the slot
        # that had just been cleared, leaving the button unable to do
        # its job at any position. Arriving at an unassigned slot still
        # asks, and so does startup, so no position goes unasked before
        # its objective matters.

    def prompt_if_objective_unknown(self, on_resolved=None):
        """Ask the Session whether the objective needs confirming; render the answer.

        The decision is the Session's (first run, or an unassigned slot at
        the current position; withheld with no hardware and while settings
        are provisional). This widget only shows the question and hands
        the choice back. A raise anywhere on the path becomes a
        notification: this runs on Clock callbacks, where a raise exits
        the app.

        Args:
            on_resolved: Run once the objective is settled, however it
                settles -- answered, not owed, or the question itself
                failing. The startup sequence hangs the persisted protocol
                load on it, because what the turret carries decides
                whether that protocol can be performed at all. It is NOT
                run while settings are provisional: the question is owed
                but unanswerable, and the host re-asks when they resolve.
                Hanging it only on the answer would strand the load behind
                the two failure paths here, which report to the user and
                return.
        """
        try:
            question = _app_ctx.ctx.session.objective_question()
            if question is None:
                if not _app_ctx.ctx.session.settings_are_provisional():
                    self._resolve_objective(on_resolved)
                return
            self._render_objective_question(question, on_resolved=on_resolved)
        except Exception as e:
            logger.error(f'[UI] objective question failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            # An objective that cannot be confirmed is unknown, and captures
            # refuse while it is: no file is written with a guessed scale.
            show_notification_popup(
                title='Objective not confirmed',
                message=(
                    f'The installed objective could not be confirmed: {e}\n'
                    'Captures are refused until the objective, and so the image scale, '
                    'is known.'
                ),
            )
            self._resolve_objective(on_resolved)

    def _resolve_objective(self, on_resolved) -> None:
        """Run the continuation, and never let it take the caller down.

        This runs on a Clock callback and inside except branches, where a
        raise exits the app or replaces one reported failure with another.
        """
        if on_resolved is None:
            return
        try:
            on_resolved()
        except Exception as e:
            logger.error(f'[UI] post-objective startup step failed: {e}', exc_info=True)

    def _render_objective_question(self, question, on_resolved=None):
        """The one popup for the objective question; the answer applies below."""
        if question.turret_position is not None:
            first_line = (
                'Please confirm the objective installed at turret position '
                f'{question.turret_position}.'
            )
        else:
            first_line = 'Please confirm the installed objective.'
        message = f'{first_line}\nThis sets the image scale recorded with every capture.'

        from ui.notification_popup import show_objective_selection_popup

        show_objective_selection_popup(
            title='Objective',
            message=message,
            objectives=list(question.choices),
            current_objective_id=question.proposed,
            on_confirm=lambda chosen: self._apply_objective_answer(
                chosen, question.turret_position, on_resolved=on_resolved
            ),
            on_folded=lambda: self._resolve_objective(on_resolved),
        )

    def _apply_objective_answer(self, chosen, turret_position, on_resolved=None):
        """Hand the answer to the Session and render what it did."""

        def _redraw():
            try:
                # Asks again only if the objective is still unknown -- the
                # turret moved to an unassigned slot while this question was
                # on screen -- and then about that slot, not this one.
                self.show_turret_state()
            finally:
                # Whatever the rendering above did, the objective question
                # is answered and the Session has it. A startup step waiting
                # on that must not be stranded by a widget write that failed.
                self._resolve_objective(on_resolved)

        run_reported(
            lambda: _app_ctx.ctx.session.confirm_objective(chosen, turret_position=turret_position),
            redraw=_redraw,
            label='OBJECTIVE_ANSWER',
        )

    @debounce(0.5)
    def turret_gesture(self, selected_position):
        """A person pressed a turret position button.

        Absorbing a double-press and recording a press are properties of
        the GESTURE, not of the turret move, so they live here and not on
        turret_select. While they sat on turret_select, the debounce keyed
        on the instance plus the method name, which is ONE window shared by
        these four buttons, the step-navigation path, the XY home and the
        protocol lane: a click within half a second of a run's turret move
        silently dropped the run's move while X, Y and Z went on to the
        step's coordinates. The record had the matching fault in the other
        direction, naming program-initiated moves as presses nobody made.
        """
        gui_logger.button(f'TURRET_POS_{selected_position}')
        self.turret_select(selected_position)

    def turret_select(self, selected_position):
        """Ask the API to turn the turret to a slot, then show where it is.

        The gesture above and step navigation reach this. It is therefore
        not debounced and writes no interaction record -- a program-initiated
        move is neither a double-press to absorb nor a press to report. The
        API refuses a slot that is not one and a turret that is not homed;
        the display runs after the move either way, so a failed move shows
        the turret in no known slot rather than in the one that was asked for.
        """
        ctx = _app_ctx.ctx
        motion = ctx.lumaview.scope.motion
        submit_reported(
            lambda: motion.move_turret(selected_position),
            self.show_turret_state,
            f'TURRET_POS_{selected_position}',
            lane=ctx.io_executor,
        )
