# Copyright Etaluma, Inc.
import logging

from kivy.clock import Clock
from kivy.properties import BooleanProperty
from kivy.uix.boxlayout import BoxLayout

import _lumascope.modules.app_context as _app_ctx
from _lumascope.modules import gui_logger
from _lumascope.modules.debounce import debounce
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _lumascope.modules.operation import Operation
from _lumascope.modules.run_events import RunEvents
from ui.ui_helpers import (
    go_to_bookmark,
    jog,
    move_home,
    move_z,
    run_reported,
    run_unasked,
    show_captured_frame,
    submit_reported,
    typed_number,
)

logger = logging.getLogger('LVP.ui.vertical_control')


# ============================================================================
# VerticalControl -- Z-Axis, Objectives, Turret, and Autofocus
# ============================================================================


class VerticalControl(BoxLayout):
    # True while this button's own request is on its way to the engine; the
    # Autofocus button is disabled until that request's redraw.
    autofocus_pending = BooleanProperty(False)
    # True while anything but the Autofocus button's own run holds the scope --
    # another run, a recording, a diagnostic -- as the Session answers; greys
    # the button, and its own run leaves it live as that run's Stop.
    autofocus_held = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] VerticalControl.__init__()')

        # boolean describing whether the scope is currently in the process of autofocus
        self.is_autofocus = False
        self.record_autofocus_to_file = False
        self._next_pos = None
        # The handle this button's last start returned: what its Stop
        # names. The engine answers whether it is live.
        self._autofocus_run: Operation | None = None

        self.queue_slider_position_trigger = Clock.create_trigger(
            lambda dt: self.queue_slider_position(), 0.1
        )

    def update_gui(self):
        # The reads are cache reads -- no lane, no serial I/O. Every caller
        # is on the UI thread; the draw is deferred to its next frame.
        Clock.schedule_once(lambda dt: run_reported(None, self.redraw_z, 'Z_POSITION'), 0)

    def _write_z_text(self, pos):
        """Write a Z read-back into its box.

        While the user is in the box it holds the read-back and shows it once
        left unedited (``CommitInput``), so a click in and out moves nothing
        and a part-typed entry is never written over. ``pos`` None, a Z the
        API does not know (homing, or lost), is shown as ``--``; committed
        from the box, it moves nothing.
        """
        # The figures are a readout; whether the API knows Z is the change.
        gui_logger.display('Z_POSITION_KNOWN', pos is not None)
        self.ids['z_position_id'].text = '--' if pos is None else format(pos, '.2f')

    def redraw_z(self):
        """Draw the Z panel as the API answers it now: the slider at the target, the text where Z is.

        The text is the position the API knows, never the target: while Z
        homes or is lost the target still answers the last number Z
        reported, and the box would show a Z nobody established. A scope
        without Z is left as it is.
        """
        scope = _app_ctx.ctx.lumaview.scope
        if not scope.capabilities.has_focus:
            return
        self.show_z(scope.motion.axis_positions()['Z'].position_um)

    def show_z_travel(self) -> None:
        """Span the focus slider over the scope's Z travel.

        The travel is ``MotionAPI.axis_limits('Z')``, the one owner of
        it; the slider's kv range is a placeholder until this runs at
        bring-up. The range is display: a move past travel is the API's
        refusal, whatever the slider offers.
        """
        scope = _app_ctx.ctx.lumaview.scope
        if not scope.capabilities.has_focus:
            return
        travel = scope.motion.axis_limits('Z')
        slider = self.ids['obj_position']
        slider.min = travel.min_um
        slider.max = travel.max_um

    def show_z(self, polled):
        """Show a Z position change: the slider at the API's Z target, the text at ``polled``.

        ``polled`` is None while Z's position is not known, and the text
        then says so rather than showing the last number Z reported.

        The listener bridge calls this, on the UI thread, for every Z move,
        run or not: a step's move, each autofocus sample, the restore when
        an autofocus gives up; ``redraw_z`` calls it for every other redraw.
        The slider is where Z is going and the text where it is. The slider
        is left alone while the user holds it, as the text is while the box
        has focus, so a drag is not overwritten.
        """
        slider = self.ids['obj_position']
        target = _app_ctx.ctx.lumaview.scope.motion.target_positions().get('Z')
        if target is not None and not slider.user_interacting:
            slider.value = target
        self._write_z_text(polled)

    def _z_jog(self, direction: int, coarse: bool):
        """Shared Z-axis jog handler.

        Args:
            direction: +1 for up, -1 for down.
            coarse: True for coarse step, False for fine step.
        """
        label = f'Z_{"COARSE" if coarse else "FINE"}_{"UP" if direction > 0 else "DOWN"}'
        gui_logger.button(label)
        logger.info(f'[LVP Main  ] VerticalControl._z_jog({label})')
        jog('Z', direction, coarse)

    @debounce(0.2)
    def coarse_up(self):
        self._z_jog(+1, coarse=True)

    @debounce(0.2)
    def fine_up(self):
        self._z_jog(+1, coarse=False)

    @debounce(0.2)
    def fine_down(self):
        self._z_jog(-1, coarse=False)

    @debounce(0.2)
    def coarse_down(self):
        self._z_jog(-1, coarse=True)

    def _queue_z_move(self, pos: float) -> None:
        """Queue the move to a committed Z value.

        The slider and the text box share the move but not the record. The
        slider reports the value it resolved to; the box reports what the user
        typed, before anything parsed it. A drag and a keystroke have to stay
        distinguishable in the bundle, so each caller writes its own line and
        only the move lives here.
        """
        logger.info('[LVP Main  ] VerticalControl.set_position()')
        self._next_pos = pos
        self.queue_slider_position_trigger()

    def set_position(self, pos):
        """The SLIDER's commit -- the kv binds this to its on_release."""
        self._queue_z_move(pos)
        gui_logger.slider('Z_POSITION', pos)

    def set_position_text(self, text):
        """The BOX's commit -- what was typed is recorded before the move.

        Separate from ``set_position`` so a typed commit does not report
        itself as a drag; the record name is shared because it is the same
        setting, and the verb is what tells them apart.
        """
        gui_logger.text_input('Z_POSITION', text)
        pos = typed_number(text, float, self.redraw_z)
        if pos is None:
            gui_logger.text_input('Z_POSITION_APPLIED', self.ids['z_position_id'].text)
            return
        self._queue_z_move(pos)

    def queue_slider_position(self):
        move_z(self._next_pos)
        self._next_pos = None

    def set_bookmark(self):
        gui_logger.button('SET_Z_BOOKMARK')
        logger.info('[LVP Main  ] VerticalControl.set_bookmark()')
        run_reported(self.ex_set_bookmark, None, 'SET_Z_BOOKMARK')

    def ex_set_bookmark(self):
        _app_ctx.ctx.session.save_bookmark(('Z',))

    def set_all_bookmarks(self):
        gui_logger.button('SET_ALL_BOOKMARKS')
        logger.info('[LVP Main  ] VerticalControl.set_all_bookmarks()')
        run_reported(self.ex_set_all_bookmarks, None, 'SET_ALL_BOOKMARKS')

    def ex_set_all_bookmarks(self):
        # Also writes every channel's focus from the Z.
        _app_ctx.ctx.session.save_all_bookmarks()

    def goto_bookmark(self):
        gui_logger.button('GOTO_Z_BOOKMARK')
        logger.info('[LVP Main  ] VerticalControl.goto_bookmark()')
        go_to_bookmark('GOTO_Z_BOOKMARK', ('Z',))

    @debounce(1.0)
    def home(self):
        gui_logger.button('HOME_Z')
        logger.info('[LVP Main  ] VerticalControl.home()')
        run_reported(lambda: move_home(axis='Z'), None, 'HOME_Z')

    def show_turret_state(self, prompt=True):
        """Show what the API says: the turret slot in the light path, each
        slot's assignment, and the active objective.

        Every turret action ends here, a failed one included -- a press,
        a home, a run's step, startup, an objective answer -- because only
        the API knows where the turret is. The button down is the slot the
        API reports and none is down while that slot is unknown, so the
        working objective is the name on the pressed button; each button
        names the objective assigned to its slot. On a scope without a
        turret, Assign Objective names the active objective.

        Args:
            prompt: Ask the objective question when the objective is
                unknown. The Session decides whether a question is owed,
                and owes none while an activity holds the scope: a prompt
                must never interrupt an unattended run, and a recording
                would refuse the answer.
        """
        ctx = _app_ctx.ctx
        turret = ctx.scope.motion.turret_status()
        slot = turret.slot if turret is not None else None
        for position, assigned in ctx.scope.runtime_state.turret_config().items():
            button = self.ids[f'turret_pos_{position}_btn']
            button.state = 'down' if position == slot else 'normal'
            # A name outside the catalogue is shown as assigned, because it
            # is; the active objective there is unknown.
            button.text = f'< {position} >' if assigned is None else assigned

        # The objective, or the refusal that says why it is unknown.
        objective = ctx.scope.runtime_state.optics_status().objective
        objective_id = objective if isinstance(objective, str) else None
        self.ids['assign_objective_btn'].text = objective_id or 'Assign Objective'
        shown = objective_id or f'Unknown ({objective.reason})'
        if ctx.scope.capabilities.has_turret:
            # The turret row is shown only on a scope that has one.
            slots = ' '.join(
                f'{position}={self.ids[f"turret_pos_{position}_btn"].text}'
                for position in ctx.scope.runtime_state.turret_config()
            )
            shown += f' slot={slot} turret=[{slots}]'
        gui_logger.display('OBJECTIVE', shown)
        ctx.motion_settings.ids['microscope_settings_id'].refresh_fov_labels()
        if objective_id is None and prompt:
            # Scheduled, never opened from here: the startup home's display
            # runs before the event loop, and a popup opened then is painted
            # under the app root -- open, and invisible. On the Clock it
            # waits for the loop and folds into the startup question.
            Clock.schedule_once(lambda dt: self.prompt_if_objective_unknown(), 0)

    def run_autofocus_from_ui(self):
        """Autofocus on the open channel, or stop the autofocus this button started.

        The run is ProtocolRunner.start_autofocus, the one a script or REST
        calls: everything about it -- the objective, the capture, every
        refusal -- is the member's, and the boundary shows a refusal once.
        This button states only what a running GUI knows: the open drawer
        and the live engineering flag, which is also whether the sweep's
        characterization data is saved. The run records the GUI as its
        starter from the host's stamp on the worker that starts it.

        Whether the press means Stop is the engine's answer -- is the run
        this button started still live -- never the toggle's, which Kivy has
        already flipped. The button changes nothing ahead of the engine's
        answer; draw_autofocus_button shows it.
        """
        gui_logger.button('AUTOFOCUS')
        logger.info('[LVP Main  ] VerticalControl.run_autofocus_from_ui()')
        ctx = _app_ctx.ctx
        run = self._autofocus_run
        if run is not None and run.progress().is_live:
            self._stop_autofocus(run)
            return

        member = ctx.session.protocol_runner()
        channel = ctx.session.shown_channel.channel
        engineering_mode = ctx.session.engineering_mode
        events = RunEvents(
            frame_captured=show_captured_frame,
            run_ended=lambda *_ended: self._autofocus_run_ended(channel),
            on_ui_thread=True,
        )

        def _start():
            self._autofocus_run = member.start_autofocus(
                channel=channel,
                save_characterization_data=engineering_mode,
                events=events,
            )

        self._submit_autofocus_request(_start)

    def _stop_autofocus(self, run: 'Operation') -> None:
        self._submit_autofocus_request(run.stop, stop=True)

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
        ctx = _app_ctx.ctx
        run = self._autofocus_run
        self.autofocus_held = ctx.session.held_by_other(run)
        button = self.ids['autofocus_id']
        progress = None if run is None else run.progress()
        if progress is None or not progress.is_live:
            button.state = 'normal'
            button.text = 'Autofocus'
            gui_logger.display('AUTOFOCUS_BUTTON', f'{button.text} held={self.autofocus_held}')
            return
        button.state = 'down'
        button.text = 'Stopping...' if progress.is_stopping else 'Focusing...'
        gui_logger.display('AUTOFOCUS_BUTTON', f'{button.text} held={self.autofocus_held}')

    def _autofocus_run_ended(self, channel: str | None) -> None:
        """Show the camera settings of the channel this button's autofocus focused, as the run left them.

        ``channel`` is the drawer open when the button was pressed, which is
        the channel the run focused, whatever is open now. The run itself
        saved the Z it chose as that channel's focus (``start_autofocus``),
        before it let go of the scope; this handler displays.
        """
        ctx = _app_ctx.ctx
        if channel is None:
            return
        # AF restored the camera from committed settings; an uncommitted
        # text edit (typed, no Enter) would keep showing a value the
        # hardware no longer has. Re-point the widgets at the truth. Not
        # conditional on a result: the camera restore happens on every
        # terminal path, and these widgets show illumination, gain and
        # exposure, not focus.
        ctx.image_settings.channel_control_lookup(channel=channel).sync_widgets_from_settings()

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
        # this slot is empty; asking straight back which objective is in
        # it is a question they have just answered. Pressing the slot
        # names its glass when they want to.

    def assign_objective(self):
        """Ask which objective is installed, on a scope without a turret.

        The question is the Session's, asked whether or not one is owed,
        and is shown by the one objective popup; its answer and its
        Cancel go where the owed question's do. A refusal is shown.
        """
        gui_logger.button('ASSIGN_OBJECTIVE')
        # The popup is opened inside the reported call: run_reported drops
        # the call's return, and the question is that return.
        run_reported(
            lambda: self._render_objective_question(
                _app_ctx.ctx.session.current_slot_objective_question()
            ),
            redraw=None,
            label='ASSIGN_OBJECTIVE',
        )

    def prompt_if_objective_unknown(self, on_resolved=None):
        """Ask the Session whether the objective needs confirming; render the answer.

        The decision is the Session's (first run, or an unassigned slot at
        the current position; withheld with no hardware and while settings
        are provisional). This widget only shows the question and hands
        the choice back. The question runs through the boundary: this runs
        on Clock callbacks, where a raise exits the app, and the Session's
        refusal -- the turret in no known slot, an unreadable catalogue --
        is shown once by the one reporter, in its own words.

        Args:
            on_resolved: Run once the objective is settled, however it
                settles -- answered, not owed, or the question itself
                failing. The startup sequence hangs the persisted protocol
                load on it, because what the turret carries decides
                whether that protocol can be performed at all. It is NOT
                run while settings are provisional: the question is owed
                but unanswerable, and the host re-asks when they resolve.
                Hanging it only on the answer would strand the load behind
                a question that raised, which is reported and answers
                nothing.
        """

        def _ask():
            question = _app_ctx.ctx.session.objective_question()
            if question is None:
                if not _app_ctx.ctx.session.settings_are_provisional():
                    self._resolve_objective(on_resolved)
                return
            self._render_objective_question(question, on_resolved=on_resolved)

        # Runs only when the question raised. A continuation that ran inside
        # and raised was contained by _resolve_objective, so it is never run
        # twice.
        if not run_unasked(_ask, 'OBJECTIVE_QUESTION'):
            self._resolve_objective(on_resolved)

    def _resolve_objective(self, on_resolved) -> None:
        """Run the continuation through the boundary; a raise is reported, not raised.

        This runs on a Clock callback, where a raise exits the app, and
        after a question that raised, where it would replace one reported
        failure with another. Nothing waits on the step it runs, so its
        failure is shown as one.
        """
        if on_resolved is None:
            return
        run_unasked(on_resolved, 'OBJECTIVE_CONTINUATION')

    def _render_objective_question(self, question, on_resolved=None):
        """The one popup for the objective question; the answer applies below."""
        if question.turret_slot is not None:
            first_line = (
                f'Please confirm the objective installed at turret position {question.turret_slot}.'
            )
        else:
            first_line = 'Please confirm the installed objective.'
        message = f'{first_line}\nThis sets the image scale recorded with every capture.'

        from ui.notification_popup import show_objective_selection_popup

        show_objective_selection_popup(
            title='Objective',
            message=message,
            objectives=[_app_ctx.ctx.session.scope.objective(o) for o in question.choices],
            current_objective_id=question.proposed,
            on_confirm=lambda chosen: self._apply_objective_answer(
                chosen, question.turret_slot, on_resolved=on_resolved
            ),
            on_cancel=lambda: self._resolve_objective(on_resolved),
            on_folded=lambda: self._resolve_objective(on_resolved),
        )

    def _apply_objective_answer(self, chosen, slot, on_resolved=None):
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
            lambda: _app_ctx.ctx.session.confirm_objective(chosen, slot=slot),
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
