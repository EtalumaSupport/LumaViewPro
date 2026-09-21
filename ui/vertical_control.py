# Copyright Etaluma, Inc.
import functools
import logging
import pathlib

from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_ui_getters as config_ui_getters
import modules.config_helpers as config_helpers
from modules.config_ui_getters import (
    get_active_layer_config,
    get_auto_gain_settings,
    get_binning_from_ui,
    get_image_capture_config_from_ui,
    get_selected_labware,
)
from modules import gui_logger
from modules.debounce import debounce
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.sequential_io_executor import IOTask, PRIORITY_HIGH
from modules.tiling_config import TilingConfig
from ui.protocol_settings import require_file_writes_idle
from ui.ui_helpers import (
    _handle_ui_update_for_axis,
    live_display_callbacks,
    live_histo_off,
    live_histo_reverse,
    move_absolute,
    move_home,
    move_relative,
    run_with_refusal_boundary,
    reset_with_refusal_boundary,
)

logger = logging.getLogger('LVP.ui.vertical_control')

AF_SAFETY_TIMEOUT_S = 15  # Seconds before AF is considered stuck and force-reset


# ============================================================================
# VerticalControl -- Z-Axis, Objectives, Turret, and Autofocus
# ============================================================================


class VerticalControl(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] VerticalControl.__init__()')

        # boolean describing whether the scope is currently in the process of autofocus
        self.is_autofocus = False
        self.is_complete = False
        self.record_autofocus_to_file = False
        self._next_pos = None
        self._af_safety_event = None

        self.queue_slider_position_trigger = Clock.create_trigger(
            lambda dt: self.queue_slider_position(), 0.1
        )

    def update_gui(self, vertical_control=False):
        ctx = _app_ctx.ctx
        if ctx.sequenced_capture_runner.run_in_progress():
            return
        if not vertical_control:
            ctx.io_executor.put(
                IOTask(
                    action=ctx.lumaview.scope.motion.get_target_position,
                    args=('Z'),
                    callback=self.execute_kivy_gui,
                    cb_kwargs={'vertical_control': vertical_control},
                    pass_result=True,
                )
            )
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

    def execute_kivy_gui(self, vertical_control=False, result=None, exception=None):
        """IOTask callback -- runs on worker thread. Must schedule widget access."""
        if exception is not None:
            raise exception

        if result is None:
            return

        set_pos = result

        # Widget access must happen on the main Kivy thread (H24).
        # This callback runs on the IO worker thread.
        from kivy.clock import Clock

        if not vertical_control:
            Clock.schedule_once(lambda dt, p=set_pos: self._update_z_position(p), 0)
        else:
            Clock.schedule_once(lambda dt, p=set_pos: self._update_z_text(p), 0)

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
        try:
            _, objective = ctx.session.get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] {label}: no objective info: {e}')
            return
        step = objective['z_coarse' if coarse else 'z_fine']
        move_relative('Z', direction * step, overshoot_enabled=overshoot_enabled)

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
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.set_bookmark()')
        ctx.io_executor.put(IOTask(action=self.ex_set_bookmark))

    def ex_set_bookmark(self):
        ctx = _app_ctx.ctx
        height = ctx.lumaview.scope.motion.get_current_position('Z')  # Get current z height in um
        with ctx.settings_lock:
            ctx.settings['bookmark']['z'] = height

    def set_all_bookmarks(self):
        gui_logger.button('SET_ALL_BOOKMARKS')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.set_all_bookmarks()')
        ctx.io_executor.put(IOTask(action=self.ex_set_all_bookmarks))

    def ex_set_all_bookmarks(self):
        ctx = _app_ctx.ctx
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
        try:
            gui_logger.button('HOME_Z')
            ctx = _app_ctx.ctx
            if ctx.session.controls_locked:
                return
            logger.info('[LVP Main  ] VerticalControl.home()')
            move_home(axis='Z')
        except Exception as e:
            logger.error(f'[UI] home failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def load_objectives(self):
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.load_objectives()')
        spinner = self.ids['objective_spinner2']
        spinner.values = ctx.objective_helper.get_objectives_list()

    def select_objective(self):
        try:
            ctx = _app_ctx.ctx
            objective_id = self.ids['objective_spinner2'].text

            # on_text fires for programmatic text writes too (settings load,
            # a turret move, the prompt's own answer); the Session reports
            # whether anything changed and writes nothing when it did not.
            changed = ctx.session.select_objective(objective_id)
            if not changed:
                return

            # Only log objective changes from user interaction, not protocol
            if not ctx.session.is_protocol_running:
                gui_logger.select('OBJECTIVE', objective_id)
            logger.info('[LVP Main  ] VerticalControl.select_objective()')

            self._refresh_fov(objective_id)
        except Exception as e:
            logger.error(f'[UI] select_objective failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def _refresh_fov(self, objective_id):
        """Show the field of view the objective gives at the current frame."""
        ctx = _app_ctx.ctx
        objective = ctx.session.get_objective_info(objective_id=objective_id)
        microscope_settings_id = ctx.motion_settings.ids['microscope_settings_id']
        fov_size = config_ui_getters.get_field_of_view(
            focal_length=objective['focal_length'],
            frame_size=ctx.settings['frame'],
            binning_size=get_binning_from_ui(),
        )
        fov_w_text, fov_h_text = common_utils.format_field_of_view(fov_size)
        microscope_settings_id.ids['field_of_view_width_id'].text = fov_w_text
        microscope_settings_id.ids['field_of_view_height_id'].text = fov_h_text

    def _reset_run_autofocus_button_cosmetics(self, **kwargs):
        self.ids['autofocus_id'].state = 'normal'
        self.ids['autofocus_id'].text = 'Autofocus'

    def _reset_run_autofocus_button(self, **kwargs):
        # Protocol AF steps route their completion through this funnel
        # too, so it must never touch the shared lockout state -- the
        # standalone release is generation-owned and lives with the
        # standalone exits.
        #
        # Cosmetics only, deliberately. This runs as the completion
        # callback of the teardown task, and a completion callback fires
        # whatever the outcome -- including a teardown the engine
        # REFUSED. An abort here therefore killed the autofocus of a run
        # the caller had just been told it did not own. Unwinding the
        # autofocus belongs to the engine's cleanup, which does it for
        # the run's owner and waits for the sweep to finish before
        # restoring the LEDs.
        self._reset_run_autofocus_button_cosmetics()

    def _set_run_autofocus_button(self, **kwargs):
        self.ids['autofocus_id'].state = 'down'
        self.ids['autofocus_id'].text = 'Focusing...'

    def _cleanup_at_end_of_autofocus(self):
        ctx = _app_ctx.ctx

        # SequencedCaptureRunner.reset() unwinds any running protocol
        # (its _cleanup chain calls autofocus_thread.abort() on the AF
        # thread and fires the run/files-complete callbacks, which
        # release the lockout). AFE state is reset implicitly on the
        # next AFE.run().
        ctx.worker_pool.put(
            IOTask(
                # Through the boundary, which returns the outcome instead
                # of raising: a teardown this button does not own is
                # refused, and the callback below must still run to put
                # the button back -- a refused stop is not a stop, and a
                # button left mid-stop is dead until the process ends.
                action=functools.partial(
                    reset_with_refusal_boundary,
                    ctx.sequenced_capture_runner,
                    requester='autofocus',
                ),
                callback=self._reset_run_autofocus_button,
                # The engine logs and notifies a refusal exactly once.
                # Without this the executor's generic failure popup fires a
                # SECOND notification for the same event -- and titles it
                # from the action, which for a partial is its repr, heap
                # address and all.
                silent_on_failure=True,
                priority=PRIORITY_HIGH,
            )
        )

    def _unschedule_af_safety_timer(self):
        if self._af_safety_event is not None:
            Clock.unschedule(self._af_safety_event)
            self._af_safety_event = None

    def _schedule_af_safety_timer(self):
        """Arm the stuck-AF bound: a standalone AF that stops progressing
        is force-aborted rather than holding the lockout until the user
        notices. The run pipeline bounds stalled MOTION, not a stalled
        AF algorithm, so the bound lives with this starter.
        """
        ctx = _app_ctx.ctx
        self._unschedule_af_safety_timer()

        def _af_safety(dt):
            runner = ctx.sequenced_capture_runner
            # Key on this button's own run, not on the AF thread being
            # busy: a rival run's AF step in flight when a stale timer
            # fires must stay out of reach.
            if runner.run_in_progress() and runner.run_trigger_source() == 'autofocus':
                logger.warning('[AF Safety] Autofocus appeared stuck. Forced abort.')
                self._cleanup_at_end_of_autofocus()

        self._af_safety_event = Clock.schedule_once(_af_safety, AF_SAFETY_TIMEOUT_S)

    def _autofocus_run_complete(self, **kwargs):
        ctx = _app_ctx.ctx
        self._unschedule_af_safety_timer()
        live_histo_reverse()
        Clock.schedule_once(lambda dt: self._reset_run_autofocus_button(), 0)

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

    def run_autofocus_from_ui(self):
        try:
            gui_logger.button('AUTOFOCUS')
            ctx = _app_ctx.ctx
            logger.info('[LVP Main  ] VerticalControl.run_autofocus_from_ui()')
            settings = ctx.settings
            trigger_source = 'autofocus'
            runner = ctx.sequenced_capture_runner
            run_trigger_source = runner.run_trigger_source()

            # A click during someone else's run falls through to the stop
            # branch below and the engine refuses the teardown, naming the
            # run that holds the scope. This widget asks nothing about
            # rival runs: the same refusal has to reach a script and REST,
            # so it is the engine's to give.

            # Stop click: the toggle is back to 'normal', or re-clicked
            # while this button's own run is live. The ownership term is
            # load-bearing -- a run callback can reset this button to
            # 'normal' mid-run, and Kivy flips a toggle at touch-down, so
            # the user's own Stop can arrive reading 'down'.
            if self.ids['autofocus_id'].state == 'normal' or (
                runner.run_in_progress() and run_trigger_source == trigger_source
            ):
                self._cleanup_at_end_of_autofocus()
                return

            # The post-run file drain outlives the run by design: writes
            # keep landing after the run itself has ended, so it needs a
            # gate of its own here rather than riding on the run's. The
            # gate helper owns the stalled-writer recovery popup.
            if not require_file_writes_idle('start autofocus'):
                self._reset_run_autofocus_button_cosmetics()
                return

            if ctx.engineering_mode:
                save_autofocus_data = True
                parent_dir = (
                    pathlib.Path(settings['live_folder']).resolve() / 'Autofocus Characterization'
                )
            else:
                save_autofocus_data = False
                parent_dir = None

            live_histo_off()

            def run_refused_func():
                self._reset_run_autofocus_button_cosmetics()
                live_histo_reverse()

            self._set_run_autofocus_button()

            # A one-position run at the current location: the active
            # layer with autofocus enabled, nothing saved. The same
            # degenerate-plan recipe as the z-stack starter, so the
            # standalone button and a protocol AF step share one engine.
            labware_id, _ = get_selected_labware()
            objective_id, _ = ctx.session.get_current_objective_info()
            active_layer, active_layer_config = get_active_layer_config(
                common_utils.get_opened_layer(ctx.image_settings)
            )
            active_layer_config['acquire'] = 'image'
            active_layer_config['autofocus'] = True

            curr_position = ctx.session.get_current_plate_position()
            curr_position.update({'name': 'Autofocus'})

            tiling_config = TilingConfig(
                tiling_configs_file_loc=pathlib.Path(ctx.source_path) / 'data' / 'tiling.json',
            )
            config = config_helpers.build_sequenced_capture_config(
                {
                    'labware_id': labware_id,
                    'positions': [curr_position],
                    'objective_id': objective_id,
                    'zstack_params': {},
                    'use_zstacking': False,
                    'tiling': tiling_config.no_tiling_label(),
                    'tiling_overlap_percent': 0.0,
                    'layer_configs': {active_layer: active_layer_config},
                    'period': None,
                    'duration': None,
                    'frame_dimensions': config_helpers.get_frame_dimensions_from_settings(settings),
                    'binning_size': get_binning_from_ui(),
                    # A standalone autofocus never pulses stimulation;
                    # an empty config keeps the built step stim-free.
                    'stim_config': {},
                }
            )
            af_sequence = ctx.scope.protocols.create_protocol(input_config=config)

            callbacks = {
                **live_display_callbacks(),
                'move_position': _handle_ui_update_for_axis,
                'run_complete': self._autofocus_run_complete,
            }

            def prepare_and_start():
                plan = runner.prepare(
                    protocol=af_sequence,
                    run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
                    run_trigger_source=trigger_source,
                    max_scans=1,
                    sequence_name='autofocus',
                    parent_dir=parent_dir,
                    image_capture_config=get_image_capture_config_from_ui(),
                    enable_image_saving=False,
                    # The run saves no protocol artifacts; in
                    # engineering mode the AF characterization data
                    # allocates its own timestamped folder under
                    # parent_dir, the on-disk shape the standalone
                    # button has always produced.
                    disable_saving_artifacts=True,
                    save_autofocus_data=save_autofocus_data,
                    autogain_settings=get_auto_gain_settings(),
                    callbacks=callbacks,
                    update_z_pos_from_autofocus=False,
                    # A standalone autofocus is a one-shot at the field the
                    # user is already watching, so it ends by putting the live
                    # view back exactly as they had it, illumination included.
                    # Ending dark is the right policy for an acquisition that
                    # traverses the plate (the sample must not be left lit
                    # between positions), and the wrong one for a run that
                    # never leaves the current position. A fatal abort still
                    # forces dark regardless of this policy.
                    leds_state_at_end='return_to_original',
                    engineering_mode=ctx.engineering_mode,
                    autofocus_snapshot=config_helpers.autofocus_snapshot_from_settings(
                        settings, ctx.settings_lock
                    ),
                    **config_helpers.get_sequenced_run_settings(
                        settings, run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN
                    ),
                )
                runner.start(plan)
                # Armed by the run it bounds, never before it. Arming
                # ahead of prepare() outlived every exit between the arm
                # and a committed run -- a refusal, a raise out of the
                # builder, anything the blanket handler below catches --
                # and the bound cannot tell those apart: it fires on the
                # trigger source alone, so a click that started nothing
                # reached forward and aborted the next autofocus that did.
                self._schedule_af_safety_timer()

            run_with_refusal_boundary(prepare_and_start, on_refused=run_refused_func)
        except Exception as e:
            logger.error(f'[UI] run_autofocus_from_ui failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    @debounce(1.0)
    def turret_home(self):
        gui_logger.button('HOME_TURRET')
        ctx = _app_ctx.ctx
        if ctx.session.controls_locked:
            return

        def _on_turret_homed():
            Clock.schedule_once(lambda dt: self._reset_turret_buttons(), 0)

        ctx.io_executor.put(
            IOTask(
                action=ctx.lumaview.scope.motion._home_turret_impl,
                callback=_on_turret_homed,
            )
        )

    def _reset_turret_buttons(self):
        self.ids['turret_pos_1_btn'].state = 'normal'
        self.ids['turret_pos_2_btn'].state = 'normal'
        self.ids['turret_pos_3_btn'].state = 'normal'
        self.ids['turret_pos_4_btn'].state = 'normal'

    def _selected_turret_position(self):
        """The slot whose button is down, or None when none is."""
        for position in range(1, 5):
            if self.ids[f'turret_pos_{position}_btn'].state == 'down':
                return position
        return None

    def set_turret_objective(self):
        ctx = _app_ctx.ctx
        desired_objective_id = self.ids['objective_spinner2'].text
        gui_logger.select('TURRET_OBJECTIVE', desired_objective_id)

        selected_turret = self._selected_turret_position()
        if selected_turret is None:
            logger.error('VerticalControl] SetTurretObjective] No turret button selected')
            return

        try:
            magnification = ctx.session.get_objective_info(objective_id=desired_objective_id)[
                'magnification'
            ]
            self.ids[f'turret_pos_{selected_turret}_btn'].text = f'{magnification}x'
            ctx.session.assign_turret_objective(selected_turret, desired_objective_id)
        except Exception as e:
            logger.exception(f'SetTurretObjective] Error: {e}')
            return

    def reset_turret_objective(self):
        gui_logger.button('RESET_TURRET_OBJECTIVE')

        selected_turret = self._selected_turret_position()
        if selected_turret is None:
            logger.error('VerticalControl] ResetTurretObjective] No turret button selected')
            return

        try:
            self.ids[f'turret_pos_{selected_turret}_btn'].text = str(selected_turret)
            _app_ctx.ctx.session.clear_turret_objective(selected_turret)
        except Exception as e:
            logger.exception(f'ResetTurretObjective] Error: {e}')
            return

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

            show_notification_popup(
                title='Objective not confirmed',
                message=(
                    'The installed objective could not be confirmed, so the image scale '
                    f'recorded with captures may be wrong: {e}'
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
        )

    def _apply_objective_answer(self, chosen, turret_position, on_resolved=None):
        """Hand the answer to the Session and render what it did."""
        try:
            ctx = _app_ctx.ctx
            changed = ctx.session.confirm_objective(chosen, turret_position=turret_position)
            if changed and not ctx.session.is_protocol_running:
                gui_logger.select('OBJECTIVE', chosen)
            if changed:
                logger.info('[LVP Main  ] VerticalControl.select_objective()')
            # on_text reaches select_objective, whose Session call reports
            # no change and does nothing further.
            self.ids['objective_spinner2'].text = chosen
            if changed:
                self._refresh_fov(chosen)
            if turret_position is not None:
                gui_logger.select('TURRET_OBJECTIVE', chosen)
                self.update_all_turret_btn_states(turret_position)
                magnification = ctx.session.get_objective_info(objective_id=chosen)['magnification']
                self.ids[f'turret_pos_{turret_position}_btn'].text = f'{magnification}x'
        except Exception as e:
            logger.error(f'[UI] objective answer failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))
        finally:
            # Whatever the rendering above did, the objective question is
            # answered and the Session has it. A startup step waiting on
            # that must not be stranded by a widget write that failed.
            self._resolve_objective(on_resolved)

    @debounce(0.5)
    def turret_select(self, selected_position, protocol=False, restore_z=True):
        try:
            if not protocol:
                gui_logger.button(f'TURRET_POS_{selected_position}')
            ctx = _app_ctx.ctx
            settings = ctx.settings
            if not ctx.lumaview.scope.motion.has_turret_homed():
                if not protocol:
                    ctx.io_executor.put(IOTask(ctx.lumaview.scope.motion._home_turret_impl))
                else:
                    # Protocol context runs on protocol_thread, not the io
                    # worker -- route the turret home through the protocol queue so it
                    # stays ordered ahead of the subsequent move_turret/X/Y/Z and
                    # behind the prior step's leds_off on the single worker.
                    fut = ctx.io_executor.protocol_put(
                        IOTask(ctx.lumaview.scope.motion._home_turret_impl), return_future=True
                    )
                    if fut:
                        fut.result(timeout=120)

            if not isinstance(selected_position, int) and not isinstance(selected_position, float):
                # A digit string names a slot; anything else falls back to
                # slot 1. Left as a string, a digit would match no slot in the
                # loop below and skip the spinner sync and the prompt.
                selected_position = int(selected_position) if selected_position.isdigit() else 1
            else:
                selected_position = int(selected_position)

            if not protocol:
                ctx.io_executor.put(
                    IOTask(
                        ctx.lumaview.scope.motion._move_turret_impl,
                        kwargs={'position': selected_position},
                    )
                )
            else:
                # See the turret-home branch above: route the protocol-context
                # move_turret through the protocol queue so it serializes with the
                # step's other moves and LED ops on the single io worker
                # instead of racing them from protocol_thread.
                fut = ctx.io_executor.protocol_put(
                    IOTask(
                        ctx.lumaview.scope.motion._move_turret_impl,
                        kwargs={'position': selected_position, 'restore_z': restore_z},
                    ),
                    return_future=True,
                )
                if fut:
                    fut.result(timeout=60)

            # Record the user's explicit turret choice so the next session
            # (or any post-home lookup) prefers this position when the
            # objective at this slot is duplicated elsewhere on the turret.
            ctx.session.set_turret_position(selected_position)

            for available_position in range(1, 5):
                if selected_position == available_position:
                    # Check if an objective has been saved to that turret
                    turret_position_objective = settings['turret_objectives'][selected_position]
                    if turret_position_objective is not None:
                        # If an objective has been assigned to the turret position, change to that objective
                        Clock.schedule_once(
                            lambda dt: self.update_spinner_text(selected_position), 0
                        )
                        Clock.schedule_once(lambda dt: self.select_objective(), 0)
                    elif not protocol:
                        # The turret is moving to a position with no
                        # assignment: the previous objective would keep
                        # setting the image scale silently. The Session
                        # decides whether to ask (a declared non-turret
                        # model, e.g. the XY-home resync, is never asked)
                        # and has already warned for every host. A prompt
                        # must never interrupt an unattended run.
                        Clock.schedule_once(lambda dt: self.prompt_if_objective_unknown(), 0)

            Clock.schedule_once(lambda dt: self.update_all_turret_btn_states(selected_position), 0)
        except Exception as e:
            logger.error(f'[UI] turret_select failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def update_spinner_text(self, selected_position):
        settings = _app_ctx.ctx.settings
        self.ids['objective_spinner2'].text = settings['turret_objectives'][selected_position]

    def update_turret_btn_state(self, position, state):
        self.ids[f'turret_pos_{position}_btn'].state = state

    def update_all_turret_btn_states(self, selected_position):
        for available_position in range(1, 5):
            if selected_position == available_position:
                state = 'down'
            else:
                state = 'normal'
            self.update_turret_btn_state(available_position, state)

    def update_turret_gui(self, turret_position):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        # Record the position the turret physically ended up at -- this
        # is called after every protocol-driven or step-navigation T
        # move, so the recorded value tracks reality across moves.
        ctx.session.set_turret_position(int(turret_position))
        for available_position in range(1, 5):
            if turret_position == available_position:
                state = 'down'

                # Check if an objective has been saved to that turret
                turret_position_objective = settings['turret_objectives'][turret_position]
                if turret_position_objective is not None:
                    # If an objective has been assigned to the turret position, change to that objective
                    self.ids['objective_spinner2'].text = settings['turret_objectives'][
                        turret_position
                    ]
                    self.select_objective()

            else:
                state = 'normal'

            self.ids[f'turret_pos_{available_position}_btn'].state = state
