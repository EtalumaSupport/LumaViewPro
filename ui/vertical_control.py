# Copyright Etaluma, Inc.
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
    get_current_frame_dimensions,
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

    def update_autofocus_gui(self, pos=None):
        if pos is None:
            return

        self.ids['obj_position'].value = max(0, pos)
        # Cache text to prevent redundant ScrollView updates
        new_text = format(max(0, pos), '.2f')
        if self.ids['z_position_id'].text != new_text:
            self.ids['z_position_id'].text = new_text

    def update_text_only(self):
        # Cache text to prevent redundant ScrollView updates
        if not self.ids['z_position_id'].focus:
            new_text = format(max(0, self.ids['obj_position'].value), '.2f')
            if self.ids['z_position_id'].text != new_text:
                self.ids['z_position_id'].text = new_text

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
        if not self.ids['z_position_id'].focus:
            new_text = format(max(0, pos), '.2f')
            if self.ids['z_position_id'].text != new_text:
                self.ids['z_position_id'].text = new_text

    def _update_z_text(self, pos):
        """Update Z text only -- must be called on main thread."""
        if not self.ids['z_position_id'].focus:
            new_text = format(max(0, pos), '.2f')
            if self.ids['z_position_id'].text != new_text:
                self.ids['z_position_id'].text = new_text

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

    def set_position(self, pos):
        ctx = _app_ctx.ctx
        if ctx.session.controls_locked:
            return

        logger.info('[LVP Main  ] VerticalControl.set_position()')
        try:
            self._next_pos = float(pos)
        except Exception:
            return
        gui_logger.slider('Z_POSITION', self._next_pos)
        self.queue_slider_position_trigger()

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
            settings = ctx.settings
            objective_id = self.ids['objective_spinner2'].text

            # Idempotent: text matching current settings is not a change, so
            # skip the hardware calls and notifications. Defends against on_text
            # firing for programmatic text writes (settings load, restore).
            if objective_id == settings.get('objective_id'):
                return

            # Only log objective changes from user interaction, not protocol
            if not ctx.session.is_protocol_running:
                gui_logger.select('OBJECTIVE', objective_id)
            logger.info('[LVP Main  ] VerticalControl.select_objective()')

            # Selecting an objective the turret does not hold is a normal step
            # of assigning it: the user picks the objective, then presses Set to
            # bind it to the current position. Interrupting the first half to
            # complain about the second is why this used to pop a dialog saying
            # the selection had been refused -- which it never was; the write
            # below always happened. Logged, not raised: the moments where an
            # unassigned objective actually blocks something (creating,
            # modifying, adding to and running a protocol) each refuse there,
            # and those refusals are real.
            if ctx.lumaview.scope.capabilities.has_turret:
                turret_objectives = list(settings.get('turret_objectives', {}).values())
                assigned = [obj for obj in turret_objectives if obj is not None]
                if assigned and objective_id not in assigned:
                    logger.info(
                        f'[LVP Main  ] Objective {objective_id!r} selected with no turret '
                        f'position assigned; assigned objectives are {assigned}'
                    )

            # Update objective stored in settings
            objective = ctx.session.get_objective_info(objective_id=objective_id)
            with ctx.settings_lock:
                settings['objective_id'] = objective_id

            # Set objective in lumascope
            if ctx.lumaview.scope.capabilities.has_turret:
                ctx.lumaview.scope.runtime_state.set_turret_config(
                    turret_config=settings['turret_objectives']
                )

            ctx.lumaview.scope.runtime_state.set_objective(objective_id=objective_id)

            binning_size = get_binning_from_ui()

            # Update UI FOV
            microscope_settings_id = ctx.motion_settings.ids['microscope_settings_id']
            fov_size = config_ui_getters.get_field_of_view(
                focal_length=objective['focal_length'],
                frame_size=settings['frame'],
                binning_size=binning_size,
            )
            fov_w_text, fov_h_text = common_utils.format_field_of_view(fov_size)
            microscope_settings_id.ids['field_of_view_width_id'].text = fov_w_text
            microscope_settings_id.ids['field_of_view_height_id'].text = fov_h_text
        except Exception as e:
            logger.error(f'[UI] select_objective failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def _reset_run_autofocus_button_cosmetics(self, **kwargs):
        self.ids['autofocus_id'].state = 'normal'
        self.ids['autofocus_id'].text = 'Autofocus'

    def _reset_run_autofocus_button(self, **kwargs):
        # Protocol AF steps route their completion through this funnel
        # too, so it must never touch the shared lockout state -- the
        # standalone release is generation-owned and lives with the
        # standalone exits.
        ctx = _app_ctx.ctx
        if ctx.autofocus_thread is not None:
            ctx.autofocus_thread.abort()
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
                action=ctx.sequenced_capture_runner.reset,
                callback=self._reset_run_autofocus_button,
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

        # Update per-layer focus in settings so new protocol steps use the
        # AF result, not the stale pre-AF Z value.
        try:
            focus_z = ctx.scope.motion.get_current_position('Z')
            layer = common_utils.get_opened_layer(ctx.image_settings)
            if layer is not None:
                with ctx.settings_lock:
                    ctx.settings[layer]['focus'] = focus_z
                logger.info(f'[AF] Updated {layer} focus to {focus_z:.2f}um')
                # AF restored the camera from committed settings; an
                # uncommitted text edit (typed, no Enter) would keep
                # showing a value the hardware no longer has. Re-point
                # the widgets at the truth.
                try:
                    layer_obj = ctx.image_settings.layer_lookup(layer=layer)
                    layer_obj.sync_camera_widgets_from_settings()
                except Exception as e:
                    logger.warning(f'[AF] Widget sync after AF failed: {e}')
        except Exception as e:
            logger.warning(f'[AF] Failed to update layer focus after AF: {e}')

        # Defensive abort -- if the AF thread is somehow still in flight
        # at the completion path, this is a no-op; if not, it unwinds.
        try:
            if ctx.autofocus_thread is not None:
                ctx.autofocus_thread.abort()
        except Exception:
            logger.debug('[AF] defensive AF-thread abort at completion failed', exc_info=True)

    def run_autofocus_from_ui(self):
        try:
            gui_logger.button('AUTOFOCUS')
            ctx = _app_ctx.ctx
            logger.info('[LVP Main  ] VerticalControl.run_autofocus_from_ui()')
            settings = ctx.settings
            trigger_source = 'autofocus'
            runner = ctx.sequenced_capture_runner
            run_trigger_source = runner.run_trigger_source()

            # Abort click: the toggle is back to 'normal', or re-clicked
            # while this button's own run is live.
            if self.ids['autofocus_id'].state == 'normal' or (
                runner.run_in_progress() and run_trigger_source == trigger_source
            ):
                self._cleanup_at_end_of_autofocus()
                return

            # A rival run owns the scope; undo cosmetics ONLY -- the
            # lockout is that run's to keep.
            if runner.run_in_progress():
                self._reset_run_autofocus_button_cosmetics()
                logger.warning(
                    'Cannot start autofocus: run already in progress '
                    f'(trigger={run_trigger_source})'
                )
                return

            # The post-run file drain deliberately holds the lockout
            # while run_in_progress() is already False; the gate helper
            # owns the stalled-writer recovery popup.
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
            self._schedule_af_safety_timer()

            # A one-position run at the current location: the active
            # layer with autofocus enabled, nothing saved. The same
            # degenerate-plan recipe as the z-stack starter, so the
            # standalone button and a protocol AF step share one engine.
            labware_id, _ = get_selected_labware()
            objective_id, _ = ctx.session.get_current_objective_info()
            active_layer, active_layer_config = get_active_layer_config()
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
                    'frame_dimensions': get_current_frame_dimensions(),
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

    def set_turret_objective(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        gui_logger.select('TURRET_OBJECTIVE', self.ids['objective_spinner2'].text)

        selected_turret = None
        for position in range(1, 5):
            if self.ids[f'turret_pos_{position}_btn'].state == 'down':
                selected_turret = position

        if selected_turret is None:
            logger.error('VerticalControl] SetTurretObjective] No turret button selected')
            return

        try:
            selected_turret_id = self.ids[f'turret_pos_{selected_turret}_btn']

            # Find magnification of the selected objective
            desired_objective_id = self.ids['objective_spinner2'].text
            magnification = ctx.session.get_objective_info(objective_id=desired_objective_id)[
                'magnification'
            ]

            # Change turret text
            selected_turret_id.text = f'{magnification}x'

            # Update settings. The slot key is the STRING form: this dict is
            # loaded from JSON, whose object keys are always strings, while
            # range() hands out ints. Writing the int added a second entry
            # beside the string one, so every reader keyed by string kept
            # seeing the old value and the saved file carried a duplicate key.
            with _app_ctx.ctx.settings_lock:
                settings['turret_objectives'][selected_turret] = desired_objective_id

            # Push the new assignment to the microscope -- the settings write
            # alone does not reach hardware (mirrors select_objective).
            if ctx.lumaview.scope.capabilities.has_turret:
                ctx.lumaview.scope.runtime_state.set_turret_config(
                    turret_config=settings['turret_objectives']
                )

        except Exception as e:
            logger.exception(f'SetTurretObjective] Error: {e}')
            return

    def reset_turret_objective(self):
        settings = _app_ctx.ctx.settings

        selected_turret = None
        for position in range(1, 5):
            if self.ids[f'turret_pos_{position}_btn'].state == 'down':
                selected_turret = position

        if selected_turret is None:
            logger.error('VerticalControl] ResetTurretObjective] No turret button selected')
            return

        try:
            selected_turret_id = self.ids[f'turret_pos_{selected_turret}_btn']

            # Change turret text
            selected_turret_id.text = str(selected_turret)

            # Update settings -- string key, for the reason in
            # set_turret_objective: an int key writes a second entry instead of
            # clearing the one readers look at.
            with _app_ctx.ctx.settings_lock:
                settings['turret_objectives'][selected_turret] = None

            # Push the cleared slot to the microscope -- the settings write
            # alone does not reach hardware (mirrors select_objective).
            if _app_ctx.ctx.lumaview.scope.capabilities.has_turret:
                _app_ctx.ctx.lumaview.scope.runtime_state.set_turret_config(
                    turret_config=settings['turret_objectives']
                )

        except Exception as e:
            logger.exception(f'ResetTurretObjective] Error: {e}')
            return

        # Clearing the assignment at the position the turret is sitting
        # on leaves the app unable to say what is in the light path --
        # the previous objective would keep setting the image scale
        # silently. Ask the user instead.
        if selected_turret == settings.get('turret_position'):
            Clock.schedule_once(
                lambda dt: self.prompt_objective_selection(turret_position=selected_turret), 0
            )

    def prompt_objective_selection(self, turret_position=None):
        """Ask which objective is in the light path, and apply the answer.

        The pixel size derived from the objective is stamped into the
        scale bar and every saved image's metadata, and a wrong scale
        cannot be told from a measured one afterwards -- so when the app
        cannot know the objective, it asks instead of assuming silently.

        The answer performs the same actions a user does manually:
        select the objective (spinner -> select_objective), and for a
        turret position press Set (set_turret_objective) -- every write
        stays on the production path.

        Args:
            turret_position: Position the answer binds to (the answer
                also assigns that slot), or None on non-turret models.
        """
        ctx = _app_ctx.ctx
        # Asking is only useful when the answer can matter: with no
        # hardware there is nothing in the light path and no capture to
        # stamp, and this cancel-less modal would cover the notice that
        # explains why nothing works. The flag stays unconfirmed, so the
        # next hardware session asks.
        if ctx.lumaview.scope.no_hardware:
            logger.info('[LVP Main  ] Objective prompt suppressed -- no hardware this session')
            return
        # While settings are provisional every settings write is refused,
        # so an answer given now would be silently lost -- and the modal
        # would cover the provisional-settings question whose resolution
        # is what makes the answer saveable. Resolution re-asks.
        if ctx.session.settings_are_provisional():
            logger.info(
                '[LVP Main  ] Objective prompt deferred -- settings are provisional and '
                'the answer could not be kept'
            )
            return

        settings = ctx.settings
        if turret_position is not None:
            first_line = (
                f'Please confirm the objective installed at turret position {turret_position}.'
            )
        else:
            first_line = 'Please confirm the installed objective.'
        message = f'{first_line}\nThis sets the image scale recorded with every capture.'

        slots = settings.get('turret_objectives') or {}
        current = (
            slots.get(turret_position) if turret_position is not None else None
        ) or settings.get('objective_id')

        self.load_objectives()
        objectives = list(self.ids['objective_spinner2'].values)
        if not objectives:
            logger.error('[LVP Main  ] Objective catalogue empty; cannot prompt for a selection')
            return

        def _apply(chosen: str):
            self.ids['objective_spinner2'].text = chosen
            if turret_position is not None:
                self.update_all_turret_btn_states(turret_position)
                self.set_turret_objective()
            ctx.session.update_settings('objective_confirmed', True)

        from ui.notification_popup import show_objective_selection_popup

        show_objective_selection_popup(
            title='Objective',
            message=message,
            objectives=objectives,
            current_objective_id=current if current in objectives else objectives[0],
            on_confirm=_apply,
        )

    def maybe_prompt_objective_selection(self, model_has_turret: bool):
        """Fire the objective prompt if the objective is unknowable.

        Two ways the app cannot know what is in the light path: no
        person has ever confirmed the objective on this install (the
        settings template ships a 20x default that would otherwise set
        image scale silently forever), or the session's turret position
        has no assignment.
        """
        settings = _app_ctx.ctx.settings
        first_run = not settings.get('objective_confirmed', False)
        turret_position = None
        slot_unassigned = False
        if model_has_turret:
            position = settings.get('turret_position') or 1
            slots = settings.get('turret_objectives') or {}
            slot_unassigned = slots.get(position) is None
            turret_position = position
        if first_run or slot_unassigned:
            self.prompt_objective_selection(turret_position=turret_position)

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
                if not selected_position.isdigit():
                    selected_position = 1
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

            # Persist user's explicit turret choice so the next session
            # (or any post-home lookup) prefers this position when the
            # objective at this slot is duplicated elsewhere on the
            # turret. (#488)
            settings['turret_position'] = selected_position

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
                        # setting the image scale silently. Ask the user
                        # what is installed there. (Programmatic
                        # turret_select on a scope with no turret --
                        # e.g. the XY-home resync -- must not prompt.)
                        if ctx.lumaview.scope.capabilities.has_turret:
                            Clock.schedule_once(
                                lambda dt: self.prompt_objective_selection(
                                    turret_position=selected_position
                                ),
                                0,
                            )
                    else:
                        # Run validation refuses unassigned step
                        # objectives, so a protocol move cannot legally
                        # land here -- and a prompt must never interrupt
                        # an unattended run. Log loudly instead.
                        logger.warning(
                            f'[LVP Main  ] Protocol turret move landed on position '
                            f'{selected_position} with no objective assigned'
                        )

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
        settings = _app_ctx.ctx.settings
        # Persist the position the turret physically ended up at -- this
        # is called after every protocol-driven or step-navigation T
        # move, so the persisted value tracks reality across moves. (#488)
        try:
            settings['turret_position'] = int(turret_position)
        except (TypeError, ValueError):
            pass
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
