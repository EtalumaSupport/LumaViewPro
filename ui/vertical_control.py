# Copyright Etaluma, Inc.
import logging
import pathlib

from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules.config_ui_getters import (
    get_active_layer_config,
    get_binning_from_ui,
    get_current_objective_info,
)
from modules import gui_logger
from modules.debounce import debounce
from modules.kivy_utils import schedule_ui as _schedule_ui
from modules.sequential_io_executor import IOTask, PRIORITY_HIGH
from ui.protocol_settings import require_file_writes_idle
from ui.ui_helpers import (
    _handle_ui_update_for_axis,
    live_histo_off,
    live_histo_reverse,
    move_absolute,
    move_home,
    move_relative,
    publish_protocol_running,
)

logger = logging.getLogger('LVP.ui.vertical_control')

AF_SAFETY_TIMEOUT_S = 15  # Seconds before AF is considered stuck and force-reset


class _AfLockoutState:
    """Main-thread-only ownership record for the standalone-AF lockout.

    A standalone AF engages the same guard set a protocol run does (the
    protocol_running Event, its kv mirror, the stage motion lock). The
    generation counter makes every release single-shot and stale-proof:
    an AF run's exit paths (completion callback, abort cleanup, safety
    timer) can each fire, in any order, possibly AFTER a newer AF has
    acquired -- only the release carrying the CURRENT generation acts.
    The snapshot restores prior state rather than clearing, so a rival
    holder's Event stays held; it reads the Event, never the
    Clock-deferred mirror property (one frame stale under a same-frame
    click).
    """

    __slots__ = ('generation', 'snapshot')

    def __init__(self):
        self.generation = 0
        self.snapshot = None


_af_lockout = _AfLockoutState()


def _acquire_af_lockout() -> int:
    ctx = _app_ctx.ctx
    _af_lockout.generation += 1
    _af_lockout.snapshot = (
        ctx.protocol_running.is_set(),
        ctx.stage.motion_capability(),
    )
    ctx.protocol_running.set()
    publish_protocol_running(True)
    ctx.stage.set_motion_capability(False)
    return _af_lockout.generation


def _release_af_lockout(generation: int) -> None:
    if generation != _af_lockout.generation or _af_lockout.snapshot is None:
        return
    event_was_set, motion_was_enabled = _af_lockout.snapshot
    _af_lockout.snapshot = None
    ctx = _app_ctx.ctx
    if event_was_set:
        ctx.protocol_running.set()
    else:
        ctx.protocol_running.clear()
    publish_protocol_running(event_was_set)
    ctx.stage.set_motion_capability(motion_was_enabled)


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
        # Generation of the standalone-AF lockout this widget last
        # acquired; 0 = never acquired (release(0) no-ops on the empty
        # snapshot).
        self._af_lockout_gen = 0
        self.record_autofocus_to_file = False
        self._next_pos = None

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
        if ctx.protocol_running.is_set():
            return
        label = f'Z_{"COARSE" if coarse else "FINE"}_{"UP" if direction > 0 else "DOWN"}'
        gui_logger.button(label)
        logger.info(f'[LVP Main  ] VerticalControl._z_jog({label})')
        try:
            _, objective = get_current_objective_info()
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
        if ctx.protocol_running.is_set():
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
        if ctx.protocol_running.is_set():
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
            if ctx.protocol_running.is_set():
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
            if not ctx.protocol_running or not ctx.protocol_running.is_set():
                gui_logger.select('OBJECTIVE', objective_id)
            logger.info('[LVP Main  ] VerticalControl.select_objective()')

            # With a turret, the objective must be assigned to a turret position
            # before it can be selected; warn (but still allow) if it is not.
            if ctx.lumaview.scope.capabilities.has_turret:
                turret_objectives = list(settings.get('turret_objectives', {}).values())
                assigned = [obj for obj in turret_objectives if obj is not None]
                if assigned and objective_id not in assigned:
                    from modules.notification_center import notifications

                    notifications.warning(
                        'Objective',
                        'Objective Not in Turret',
                        f"[Objective] Cannot select '{objective_id}' -- not assigned "
                        f'to any turret position. Assign it in Objective Control > '
                        f'Turret before using.',
                    )

            # Update objective stored in settings
            objective = ctx.objective_helper.get_objective_info(objective_id=objective_id)
            with ctx.settings_lock:
                settings['objective_id'] = objective_id

            # Set objective in lumascope
            if ctx.lumaview.scope.capabilities.has_turret:
                ctx.lumaview.scope.runtime_state.set_turret_config(
                    turret_config=settings['turret_objectives']
                )

            ctx.lumaview.scope.runtime_state.set_objective(objective_id=objective_id)

            binning_size = get_binning_from_ui()

            # The optics that set image scale change here and nowhere else in
            # a session; recording them now is what lets a returned bundle
            # explain the scale baked into its own images.
            common_utils.log_resolved_optics(
                objective_id=objective_id,
                focal_length=objective['focal_length'],
                binning_size=binning_size,
            )

            # Update UI FOV
            microscope_settings_id = ctx.motion_settings.ids['microscope_settings_id']
            fov_size = common_utils.get_field_of_view(
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

        # The abort exit releases the standalone lockout it owns; the
        # completion callback's release then no-ops (single-shot).
        _release_af_lockout(self._af_lockout_gen)

        # SequencedCaptureRunner.reset() unwinds any running protocol
        # (its _cleanup chain calls autofocus_thread.abort() on the AF
        # thread). AFE state is reset implicitly on the next AFE.run().
        ctx.worker_pool.put(
            IOTask(
                action=ctx.sequenced_capture_runner.reset,
                callback=self._reset_run_autofocus_button,
                priority=PRIORITY_HIGH,
            )
        )

    def _autofocus_run_complete(self, **kwargs):
        ctx = _app_ctx.ctx
        live_histo_reverse()
        Clock.schedule_once(lambda dt: self._reset_run_autofocus_button(), 0)

        # Update per-layer focus in settings so new protocol steps use the
        # AF result, not the stale pre-AF Z value.
        try:
            focus_z = ctx.scope.motion.get_current_position('Z')
            for layer in common_utils.get_layers():
                accordion_item = ctx.image_settings.accordion_item_lookup(layer=layer)
                if not accordion_item.collapse:
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
                    break
        except Exception as e:
            logger.warning(f'[AF] Failed to update layer focus after AF: {e}')

        # Defensive abort -- if the AF thread is somehow still in flight
        # at the completion path, this is a no-op; if not, it unwinds.
        try:
            if ctx.autofocus_thread is not None:
                ctx.autofocus_thread.abort()
        except Exception:
            pass

    def _build_standalone_af_args(
        self,
        active_layer: str,
        active_layer_config: dict,
        save_autofocus_data: bool,
        parent_dir,
    ) -> dict:
        """Build the kwargs for AutofocusThread.run_autofocus() from the
        currently-active layer and UI state.

        Mirrors the per-step kwarg build in protocol_step_runner so a
        standalone AF run uses the same AFE entry as a protocol step.
        """
        objective_id, _ = get_current_objective_info()
        return {
            'objective_id': objective_id,
            'save_results_to_file': save_autofocus_data,
            'results_dir': parent_dir,
            'run_trigger_source': 'autofocus',
            'led_color': active_layer,
            'led_illumination': float(active_layer_config.get('illumination_ma', 0)),
            'camera_gain': float(active_layer_config.get('gain_db', 0)),
            'camera_exposure': float(active_layer_config.get('exposure_ms', 1)),
            'callbacks': {'move_position': _handle_ui_update_for_axis},
        }

    def run_autofocus_from_ui(self):
        gui_logger.button('AUTOFOCUS')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.run_autofocus_from_ui()')
        settings = ctx.settings

        if ctx.engineering_mode:
            save_autofocus_data = True
            parent_dir = (
                pathlib.Path(settings['live_folder']).resolve() / 'Autofocus Characterization'
            )
        else:
            save_autofocus_data = False
            parent_dir = None

        live_histo_off()

        # Block standalone AF if a protocol is running OR if AF is
        # already in flight. Refusals undo cosmetics ONLY: a full reset
        # here would abort a protocol's own AF step in the one-frame
        # window before the kv mirror catches up, and the lockout is a
        # rival run's to keep.
        if ctx.sequenced_capture_runner.run_in_progress():
            self._reset_run_autofocus_button_cosmetics()
            logger.warning(
                'Cannot start autofocus: protocol run already in progress '
                f'(trigger={ctx.sequenced_capture_runner.run_trigger_source()})'
            )
            return

        # The post-run file drain deliberately holds the lockout while
        # run_in_progress() is already False -- AF must not run under it
        # and, worse, must not RELEASE it mid-drain.
        if not require_file_writes_idle('start autofocus'):
            self._reset_run_autofocus_button_cosmetics()
            return

        if ctx.autofocus_thread is not None and ctx.autofocus_thread.is_running:
            self._cleanup_at_end_of_autofocus()
            return

        if self.ids['autofocus_id'].state == 'normal':
            self._cleanup_at_end_of_autofocus()
            return

        self._set_run_autofocus_button()

        # Safety timer to revert AF UI if AF doesn't progress within a timeout
        try:
            if hasattr(self, '_af_safety_event') and self._af_safety_event is not None:
                Clock.unschedule(self._af_safety_event)
        except Exception:
            pass

        def _af_safety(dt):
            try:
                if ctx.autofocus_thread is not None and ctx.autofocus_thread.is_running:
                    ctx.autofocus_thread.abort()
                    _schedule_ui(lambda _dt: self._reset_run_autofocus_button(), 0)
                    logger.warning('[AF Safety] Autofocus appeared stuck. Forced abort.')
            except Exception:
                pass

        self._af_safety_event = Clock.schedule_once(_af_safety, AF_SAFETY_TIMEOUT_S)

        active_layer, active_layer_config = get_active_layer_config()
        args = self._build_standalone_af_args(
            active_layer=active_layer,
            active_layer_config=active_layer_config,
            save_autofocus_data=save_autofocus_data,
            parent_dir=parent_dir,
        )

        # The standalone scan engages the full protocol guard set (Event,
        # kv mirror, stage motion) for its duration -- turret, jog, record
        # and dialogs all lock exactly as during a run. Acquired only after
        # every refusal gate above; released single-shot by whichever exit
        # fires first (completion callback, abort cleanup, safety timer),
        # generation-checked so a stale exit cannot unlock a newer AF.
        lockout_gen = self._af_lockout_gen = _acquire_af_lockout()
        try:
            future = ctx.autofocus_thread.run_autofocus(**args)
            # Cleanup UI state on completion (success, abort, or failure).
            # _autofocus_run_complete handles per-layer focus persistence
            # and the AF safety timer reset; it tolerates either outcome.
            future.add_done_callback(
                lambda _f: _schedule_ui(
                    lambda _dt: (
                        self._autofocus_run_complete(),
                        _release_af_lockout(lockout_gen),
                    ),
                    0,
                )
            )
        except Exception:
            _release_af_lockout(lockout_gen)
            raise

    @debounce(1.0)
    def turret_home(self):
        gui_logger.button('HOME_TURRET')
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return

        def _on_turret_homed():
            Clock.schedule_once(lambda dt: self._reset_turret_buttons(), 0)

        ctx.io_executor.put(
            IOTask(
                action=ctx.lumaview.scope.motion._thome_impl,
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
            magnification = ctx.objective_helper.get_objective_info(
                objective_id=desired_objective_id
            )['magnification']

            # Change turret text
            selected_turret_id.text = f'{magnification}x'

            # Update settings
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

            # Update settings
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

    @debounce(0.5)
    def turret_select(self, selected_position, protocol=False, restore_z=True):
        try:
            if not protocol:
                gui_logger.button(f'TURRET_POS_{selected_position}')
            ctx = _app_ctx.ctx
            settings = ctx.settings
            if not ctx.lumaview.scope.motion.has_thomed():
                if not protocol:
                    ctx.io_executor.put(IOTask(ctx.lumaview.scope.motion._thome_impl))
                else:
                    # Protocol context runs on protocol_thread, not the io
                    # worker -- route the turret home through the protocol queue so it
                    # stays ordered ahead of the subsequent tmove/X/Y/Z and
                    # behind the prior step's leds_off on the single worker.
                    fut = ctx.io_executor.protocol_put(
                        IOTask(ctx.lumaview.scope.motion._thome_impl), return_future=True
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
                        ctx.lumaview.scope.motion._tmove_impl,
                        kwargs={'position': selected_position},
                    )
                )
            else:
                # See the turret-home branch above: route the protocol-context
                # tmove through the protocol queue so it serializes with the
                # step's other moves and LED ops on the single io worker
                # instead of racing them from protocol_thread.
                fut = ctx.io_executor.protocol_put(
                    IOTask(
                        ctx.lumaview.scope.motion._tmove_impl,
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
