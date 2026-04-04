# Copyright Etaluma, Inc.
import logging
import pathlib

from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules.config_ui_getters import (
    get_active_layer_config,
    get_auto_gain_settings,
    get_binning_from_ui,
    get_current_frame_dimensions,
    get_current_objective_info,
    get_current_plate_position,
    get_image_capture_config_from_ui,
    get_selected_labware,
    get_stim_configs,
)
from modules import gui_logger
from modules.debounce import debounce
from modules.protocol import Protocol
from modules.sequenced_capture_executor import SequencedCaptureRunMode
from modules.sequential_io_executor import IOTask
from modules.tiling_config import TilingConfig
from ui.ui_helpers import (
    _handle_ui_update_for_axis,
    live_histo_off,
    live_histo_reverse,
    move_absolute_position,
    move_home,
    move_relative_position,
    reset_title,
    set_recording_title,
    set_writing_title,
    update_autofocus_selection_after_protocol,
)

logger = logging.getLogger('LVP.ui.vertical_control')

AF_SAFETY_TIMEOUT_S = 15  # Seconds before AF is considered stuck and force-reset


# ============================================================================
# VerticalControl — Z-Axis, Objectives, Turret, and Autofocus
# ============================================================================

class VerticalControl(BoxLayout):

    def __init__(self, **kwargs):
        super(VerticalControl, self).__init__(**kwargs)
        logger.debug('[LVP Main  ] VerticalControl.__init__()')

        # boolean describing whether the scope is currently in the process of autofocus
        self.is_autofocus = False
        self.is_complete = False
        self.record_autofocus_to_file = False
        self._next_pos = None

        self.queue_slider_position_trigger = Clock.create_trigger(lambda dt: self.queue_slider_position(), 0.1)


    def update_gui(self, vertical_control=False):
        ctx = _app_ctx.ctx
        if ctx.sequenced_capture_executor.run_in_progress():
            return
        if not vertical_control:
            ctx.io_executor.put(IOTask(
                action=ctx.lumaview.scope.get_target_position,
                args=('Z'),
                callback=self.execute_kivy_gui,
                cb_kwargs={"vertical_control":vertical_control},
                pass_result=True
            ))
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
        """IOTask callback — runs on worker thread. Must schedule widget access."""
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
        """Update Z slider and text — must be called on main thread.

        Only updates text field when user is not typing (focus check),
        matching XY behavior. Without this, the text shows current
        position during motion then snaps to target — confusing.
        """
        self.ids['obj_position'].value = max(0, pos)
        if not self.ids['z_position_id'].focus:
            new_text = format(max(0, pos), '.2f')
            if self.ids['z_position_id'].text != new_text:
                self.ids['z_position_id'].text = new_text

    def _update_z_text(self, pos):
        """Update Z text only — must be called on main thread."""
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
        ctx.io_executor.put(IOTask(
            action=move_relative_position,
            args=('Z', direction * step),
            kwargs={"overshoot_enabled": overshoot_enabled},
        ))

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
        self.queue_slider_position_trigger()

    def queue_slider_position(self):
        ctx = _app_ctx.ctx
        ctx.io_executor.put(IOTask(
            action=move_absolute_position,
            args=('Z', self._next_pos)
        ))
        self._next_pos = None

    def set_bookmark(self):
        gui_logger.button('SET_Z_BOOKMARK')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.set_bookmark()')
        ctx.io_executor.put(IOTask(action=self.ex_set_bookmark))

    def ex_set_bookmark(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        height = ctx.lumaview.scope.get_current_position('Z')  # Get current z height in um
        settings['bookmark']['z'] = height

    def set_all_bookmarks(self):
        gui_logger.button('SET_ALL_BOOKMARKS')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.set_all_bookmarks()')
        ctx.io_executor.put(IOTask(action=self.ex_set_all_bookmarks))

    def ex_set_all_bookmarks(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        height = ctx.lumaview.scope.get_current_position('Z')  # Get current z height in um
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
        settings = ctx.settings
        logger.info('[LVP Main  ] VerticalControl.goto_bookmark()')
        pos = settings['bookmark']['z']
        ctx.io_executor.put(IOTask(action=move_absolute_position, args=('Z', pos)))

    @debounce(1.0)
    def home(self):
        try:
            gui_logger.button('HOME_Z')
            ctx = _app_ctx.ctx
            if ctx.protocol_running.is_set():
                return
            logger.info('[LVP Main  ] VerticalControl.home()')
            ctx.io_executor.put(IOTask(action=move_home, kwargs={"axis":'Z'}))
        except Exception as e:
            logger.error(f'[UI] home failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup
            show_notification_popup(title="Error", message=str(e))

    def load_objective_from_settings(self):
        settings = _app_ctx.ctx.settings
        self.ids['objective_spinner2'] = settings['objective_id']

    def load_objectives(self):
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.load_objectives()')
        spinner = self.ids['objective_spinner2']
        spinner.values = ctx.objective_helper.get_objectives_list()


    def select_objective(self):
        try:
            ctx = _app_ctx.ctx
            # Only log objective changes from user interaction, not protocol
            if not ctx.protocol_running or not ctx.protocol_running.is_set():
                gui_logger.select('OBJECTIVE', self.ids['objective_spinner2'].text)
            logger.info('[LVP Main  ] VerticalControl.select_objective()')
            settings = ctx.settings

            # Update objective stored in settings
            objective_id = self.ids['objective_spinner2'].text
            objective = ctx.objective_helper.get_objective_info(objective_id=objective_id)
            settings['objective_id'] = objective_id

            # Update magnification UI info
            microscope_settings_id = ctx.motion_settings.ids['microscope_settings_id']
            microscope_settings_id.ids['magnification_id'].text = f"{objective['magnification']}"

            # Update selected to be consistent with other selector
            ms_objective_spinner = microscope_settings_id.ids['objective_spinner']
            ms_objective_spinner.text = objective_id

            # Set objective in lumascope
            if ctx.lumaview.scope.has_turret():
                ctx.lumaview.scope.set_turret_config(turret_config=settings["turret_objectives"])

            ctx.lumaview.scope.set_objective(objective_id=objective_id)

            # Update UI FOV
            fov_size = common_utils.get_field_of_view(
                focal_length=objective['focal_length'],
                frame_size=settings['frame'],
                binning_size=get_binning_from_ui(),
            )
            microscope_settings_id.ids['field_of_view_width_id'].text = str(round(fov_size['width'],0))
            microscope_settings_id.ids['field_of_view_height_id'].text = str(round(fov_size['height'],0))
        except Exception as e:
            logger.error(f'[UI] select_objective failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup
            show_notification_popup(title="Error", message=str(e))


    def _reset_run_autofocus_button(self, **kwargs):
        ctx = _app_ctx.ctx
        ctx.autofocus_thread_executor.protocol_end()
        ctx.autofocus_thread_executor.clear_protocol_pending()
        self.ids['autofocus_id'].state = 'normal'
        self.ids['autofocus_id'].text = 'Autofocus'


    def _set_run_autofocus_button(self, **kwargs):
        self.ids['autofocus_id'].state = 'down'
        self.ids['autofocus_id'].text = 'Focusing...'


    def _cleanup_at_end_of_autofocus(self):
        ctx = _app_ctx.ctx

        ctx.reset_executor.put(IOTask(
            action=ctx.sequenced_capture_executor.reset,
            callback=self._reset_run_autofocus_button
        ))

        ctx.reset_executor.put(IOTask(
            action=ctx.autofocus_executor.reset
        ))

        # Resetting autofocus_executor before sequenced_capture_executor leads to possibility
        # of sequenced_capture accidentally re-starting AF (sees it is finished, sequenced
        # iterate is running, restarts AF). Reset order matters.


    def _autofocus_run_complete(self, **kwargs):
        ctx = _app_ctx.ctx
        live_histo_reverse()
        Clock.schedule_once(lambda dt: self._reset_run_autofocus_button(), 0)

        # Update per-layer focus in settings so new protocol steps use the
        # AF result, not the stale pre-AF Z value.
        try:
            focus_z = ctx.scope.get_current_position('Z')
            for layer in common_utils.get_layers():
                accordion_item = ctx.image_settings.accordion_item_lookup(layer=layer)
                if not accordion_item.collapse:
                    with ctx.settings_lock:
                        ctx.settings[layer]['focus'] = focus_z
                    logger.info(f'[AF] Updated {layer} focus to {focus_z:.2f}um')
                    break
        except Exception as e:
            logger.warning(f'[AF] Failed to update layer focus after AF: {e}')

        # Clear any stuck AF protocol queue entries after completion
        try:
            ctx.autofocus_thread_executor.protocol_end()
            ctx.autofocus_thread_executor.clear_protocol_pending()
        except Exception:
            pass


    def run_autofocus_from_ui(self):
        gui_logger.button('AUTOFOCUS')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] VerticalControl.run_autofocus_from_ui()')
        settings = ctx.settings

        if ctx.engineering_mode:
            save_autofocus_data = True
            parent_dir = pathlib.Path(settings['live_folder']).resolve() / "Autofocus Characterization"
        else:
            save_autofocus_data = False
            parent_dir = None

        live_histo_off()

        trigger_source = 'autofocus'
        run_complete_func = self._autofocus_run_complete
        run_not_started_func = self._reset_run_autofocus_button

        run_trigger_source = ctx.sequenced_capture_executor.run_trigger_source()
        if ctx.sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            logger.warning(f"Cannot start autofocus. Run already in progress from {run_trigger_source}")
            return

        if ctx.autofocus_executor.run_in_progress() or ctx.sequenced_capture_executor.run_in_progress():
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
                if ctx.sequenced_capture_executor.run_trigger_source() == 'autofocus' and ctx.sequenced_capture_executor.run_in_progress():
                    # If AF is still stuck after timeout, attempt a protocol reset and revert UI
                    ctx.reset_executor.put(IOTask(
                        action=ctx.sequenced_capture_executor.reset,
                        callback=self._reset_run_autofocus_button
                    ))
                    logger.warning('[AF Safety] Autofocus appeared stuck. Forced reset.')
            except Exception:
                pass
        self._af_safety_event = Clock.schedule_once(_af_safety, AF_SAFETY_TIMEOUT_S)

        objective_id, _ = get_current_objective_info()
        labware_id, _ = get_selected_labware()
        active_layer, active_layer_config = get_active_layer_config()
        active_layer_config['autofocus'] = True
        active_layer_config['acquire'] = "image"
        ctx.io_executor.put(IOTask(
            action=get_current_plate_position,
            callback=self.intermediary_autofocus,
            cb_args=(
                labware_id,
                objective_id,
                active_layer,
                active_layer_config,
                run_complete_func,
                trigger_source,
                parent_dir,
                save_autofocus_data
            ),
            pass_result=True
        ))

    def intermediary_autofocus(self, labware_id,
                objective_id,
                active_layer,
                active_layer_config,
                run_complete_func,
                trigger_source,
                parent_dir,
                save_autofocus_data,
                result=None,
                exception=None):

        if exception is not None:
            raise exception

        if result is None:
            return

        curr_position = result

        ctx = _app_ctx.ctx
        ctx.io_executor.put(IOTask(
            action=self.curr_position_autofocus,
            args= (
                curr_position,
                labware_id,
                objective_id,
                active_layer,
                active_layer_config,
                run_complete_func,
                trigger_source,
                parent_dir,
                save_autofocus_data
            )
        ))


    def curr_position_autofocus(self,
                                curr_position,
                                labware_id,
                                objective_id,
                                active_layer,
                                active_layer_config,
                                run_complete_func,
                                trigger_source,
                                parent_dir,
                                save_autofocus_data,
                                result=None, exception=None):

        ctx = _app_ctx.ctx
        settings = ctx.settings

        curr_position.update({'name': 'AF'})

        positions = [
            curr_position,
        ]

        tiling_config = TilingConfig(
            tiling_configs_file_loc=pathlib.Path(ctx.source_path) / "data" / "tiling.json",
        )

        config = {
            'labware_id': labware_id,
            'positions': positions,
            'objective_id': objective_id,
            'zstack_params': {'range': 0, 'step_size': 0},
            'use_zstacking': False,
            'tiling': tiling_config.no_tiling_label(),
            'layer_configs': {active_layer: active_layer_config},
            'period': None,
            'duration': None,
            'frame_dimensions': get_current_frame_dimensions(),
            'binning_size': get_binning_from_ui(),
            'stim_config': get_stim_configs(),
        }

        autofocus_sequence = Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=pathlib.Path(ctx.source_path) / "data" / "tiling.json",
        )

        autogain_settings = get_auto_gain_settings()

        callbacks = {
            'move_position': _handle_ui_update_for_axis,
            'update_scope_display': ctx.scope_display.update_scopedisplay,
            'scan_iterate_post': run_complete_func,
            'run_complete': run_complete_func,
            # LED observer handles UI sync — no manual callbacks needed
            'reset_autofocus_btns': update_autofocus_selection_after_protocol,
            'set_recording_title': set_recording_title,
            'set_writing_title': set_writing_title,
            'reset_title': reset_title,
            'autofocus_completed': self._cleanup_at_end_of_autofocus,
        }

        ctx.protocol_executor.put(IOTask(
            action=ctx.sequenced_capture_executor.run,
            kwargs={
                "protocol":autofocus_sequence,
                "run_mode":SequencedCaptureRunMode.SINGLE_AUTOFOCUS,
                "run_trigger_source":trigger_source,
                "max_scans":1,
                "sequence_name":'af',
                "parent_dir":parent_dir,
                "image_capture_config":get_image_capture_config_from_ui(),
                "enable_image_saving":False,
                "disable_saving_artifacts":True,
                "separate_folder_per_channel":False,
                "autogain_settings":autogain_settings,
                "callbacks":callbacks,
                "return_to_position":None,
                "save_autofocus_data":save_autofocus_data,
                "leds_state_at_end":"return_to_original",
                "video_as_frames":settings['video_as_frames']
            }
        ))


    @debounce(1.0)
    def turret_home(self):
        gui_logger.button('HOME_TURRET')
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        def _on_turret_homed():
            Clock.schedule_once(lambda dt: self._reset_turret_buttons(), 0)

        ctx.io_executor.put(IOTask(
            action=ctx.lumaview.scope.thome,
            callback=_on_turret_homed,
        ))

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
        for position in range(1,5):
            if self.ids[f"turret_pos_{position}_btn"].state == 'down':
                selected_turret = position

        if selected_turret is None:
            logger.error("VerticalControl] SetTurretObjective] No turret button selected")
            return

        try:
            selected_turret_id = self.ids[f"turret_pos_{selected_turret}_btn"]

            # Find magnification of the selected objective
            desired_objective_id = self.ids['objective_spinner2'].text
            magnification = ctx.objective_helper.get_objective_info(objective_id=desired_objective_id)['magnification']

            # Change turret text
            selected_turret_id.text = f"{magnification}x"

            # Update settings
            settings["turret_objectives"][selected_turret] = desired_objective_id

        except Exception as e:
            logger.exception(f"SetTurretObjective] Error: {e}")
            return

    def reset_turret_objective(self):
        settings = _app_ctx.ctx.settings

        selected_turret = None
        for position in range(1,5):
            if self.ids[f"turret_pos_{position}_btn"].state == 'down':
                selected_turret = position

        if selected_turret is None:
            logger.error("VerticalControl] ResetTurretObjective] No turret button selected")
            return

        try:
            selected_turret_id = self.ids[f"turret_pos_{selected_turret}_btn"]

            # Change turret text
            selected_turret_id.text = str(selected_turret)

            # Update settings
            settings["turret_objectives"][selected_turret] = None

        except Exception as e:
            logger.exception(f"ResetTurretObjective] Error: {e}")
            return


    @debounce(0.5)
    def turret_select(self, selected_position, protocol=False):
        try:
            if not protocol:
                gui_logger.button(f'TURRET_POS_{selected_position}')
            ctx = _app_ctx.ctx
            settings = ctx.settings
            if not ctx.lumaview.scope.has_thomed():
                if not protocol:
                    ctx.io_executor.put(IOTask(ctx.lumaview.scope.thome))
                else:
                    ctx.lumaview.scope.thome()

            if not isinstance(selected_position, int) and not isinstance(selected_position, float):
                if not selected_position.isdigit():
                    selected_position = 1
            else:
                selected_position = int(selected_position)

            if not protocol:
                ctx.io_executor.put(IOTask(ctx.lumaview.scope.tmove, kwargs={'position':selected_position}))
            else:
                ctx.lumaview.scope.tmove(position=selected_position)

            for available_position in range(1,5):
                if selected_position == available_position:
                    state = 'down'

                    # Check if an objective has been saved to that turret
                    turret_position_objective = settings["turret_objectives"][selected_position]
                    if turret_position_objective is not None:
                        # If an objective has been assigned to the turret position, change to that objective
                        Clock.schedule_once(lambda dt: self.update_spinner_text(selected_position), 0)
                        Clock.schedule_once(lambda dt: self.select_objective(), 0)

                else:
                    state = 'normal'

            Clock.schedule_once(lambda dt: self.update_all_turret_btn_states(selected_position), 0)
        except Exception as e:
            logger.error(f'[UI] turret_select failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup
            show_notification_popup(title="Error", message=str(e))

    def update_spinner_text(self, selected_position):
        settings = _app_ctx.ctx.settings
        self.ids["objective_spinner2"].text = settings["turret_objectives"][selected_position]

    def update_turret_btn_state(self, position, state):
        self.ids[f'turret_pos_{position}_btn'].state = state

    def update_all_turret_btn_states(self, selected_position):
        for available_position in range(1,5):
            if selected_position == available_position:
                state = 'down'
            else:
                state = 'normal'
            self.update_turret_btn_state(available_position, state)

    def update_turret_gui(self, turret_position):
        settings = _app_ctx.ctx.settings
        for available_position in range(1,5):
            if turret_position == available_position:
                state = 'down'

                # Check if an objective has been saved to that turret
                turret_position_objective = settings["turret_objectives"][turret_position]
                if turret_position_objective is not None:
                    # If an objective has been assigned to the turret position, change to that objective
                    self.ids["objective_spinner2"].text = settings["turret_objectives"][turret_position]
                    self.select_objective()

            else:
                state = 'normal'

            self.ids[f'turret_pos_{available_position}_btn'].state = state
