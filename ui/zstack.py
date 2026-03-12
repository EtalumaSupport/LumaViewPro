# Copyright Etaluma, Inc.
import logging
import pathlib

from kivy.clock import Clock

from kivy.uix.floatlayout import FloatLayout

import modules.common_utils as common_utils
import modules.app_context as _app_ctx
from modules.config_ui_getters import (
    create_hyperstacks_if_needed,
    get_active_layer_config,
    get_auto_gain_settings,
    get_binning_from_ui,
    get_current_frame_dimensions,
    get_current_objective_info,
    get_current_plate_position,
    get_image_capture_config_from_ui,
    get_selected_labware,
    get_sequenced_capture_config_from_ui,
    get_stim_configs,
    get_zstack_params,
    get_zstack_positions,
    is_image_saving_enabled,
)
from modules.protocol import Protocol
from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
from modules.tiling_config import TilingConfig
from modules.ui_helpers import (
    _handle_ui_for_led,
    _handle_ui_for_leds_off,
    _handle_ui_update_for_axis,
    live_histo_off,
    live_histo_reverse,
    reset_title,
    set_last_save_folder,
    set_recording_title,
    set_writing_title,
    update_autofocus_selection_after_protocol,
)
from modules.zstack_config import ZStackConfig

logger = logging.getLogger('LVP.ui.zstack')


class ZStack(FloatLayout):
    def set_steps(self):
        logger.info('[LVP Main  ] ZStack.set_steps()')
        settings = _app_ctx.ctx.settings

        try:
            step_size = float(self.ids['zstack_stepsize_id'].text)
            if step_size < 0:
                step_size = 0
                self.ids['zstack_stepsize_id'].text = str(step_size)
        except Exception:
            step_size = 0
            self.ids['zstack_stepsize_id'].text = str(step_size)
        finally:
            settings['zstack']['step_size'] = step_size

        try:
            step_range = float(self.ids['zstack_range_id'].text)
            if step_range < 0:
                step_range = 0
                self.ids['zstack_range_id'].text = str(step_range)
        except Exception:
            step_range = 0
            self.ids['zstack_range_id'].text = str(step_range)
        finally:
            settings['zstack']['range'] = step_range

        z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
            text_label=self.ids['zstack_spinner'].text
        )

        zstack_config = ZStackConfig(
            range=settings['zstack']['range'],
            step_size=settings['zstack']['step_size'],
            current_z_reference=z_reference,
            current_z_value=None
        )

        self.ids['zstack_steps_id'].text = str(zstack_config.number_of_steps())


    def set_position(self):
        settings = _app_ctx.ctx.settings
        settings['zstack']['position'] = self.ids['zstack_spinner'].text


    def _reset_run_zstack_acquire_button(self, **kwargs):
        self.ids['zstack_aqr_btn'].state = 'normal'
        self.ids['zstack_aqr_btn'].text = 'Acquire'
        live_histo_reverse()


    def _cleanup_at_end_of_acquire(self):
        ctx = _app_ctx.ctx
        ctx.sequenced_capture_executor.reset()
        self._reset_run_zstack_acquire_button()
        live_histo_reverse()


    def _zstack_run_complete(self, **kwargs):
        self._reset_run_zstack_acquire_button()
        create_hyperstacks_if_needed()
        live_histo_reverse()


    def run_zstack_acquire_from_ui(self):
        logger.info('[LVP Main  ] ZStack.run_zstack_acquire_from_ui()')
        ctx = _app_ctx.ctx

        live_histo_off()

        settings = ctx.settings

        trigger_source = 'zstack'
        run_not_started_func = self._reset_run_zstack_acquire_button
        run_complete_func = self._zstack_run_complete

        run_trigger_source = ctx.sequenced_capture_executor.run_trigger_source()
        if ctx.sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            logger.warning(f"Cannot start Z-Stack acquire. Run already in progress from {run_trigger_source}")
            return

        if self.ids['zstack_aqr_btn'].state == 'normal':
            self._cleanup_at_end_of_acquire()
            return

        # Note: This will be quickly overwritten by the remaining number of scans
        self.ids['zstack_aqr_btn'].text = 'Running Z-Stack'

        config = get_sequenced_capture_config_from_ui()

        labware_id, _ = get_selected_labware()
        objective_id, _ = get_current_objective_info()
        zstack_positions_valid, _ = get_zstack_positions()
        zstack_params = get_zstack_params()
        active_layer, active_layer_config = get_active_layer_config()
        active_layer_config['acquire'] = "image"

        if not zstack_positions_valid:
            logger.info('[LVP Main  ] ZStack.acquire_zstack() -> No Z-Stack positions configured')
            run_not_started_func()
            return

        curr_position = get_current_plate_position()
        curr_position.update({'name': 'ZStack'})

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
            'zstack_params': zstack_params,
            'use_zstacking': True,
            'tiling': tiling_config.no_tiling_label(),
            'layer_configs': {active_layer: active_layer_config},
            'period': None,
            'duration': None,
            'frame_dimensions': get_current_frame_dimensions(),
            'binning_size': get_binning_from_ui(),
            'stim_config': get_stim_configs(),
        }

        zstack_sequence = Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=pathlib.Path(ctx.source_path) / "data" / "tiling.json"
        )

        autogain_settings = get_auto_gain_settings()

        callbacks = {
            'move_position': _handle_ui_update_for_axis,
            'update_scope_display': ctx.scope_display.update_scopedisplay,
            'run_complete': run_complete_func,
            'leds_off': _handle_ui_for_leds_off,
            'led_state': _handle_ui_for_led,
            'reset_autofocus_btns': update_autofocus_selection_after_protocol,
            'set_recording_title': set_recording_title,
            'set_writing_title': set_writing_title,
            'reset_title': reset_title,
            'pause_live_ui': lambda: (
                Clock.unschedule(ctx.scope_display.update_scopedisplay),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui)
            ),
            'resume_live_ui': lambda: (
                ctx.scope_display.start(),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
                Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1)
            ),
        }

        parent_dir = pathlib.Path(settings['live_folder']).resolve() / "Manual" / "Z-Stacks"

        initial_position = get_current_plate_position()
        image_capture_config = get_image_capture_config_from_ui()

        ctx.sequenced_capture_executor.run(
            protocol=zstack_sequence,
            run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
            run_trigger_source=trigger_source,
            max_scans=1,
            sequence_name='zstack',
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=is_image_saving_enabled(),
            separate_folder_per_channel=False,
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            return_to_position=initial_position,
            leds_state_at_end="off",
            video_as_frames = settings['video_as_frames']
        )

        set_last_save_folder(dir=ctx.sequenced_capture_executor.run_dir())
