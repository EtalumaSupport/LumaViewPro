# Copyright Etaluma, Inc.
import logging
import pathlib

from kivy.clock import Clock

from lumaviewpro import CompositeCapture

import modules.common_utils as common_utils
import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.zstack')


class ZStack(CompositeCapture):
    def set_steps(self):
        logger.info('[LVP Main  ] ZStack.set_steps()')
        import lumaviewpro
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

        zstack_config = lumaviewpro.ZStackConfig(
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
        import lumaviewpro
        self.ids['zstack_aqr_btn'].state = 'normal'
        self.ids['zstack_aqr_btn'].text = 'Acquire'
        lumaviewpro.live_histo_reverse()


    def _cleanup_at_end_of_acquire(self):
        import lumaviewpro
        lumaviewpro.sequenced_capture_executor.reset()
        self._reset_run_zstack_acquire_button()
        lumaviewpro.live_histo_reverse()


    def _zstack_run_complete(self, **kwargs):
        import lumaviewpro
        self._reset_run_zstack_acquire_button()
        lumaviewpro.create_hyperstacks_if_needed()
        lumaviewpro.live_histo_reverse()


    def run_zstack_acquire_from_ui(self):
        logger.info('[LVP Main  ] ZStack.run_zstack_acquire_from_ui()')
        import lumaviewpro

        lumaviewpro.live_histo_off()

        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx

        trigger_source = 'zstack'
        run_not_started_func = self._reset_run_zstack_acquire_button
        run_complete_func = self._zstack_run_complete

        run_trigger_source = lumaviewpro.sequenced_capture_executor.run_trigger_source()
        if lumaviewpro.sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            logger.warning(f"Cannot start Z-Stack acquire. Run already in progress from {run_trigger_source}")
            return

        if self.ids['zstack_aqr_btn'].state == 'normal':
            self._cleanup_at_end_of_acquire()
            return

        # Note: This will be quickly overwritten by the remaining number of scans
        self.ids['zstack_aqr_btn'].text = 'Running Z-Stack'

        config = lumaviewpro.get_sequenced_capture_config_from_ui()

        labware_id, _ = lumaviewpro.get_selected_labware()
        objective_id, _ = lumaviewpro.get_current_objective_info()
        zstack_positions_valid, _ = lumaviewpro.get_zstack_positions()
        zstack_params = lumaviewpro.get_zstack_params()
        active_layer, active_layer_config = lumaviewpro.get_active_layer_config()
        active_layer_config['acquire'] = "image"

        if not zstack_positions_valid:
            logger.info('[LVP Main  ] ZStack.acquire_zstack() -> No Z-Stack positions configured')
            run_not_started_func()
            return

        curr_position = lumaviewpro.get_current_plate_position()
        curr_position.update({'name': 'ZStack'})

        positions = [
            curr_position,
        ]

        tiling_config = lumaviewpro.TilingConfig(
            tiling_configs_file_loc=pathlib.Path(lumaviewpro.source_path) / "data" / "tiling.json",
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
            'frame_dimensions': lumaviewpro.get_current_frame_dimensions(),
            'binning_size': lumaviewpro.get_binning_from_ui(),
            'stim_config': lumaviewpro.get_stim_configs(),
        }

        zstack_sequence = lumaviewpro.Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=pathlib.Path(lumaviewpro.source_path) / "data" / "tiling.json"
        )

        autogain_settings = lumaviewpro.get_auto_gain_settings()

        callbacks = {
            'move_position': lumaviewpro._handle_ui_update_for_axis,
            'update_scope_display': ctx.scope_display.update_scopedisplay,
            'run_complete': run_complete_func,
            'leds_off': lumaviewpro._handle_ui_for_leds_off,
            'led_state': lumaviewpro._handle_ui_for_led,
            'reset_autofocus_btns': lumaviewpro.update_autofocus_selection_after_protocol,
            'set_recording_title': lumaviewpro.set_recording_title,
            'set_writing_title': lumaviewpro.set_writing_title,
            'reset_title': lumaviewpro.reset_title,
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

        initial_position = lumaviewpro.get_current_plate_position()
        image_capture_config = lumaviewpro.get_image_capture_config_from_ui()

        lumaviewpro.sequenced_capture_executor.run(
            protocol=zstack_sequence,
            run_mode=lumaviewpro.SequencedCaptureRunMode.SINGLE_ZSTACK,
            run_trigger_source=trigger_source,
            max_scans=1,
            sequence_name='zstack',
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=lumaviewpro.is_image_saving_enabled(),
            separate_folder_per_channel=False,
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            return_to_position=initial_position,
            leds_state_at_end="off",
            video_as_frames = settings['video_as_frames']
        )

        lumaviewpro.set_last_save_folder(dir=lumaviewpro.sequenced_capture_executor.run_dir())
