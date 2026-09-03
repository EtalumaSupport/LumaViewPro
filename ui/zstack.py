# Copyright Etaluma, Inc.
import logging
import pathlib

from kivy.clock import Clock

from kivy.uix.floatlayout import FloatLayout

import modules.common_utils as common_utils
import modules.app_context as _app_ctx
import modules.config_helpers as config_helpers
from modules import gui_logger
from modules.config_ui_getters import (
    get_active_layer_config,
    get_auto_gain_settings,
    get_binning_from_ui,
    get_current_frame_dimensions,
    get_image_capture_config_from_ui,
    get_selected_labware,
    get_stim_configs,
    get_zstack_params,
    get_zstack_positions,
    is_image_saving_enabled,
)
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.tiling_config import TilingConfig
from ui.ui_helpers import (
    _handle_ui_update_for_axis,
    live_display_callbacks,
    live_histo_off,
    live_histo_reverse,
    reset_title,
    run_with_refusal_boundary,
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
            with _app_ctx.ctx.settings_lock:
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
            with _app_ctx.ctx.settings_lock:
                settings['zstack']['range'] = step_range

        z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
            text_label=self.ids['zstack_spinner'].text
        )

        zstack_config = ZStackConfig(
            range=settings['zstack']['range'],
            step_size=settings['zstack']['step_size'],
            current_z_reference=z_reference,
            current_z_value=None,
        )

        self.ids['zstack_steps_id'].text = str(zstack_config.number_of_steps())

    def set_position(self):
        ctx = _app_ctx.ctx
        with ctx.settings_lock:
            ctx.settings['zstack']['position'] = self.ids['zstack_spinner'].text

    def _reset_run_zstack_acquire_button(self, **kwargs):
        self.ids['zstack_aqr_btn'].state = 'normal'
        self.ids['zstack_aqr_btn'].text = 'Acquire'
        live_histo_reverse()

    def _cleanup_at_end_of_acquire(self):
        ctx = _app_ctx.ctx
        runner = ctx.sequenced_capture_runner
        # On an abort, reset() returns immediately and the hardware
        # teardown runs on the protocol thread; _zstack_run_complete
        # (fired by cleanup) resets the button when it ends. Restoring
        # the button here on the abort flavor would invite a new acquire
        # while the old one is still tearing down (the start guard
        # refuses it, but the label would lie about readiness).
        deferred_to_cleanup = runner.run_in_progress()
        runner.reset()
        if deferred_to_cleanup:
            self.ids['zstack_aqr_btn'].text = 'Stopping...'
            return
        self._reset_run_zstack_acquire_button()
        live_histo_reverse()

    def _zstack_run_complete(self, **kwargs):
        self._reset_run_zstack_acquire_button()
        live_histo_reverse()

    def run_zstack_acquire_from_ui(self):
        try:
            gui_logger.button('ZSTACK')
            logger.info('[LVP Main  ] ZStack.run_zstack_acquire_from_ui()')
            ctx = _app_ctx.ctx

            live_histo_off()

            settings = ctx.settings

            trigger_source = 'zstack'
            run_not_started_func = self._reset_run_zstack_acquire_button
            run_complete_func = self._zstack_run_complete

            run_trigger_source = ctx.sequenced_capture_runner.run_trigger_source()
            if ctx.sequenced_capture_runner.run_in_progress() and (
                run_trigger_source != trigger_source
            ):
                run_not_started_func()
                logger.warning(
                    f'Cannot start Z-Stack acquire. Run already in progress from {run_trigger_source}'
                )
                return

            if self.ids['zstack_aqr_btn'].state == 'normal':
                self._cleanup_at_end_of_acquire()
                return

            # Immediate text while the first slice is being prepared.
            # _zstack_progress (below) overwrites this with "Z {n}/{total}"
            # as soon as the protocol_step_runner starts the first slice.
            self.ids['zstack_aqr_btn'].text = 'Running Z-Stack'

            labware_id, _ = get_selected_labware()
            objective_id, _ = ctx.session.get_current_objective_info()
            zstack_positions_valid, _ = get_zstack_positions()
            zstack_params = get_zstack_params()
            active_layer, active_layer_config = get_active_layer_config()
            active_layer_config['acquire'] = 'image'
            # Z-stack manages Z positions explicitly -- AF would override them
            active_layer_config['autofocus'] = False

            if not zstack_positions_valid:
                _range = zstack_params.get('range', 0)
                _step = zstack_params.get('step_size', 0)
                if _range <= 0 or _step <= 0:
                    msg = (
                        f'Z-stack range ({_range}) and step size ({_step}) '
                        f'must both be greater than zero.'
                    )
                else:
                    msg = 'No Z-stack positions configured.'
                logger.warning(f'[LVP Main  ] ZStack: {msg}')
                from modules.notification_center import notifications

                notifications.warning('Z-Stack', 'Z-Stack Not Configured', msg)
                run_not_started_func()
                return

            curr_position = ctx.session.get_current_plate_position()
            curr_position.update({'name': 'ZStack'})

            positions = [
                curr_position,
            ]

            tiling_config = TilingConfig(
                tiling_configs_file_loc=pathlib.Path(ctx.source_path) / 'data' / 'tiling.json',
            )

            config = config_helpers.build_sequenced_capture_config(
                {
                    'labware_id': labware_id,
                    'positions': positions,
                    'objective_id': objective_id,
                    'zstack_params': zstack_params,
                    'use_zstacking': True,
                    'tiling': tiling_config.no_tiling_label(),
                    'tiling_overlap_percent': 0.0,
                    'layer_configs': {active_layer: active_layer_config},
                    'period': None,
                    'duration': None,
                    'frame_dimensions': get_current_frame_dimensions(),
                    'binning_size': get_binning_from_ui(),
                    'stim_config': get_stim_configs(),
                }
            )

            zstack_sequence = ctx.scope.protocols.create_protocol(input_config=config)

            autogain_settings = get_auto_gain_settings()

            # Per-step progress indicator on the Acquire button. The
            # protocol_step_runner fires update_step_number(step) per slice,
            # where step is 1-indexed; the lambda captures total once at
            # construction time. Clock.schedule_once marshals the text
            # update back to the main thread because update_step_number
            # fires from the protocol thread.
            total_slices = zstack_sequence.num_steps()
            zstack_btn = self.ids['zstack_aqr_btn']

            def _zstack_progress(step_num):
                Clock.schedule_once(
                    lambda dt: setattr(zstack_btn, 'text', f'Z {step_num}/{total_slices}'),
                    0,
                )

            callbacks = {
                **live_display_callbacks(),
                'move_position': _handle_ui_update_for_axis,
                'run_complete': run_complete_func,
                'update_step_number': _zstack_progress,
                # LED observer handles UI sync -- no manual callbacks needed
                'reset_autofocus_btns': update_autofocus_selection_after_protocol,
                'set_recording_title': set_recording_title,
                'set_writing_title': set_writing_title,
                'reset_title': reset_title,
                'pause_live_ui': lambda: (
                    ctx.scope_display.stop(),
                    Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
                ),
                'resume_live_ui': lambda: (
                    ctx.scope_display.start(),
                    Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
                    Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1),
                ),
            }

            parent_dir = pathlib.Path(settings['live_folder']).resolve() / 'Manual' / 'Z-Stacks'

            initial_position = ctx.session.get_current_plate_position()
            image_capture_config = get_image_capture_config_from_ui()

            def prepare_and_start():
                plan = ctx.sequenced_capture_runner.prepare(
                    protocol=zstack_sequence,
                    run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
                    run_trigger_source=trigger_source,
                    max_scans=1,
                    sequence_name='zstack',
                    parent_dir=parent_dir,
                    image_capture_config=image_capture_config,
                    enable_image_saving=is_image_saving_enabled(),
                    autogain_settings=autogain_settings,
                    callbacks=callbacks,
                    return_to_position=initial_position,
                    leds_state_at_end='return_to_original',
                    engineering_mode=ctx.engineering_mode,
                    autofocus_snapshot=config_helpers.autofocus_snapshot_from_settings(
                        settings, ctx.settings_lock
                    ),
                    **config_helpers.get_sequenced_run_settings(
                        settings, run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK
                    ),
                )
                ctx.sequenced_capture_runner.start(plan)
                # A refusal raises out of prepare before this line, so the
                # save folder can only ever point at THIS run's directory,
                # never a previous run's stale data.
                set_last_save_folder(dir=ctx.sequenced_capture_runner.run_dir())

            run_with_refusal_boundary(prepare_and_start, on_refused=run_not_started_func)
        except Exception as e:
            logger.error(f'[UI] run_zstack_acquire_from_ui failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))
