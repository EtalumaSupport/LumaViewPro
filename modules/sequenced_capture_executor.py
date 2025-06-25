
import datetime
import pathlib
import time
import typing

import numpy as np
import cv2

from kivy.clock import Clock

from lumascope_api import Lumascope

import image_utils

import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations

import modules.labware_loader as labware_loader
from modules.autofocus_executor import AutofocusExecutor
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
from modules.video_writer import VideoWriter
from modules.sequential_io_executor import SequentialIOExecutor, IOTask
from lvp_logger import logger

from settings_init import settings


class SequencedCaptureExecutor:

    LOGGER_NAME = "SeqCapExec"

    def __init__(
        self,
        scope: Lumascope,
        stage_offset: dict,
        io_executor: SequentialIOExecutor,
        protocol_executor: SequentialIOExecutor,
        file_io_executor: SequentialIOExecutor,
        autofocus_executor: AutofocusExecutor | None = None,
    ):
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()
        self._wellplate_loader = labware_loader.WellPlateLoader()
        self._stage_offset = stage_offset
        self._io_executor = io_executor
        self.protocol_executor = protocol_executor
        self.file_io_executor = file_io_executor

        if autofocus_executor is None:
            self._autofocus_executor = AutofocusExecutor(
                scope=scope,
                use_kivy_clock=True,
            )
        else:
            self._autofocus_executor = autofocus_executor

        self._scope = scope
        self._run_trigger_source = None
        self._reset_vars()
        self._grease_redistribution_done = True


    def _reset_vars(
        self
    ):
        self._run_dir = None
        self._run_trigger_source = None
        self._run_in_progress = False
        self._curr_step = 0
        self._n_scans = 0
        self._scan_count = 0
        self._scan_in_progress = False
        self._autofocus_count = 0
        self._autogain_countdown = 0
        self._grease_redistribution_done = True


    @staticmethod
    def _calculate_num_scans(
        protocol: Protocol,
        run_mode: SequencedCaptureRunMode,
        max_scans: int | None,
    ) -> int:
        if run_mode in (
            SequencedCaptureRunMode.FULL_PROTOCOL,
        ):
            n_scans = int(protocol.duration()/protocol.period())

            if max_scans is not None:
                n_scans = min(n_scans, max_scans)
        else:
            n_scans = max_scans

        return n_scans
    

    def num_scans(self) -> int:
        return self._n_scans
    

    def scan_count(self) -> int:
        return self._scan_count
    

    def remaining_scans(self) -> int:
        return self._n_scans - self._scan_count
    

    def _init_for_new_scan(
        self,
        max_scans: int
    ) -> bool:
        self._reset_vars()
        self._n_scans = self._calculate_num_scans(
            protocol=self._protocol,
            run_mode=self._run_mode,
            max_scans=max_scans,
        )

        self._start_t = datetime.datetime.now()

        if self._disable_saving_artifacts:
            return {
                'status': True,
                'data': None,
                'error': None
            }

        try:
            self._parent_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            err_str = f"Unable to save data to {str(self._parent_dir)}. Please select an accessible capture location."
            return {
                'status': False,
                'data': None,
                'error': err_str,
            }
    
        result = self._create_run_dir()
        if not result['status']:
            return result
        
        try:
            self._initialize_run_dir()
        except Exception as ex:
            err_str = f"Unable to initialize sequenced run directory: {ex}"
            return {
                'status': False,
                'data': None,
                'error': err_str
            }

        return {
            'status': True,
            'data': None,
            'error': None
        }
    

    def run_dir(self):
        return self._run_dir
        

    def _create_run_dir(self):
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d_%H%M%S")
        self._run_dir = self._parent_dir / time_string

        try:
            self._run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            err_str = f"Unable to save data to {str(self._run_dir)}, already exists."
            return {
                'status': False,
                'data': None,
                'error': err_str,
            }
        except FileNotFoundError:
            err_str = f"Unable to save data to {str(self._run_dir)}. Please select an accessible capture location."
            return {
                'status': False,
                'data': None,
                'error': err_str,
            }
        
        return {
            'status': True,
            'data': None,
            'error': None,
        }
    

    def _initialize_run_dir(self):
        if self._sequence_name in (None, ""):
            self._sequence_name = 'unsaved_protocol'
            
        protocol_filename = self._sequence_name
        if not protocol_filename.endswith(".tsv"):
            protocol_filename += ".tsv"

        protocol_file_loc = self._run_dir / protocol_filename
        self._protocol.to_file(
            file_path=protocol_file_loc
        )

        protocol_record_file_loc = self._run_dir / ProtocolExecutionRecord.DEFAULT_FILENAME
        self._protocol_execution_record = ProtocolExecutionRecord(
            outfile=protocol_record_file_loc,
            protocol_file_loc=protocol_filename,
        )

        return True


    def reset(self):
        if not self._run_in_progress:
            return
        
        self._cleanup()


    def protocol_interval(self):
        return self._protocol.period()
    
    def get_initial_autofocus_states(self):
        states = {}
        for layer in common_utils.get_layers():
            states[layer] = settings[layer]["autofocus"]
        return states


    def run(
        self,
        protocol: Protocol,
        run_trigger_source: str,
        run_mode: SequencedCaptureRunMode,
        sequence_name: str,
        image_capture_config: dict,
        autogain_settings: dict,
        parent_dir: pathlib.Path | None = None,
        enable_image_saving: bool = True,
        separate_folder_per_channel: bool = False,
        callbacks: dict[str, typing.Callable] | None = None,
        max_scans: int | None = None,
        return_to_position: dict | None = None,
        disable_saving_artifacts: bool = False,
        save_autofocus_data: bool = False,
        update_z_pos_from_autofocus: bool = False,
        leds_state_at_end: str = "off",
        video_as_frames: bool = False,
    ):
        if self._run_in_progress:
            logger.error(f"[{self.LOGGER_NAME} ] Cannot start new run, run already in progress")
            return
        
        if leds_state_at_end not in ("off", "return_to_original",):
            raise ValueError(f"Unsupported value for leds_state_at_end: {leds_state_at_end}")
        
        # Not IO
        self._original_led_states = self._scope.get_led_states()
        self._original_autofocus_states = self.get_initial_autofocus_states()

        self._protocol = protocol
        self._run_mode = run_mode
        self._sequence_name = sequence_name
        self._parent_dir = parent_dir
        self._image_capture_config = image_capture_config
        self._enable_image_saving = enable_image_saving
        self._separate_folder_per_channel = separate_folder_per_channel
        self._autogain_settings = autogain_settings
        self._callbacks = callbacks
        self._return_to_position = return_to_position
        self._disable_saving_artifacts = disable_saving_artifacts
        self._save_autofocus_data = save_autofocus_data
        self._update_z_pos_from_autofocus = update_z_pos_from_autofocus
        self._leds_state_at_end = leds_state_at_end
        self._video_as_frames = video_as_frames
        self._autofocus_executor.reset()

        if self._parent_dir is None:
            self._disable_saving_artifacts = True

        self._cancel_all_scheduled_events()
        result = self._init_for_new_scan(max_scans=max_scans)
        if not result['status']:
            self._run_in_progress = False
            logger.error(f"[{self.LOGGER_NAME} ] {result['error']}")
            return 
        
        self._run_trigger_source = run_trigger_source
        self._run_in_progress = True
        # Not IO
        self._scope.camera.update_auto_gain_target_brightness(self._autogain_settings['target_brightness'])

        self._run_scan()
        Clock.schedule_interval(self.protocol_executor.put(IOTask(action=self._protocol_iterate)), 1)

    
    def _protocol_iterate(self, dt):
        if self._scan_in_progress:
            return
        
        remaining_scans=self.remaining_scans()
        if remaining_scans == 0:
            self._cleanup()
            return 

        if 'protocol_iterate_pre' in self._callbacks:
            self._callbacks['protocol_iterate_pre'](remaining_scans=remaining_scans, interval=self._protocol.period())

        current_time = datetime.datetime.now()
        elapsed_time = current_time - self._start_t

        # If the next period hasn't been reached, then return
        if elapsed_time <= self._protocol.period():
            return

        # reset the start time and update number of scans remaining
        self._start_t = current_time
                
        self._run_scan()


    def _run_scan(
        self
    ):
        if self._scan_in_progress == True:
            self._scan_count += 1

        self._scan_in_progress = True 
        self._curr_step = 0

        # reset the is_complete flag on autofocus
        if 'run_scan_pre' in self._callbacks:
            self._callbacks['run_scan_pre']()

        self._go_to_step(step_idx=self._curr_step)

        self._auto_gain_countdown = self._autogain_settings['max_duration'].total_seconds()

        Clock.schedule_interval(self.protocol_executor.put(IOTask(action=self._scan_iterate)), 0.1)
    

    def _scan_iterate(self, dt):
        if self._autofocus_executor.in_progress():
            return

        # Check if at desired position
        self._io_executor.protocol_put(IOTask(
            action=self._get_target_status,
            callback=self._scan_iterate_with_targets,
            pass_result=True
        ))

    def _get_target_status(self):
        x_status = self._scope.get_target_status('X')
        y_status = self._scope.get_target_status('Y')
        z_status = self._scope.get_target_status('Z')

        return (x_status, y_status, z_status)

    def _scan_iterate_with_targets(self, result=None, exception=None):
        if exception is not None:
            raise exception
        if result is None:
            return
        
        x_status = result[0]
        y_status = result[1]
        z_status = result[2]

        # Check if target location has not been reached yet
        if (not x_status) or (not y_status) or (not z_status) or self._scope.get_overshoot():
            return
        
        if not self._grease_redistribution_done:
            return
        
        step = self._protocol.step(idx=self._curr_step)
        
        # Set camera settings
        self._io_executor.protocol_put(IOTask(
            action=self._scope.set_auto_gain,
            kwargs={
                "state": step['Auto_Gain'],
                "settings": self._autogain_settings,
            }
        ))
    # self._scope.set_auto_gain(
    #     state=step['Auto_Gain'],
    #     settings=self._autogain_settings,
    # )
        # Internally uses io_executor
        self._led_on(color=step['Color'], illumination=step['Illumination'])

        if not step['Auto_Gain']:
            self._io_executor.protocol_put(IOTask(action=self._scope.set_gain, args=(step['Gain'])))
            #self._scope.set_gain(step['Gain'])
            # 2023-12-18 Instead of using only auto gain, now it's auto gain + exp. If auto gain is enabled, then don't set exposure time
            self._io_executor.protocol_put(IOTask(action=self._scope.set_exposure_time, args=(step['Exposure'])))
            #self._scope.set_exposure_time(step['Exposure'])

        if step['Auto_Gain'] and self._auto_gain_countdown > 0:
            self._auto_gain_countdown -= 0.1
        
        # If the autofocus is selected, is not currently running and has not completed, begin autofocus
        if step['Auto_Focus'] and not self._autofocus_executor.complete() and not self._autofocus_executor.in_progress():

            if 'autofocus_in_progress' in self._callbacks:
                self._callbacks['autofocus_in_progress']()

            af_executor_callbacks = {}
            if 'move_position' in self._callbacks:
                af_executor_callbacks['move_position'] = self._callbacks['move_position']

            #TODO: Make sure all of this IO is handled outside of Kivy main thread
            self._autofocus_executor.run(
                objective_id=step['Objective'],
                save_results_to_file=self._save_autofocus_data,
                results_dir=self._parent_dir,
                callbacks=af_executor_callbacks,
            )
            
            return
        
        # Still executing autofocus (not complete)
        if step['Auto_Focus'] and self._autofocus_executor.in_progress():
            return
        
        # Check if autogain has time-finished after auto-focus so that they can run in parallel
        if step['Auto_Gain'] and self._auto_gain_countdown > 0:
            return
        
        # Reset the autogain countdown
        self._auto_gain_countdown = self._autogain_settings['max_duration'].total_seconds()
        
        # Update the Z position with autofocus results
        if step['Auto_Focus'] and self._update_z_pos_from_autofocus:
            new_z_pos = self._autofocus_executor.best_focus_position()
            self._protocol.modify_step_z_height(
                step_idx=self._curr_step,
                z=new_z_pos,
            )

        # reset the is_complete flag on autofocus
        if 'autofocus_complete' in self._callbacks:
            self._callbacks['autofocus_complete']()

        if step["Auto_Focus"]:
            self._autofocus_count += 1

        if not self._disable_saving_artifacts:

            if self._separate_folder_per_channel:
                save_folder = self._run_dir / step["Color"]
                save_folder.mkdir(parents=True, exist_ok=True)
            else:
                save_folder = self._run_dir


            output_format=self._image_capture_config['output_format']['sequenced']

            # Handle hyperstack creation as a post-processing function for now. Capture images in TIFF.
            if output_format == 'ImageJ Hyperstack':
                output_format = 'TIFF'

            capture_result = self._capture(
                save_folder=save_folder,
                step=step,
                scan_count=self._scan_count,
                output_format=output_format,
                sum_count=step["Sum"],
            )


            if self._enable_image_saving:
                if capture_result is None:
                    capture_result_filepath_name = "unsaved"

                elif type(capture_result) == dict:
                    capture_result_filepath_name = capture_result['metadata']['file_loc']

                elif self._separate_folder_per_channel:
                    capture_result_filepath_name = pathlib.Path(step["Color"]) / capture_result.name

                else:
                    capture_result_filepath_name = capture_result.name
            else:
                capture_result_filepath_name = "unsaved"

            self._protocol_execution_record.add_step(
                capture_result_file_name=capture_result_filepath_name,
                step_name=step['Name'],
                step_index=self._curr_step,
                scan_count=self._scan_count,
                timestamp=datetime.datetime.now()
            )
        else:
            # Normally LEDs are turned off at the end of a capture. However, when not capturing, need to manually turn
            # off LEDs (such as in autofocus scan)
            self._leds_off()

        self._autofocus_executor.reset()
        # Disable autogain when moving between steps
        if step['Auto_Gain']:
            self._io_executor.protocol_put(IOTask(
                action=self._scope.set_auto_gain,
                kwargs={
                    "state":False,
                    "settings":self._autogain_settings
                }
            ))
            #self._scope.set_auto_gain(state=False, settings=self._autogain_settings,)

        num_steps = self._protocol.num_steps()
        if self._curr_step < num_steps-1:

            # increment to the next step. Don't let it exceed the number of steps in the protocol
            self._curr_step = min(self._curr_step+1, num_steps-1)

            if 'update_step_number' in self._callbacks:
                self._callbacks['update_step_number'](self._curr_step+1)
            self._go_to_step(step_idx=self._curr_step)
            return

        # At the end of a scan, if we've performed more than 100 AFs, cycle the Z-axis to re-distribute grease
        if self._autofocus_count >= 100:
            self._perform_grease_redistribution()
            self._autofocus_count = 0

        self._scan_count += 1
        
        if 'scan_iterate_post' in self._callbacks is not None:
            self._callbacks['scan_iterate_post']()

        Clock.unschedule(self._scan_iterate)
        self._scan_in_progress = False


    def run_in_progress(self) -> bool:
        return self._run_in_progress
    

    def run_trigger_source(self) -> str:
        return self._run_trigger_source
    
    
    def _default_move(
        self,
        px: float | None = None,
        py: float | None = None,
        z: float | None = None,
    ):
        labware = self._wellplate_loader.get_plate(plate_key=self._protocol.labware())

        if (px is not None) and (py is not None):
            sx, sy = self._coordinate_transformer.plate_to_stage(
                labware=labware,
                stage_offset=self._stage_offset,
                px=px,
                py=py
            )

            self._io_executor.protocol_put(IOTask(action=self._default_move_ex, args=(sx, sy, z), callback=self._default_move_callbacks_ex, cb_args=(z)))
        
        

    def _default_move_ex(self, sx, sy, z):
        self._scope.move_absolute_position('X', sx)
        self._scope.move_absolute_position('Y', sy)
        if z is not None:
            self._scope.move_absolute_position('Z', z)

    def _default_move_callbacks_ex(self, z):
        
        self._callbacks['move_position']('X')
        self._callbacks['move_position']('Y')

        if z is not None:
            self._callbacks['move_position']('Z')


    def _go_to_step(
        self,
        step_idx: int,
    ):
        if 'go_to_step' in self._callbacks:
            self._callbacks['go_to_step'](
                protocol=self._protocol,
                step_idx=step_idx,
                include_move=True,
                ignore_auto_gain=True
            )

        else:
            step = self._protocol.step(idx=step_idx)
            self._default_move(
                px=step['X'],
                py=step['Y'],
                z=step['Z'],
            )


    def _perform_grease_redistribution(self):
        self._grease_redistribution_done = False
        self._io_executor.protocol_put(IOTask(action=self._grease_redist_w_pos))

    def _grease_redist_w_pos(self):
        axis='Z'
        z_orig = self._scope.get_current_position(axis=axis)
        self._scope.move_absolute_position(
            axis=axis,
            pos=0,
            wait_until_complete=True,
            overshoot_enabled=True
        )

        if 'move_position' in self._callbacks:
            self._callbacks['move_position'](axis)

        self._scope.move_absolute_position(
            axis=axis,
            pos=z_orig,
            wait_until_complete=True,
            overshoot_enabled=True
        )

        if 'move_position' in self._callbacks:
            self._callbacks['move_position'](axis)

        self._grease_redistribution_done = True


    def _cancel_all_scheduled_events(self):
        Clock.unschedule(self._protocol_iterate)
        Clock.unschedule(self._scan_iterate)
        self._io_executor.clear_protocol_pending()


    def _leds_off(self):
        #self._scope.leds_off()
        if 'leds_off' in self._callbacks:
            self._io_executor.protocol_put(IOTask(
                action=self._scope.leds_off,
                callback=self._callbacks['leds_off']
            ))
            #self._callbacks['leds_off']()
        else:
            self._io_executor.protocol_put(action=self._scope.leds_off)

    
    def _led_on(self, color: str, illumination: float):
        # self._scope.led_on(
        #     channel=self._scope.color2ch(color),
        #     mA=illumination,
        # )

        if 'led_state' in self._callbacks:
            self._io_executor.protocol_put(IOTask(
                action=self._scope.led_on,
                kwargs={
                    "channel":self._scope.color2ch(color),
                    "mA":illumination
                },
                callback=self._callbacks['led_state'],
                cb_kwargs={
                    "layer":color,
                    "enabled":True
                }
            ))
            #self._callbacks['led_state'](layer=color, enabled=True)

        else:
            self._io_executor.protocol_put(IOTask(
                action=self._scope.led_on,
                kwargs={
                    "channel":self._scope.color2ch(color),
                    "mA":illumination
                },
            ))


    def _cleanup(self):
        if not self._run_in_progress:
            return
        
        # if self._use_tiff_stacks:
        #     self._protocol.remove_zstack_starts_and_ends()
        
        self._cancel_all_scheduled_events()

        if self._leds_state_at_end == "off":
            self._leds_off()
        elif self._leds_state_at_end == "return_to_original":
            self._leds_off()
            for color, color_data in self._original_led_states.items():
                if color_data['enabled']:
                    self._led_on(color=color, illumination=color_data['illumination'])
        else:
            raise NotImplementedError(f"Unsupported LEDs state at end value: {self._leds_state_at_end}")
        
        # Always return autofocus states to intial
        for layer, layer_data in self._original_autofocus_states.items():
            settings[layer]["autofocus"] = layer_data

        if not self._disable_saving_artifacts:
            self._protocol_execution_record.complete()

        if self._return_to_position is not None:
            self._default_move(
                px=self._return_to_position['x'],
                py=self._return_to_position['y'],
                z=self._return_to_position['z'],
            )

        self._run_in_progress = False

        self._io_executor.protocol_end()


        if 'run_complete' in self._callbacks:
            self._callbacks['run_complete'](protocol=self._protocol)


    def _capture(
        self,
        save_folder,
        step,
        output_format: str,
        scan_count = None,
        sum_count: int = 1,
    ):
        
        is_video = True if step['Acquire'] == "video" else False
        video_as_frames = self._video_as_frames

        if not step['Auto_Gain']:
            self._scope.set_gain(step['Gain'])
            self._scope.set_exposure_time(step['Exposure'])

        if self._scope.has_turret():
            objective_short_name = self._scope.get_objective_info(objective_id=step["Objective"])['short_name']
        else:
            objective_short_name = None
        
        name = common_utils.generate_default_step_name(
            well_label=step['Well'],
            color=step['Color'],
            z_height_idx=step['Z-Slice'],
            scan_count=scan_count,
            custom_name_prefix=step['Name'],
            objective_short_name=objective_short_name,
            tile_label=step['Tile'],
            video=is_video,
        )

        # Illuminate
        if self._scope.led:
            self._led_on(color=step['Color'], illumination=step['Illumination'])
            logger.info(f"[{self.LOGGER_NAME} ] scope.led_on({step['Color']}, {step['Illumination']})")
        else:
            logger.warning('LED controller not available.')

        # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
        # Grab image and save

        time.sleep(2*step['Exposure']/1000+0.2)
        earliest_image_ts = datetime.datetime.now()
        if 'update_scope_display' in self._callbacks:
            self._callbacks['update_scope_display']()
            sum_iteration_callback=self._callbacks['update_scope_display']
        else:
            sum_iteration_callback=None

        use_color = step['Color'] if step['False_Color'] else 'BF'

        if self._enable_image_saving == True:
            use_full_pixel_depth = self._image_capture_config['use_full_pixel_depth']

            if is_video:
                # Disable autogain and then reenable it only for the first frame
                if step["Auto_Gain"]:
                    self._scope.set_auto_gain(state=False, settings=self._autogain_settings)
                    self._scope.camera.auto_gain_once(state=True, 
                                                      target_brightness=self._autogain_settings['target_brightness'], 
                                                      min_gain=self._autogain_settings['min_gain'], 
                                                      max_gain=self._autogain_settings['max_gain']
                                                      )

                duration_sec = step['Video Config']['duration']

                # Clamp the FPS to be no faster than the exposure rate
                exposure = step['Exposure']
                exposure_freq = 1.0 / (exposure / 1000)
                fps = min(exposure_freq, 40)
                
                if video_as_frames:
                    save_folder = save_folder / f"{name}"

                else:    
                    output_file_loc = save_folder / f"{name}.mp4v"

                start_ts = time.time()
                stop_ts = start_ts + duration_sec
                expected_frames = fps * duration_sec
                captured_frames = 0
                seconds_per_frame = 1.0 / fps
                video_images = []

                logger.info(f"Protocol-Video] Capturing video...")

                while time.time() < stop_ts:
                    
                    """     **NOT BEING REFRESHED ON SCREEN EVEN IF ENABLED**
                    if 'update_scope_display' in self._callbacks:
                        self._callbacks['update_scope_display']()
                    """
                        
                    # Currently only support 8-bit images for video
                    force_to_8bit = True
                    image = self._scope.get_image(force_to_8bit=force_to_8bit)


                    if type(image) == np.ndarray:
                        
                        # Should never be used since forcing images to 8-bit
                        if image.dtype == np.uint16:
                            image = image_utils.convert_12bit_to_16bit(image)

                        # Note: Currently, if image is 12/16-bit, then we ignore false coloring for video captures.
                        if (image.dtype != np.uint16) and (step['False_Color']):
                            image = image_utils.add_false_color(array=image, color=use_color)

                        image = np.flip(image, 0)

                        video_images.append((image, datetime.datetime.now()))

                        captured_frames += 1
                    
                    # Some process is slowing the video-process down (getting fewer frames than expected if delay of seconds_per_frame), so a shorter sleep time can be used
                    time.sleep(seconds_per_frame*0.9)

                calculated_fps = captured_frames//duration_sec

                logger.info(f"Protocol-Video] Images present in video array: {len(video_images) > 0}")
                logger.info(f"Protocol-Video] Captured Frames: {captured_frames}")
                logger.info(f"Protocol-Video] Video FPS: {calculated_fps}")
                logger.info("Protocol-Video] Writing video...")

                if video_as_frames:
                    frame_num = 0
                    result = save_folder
                    if not save_folder.exists():
                        save_folder.mkdir(exist_ok=True, parents=True)

                    for image_ts_pair in video_images:
                        frame_num += 1

                        image = image_ts_pair[0]
                        ts = image_ts_pair[1]
                        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                        image = image_utils.add_timestamp(image=image, timestamp_str=ts_str)

                        frame_name = f"{name}_Frame_{frame_num:04}"

                        output_file_loc = save_folder / f"{frame_name}.tiff"

                        metadata = {
                            "datetime": ts.strftime("%Y:%m:%d %H:%M:%S"),
                            "timestamp": ts.strftime("%Y:%m:%d %H:%M:%S.%f"),
                            "frame_num": frame_num
                        }
                        
                        try:
                            image_utils.write_tiff(
                                data=image,
                                metadata=metadata,
                                file_loc=output_file_loc,
                                video_frame=True,
                                ome=False,
                            )
                        except Exception as e:
                            logger.error(f"Protocol-Video] Failed to write frame {frame_num}: {e}")
                            
                        """if not cv2.imwrite(filename=str(output_file_loc), img=image):
                            logger.error(f"Protocol-Video] Failed to write frame {frame_num}")
"""
                else:
                    video_writer = VideoWriter(
                        output_file_loc=output_file_loc,
                        fps=calculated_fps,
                        include_timestamp_overlay=True
                    )

                    for image_ts_pair in video_images:
                        try:
                            video_writer.add_frame(image=image_ts_pair[0], timestamp=image_ts_pair[1])
                        except:
                            logger.error("Protocol-Video] FAILED TO WRITE FRAME")

                    video_writer.finish()
                    result = output_file_loc
                
                logger.info("Protocol-Video] Video writing finished.")
                logger.info(f"Protocol-Video] Video saved at {result}")
            
            else:

                result = self._scope.save_live_image(
                    save_folder=save_folder,
                    file_root=None,
                    append=name,
                    color=use_color,
                    tail_id_mode=None,
                    force_to_8bit=not use_full_pixel_depth,
                    output_format=output_format,
                    true_color=step['Color'],
                    earliest_image_ts=earliest_image_ts,
                    timeout=datetime.timedelta(seconds=1.0),
                    all_ones_check=True,
                    sum_count=sum_count,
                    sum_delay_s=step["Exposure"]/1000,
                    sum_iteration_callback=sum_iteration_callback,
                    turn_off_all_leds_after=True,
                )
        else:
            result = None

        self._leds_off()

        return result
