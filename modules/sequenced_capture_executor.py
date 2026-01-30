
import datetime
import pathlib
import time
import sys
import ctypes
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
from lvp_logger import logger

from settings_init import settings

import threading


class SequencedCaptureExecutor:

    LOGGER_NAME = "SeqCapExec"

    def __init__(
        self,
        scope: Lumascope,
        stage_offset: dict,
        autofocus_executor: AutofocusExecutor | None = None,
    ):
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()
        self._wellplate_loader = labware_loader.WellPlateLoader()
        self._stage_offset = stage_offset

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
        self._scope.camera.update_auto_gain_target_brightness(self._autogain_settings['target_brightness'])

        self._run_scan()
        Clock.schedule_interval(self._protocol_iterate, 1)

    
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

        Clock.schedule_interval(self._scan_iterate, 0.1)
    

    def _scan_iterate(self, dt):
        if self._autofocus_executor.in_progress():
            return

        # Check if at desired position
        x_status = self._scope.get_target_status('X')
        y_status = self._scope.get_target_status('Y')
        z_status = self._scope.get_target_status('Z')

        # Check if target location has not been reached yet
        if (not x_status) or (not y_status) or (not z_status) or self._scope.get_overshoot():
            return
        
        step = self._protocol.step(idx=self._curr_step)
        
        # Set camera settings
        self._scope.set_auto_gain(
            state=step['Auto_Gain'],
            settings=self._autogain_settings,
        )
        self._led_on(color=step['Color'], illumination=step['Illumination'])

        if not step['Auto_Gain']:
            self._scope.set_gain(step['Gain'])
            # 2023-12-18 Instead of using only auto gain, now it's auto gain + exp. If auto gain is enabled, then don't set exposure time
            self._scope.set_exposure_time(step['Exposure'])

        if step['Auto_Gain'] and self._auto_gain_countdown > 0:
            self._auto_gain_countdown -= 0.1
        
        # If the autofocus is selected, is not currently running and has not completed, begin autofocus
        if step['Auto_Focus'] and not self._autofocus_executor.complete():

            if 'autofocus_in_progress' in self._callbacks:
                self._callbacks['autofocus_in_progress']()

            af_executor_callbacks = {}
            if 'move_position' in self._callbacks:
                af_executor_callbacks['move_position'] = self._callbacks['move_position']

            self._autofocus_executor.run(
                objective_id=step['Objective'],
                save_results_to_file=self._save_autofocus_data,
                results_dir=self._parent_dir,
                callbacks=af_executor_callbacks,
            )
            
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
            self._scope.set_auto_gain(state=False, settings=self._autogain_settings,)

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

            self._scope.move_absolute_position('X', sx)
            self._scope.move_absolute_position('Y', sy)
            self._callbacks['move_position']('X')
            self._callbacks['move_position']('Y')

        if z is not None:
            self._scope.move_absolute_position('Z', z)
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


    def _cancel_all_scheduled_events(self):
        Clock.unschedule(self._protocol_iterate)
        Clock.unschedule(self._scan_iterate)


    def _leds_off(self):
        self._scope.leds_off()
        if 'leds_off' in self._callbacks:
            self._callbacks['leds_off']()

    
    def _led_on(self, color: str, illumination: float):
        self._scope.led_on(
            channel=self._scope.color2ch(color),
            mA=illumination,
        )

        if 'led_state' in self._callbacks:
            self._callbacks['led_state'](layer=color, enabled=True)


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

        # Generate stimulation timing summary
        self._generate_stimulation_summary()

        if self._return_to_position is not None:
            self._default_move(
                px=self._return_to_position['x'],
                py=self._return_to_position['y'],
                z=self._return_to_position['z'],
            )

        self._run_in_progress = False


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

                stop_event = threading.Event()
                start_event = threading.Event()

                stim_threads = []

                for color in step['Stim_Config']:
                    stim_config = step['Stim_Config'][color]
                    if stim_config['enabled']:
                        stim_thread = threading.Thread(target=self._stimulate, args=(color, stim_config, start_event, stop_event))
                        stim_threads.append(stim_thread)
                        stim_thread.start()

                start_event.set()

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

                stop_event.set()
                for stim_thread in stim_threads:
                    stim_thread.join()

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

    # stim_config = {
    #         "Red": {
    #             "enabled": True,
    #             "illumination": 100,
    #             "frequency": 1,
    #             "pulse_width": 10,
    #             "pulse_count": 1,
    #         },
    #         "Green": {
    #             "enabled": True,
    #             "illumination": 100,
    #             "frequency": 1,
    #             "pulse_width": 10,
    #             "pulse_count": 1,
    #         },
    #         "Blue": {
    #             "enabled": True,
    #             "illumination": 100,
    #             "frequency": 1,
    #             "pulse_width": 10,
    #             "pulse_count": 1,
    #         }
    #     }

    def _stimulate(self, color: str, stim_config: dict, start_event: threading.Event, stop_event: threading.Event):
        if not stim_config['enabled']:
            return

        logger.info(f"[STIMULATOR] Stimulating {color} with {stim_config}")

        illumination = stim_config['illumination']
        frequency = stim_config['frequency']
        pulse_width = stim_config['pulse_width']
        pulse_count = stim_config['pulse_count']

        # Optional: reduce Windows timer quantum to 1 ms during stimulation
        time_period_set = False
        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
                time_period_set = True
            except Exception:
                time_period_set = False

        # DEBUG: Profiling timing data for led_on and led_off
        led_on_timings = []
        led_off_timings = []
        led_actual_on_timings = []  # Time from LED ON command return to LED OFF command call
        profiling_enabled = True
        start_epoch = None  # Will be set after start_event
        pulses_executed = 0
        end_reason = "not_started"

        try:
            # Validate frequency to prevent division by zero
            if frequency <= 0:
                logger.error(f"[STIMULATOR] Invalid frequency {frequency} Hz for {color}. Must be > 0.")
                end_reason = "invalid_frequency"
                return
            period_s = 1.0 / float(frequency)
            pulse_s = float(pulse_width) / 1000.0

            # # Measure approximate command latency to compensate in scheduling
            # # Keep this minimal to avoid warm-up effects
            # t0 = time.perf_counter()
            # self._led_on(color=color, illumination=illumination)
            # t1 = time.perf_counter()
            # self._led_off(color=color)
            # t2 = time.perf_counter()
            # on_latency = max(0.0, t1 - t0)
            # off_latency = max(0.0, t2 - t1)

            # Use fast path LED toggles if available via API
            def led_on_fast():
                #logger.info(f"[STIMULATOR] {color} LED ON")
                if profiling_enabled and start_epoch is not None:
                    t_start = time.perf_counter()
                if hasattr(self._scope, 'led_on_fast'):
                    self._scope.led_on_fast(channel=self._scope.color2ch(color=color), mA=illumination)
                else:
                    self._scope.led_on(channel=self._scope.color2ch(color=color), mA=illumination)
                if profiling_enabled and start_epoch is not None:
                    t_end = time.perf_counter()
                    # Store: (absolute timestamp in ms, duration in ms, relative time since start in ms)
                    led_on_timings.append({
                        'start_time': (t_start - start_epoch) * 1000.0,
                        'end_time': (t_end - start_epoch) * 1000.0,
                        'duration': (t_end - t_start) * 1000.0
                    })
                    return t_end  # Return end time for actual on-time tracking
                return None

            def led_off_fast(led_on_end_time=None):
                #logger.info(f"[STIMULATOR] {color} LED OFF")
                if profiling_enabled and start_epoch is not None:
                    t_start = time.perf_counter()
                    # Track actual LED on-time (from when LED ON command returned to when LED OFF is called)
                    if led_on_end_time is not None:
                        actual_on_duration = (t_start - led_on_end_time) * 1000.0
                        led_actual_on_timings.append({
                            'start_time': (led_on_end_time - start_epoch) * 1000.0,
                            'end_time': (t_start - start_epoch) * 1000.0,
                            'duration': actual_on_duration
                        })
                if hasattr(self._scope, 'led_off_fast'):
                    self._scope.led_off_fast(channel=self._scope.color2ch(color=color))
                else:
                    self._scope.led_off(channel=self._scope.color2ch(color=color))
                if profiling_enabled and start_epoch is not None:
                    t_end = time.perf_counter()
                    # Store: (absolute timestamp in ms, duration in ms, relative time since start in ms)
                    led_off_timings.append({
                        'start_time': (t_start - start_epoch) * 1000.0,
                        'end_time': (t_end - start_epoch) * 1000.0,
                        'duration': (t_end - t_start) * 1000.0
                    })

            start_event.wait()
            start_epoch = time.perf_counter()
            logger.info(f"[STIMULATOR] stim_start_event set for {color} at t=0.000 ms")

            end_reason = "pulse_count_reached"

            for i in range(pulse_count):
                if stop_event.is_set():
                    elapsed = (time.perf_counter() - start_epoch) * 1000.0
                    logger.info(f"[STIMULATOR] {color} stop event set at t={elapsed:.3f} ms, ending stimulation.")
                    end_reason = "stop_event_set"
                    break

                # Target times for this pulse
                on_time = start_epoch + i * period_s
                off_time = on_time + pulse_s
                next_period_time = start_epoch + (i + 1) * period_s

                # Sleep until on_time (coarse) then spin
                while True:
                    now = time.perf_counter()
                    remaining = on_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        # short busy-wait to reduce jitter
                        pass

                led_on_end_time = led_on_fast()

                pulses_executed += 1

                # Sleep/spin until off_time
                while True:
                    now = time.perf_counter()
                    remaining = off_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        pass

                led_off_fast(led_on_end_time)

                # Maintain period; wait until next_period_time
                while True:
                    now = time.perf_counter()
                    remaining = next_period_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        pass

        finally:
            if sys.platform.startswith('win') and time_period_set:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass
            
            elapsed_time = (time.perf_counter() - start_epoch) * 1000.0 if start_epoch is not None else 0.0
            logger.info(f"[STIMULATOR] {color} stimulation ended after executing {pulses_executed} pulses. Elapsed: {elapsed_time:.3f} ms")
            logger.info(f"[STIMULATOR] {color} Ended due to {end_reason}")
            # Ensure LED off at the end
            try:
                if hasattr(self._scope, 'led_off_fast'):
                    self._scope.led_off_fast(channel=self._scope.color2ch(color=color))
                else:
                    self._scope.led_off(channel=self._scope.color2ch(color=color))
            except Exception:
                pass

            # DEBUG: Save profiling statistics
            if profiling_enabled and (led_on_timings or led_off_timings or led_actual_on_timings) and self._run_dir is not None:
                try:
                    profile_dir = self._run_dir / "stimulation_profile"
                    profile_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    profile_file = profile_dir / f"stimulation_profile_{color}_{timestamp}.txt"
                    
                    with open(profile_file, 'w', encoding='utf-8') as f:
                        f.write(f"Stimulation Profiling Report - {color}\n")
                        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                        f.write(f"Frequency: {frequency} Hz\n")
                        f.write(f"Pulse Width: {pulse_width} ms\n")
                        f.write(f"Pulses Executed: {pulses_executed}\n")
                        f.write(f"End Reason: {end_reason}\n")
                        f.write("=" * 60 + "\n\n")
                        
                        if led_actual_on_timings:
                            f.write("LED ACTUAL ON-TIME (ms) - From LED ON return to LED OFF call:\n")
                            f.write("-" * 60 + "\n")
                            durations = [t['duration'] for t in led_actual_on_timings]
                            actual_array = np.array(durations)
                            actual_mean = np.mean(actual_array)
                            actual_std = np.std(actual_array)
                            actual_min = np.min(actual_array)
                            actual_max = np.max(actual_array)
                            
                            f.write(f"Count:            {len(led_actual_on_timings)}\n")
                            f.write(f"Expected:         {pulse_width:.3f} ms\n")
                            f.write(f"Mean:             {actual_mean:.6f} ms\n")
                            f.write(f"Std Dev:          {actual_std:.6f} ms\n")
                            f.write(f"Min:              {actual_min:.6f} ms\n")
                            f.write(f"Max:              {actual_max:.6f} ms\n")
                            f.write(f"Mean Deviation:   {(actual_mean - pulse_width):.6f} ms\n")
                            
                            # Detect outliers using 3-sigma rule
                            outlier_threshold = actual_mean + 3 * actual_std
                            outliers = [t for t in durations if t > outlier_threshold]
                            f.write(f"Outliers (>3stddev):   {len(outliers)} ({len(outliers)/len(led_actual_on_timings)*100:.2f}%)\n")
                            if outliers:
                                f.write(f"Outlier values:   {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                            
                            f.write("\nIndividual LED Actual On-Time Events:\n")
                            for idx, timing in enumerate(led_actual_on_timings):
                                f.write(f"  Pulse {idx}: Start={timing['start_time']:.6f} ms, End={timing['end_time']:.6f} ms, Duration={timing['duration']:.6f} ms\n")
                            f.write("\n")
                        
                        if led_on_timings:
                            f.write("LED ON TIMINGS (ms):\n")
                            f.write("-" * 60 + "\n")
                            durations = [t['duration'] for t in led_on_timings]
                            on_array = np.array(durations)
                            on_mean = np.mean(on_array)
                            on_std = np.std(on_array)
                            on_min = np.min(on_array)
                            on_max = np.max(on_array)
                            
                            f.write(f"Count:            {len(led_on_timings)}\n")
                            f.write(f"Mean:             {on_mean:.6f} ms\n")
                            f.write(f"Std Dev:          {on_std:.6f} ms\n")
                            f.write(f"Min:              {on_min:.6f} ms\n")
                            f.write(f"Max:              {on_max:.6f} ms\n")
                            
                            # Detect outliers using 3-sigma rule
                            outlier_threshold = on_mean + 3 * on_std
                            outliers = [t for t in durations if t > outlier_threshold]
                            f.write(f"Outliers (>3stddev):   {len(outliers)} ({len(outliers)/len(led_on_timings)*100:.2f}%)\n")
                            if outliers:
                                f.write(f"Outlier values:   {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                            
                            f.write("\nIndividual LED ON Events (timestamps relative to start):\n")
                            for idx, timing in enumerate(led_on_timings):
                                f.write(f"  Pulse {idx}: Start={timing['start_time']:.6f} ms, End={timing['end_time']:.6f} ms, Duration={timing['duration']:.6f} ms\n")
                            f.write("\n")
                        
                        if led_off_timings:
                            f.write("LED OFF TIMINGS (ms):\n")
                            f.write("-" * 60 + "\n")
                            durations = [t['duration'] for t in led_off_timings]
                            off_array = np.array(durations)
                            off_mean = np.mean(off_array)
                            off_std = np.std(off_array)
                            off_min = np.min(off_array)
                            off_max = np.max(off_array)
                            
                            f.write(f"Count:            {len(led_off_timings)}\n")
                            f.write(f"Mean:             {off_mean:.6f} ms\n")
                            f.write(f"Std Dev:          {off_std:.6f} ms\n")
                            f.write(f"Min:              {off_min:.6f} ms\n")
                            f.write(f"Max:              {off_max:.6f} ms\n")
                            
                            # Detect outliers using 3-sigma rule
                            outlier_threshold = off_mean + 3 * off_std
                            outliers = [t for t in durations if t > outlier_threshold]
                            f.write(f"Outliers (>3stddev):   {len(outliers)} ({len(outliers)/len(led_off_timings)*100:.2f}%)\n")
                            if outliers:
                                f.write(f"Outlier values:   {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                            
                            f.write("\nIndividual LED OFF Events (timestamps relative to start):\n")
                            for idx, timing in enumerate(led_off_timings):
                                f.write(f"  Pulse {idx}: Start={timing['start_time']:.6f} ms, End={timing['end_time']:.6f} ms, Duration={timing['duration']:.6f} ms\n")
                            f.write("\n")
                    
                    logger.info(f"[STIMULATOR] Profiling data saved to {profile_file}")
                except Exception as e:
                    logger.error(f"[STIMULATOR] Failed to save profiling data: {e}")
                    # Failsafe: Output profiling data to logger if file writing fails
                    try:
                        logger.info(f"[STIMULATOR] PROFILING DATA FOR {color}:")
                        logger.info(f"[STIMULATOR] Frequency: {frequency} Hz, Pulse Width: {pulse_width} ms, Pulses: {pulses_executed}")
                        
                        if led_on_timings:
                            durations = [t['duration'] for t in led_on_timings]
                            on_mean = np.mean(durations)
                            on_std = np.std(durations)
                            logger.info(f"[STIMULATOR] LED ON - Mean: {on_mean:.6f} ms, Std: {on_std:.6f} ms, Min: {np.min(durations):.6f} ms, Max: {np.max(durations):.6f} ms")
                            for idx, timing in enumerate(led_on_timings):
                                logger.info(f"[STIMULATOR] LED ON Pulse {idx}: Start={timing['start_time']:.6f} ms, End={timing['end_time']:.6f} ms, Duration={timing['duration']:.6f} ms")
                        
                        if led_off_timings:
                            durations = [t['duration'] for t in led_off_timings]
                            off_mean = np.mean(durations)
                            off_std = np.std(durations)
                            logger.info(f"[STIMULATOR] LED OFF - Mean: {off_mean:.6f} ms, Std: {off_std:.6f} ms, Min: {np.min(durations):.6f} ms, Max: {np.max(durations):.6f} ms")
                            for idx, timing in enumerate(led_off_timings):
                                logger.info(f"[STIMULATOR] LED OFF Pulse {idx}: Start={timing['start_time']:.6f} ms, End={timing['end_time']:.6f} ms, Duration={timing['duration']:.6f} ms")
                    except Exception as log_error:
                        logger.error(f"[STIMULATOR] Failed to log profiling data: {log_error}")

    def _generate_stimulation_summary(self):
        """Generate a summary report aggregating all stimulation profiling data from the run."""
        if self._run_dir is None:
            return
        
        profile_dir = self._run_dir / "stimulation_profile"
        if not profile_dir.exists():
            return
        
        # Find all profile files
        profile_files = list(profile_dir.glob("stimulation_profile_*.txt"))
        if not profile_files:
            logger.info("[STIMULATOR] No stimulation profile files found to summarize")
            return
        
        logger.info(f"[STIMULATOR] Generating summary from {len(profile_files)} profile files")
        
        try:
            # Data structures to aggregate - separated by color
            all_data_by_color = {}  # color -> timing_type -> {'values': [], 'sources': []}
            
            file_summaries = []
            colors_found = set()
            
            # Parse each profile file
            for profile_file in sorted(profile_files):
                file_data = self._parse_stimulation_profile(profile_file)
                if file_data:
                    file_summaries.append(file_data)
                    color = file_data.get('color', 'Unknown')
                    colors_found.add(color)
                    
                    # Initialize color data structure if needed
                    if color not in all_data_by_color:
                        all_data_by_color[color] = {
                            'actual_on': {'values': [], 'sources': [], 'pulse_widths': []},
                            'led_on_cmd': {'values': [], 'sources': []},
                            'led_off_cmd': {'values': [], 'sources': []}
                        }
                    
                    # Aggregate data with source tracking, separated by color
                    for timing_type in ['actual_on', 'led_on_cmd', 'led_off_cmd']:
                        if timing_type in file_data and file_data[timing_type]:
                            for idx, value in enumerate(file_data[timing_type]):
                                all_data_by_color[color][timing_type]['values'].append(value)
                                all_data_by_color[color][timing_type]['sources'].append({
                                    'file': profile_file.name,
                                    'pulse_num': idx
                                })
                    
                    # Track expected pulse width for actual_on validation
                    if 'pulse_width' in file_data and file_data['pulse_width'] != 'N/A':
                        try:
                            pw = float(file_data['pulse_width'])
                            all_data_by_color[color]['actual_on']['pulse_widths'].extend([pw] * len(file_data.get('actual_on', [])))
                        except:
                            pass
            
            # Generate summary report
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = profile_dir / f"stimulation_summary_{timestamp}.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("STIMULATION PROFILING SUMMARY - ENTIRE RUN\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Run Directory: {self._run_dir}\n")
                f.write(f"Number of Profile Files: {len(profile_files)}\n")
                f.write(f"Colors Found: {', '.join(sorted(colors_found))}\n")
                
                # Calculate total pulses across all colors
                total_pulses = sum(len(all_data_by_color[c]['actual_on']['values']) for c in all_data_by_color)
                f.write(f"Total Pulses Analyzed: {total_pulses}\n")
                f.write("\n")
                
                # Statistics separated by color/channel
                f.write("\n" + "=" * 80 + "\n")
                f.write("STATISTICS BY COLOR/CHANNEL\n")
                f.write("=" * 80 + "\n\n")
                
                for color in sorted(colors_found):
                    if color not in all_data_by_color:
                        continue
                    
                    color_data = all_data_by_color[color]
                    num_pulses = len(color_data['actual_on']['values'])
                    
                    f.write("=" * 80 + "\n")
                    f.write(f"COLOR: {color}\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Total Pulses: {num_pulses}\n\n")
                    
                    # LED Actual On-Time
                    if color_data['actual_on']['values']:
                        f.write("LED ACTUAL ON-TIME (From LED ON return to LED OFF call):\n")
                        f.write("-" * 80 + "\n")
                        self._write_timing_stats(f, color_data['actual_on']['values'])
                        
                        # Show expected pulse width if available
                        if color_data['actual_on']['pulse_widths']:
                            expected_pw = np.mean(color_data['actual_on']['pulse_widths'])
                            actual_mean = np.mean(color_data['actual_on']['values'])
                            deviation = actual_mean - expected_pw
                            f.write(f"  Expected PW: {expected_pw:.3f} ms\n")
                            f.write(f"  Deviation:   {deviation:.6f} ms ({deviation/expected_pw*100:.2f}%)\n")
                        f.write("\n")
                    
                    # LED ON Command Duration
                    if color_data['led_on_cmd']['values']:
                        f.write("LED ON COMMAND DURATION:\n")
                        f.write("-" * 80 + "\n")
                        self._write_timing_stats(f, color_data['led_on_cmd']['values'])
                        f.write("\n")
                    
                    # LED OFF Command Duration
                    if color_data['led_off_cmd']['values']:
                        f.write("LED OFF COMMAND DURATION:\n")
                        f.write("-" * 80 + "\n")
                        self._write_timing_stats(f, color_data['led_off_cmd']['values'])
                        f.write("\n")
                
                # Outlier Analysis by color
                f.write("\n" + "=" * 80 + "\n")
                f.write("OUTLIER ANALYSIS BY COLOR\n")
                f.write("=" * 80 + "\n\n")
                
                for color in sorted(colors_found):
                    if color not in all_data_by_color:
                        continue
                    
                    color_data = all_data_by_color[color]
                    
                    f.write("=" * 80 + "\n")
                    f.write(f"COLOR: {color}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for timing_name, timing_label in [
                        ('actual_on', 'LED Actual On-Time'),
                        ('led_on_cmd', 'LED ON Command'),
                        ('led_off_cmd', 'LED OFF Command')
                    ]:
                        if color_data[timing_name]['values']:
                            f.write(f"{timing_label} Outliers:\n")
                            f.write("-" * 80 + "\n")
                            # For actual_on, pass expected pulse widths for deviation analysis
                            expected_vals = color_data['actual_on']['pulse_widths'] if timing_name == 'actual_on' else None
                            self._write_outlier_details(
                                f, 
                                color_data[timing_name]['values'],
                                color_data[timing_name]['sources'],
                                color,
                                expected_vals
                            )
                            f.write("\n")
            
            logger.info(f"[STIMULATOR] Summary report saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"[STIMULATOR] Failed to generate stimulation summary: {e}", exc_info=True)
    
    def _parse_stimulation_profile(self, filepath):
        """Parse a stimulation profile file and extract timing data."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            data = {'filename': filepath.name}
            
            lines = content.split('\n')
            for line in lines:
                if 'Color:' in line or 'Stimulation Profiling Report -' in line:
                    if '-' in line:
                        data['color'] = line.split('-')[-1].strip()
                elif 'Frequency:' in line:
                    data['frequency'] = line.split(':')[1].split('Hz')[0].strip()
                elif 'Pulse Width:' in line:
                    data['pulse_width'] = line.split(':')[1].split('ms')[0].strip()
                elif 'Pulses Executed:' in line:
                    data['pulses_executed'] = int(line.split(':')[1].strip())
                elif 'End Reason:' in line:
                    data['end_reason'] = line.split(':')[1].strip()
            
            # Extract timing data
            data['actual_on'] = []
            data['led_on_cmd'] = []
            data['led_off_cmd'] = []
            
            # Parse timing sections
            in_actual_on = False
            in_led_on = False
            in_led_off = False
            
            for line in lines:
                if 'LED ACTUAL ON-TIME' in line:
                    in_actual_on = True
                    in_led_on = False
                    in_led_off = False
                elif 'LED ON TIMINGS' in line and 'ACTUAL' not in line:
                    in_led_on = True
                    in_actual_on = False
                    in_led_off = False
                elif 'LED OFF TIMINGS' in line:
                    in_led_off = True
                    in_actual_on = False
                    in_led_on = False
                elif line.startswith('  Pulse '):
                    # Extract duration from pulse lines
                    if 'Duration=' in line:
                        duration_str = line.split('Duration=')[1].split('ms')[0].strip()
                        try:
                            duration = float(duration_str)
                            if in_actual_on:
                                data['actual_on'].append(duration)
                            elif in_led_on:
                                data['led_on_cmd'].append(duration)
                            elif in_led_off:
                                data['led_off_cmd'].append(duration)
                        except ValueError:
                            pass
            
            return data
            
        except Exception as e:
            logger.error(f"[STIMULATOR] Failed to parse {filepath}: {e}")
            return None
    
    def _write_timing_stats(self, f, values):
        """Write statistical summary for timing values."""
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        median = np.median(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)
        
        f.write(f"  Count:      {len(values)}\n")
        f.write(f"  Mean:       {mean:.6f} ms\n")
        f.write(f"  Median:     {median:.6f} ms\n")
        f.write(f"  Std Dev:    {std:.6f} ms\n")
        f.write(f"  Min:        {min_val:.6f} ms\n")
        f.write(f"  Max:        {max_val:.6f} ms\n")
        f.write(f"  95th %ile:  {p95:.6f} ms\n")
        f.write(f"  99th %ile:  {p99:.6f} ms\n")
    
    def _write_outlier_details(self, f, values, sources, color, expected_values=None):
        """Write detailed outlier information with source files."""
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        
        # 3-sigma outliers
        threshold_3sigma = mean + 3 * std
        outliers_3sigma = [(i, v, sources[i]) for i, v in enumerate(values) if v > threshold_3sigma]
        
        f.write(f"  3-Sigma Threshold: {threshold_3sigma:.6f} ms\n")
        f.write(f"  3-Sigma Outliers: {len(outliers_3sigma)} ({len(outliers_3sigma)/len(values)*100:.2f}%)\n")
        
        if outliers_3sigma:
            f.write(f"\n  3-Sigma Outlier Details (showing all):\n")
            for idx, value, source in sorted(outliers_3sigma, key=lambda x: x[1], reverse=True):
                f.write(f"    {value:.6f} ms - Pulse #{source['pulse_num']} in {source['file']}\n")
        
        # For actual on-time, show deviations from expected pulse width
        if expected_values is not None and len(expected_values) == len(values):
            expected_arr = np.array(expected_values)
            deviations = arr - expected_arr
            # Pulses that were >3ms over the intended LED on time
            over_3ms = [(i, v, sources[i], deviations[i]) for i, v in enumerate(values) if deviations[i] > 3.0]
            
            f.write(f"\n  Pulses >3ms Over Intended: {len(over_3ms)} ({len(over_3ms)/len(values)*100:.2f}%)\n")
            
            if over_3ms:
                f.write(f"\n  Details (showing all pulses >3ms over intended):\n")
                for idx, value, source, deviation in sorted(over_3ms, key=lambda x: x[3], reverse=True):
                    expected = expected_values[idx]
                    f.write(f"    {value:.6f} ms (expected {expected:.3f} ms, +{deviation:.3f} ms deviation) - Pulse #{source['pulse_num']} in {source['file']}\n")
