
import datetime
import pathlib
import time
import sys
import ctypes
import typing

import numpy as np
import cv2
import gc
import queue
import ctypes

from modules.sequenced_capture_writer import write_capture

try:
    from kivy.clock import Clock
except ImportError:
    # Dummy Clock for subprocess
    class Clock:
        @staticmethod
        def schedule_once(func, timeout): pass
        @staticmethod
        def schedule_interval(func, interval): pass

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
from concurrent.futures import ProcessPoolExecutor
import threading

from settings_init import settings

import threading


"""
step_dict = {
   "Name": name,
    "X": x,
    "Y": y,
    "Z": z,
    "Auto_Focus": af,
    "Color": color,
    "False_Color": fc,
    "Illumination": ill,
    "Gain": gain,
    "Auto_Gain": auto_gain,
    "Exposure": exp,
    "Sum": sum: int,
    "Objective": objective,
    "Well": well,
    "Tile": tile,
    "Z-Slice": zslice,
    "Custom Step": custom_step: bool,
    "Tile Group ID": tile_group_id,
    "Z-Stack Group ID": zstack_group_id,
    "Acquire": acquire,
    "Video Config": video_config,
}
"""

class SequencedCaptureExecutor:

    LOGGER_NAME = "SeqCapExec"

    def __init__(
        self,
        scope: Lumascope,
        stage_offset: dict,
        io_executor: SequentialIOExecutor,
        protocol_executor: SequentialIOExecutor,
        file_io_executor: SequentialIOExecutor,
        camera_executor: SequentialIOExecutor,
        autofocus_io_executor: SequentialIOExecutor,
        autofocus_executor: AutofocusExecutor | None = None,
        z_ui_update_func: typing.Callable | None = None,
        cpu_pool: ProcessPoolExecutor | None = None,
    ):
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()
        self._wellplate_loader = labware_loader.WellPlateLoader()
        self._stage_offset = stage_offset
        self._io_executor = io_executor
        self.protocol_executor = protocol_executor
        self.file_io_executor = file_io_executor
        #self.file_io_process_pool = ProcessPoolExecutor(max_workers=1)
        self.camera_executor = camera_executor
        self.autofocus_io_executor = autofocus_io_executor
        self._z_ui_update_func = z_ui_update_func
        self._scan_in_progress = threading.Event()
        self._protocol_ended = threading.Event()
        self._cpu_pool = cpu_pool
        self._video_write_finished = threading.Event()
        self._video_write_finished.set()
        self._stim_start_event = threading.Event()
        self._stim_stop_event = threading.Event()

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

        self.debug_counter = 0
        
        # Global running averages for LED command latency compensation
        # Shared across all colors and protocol runs for better accuracy
        self._avg_led_on_latency = 0.0  # seconds
        self._avg_led_off_latency = 0.0  # seconds
        self._led_latency_lock = threading.Lock()  # Thread-safe updates
        self._latency_ema_alpha = 0.2  # EMA smoothing factor
        
        # Process priority management for protocol execution
        self._original_process_priority = None
        self._process_priority_elevated = False
        
        # Test configuration for systematic timing optimization testing
        self._current_test_case = None  # Set by test executor
        
        #self.sleep_time = 0.02

    def set_scope(self, scope: Lumascope):
        self._scope = scope
    
    def _elevate_process_priority(self):
        """Elevate process priority to HIGH for better timing during protocols."""
        if not sys.platform.startswith('win'):
            return
        
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            process_handle = kernel32.GetCurrentProcess()
            
            # Get current priority to restore later
            # PROCESS_QUERY_INFORMATION = 0x0400
            self._original_process_priority = kernel32.GetPriorityClass(process_handle)
            
            # HIGH_PRIORITY_CLASS = 0x00000080
            # REALTIME_PRIORITY_CLASS = 0x00000100 (too aggressive, can freeze system)
            HIGH_PRIORITY_CLASS = 0x00000080
            
            if kernel32.SetPriorityClass(process_handle, HIGH_PRIORITY_CLASS):
                self._process_priority_elevated = True
                logger.info("[Protocol] Process priority elevated to HIGH for better timing consistency")
            else:
                logger.warning("[Protocol] Failed to elevate process priority")
        except Exception as e:
            logger.warning(f"[Protocol] Could not elevate process priority: {e}")
    
    def _restore_process_priority(self):
        """Restore process priority to original value after protocol completes."""
        if not sys.platform.startswith('win') or not self._process_priority_elevated:
            return
        
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            process_handle = kernel32.GetCurrentProcess()
            
            if self._original_process_priority is not None:
                if kernel32.SetPriorityClass(process_handle, self._original_process_priority):
                    logger.info("[Protocol] Process priority restored to normal")
                    self._process_priority_elevated = False
                else:
                    logger.warning("[Protocol] Failed to restore process priority")
        except Exception as e:
            logger.warning(f"[Protocol] Could not restore process priority: {e}")

    def _reset_vars(
        self
    ):
        self._run_dir = None
        self._run_trigger_source = None
        self._run_in_progress = False
        self._curr_step = 0
        self._n_scans = 0
        self._scan_count = 0
        self._scan_in_progress.clear()
        self._autofocus_count = 0
        self._autogain_countdown = 0
        self._grease_redistribution_done = True
        self._captures_taken = 0
        self._target_x_pos = -1
        self._target_y_pos = -1
        self._target_z_pos = -1
        self.debug_counter = 0
        self._protocol_ended.clear()
        self._video_write_finished.set()
        


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
        
        try:
            if not self._scope.are_all_connected():
                logger.error(f"[PROTOCOL] Not all scope components connected. Cannot start run.")
                return
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error checking scope connection: {ex}")
            return

        
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

        self._scan_iterate_running = False
        self._protocol_iterator = None
        self._scan_iterator = None

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
        
        # Elevate process priority for better timing consistency
        self._elevate_process_priority()
        
        self.camera_executor.disable()
        self.protocol_executor.protocol_start()
        self._io_executor.protocol_start()
        self.file_io_executor.protocol_start()
        # Not IO
        self._scope.camera.update_auto_gain_target_brightness(self._autogain_settings['target_brightness'])

        # Start the main run loop which manages all scan timing and execution
        self.protocol_executor.protocol_put(IOTask(action=self._run_loop))
    
    def _run_loop(self):
        """Main run loop - manages protocol execution and scan timing.
        A 'scan' is one complete iteration through all steps in the protocol.
        This loop runs until all scans are complete.
        """
        last_maintenance_time = time.monotonic()
        
        # Initial delay before first iteration
        time.sleep(0.1)
        
        while self._run_in_progress and not self._protocol_ended.is_set():
            try:
                # Check if we've completed all scans
                remaining_scans = self.remaining_scans()
                if remaining_scans <= 0:
                    self._cleanup()
                    break
                
                # Check if enough time has elapsed for the next scan
                # Skip this check for the first scan (scan_count == 0) - start immediately
                if self._scan_count > 0:
                    current_time = datetime.datetime.now()
                    elapsed_time = current_time - self._start_t
                    
                    if elapsed_time < self._protocol.period():
                        # Not time for next scan yet, sleep briefly
                        time.sleep(0.1)
                        continue
                    
                    # Reset the start time for next period
                    self._start_t = current_time
                
                # Time for next scan
                if 'protocol_iterate_pre' in self._callbacks:
                    Clock.schedule_once(
                        lambda dt: self._callbacks['protocol_iterate_pre'](
                            remaining_scans=remaining_scans, 
                            interval=self._protocol.period()
                        ), 0
                    )
                
                # Initialize scan variables
                self._curr_step = 0
                if 'run_scan_pre' in self._callbacks:
                    self._callbacks['run_scan_pre']()
                
                self._go_to_step(step_idx=self._curr_step)
                self._scan_in_progress.set()
                self._auto_gain_countdown = self._autogain_settings['max_duration'].total_seconds()
                
                start_scan_time = datetime.datetime.now()
                # Run the scan loop - this will block until scan completes
                self._scan_loop()

                end_scan_time = datetime.datetime.now()
                scan_duration = end_scan_time - start_scan_time

                logger.info(f"Protocol scan {self._scan_count} completed in {scan_duration.total_seconds():.2f} seconds", extra={"force_error": True})
                
                # Scan completed - increment counter
                self._scan_count += 1
                logger.debug(f"[{self.LOGGER_NAME}] Scan {self._scan_count}/{self._n_scans} completed")
                
                if 'scan_iterate_post' in self._callbacks and self._callbacks['scan_iterate_post'] is not None:
                    Clock.schedule_once(lambda dt: self._callbacks['scan_iterate_post'](), 0)
                
                # Clear the flag so we know scan is done
                self._scan_in_progress.clear()
                
            except Exception as ex:
                logger.error(f"[Protocol] Error during run loop: {ex}", exc_info=True)
                self._cleanup()
                break
    
    def _scan_loop(self):
        """Scan loop - iterates through all protocol steps until scan completes.
        Returns when the scan is complete (all steps executed).
        """
        last_maintenance_time = time.monotonic()
        
        # Initial delay before first iteration
        time.sleep(0.1)
        
        while self._scan_in_progress.is_set() and not self._protocol_ended.is_set():
            try:
                # Periodic cleanup and watchdog logging for long runs
                now_mono = time.monotonic()
                if now_mono - last_maintenance_time > 60:
                    last_maintenance_time = now_mono
                    
                    # Force garbage collection
                    collected = gc.collect()
                    if collected > 0:
                        logger.info(f"[Scan Watchdog] GC collected {collected} objects")
                    
                    # Log queue depths
                    try:
                        protocol_queue_size = self.protocol_executor.protocol_queue_size()
                        logger.debug(f"[Scan Watchdog] Protocol queue: {protocol_queue_size}")
                    except Exception:
                        pass
                
                # Run one step iteration
                self._scan_iterate()
                
                # Small delay to prevent CPU throttling
                time.sleep(0.01)
                
            except Exception as ex:
                logger.error(f"[Scan] Error during scan loop: {ex}", exc_info=True)
                self._scan_in_progress.clear()
                break

    def _scan_iterate(self, dt=None):
        if self._protocol_ended.is_set():
            return

        if not self._video_write_finished.is_set():
            return
        
        if not self._scan_in_progress.is_set():
            return

        if not self._run_in_progress:
            return
        
        if self._autofocus_executor.in_progress():
            return
        
        remaining_scans=self.remaining_scans()
        if remaining_scans <= 0:
            return
        
        
        step = self._protocol.step(idx=self._curr_step)

        x_status = self._scope.get_target_status('X')
        y_status = self._scope.get_target_status('Y')
        z_status = self._scope.get_target_status('Z')

        # x_target = self._scope.get_target_pos('X')
        # y_target = self._scope.get_target_pos('Y')
        # z_target = self._scope.get_target_pos('Z')

        # x_pos = self._scope.get_current_position('X')
        # y_pos = self._scope.get_current_position('Y')
        # z_pos = self._scope.get_current_position('Z')

        # if (x_target != x_pos) or (y_target != y_pos) or (z_target != z_pos):
        #     return
    
        # Check if target location has not been reached yet
        if (not x_status) or (not y_status) or (not z_status) or self._scope.get_overshoot():
            return
        
        if not self._grease_redistribution_done:
            return
        
        if self._z_ui_update_func is not None:
            Clock.schedule_once(lambda dt: self._z_ui_update_func(float(step['Z'])), 0)

        # Set camera settings
        # self._io_executor.protocol_put(IOTask(
        #     action=self._scope.set_auto_gain,
        #     kwargs={
        #         "state": step['Auto_Gain'],
        #         "settings": self._autogain_settings,
        #     }
        # ))

        if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
            return
        
        self._scope.set_auto_gain(
            state=step['Auto_Gain'],
            settings=self._autogain_settings,
        )

        if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
            return

        self._led_on(color=step['Color'], illumination=step['Illumination'], block=True)

        if not step['Auto_Gain']:
            #self._io_executor.protocol_put(IOTask(action=self._scope.set_gain, args=(step['Gain'])))
            if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
                return
            self._scope.set_gain(step['Gain'])
            # 2023-12-18 Instead of using only auto gain, now it's auto gain + exp. If auto gain is enabled, then don't set exposure time
            #self._io_executor.protocol_put(IOTask(action=self._scope.set_exposure_time, args=(step['Exposure'])))
            if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
                return
            self._scope.set_exposure_time(step['Exposure'])

        if step['Auto_Gain'] and self._auto_gain_countdown > 0:
            self._auto_gain_countdown -= 0.1
        
        # If the autofocus is selected, is not currently running and has not completed, begin autofocus
        if step['Auto_Focus'] and not self._autofocus_executor.complete() and not self._autofocus_executor.in_progress():

            if 'autofocus_in_progress' in self._callbacks:
                self._callbacks['autofocus_in_progress']()

            af_executor_callbacks = {}
            if 'move_position' in self._callbacks:
                af_executor_callbacks['move_position'] = self._callbacks['move_position']

            if 'autofocus_completed' in self._callbacks:
                af_executor_callbacks['complete'] = self._callbacks['autofocus_completed']

            if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
                return
            
            #TODO: Make sure all of this IO is handled outside of Kivy main thread
            self._autofocus_executor.run(
                objective_id=step['Objective'],
                save_results_to_file=self._save_autofocus_data,
                results_dir=self._parent_dir,
                run_trigger_source=self._run_trigger_source,
                callbacks=af_executor_callbacks,
            )
            
            return
        
        # Still executing autofocus (not complete)
        if step['Auto_Focus'] and self._autofocus_executor.in_progress():
            return
        
        # if step['Auto_Focus'] and not self._autofocus_executor.in_progress():
        #     self._leds_off()
        
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

        if remaining_scans > 0:
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

                # TODO THREAD

                if step['Acquire'] == 'video':
                    self._video_write_finished.clear()

                step_start_time = time.time()
                capture_result = self._capture(
                    save_folder=save_folder,
                    step=step,
                    scan_count=self._scan_count,
                    output_format=output_format,
                    sum_count=step["Sum"],
                )

                self._stim_stop_event.set()
                
                # For videos, wait for write to complete before reporting step completion
                if step['Acquire'] == 'video':
                    logger.info(f"[SeqCapExec] Waiting for video write to complete...")
                    self._video_write_finished.wait(timeout=60)  # Wait up to 60s for write
                
                # Calculate total step time including write
                total_step_time = time.time() - step_start_time
                
                # Notify step completion for progress tracking
                if 'step_complete' in self._callbacks:
                    self._callbacks['step_complete'](
                        step_idx=self._curr_step,
                        step_name=step['Name'],
                        duration_seconds=total_step_time
                    )


                # Protocol record creation and adding is handled in capture method


            else:
                # Normally LEDs are turned off at the end of a capture. However, when not capturing, need to manually turn
                # off LEDs (such as in autofocus scan)
                self._leds_off()

        self._autofocus_executor.reset()
        # Disable autogain when moving between steps
        if step['Auto_Gain']:
            # self._io_executor.protocol_put(IOTask(
            #     action=self._scope.set_auto_gain,
            #     kwargs={
            #         "state":False,
            #         "settings":self._autogain_settings
            #     }
            # ))
            self._scope.set_auto_gain(state=False, settings=self._autogain_settings,)

        num_steps = self._protocol.num_steps()
        if self._curr_step < num_steps-1:

            # increment to the next step. Don't let it exceed the number of steps in the protocol
            self._curr_step = min(self._curr_step+1, num_steps-1)

            if 'update_step_number' in self._callbacks:
                Clock.schedule_once(lambda dt: self._callbacks['update_step_number'](self._curr_step+1), 0)
            self._go_to_step(step_idx=self._curr_step)
            return

        # At the end of a scan, if we've performed more than 100 AFs, cycle the Z-axis to re-distribute grease
        if self._autofocus_count >= 100:
            self._perform_grease_redistribution()
            self._autofocus_count = 0

        # Scan completed - clear the flag so run_loop knows to proceed
        # The run_loop will handle incrementing scan_count and callbacks
        self._scan_in_progress.clear()


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
            self._target_x_pos = sx
            Clock.schedule_once(lambda dt: self._callbacks['move_position']('X'), 0)

            self._scope.move_absolute_position('Y', sy)
            self._target_y_pos = sy
            Clock.schedule_once(lambda dt: self._callbacks['move_position']('Y'), 0)

            if z is not None:
                self._scope.move_absolute_position('Z', z)
                self._target_z_pos = z
                Clock.schedule_once(lambda dt: self._callbacks['move_position']('Z'), 0)
            #self._io_executor.protocol_put(IOTask(action=self._default_move_ex, args=(sx, sy, z), callback=self._default_move_callbacks_ex, cb_args=(z)))
        
        

    def _default_move_ex(self, sx, sy, z):
        self._scope.move_absolute_position('X', sx)
        self._scope.move_absolute_position('Y', sy)
        if z is not None:
            self._scope.move_absolute_position('Z', z)

    def _default_move_callbacks_ex(self, z):
        
        Clock.schedule_once(lambda dt: self._callbacks['move_position']('X'), 0)
        Clock.schedule_once(lambda dt: self._callbacks['move_position']('Y'), 0)

        if z is not None:
            Clock.schedule_once(lambda dt: self._callbacks['move_position']('Z'), 0)


    def _go_to_step(
        self,
        step_idx: int,
    ):
        if self._protocol_ended.is_set():
            return
        
        #if False:
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
        """Cancel any remaining scheduled events. 
        Note: With the loop-based approach, most work happens in executor threads,
        so there's less to unschedule than before.
        """
        if self._protocol_iterator is not None:
            Clock.unschedule(self._protocol_iterator)
        if self._scan_iterator is not None:
            Clock.unschedule(self._scan_iterator)


    def _leds_off(self):
        self._scope.leds_off()
        if 'leds_off' in self._callbacks:
            self._callbacks['leds_off']()

    
    def _led_on(self, color: str, illumination: float, block: bool=True):
        if self._protocol_ended.is_set():
            return
        
        self._scope.led_on(
            channel=self._scope.color2ch(color),
            mA=illumination,
            block=block
        )
        # Sleep for 5 ms to ensure that LED properly turns on before next action
        time.sleep(0.005)

        if 'led_state' in self._callbacks:
            self._callbacks['led_state'](layer=color, enabled=True)

        # else:
        #     self._io_executor.protocol_put(IOTask(
        #         action=self._scope.led_on,
        #         kwargs={
        #             "channel":self._scope.color2ch(color),
        #             "mA":illumination
        #         },
        #     ))


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
                    self._led_on(color=color, illumination=color_data['illumination'], block=True)
        else:
            raise NotImplementedError(f"Unsupported LEDs state at end value: {self._leds_state_at_end}")
        
        # Always return autofocus states to intial
        for layer, layer_data in self._original_autofocus_states.items():
            settings[layer]["autofocus"] = layer_data
            if self._callbacks['reset_autofocus_btns']:
                # Updates autofocus buttons to their prior states
                Clock.schedule_once(lambda dt: self._callbacks['reset_autofocus_btns'](), 0)

        if not self._disable_saving_artifacts:
            # Queue to close protocol execution record (should execute after last file/protocol record written)
            self.file_io_executor.protocol_put(IOTask(
                action=self._protocol_execution_record.complete
            ))
            #self._protocol_execution_record.complete()
            
            # Aggregate stimulation profiling data if any exists
            self.file_io_executor.protocol_put(IOTask(
                action=self._aggregate_stimulation_profiling_data
            ))

        if self._return_to_position is not None:
            self._default_move(
                px=self._return_to_position['x'],
                py=self._return_to_position['y'],
                z=self._return_to_position['z'],
            )

        
        self._scan_in_progress.clear()
        self._protocol_ended.set()

        self._stim_start_event.clear()
        self._stim_stop_event.set()
        
        # Restore normal process priority
        self._restore_process_priority()

        self._io_executor.protocol_end()
        self.file_io_executor.protocol_finish_then_end()
        self.protocol_executor.protocol_end()
        self.autofocus_io_executor.protocol_end()
        self.camera_executor.enable()

        self._io_executor.clear_protocol_pending()
        self.protocol_executor.clear_protocol_pending()

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
        if self._protocol_ended.is_set():
            return
        
        #if sum_count > step['']

        if not self._run_in_progress:
            return
        
        if not self.protocol_executor.is_protocol_running():
            return
        
        self.debug_counter += 1
        
        is_video = True if step['Acquire'] == "video" else False
        video_as_frames = self._video_as_frames

        if not step['Auto_Gain']:
            with self._scope.camera.update_camera_config():
                self._scope.set_gain(step['Gain'])
                self._scope.set_exposure_time(step['Exposure'])

        if self._scope.has_turret():
            objective_short_name = self._scope.get_objective_info(objective_id=step["Objective"])['short_name']
        else:
            objective_short_name = None
        
        # Build base name from protocol's custom root + step name
        try:
            capture_root = self._protocol.capture_root()
        except Exception:
            capture_root = ''

        combined_prefix = None
        if capture_root not in (None, ''):
            combined_prefix = f"{capture_root}_{step['Name']}"
        else:
            combined_prefix = step['Name']

        name = common_utils.generate_default_step_name(
            well_label=step['Well'],
            color=step['Color'],
            z_height_idx=step['Z-Slice'],
            scan_count=scan_count,
            custom_name_prefix=combined_prefix,
            objective_short_name=objective_short_name,
            tile_label=step['Tile'],
            video=is_video,
        )
        # Ensure the filename base has no invalid path characters
        try:
            name = Protocol.sanitize_step_name(input=name)
        except Exception:
            pass

        # Illuminate
        if self._scope.led:
            self._led_on(color=step['Color'], illumination=step['Illumination'], block=True)
            logger.info(f"[{self.LOGGER_NAME} ] scope.led_on({step['Color']}, {step['Illumination']})")
        else:
            logger.warning('LED controller not available.')

        # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
        # Grab image and save

        earliest_image_ts = datetime.datetime.now()
        # if 'update_scope_display' in self._callbacks:
        #     Clock.schedule_once(lambda dt: self._callbacks['update_scope_display'](), 0)
        #     sum_iteration_callback=lambda: Clock.schedule_once(lambda dt: self._callbacks['update_scope_display'](), 0)
        # else:
        sum_iteration_callback=None

        use_color = step['Color'] if step['False_Color'] else 'BF'

        if self._enable_image_saving == True:
            use_full_pixel_depth = self._image_capture_config['use_full_pixel_depth']

            accepted_gain_range = 0.001
            accepted_exp_range = 0.001

            # # Verify that we have the correct settings in the camera
            # real_gain = self._scope.get_gain()
            # if abs(real_gain - step['Gain']) > accepted_gain_range:
            #     self._scope.set_gain(step['Gain'])

            # real_exp = self._scope.get_exposure_time()
            # if abs(real_exp - step['Exposure']) > accepted_exp_range:
            #     self._scope.set_exposure_time(step['Exposure'])

            # Sleep for at least 100ms to ensure that the camera is ready for the next capture
            time.sleep(max(step['Exposure']/1000, 0.1))

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
                
                # Pause live UI to maximize throughput during recording
                # try:
                #     if 'pause_live_ui' in self._callbacks and callable(self._callbacks['pause_live_ui']):
                #         self._callbacks['pause_live_ui']()
                # except Exception:
                #     pass

                if video_as_frames:
                    save_folder = save_folder / f"{name}"

                else:    
                    output_file_loc = save_folder / f"{name}.mp4"

                if True:
                    start_ts = time.time()
                    stop_ts = start_ts + duration_sec
                    expected_frames = fps * duration_sec
                    captured_frames = 0
                    seconds_per_frame = 1.0 / fps
                    video_images = queue.Queue()

                    stim_threads = []

                    for color in step['Stim_Config']:
                        stim_config = step['Stim_Config'][color]
                        if stim_config['enabled']:
                            stim_thread = threading.Thread(target=self._stimulate, args=(color, stim_config, self._stim_start_event, self._stim_stop_event))
                            stim_threads.append(stim_thread)
                            stim_thread.start() 

                  
                    if "set_recording_title" in self._callbacks:
                        Clock.schedule_once(lambda dt: self._callbacks['set_recording_title'](progress=0), 0)

                    logger.info(f"Protocol-Video] Capturing video...")

                    progress = 0
                    ctypes.windll.winmm.timeBeginPeriod(1)
                    self._stim_stop_event.clear()
                    self._stim_start_event.set()

                    while time.time() < stop_ts:
                        curr_time = time.time()
                        progress = (curr_time - start_ts) / duration_sec * 100
                        if "set_recording_title" in self._callbacks:
                            Clock.schedule_once(lambda dt: self._callbacks['set_recording_title'](progress=progress), 0)

                        if not self.protocol_executor.is_protocol_running():
                            self._scope.leds_off()
                            if "reset_title" in self._callbacks:
                                Clock.schedule_once(lambda dt: self._callbacks['reset_title'](), 0)
                            return
                        
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

                            video_images.put_nowait((image, datetime.datetime.now()))
                            #video_images.append((image, datetime.datetime.now()))

                            captured_frames += 1
                        
                        # Some process is slowing the video-process down (getting fewer frames than expected if delay of seconds_per_frame), so a shorter sleep time can be used
                        time.sleep(seconds_per_frame*0.9)

                    ctypes.windll.winmm.timeEndPeriod(1)
                    self._stim_stop_event.set()
                    self._stim_start_event.clear()  # Reset start event for next step

                    for stim_thread in stim_threads:
                        stim_thread.join()
                    
                    calculated_fps = captured_frames//duration_sec

                    logger.info(f"Protocol-Video] Images present in video array: {not video_images.empty()}")
                    logger.info(f"Protocol-Video] Captured Frames: {captured_frames}")
                    logger.info(f"Protocol-Video] Video FPS: {calculated_fps}")
                    logger.info("Protocol-Video] Writing video...")

                    # Resume live UI after recording finishes
                    # try:
                    #     if 'resume_live_ui' in self._callbacks and callable(self._callbacks['resume_live_ui']):
                    #         self._callbacks['resume_live_ui']()
                    # except Exception:
                    #     pass

                    self._scope.leds_off()

                    # future = self.file_io_process_pool.submit(self._write_capture, 
                    #         is_video=is_video,
                    #         video_as_frames=video_as_frames,
                    #         video_images=video_images,
                    #         save_folder=save_folder,
                    #         use_color=use_color,
                    #         name=name,
                    #         calculated_fps=calculated_fps,
                    #         output_format=output_format,
                    #         step=step,
                    #         captured_image=None)
                    
                    # future.add_done_callback(logger.info("Protocol] Video writing complete"))

                    self.file_io_executor.protocol_put(IOTask(
                        action=self._write_capture,
                        kwargs={
                            "is_video": is_video,
                            "video_as_frames": video_as_frames,
                            "video_images": video_images,
                            "save_folder": save_folder,
                            "use_color": use_color,
                            "name": name,
                            "calculated_fps": calculated_fps,
                            "output_format": output_format,
                            "step": step,
                            "captured_image": None,
                            "step_index": self._curr_step,
                            "scan_count": self._scan_count,
                            "capture_time": datetime.datetime.now(),
                            "duration_sec": duration_sec,
                            "captured_frames": captured_frames
                        }
                    ))
                
                else:
                    try:
                        record_result = self._scope.camera.record_video_vfr(
                            output_file=output_file_loc,
                            duration_sec=duration_sec,
                            overlay_timestamp=True,
                            false_color=use_color
                        )

                    except Exception as e:
                        logger.error(f"Protocol-Video] Failed to record video: {e}")
                        try:
                            self._protocol_execution_record.add_step(
                                capture_result_file_name="unsaved",
                                step_name=step['Name'],
                                step_index=self._curr_step,
                                scan_count=self._scan_count,
                                timestamp=datetime.datetime.now(),
                                frame_count=0,
                                duration_sec=0.0
                            )
                        except Exception:
                            pass
                        return
                    finally:
                        self._video_write_finished.set()
                        stop_event.set()

                    if "reset_title" in self._callbacks:
                        Clock.schedule_once(lambda dt: self._callbacks['reset_title'](), 0)

                    logger.info("Protocol-Video] Video writing finished.")
                    logger.info(f"Protocol-Video] Video saved at {output_file_loc}")

                    stop_event.set()
                    self._scope.leds_off()
                    
                    


                    self._protocol_execution_record.add_step(
                        capture_result_file_name=output_file_loc,
                        step_name=step['Name'],
                        step_index=self._curr_step,
                        scan_count=self._scan_count,
                        timestamp=datetime.datetime.now(),
                        frame_count=record_result['frames_captured'],
                        duration_sec=duration_sec
                    )



            else:
                #time.sleep(self.sleep_time)
                captured_image = self._scope.get_image(
                    force_to_8bit=not use_full_pixel_depth,
                    earliest_image_ts=earliest_image_ts,
                    timeout=datetime.timedelta(seconds=1.0),
                    all_ones_check=True,
                    sum_count=sum_count,
                    sum_delay_s=step["Exposure"]/1000,
                    sum_iteration_callback=sum_iteration_callback,
                    force_new_capture=True,
                    new_capture_timeout=1
                )



                logger.info(f"Protocol Image Captured: {name}")
                #time.sleep(0.005)

                self.file_io_executor.protocol_put(IOTask(
                    action=self._write_capture,
                    kwargs={
                        "is_video": is_video,
                        "video_as_frames": video_as_frames,
                        "video_images": None,
                        "save_folder": save_folder,
                        "use_color": use_color,
                        "name": name,
                        "calculated_fps": None,
                        "output_format": output_format,
                        "step": step,
                        "captured_image": captured_image,
                        "step_index": self._curr_step,
                        "scan_count": self._scan_count,
                        "capture_time": datetime.datetime.now(),
                        "duration_sec": 0.0,
                        "captured_frames": 1
                    }
                ))
        
        else:
            self.file_io_executor.protocol_put(IOTask(
                action=self._write_capture,
                kwargs={
                    "step": step
                }
            ))

        
        self._scope.leds_off()

            

                
    
    def _write_capture(self,
                       is_video=False, 
                       video_as_frames=False, 
                       video_images: queue.Queue=None, 
                       save_folder=None,
                       use_color=None,
                       name=None,
                       calculated_fps=None,
                       output_format=None,
                       step=None,
                       captured_image=None,
                       step_index=None,
                       scan_count=None,
                       capture_time=None,
                       duration_sec=0.0,
                       captured_frames=1
                       ):
        
        if self._enable_image_saving == True:
            if is_video:
                if "set_writing_title" in self._callbacks:
                    Clock.schedule_once(lambda dt: self._callbacks['set_writing_title'](progress=0), 0)

                if video_as_frames:
                    frame_num = 0
                    capture_result = save_folder
                    if not save_folder.exists():
                        save_folder.mkdir(exist_ok=True, parents=True)

                    while not video_images.empty():

                        progress = frame_num / captured_frames * 100
                        if "set_writing_title" in self._callbacks:
                            Clock.schedule_once(lambda dt: self._callbacks['set_writing_title'](progress=progress), 0)

                        image_pair = video_images.get_nowait()
                        frame_num += 1

                        image = image_pair[0]
                        ts = image_pair[1]

                        del image_pair

                        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                        image_w_timestamp = image_utils.add_timestamp(image=image, timestamp_str=ts_str)

                        del image
                        video_images.task_done()

                        frame_name = f"{name}_Frame_{frame_num:04}"

                        output_file_loc = save_folder / f"{frame_name}.tiff"

                        metadata = {
                            "datetime": ts.strftime("%Y:%m:%d %H:%M:%S"),
                            "timestamp": ts.strftime("%Y:%m:%d %H:%M:%S.%f"),
                            "frame_num": frame_num
                        }
                        
                        try:
                            image_utils.write_tiff(
                                data=image_w_timestamp,
                                metadata=metadata,
                                file_loc=output_file_loc,
                                video_frame=True,
                                ome=False,
                                color=step['Color']
                            )
                        except Exception as e:
                            logger.error(f"Protocol-Video] Failed to write frame {frame_num}: {e}")

                    # Ensure queue is fully drained before deletion
                    try:
                        while not video_images.empty():
                            video_images.get_nowait()
                            video_images.task_done()
                    except Exception:
                        pass
                    # Queue is not empty, delete it and force garbage collection
                    del video_images
                    gc.collect()

                        

                else:
                    output_file_loc = save_folder / f"{name}.mp4v"
                    video_writer = VideoWriter(
                        output_file_loc=output_file_loc,
                        fps=calculated_fps,
                        include_timestamp_overlay=True
                    )
                    frame_num = 0
                    while not video_images.empty():
                        progress = frame_num / captured_frames * 100
                        if "set_writing_title" in self._callbacks:
                            Clock.schedule_once(lambda dt: self._callbacks['set_writing_title'](progress=progress), 0)

                        try:
                            image_pair = video_images.get_nowait()
                            video_writer.add_frame(image=image_pair[0], timestamp=image_pair[1])
                            del image_pair
                            video_images.task_done()
                            frame_num += 1
                        except Exception as e:
                            logger.error(f"Protocol-Video] FAILED TO WRITE FRAME: {e}")

                    # Ensure queue is fully drained before deletion
                    try:
                        while not video_images.empty():
                            video_images.get_nowait()
                            video_images.task_done()
                    except Exception:
                        pass
                    # Video images queue empty. Delete it and force garbage collection
                    del video_images

                    video_writer.finish()
                    #video_writer.test_video(str(output_file_loc))
                    del video_writer
                    gc.collect()
                    
                    capture_result = output_file_loc

                self._video_write_finished.set()
                if "reset_title" in self._callbacks:
                    Clock.schedule_once(lambda dt: self._callbacks['reset_title'](), 0)
                
                logger.info("Protocol-Video] Video writing finished.")
                logger.info(f"Protocol-Video] Video saved at {capture_result}")
            
            else:
                if captured_image is False:
                    return
                
                capture_result = self._scope.save_image(
                    array=captured_image,
                    save_folder=save_folder,
                    file_root=None,
                    append=name,
                    color=use_color,
                    tail_id_mode=None,
                    output_format=output_format,
                    true_color=step['Color'],
                    x=step['X'],
                    y=step['Y'],
                    z=step['Z']
                )

                del captured_image
                gc.collect()

                
                
                # result = self._scope.save_live_image(
                #     save_folder=save_folder,
                #     file_root=None,
                #     append=name,
                #     color=use_color,
                #     tail_id_mode=None,
                #     force_to_8bit=not use_full_pixel_depth,
                #     output_format=output_format,
                #     true_color=step['Color'],
                #     earliest_image_ts=earliest_image_ts,
                #     timeout=datetime.timedelta(seconds=1.0),
                #     all_ones_check=True,
                #     sum_count=sum_count,
                #     sum_delay_s=step["Exposure"]/1000,
                #     sum_iteration_callback=sum_iteration_callback,
                #     turn_off_all_leds_after=True,
                # )
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

        gc.collect()
        
        self._protocol_execution_record.add_step(
            capture_result_file_name=capture_result_filepath_name,
            step_name=name,
            step_index=step_index,
            scan_count=scan_count,
            timestamp=capture_time,
            frame_count=captured_frames if is_video else 1,
            duration_sec=duration_sec if is_video else 0.0
        )

        logger.info(f"Protocol-Writer] Added step to protocol execution record")

    

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

        # Check if we're running in test mode with specific configuration
        test_case = getattr(self, '_current_test_case', None)
        
        # Optional: reduce Windows timer quantum to 1 ms during stimulation
        time_period_set = False
        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
                time_period_set = True
            except Exception:
                time_period_set = False
            
            # Set thread priority based on test configuration or default
            use_realtime_priority = True
            if test_case and not test_case.thread_priority_realtime:
                use_realtime_priority = False
                logger.info(f"[STIMULATOR] {color} test disables realtime thread priority")
            
            if use_realtime_priority:
                # Set thread to real-time priority for best timing accuracy
                try:
                    kernel32 = ctypes.windll.kernel32
                    thread_handle = kernel32.GetCurrentThread()
                    # THREAD_PRIORITY_TIME_CRITICAL = 15
                    if kernel32.SetThreadPriority(thread_handle, 15):
                        logger.info(f"[STIMULATOR] {color} thread set to real-time priority")
                    else:
                        logger.warning(f"[STIMULATOR] {color} failed to set real-time priority")
                except Exception as e:
                    logger.warning(f"[STIMULATOR] {color} could not set thread priority: {e}")

        # DEBUG: Profiling timing data for led_on and led_off
        led_on_timings = []
        led_off_timings = []
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

            # Read flush setting - from test case if available, otherwise from settings
            if test_case:
                flush_enabled = test_case.led_flush_enabled
                logger.info(f"[STIMULATOR] {color} using test case flush setting: {flush_enabled}")
            else:
                flush_enabled = settings.get('led_flush_enabled', True)

            # Use fast path LED toggles if available via API
            def led_on_fast():
                #logger.info(f"[STIMULATOR] {color} LED ON")
                t_start = time.perf_counter()
                if profiling_enabled and start_epoch is not None:
                    prof_start = t_start
                if hasattr(self._scope, 'led_on_fast'):
                    self._scope.led_on_fast(channel=self._scope.color2ch(color=color), mA=illumination, flush=flush_enabled)
                else:
                    self._scope.led_on(channel=self._scope.color2ch(color=color), mA=illumination)
                t_end = time.perf_counter()
                if profiling_enabled and start_epoch is not None:
                    # Store: (absolute timestamp in ms, duration in ms, relative time since start in ms)
                    led_on_timings.append({
                        'start_time': (prof_start - start_epoch) * 1000.0,
                        'end_time': (t_end - start_epoch) * 1000.0,
                        'duration': (t_end - prof_start) * 1000.0
                    })
                return t_end - t_start  # Return measured latency

            def led_off_fast():
                #logger.info(f"[STIMULATOR] {color} LED OFF")
                t_start = time.perf_counter()
                if profiling_enabled and start_epoch is not None:
                    prof_start = t_start
                if hasattr(self._scope, 'led_off_fast'):
                    self._scope.led_off_fast(channel=self._scope.color2ch(color=color), flush=flush_enabled)
                else:
                    self._scope.led_off(channel=self._scope.color2ch(color=color))
                t_end = time.perf_counter()
                if profiling_enabled and start_epoch is not None:
                    # Store: (absolute timestamp in ms, duration in ms, relative time since start in ms)
                    led_off_timings.append({
                        'start_time': (prof_start - start_epoch) * 1000.0,
                        'end_time': (t_end - start_epoch) * 1000.0,
                        'duration': (t_end - prof_start) * 1000.0
                    })
                return t_end - t_start  # Return measured latency
            
            def update_global_on_latency(measured: float):
                """Thread-safe update of global ON latency average"""
                with self._led_latency_lock:
                    self._avg_led_on_latency = (self._latency_ema_alpha * measured + 
                                                 (1 - self._latency_ema_alpha) * self._avg_led_on_latency)
            
            def update_global_off_latency(measured: float):
                """Thread-safe update of global OFF latency average"""
                with self._led_latency_lock:
                    self._avg_led_off_latency = (self._latency_ema_alpha * measured + 
                                                  (1 - self._latency_ema_alpha) * self._avg_led_off_latency)
            
            def get_avg_on_latency() -> float:
                """Thread-safe read of global ON latency average"""
                with self._led_latency_lock:
                    return self._avg_led_on_latency
            
            def get_avg_off_latency() -> float:
                """Thread-safe read of global OFF latency average"""
                with self._led_latency_lock:
                    return self._avg_led_off_latency

            start_event.wait()
            start_epoch = time.perf_counter()
            logger.info(f"[STIMULATOR] stim_start_event set for {color} at t=0.000 ms")
            
            # Apply compensations based on test case configuration
            use_pulse_width_compensation = True
            use_conservative_off = True
            use_on_compensation = True
            use_off_compensation = True
            
            if test_case:
                use_pulse_width_compensation = test_case.use_pulse_width_compensation
                use_conservative_off = test_case.use_conservative_off_estimate
                use_on_compensation = test_case.use_on_latency_compensation
                use_off_compensation = test_case.use_off_latency_compensation
                logger.info(f"[STIMULATOR] {color} test case compensations: "
                           f"pulse_width={use_pulse_width_compensation}, "
                           f"conservative_off={use_conservative_off}, "
                           f"on_comp={use_on_compensation}, "
                           f"off_comp={use_off_compensation}")
            
            # Compensate pulse width for OFF command latency
            # This ensures the LED is on for the correct duration
            compensated_pulse_s = pulse_s
            if use_pulse_width_compensation:
                initial_off_latency = get_avg_off_latency()
                if use_conservative_off:
                    # Use 2x average OFF latency for safety
                    initial_off_latency = initial_off_latency * 2.0
                compensated_pulse_s = pulse_s - initial_off_latency
                if compensated_pulse_s < 0:
                    logger.warning(f"[STIMULATOR] {color} OFF latency ({initial_off_latency*1000:.3f}ms) exceeds pulse width ({pulse_width}ms)!")
                    compensated_pulse_s = pulse_s * 0.5  # Use 50% of target as fallback
                logger.info(f"[STIMULATOR] {color} pulse width compensation: target={pulse_width}ms, OFF_latency={initial_off_latency*1000:.3f}ms, compensated={compensated_pulse_s*1000:.3f}ms")
            else:
                logger.info(f"[STIMULATOR] {color} pulse width compensation disabled by test case")

            end_reason = "pulse_count_reached"

            for i in range(pulse_count):
                if stop_event.is_set():
                    elapsed = (time.perf_counter() - start_epoch) * 1000.0
                    logger.info(f"[STIMULATOR] {color} stop event set at t={elapsed:.3f} ms, ending stimulation.")
                    end_reason = "stop_event_set"
                    break

                # Target times for this pulse
                on_time = start_epoch + i * period_s
                next_period_time = start_epoch + (i + 1) * period_s

                # Compensate for expected ON command latency - fire early (if enabled)
                compensated_on_time = on_time
                if use_on_compensation:
                    avg_on_latency = get_avg_on_latency()
                    compensated_on_time = on_time - avg_on_latency
                    
                    # Diagnostic: Check if compensation is causing issues
                    if compensated_on_time < (time.perf_counter() if i > 0 else start_epoch):
                        logger.warning(f"[STIMULATOR] {color} Pulse {i}: Compensated ON time is in the past! (compensation={avg_on_latency*1000:.3f}ms)")
                else:
                    compensated_on_time = on_time

                # Sleep until compensated on_time (coarse) then spin
                while True:
                    now = time.perf_counter()
                    remaining = compensated_on_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        # short busy-wait to reduce jitter
                        pass

                measured_on_latency = led_on_fast()
                
                # Base OFF timing on when ON actually completed (after flush)
                # Use compensated pulse width that accounts for OFF latency
                actual_on_complete_time = time.perf_counter()
                
                # Calculate absolute target OFF time (from ON start, not ON complete)
                # This ensures we never exceed the target pulse width
                absolute_target_off_time = on_time + pulse_s
                
                # Calculate ideal OFF time based on compensations (if enabled)
                if use_off_compensation and use_pulse_width_compensation:
                    # Get OFF latency estimate - be conservative to prevent overshooting
                    avg_off_latency = get_avg_off_latency()
                    
                    # Use conservative estimate if enabled
                    conservative_off_latency = avg_off_latency
                    if use_conservative_off:
                        conservative_off_latency = avg_off_latency * 2.0  # 2x is conservative
                    if conservative_off_latency < 0.001:  # At least 1ms buffer
                        conservative_off_latency = 0.001
                    
                    current_compensated_pulse_s = pulse_s - conservative_off_latency
                    if current_compensated_pulse_s < 0:
                        logger.warning(f"[STIMULATOR] {color} Pulse {i}: Conservative OFF latency ({conservative_off_latency*1000:.3f}ms) > pulse width ({pulse_width}ms)!")
                        current_compensated_pulse_s = pulse_s * 0.3  # Very conservative fallback
                    
                    # Calculate ideal OFF time from ON completion
                    ideal_off_time = actual_on_complete_time + current_compensated_pulse_s
                    
                    # SAFETY: Use whichever is sooner - ideal time or absolute max time
                    # This prevents overshooting pulse width if ON took too long
                    target_off_time = min(ideal_off_time, absolute_target_off_time)
                    
                    # Additional safety: ensure we leave enough time for OFF latency
                    # Even in worst case, LED shouldn't be on longer than target + conservative margin
                    worst_case_off_complete = target_off_time + conservative_off_latency
                    if worst_case_off_complete > absolute_target_off_time:
                        # Fire OFF even earlier to account for potential high latency
                        adjustment = worst_case_off_complete - absolute_target_off_time
                        target_off_time -= adjustment
                        if target_off_time < actual_on_complete_time:
                            target_off_time = actual_on_complete_time  # Can't fire before ON completes
                else:
                    # No compensation - just use target pulse width from ON complete
                    target_off_time = actual_on_complete_time + pulse_s
                
                # Check if we're already past the absolute target
                now = time.perf_counter()
                if now >= absolute_target_off_time:
                    logger.warning(f"[STIMULATOR] {color} Pulse {i}: Already {(now-absolute_target_off_time)*1000:.3f}ms past target OFF time due to ON latency!")
                    target_off_time = now  # Fire immediately
                
                # Compensate for expected OFF command latency - fire early
                compensated_off_time = target_off_time
                
                # Compensate for ON command latency in the next period
                # If ON took longer than expected, start next pulse earlier to maintain frequency
                on_latency = actual_on_complete_time - on_time
                next_period_time = start_epoch + (i + 1) * period_s - on_latency
                
                # Diagnostic: Check if we're violating period constraints
                if i > 0:
                    actual_period = actual_on_complete_time - (start_epoch + (i - 1) * period_s)
                    expected_period = period_s
                    if abs(actual_period - expected_period) > 0.001:  # More than 1ms off
                        logger.warning(f"[STIMULATOR] {color} Pulse {i}: Period deviation: expected={expected_period*1000:.3f}ms, actual={actual_period*1000:.3f}ms, error={abs(actual_period-expected_period)*1000:.3f}ms")

                pulses_executed += 1

                # Sleep/spin until compensated off_time
                remaining_before_off = compensated_off_time - time.perf_counter()
                if remaining_before_off < -0.001:  # More than 1ms late
                    logger.warning(f"[STIMULATOR] {color} Pulse {i}: OFF command is {abs(remaining_before_off)*1000:.3f}ms late!")
                
                while True:
                    now = time.perf_counter()
                    remaining = compensated_off_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        pass

                measured_off_latency = led_off_fast()
                
                # Diagnostic: Check if OFF latency is significantly different from ON
                if abs(measured_off_latency - measured_on_latency) > 0.001:  # More than 1ms difference
                    logger.info(f"[STIMULATOR] {color} Pulse {i}: Latency difference: ON={measured_on_latency*1000:.3f}ms, OFF={measured_off_latency*1000:.3f}ms, diff={abs(measured_off_latency-measured_on_latency)*1000:.3f}ms")

                # Update global running averages during the inter-pulse wait period
                # This is the least time-critical section, so lock acquisition won't affect timing
                update_global_on_latency(measured_on_latency)
                update_global_off_latency(measured_off_latency)

                # Maintain period with latency compensation; wait until next_period_time
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
            if profiling_enabled and (led_on_timings or led_off_timings) and self._run_dir is not None:
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
                        f.write(f"LED Flush Enabled: {flush_enabled}\n")
                        f.write(f"End Reason: {end_reason}\n")
                        f.write("=" * 60 + "\n\n")
                        
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
                        
                        # Calculate actual LED on time (from end of led_on command to end of led_off command)
                        # This represents when ON flush completes to when OFF flush completes
                        if led_on_timings and led_off_timings and len(led_on_timings) == len(led_off_timings):
                            f.write("ACTUAL LED ON TIME (from Windows side):\n")
                            f.write("-" * 60 + "\n")
                            actual_on_times = []
                            for idx in range(len(led_on_timings)):
                                actual_on = led_off_timings[idx]['end_time'] - led_on_timings[idx]['end_time']
                                actual_on_times.append(actual_on)
                            
                            actual_array = np.array(actual_on_times)
                            actual_mean = np.mean(actual_array)
                            actual_std = np.std(actual_array)
                            actual_min = np.min(actual_array)
                            actual_max = np.max(actual_array)
                            
                            f.write(f"Count:            {len(actual_on_times)}\n")
                            f.write(f"Mean:             {actual_mean:.6f} ms\n")
                            f.write(f"Std Dev:          {actual_std:.6f} ms\n")
                            f.write(f"Min:              {actual_min:.6f} ms\n")
                            f.write(f"Max:              {actual_max:.6f} ms\n")
                            f.write(f"Target:           {pulse_width} ms\n")
                            f.write(f"Deviation:        {actual_mean - pulse_width:.6f} ms\n")
                            
                            # Detect outliers using 3-sigma rule
                            outlier_threshold = actual_mean + 3 * actual_std
                            outliers = [t for t in actual_on_times if t > outlier_threshold]
                            f.write(f"Outliers (>3σ):   {len(outliers)} ({len(outliers)/len(actual_on_times)*100:.2f}%)\n")
                            if outliers:
                                f.write(f"Outlier values:   {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                            
                            f.write("\nIndividual Actual LED ON Times:\n")
                            for idx, actual_on in enumerate(actual_on_times):
                                f.write(f"  Pulse {idx}: {actual_on:.6f} ms (Target: {pulse_width} ms, Deviation: {actual_on - pulse_width:.6f} ms)\n")
                            f.write("\n")
                        
                        # Calculate frequency accuracy (time between consecutive LED ON start times)
                        if led_on_timings and len(led_on_timings) > 1:
                            f.write("FREQUENCY ACCURACY:\n")
                            f.write("-" * 60 + "\n")
                            periods = []
                            actual_frequencies = []
                            target_period_ms = 1000.0 / frequency  # Convert Hz to ms
                            
                            for idx in range(1, len(led_on_timings)):
                                period = led_on_timings[idx]['start_time'] - led_on_timings[idx-1]['start_time']
                                periods.append(period)
                                actual_freq = 1000.0 / period  # Convert ms to Hz
                                actual_frequencies.append(actual_freq)
                            
                            period_array = np.array(periods)
                            freq_array = np.array(actual_frequencies)
                            
                            period_mean = np.mean(period_array)
                            period_std = np.std(period_array)
                            freq_mean = np.mean(freq_array)
                            freq_std = np.std(freq_array)
                            
                            f.write(f"Count:            {len(periods)}\n")
                            f.write(f"Target Frequency: {frequency} Hz (Period: {target_period_ms:.6f} ms)\n")
                            f.write(f"Actual Frequency: {freq_mean:.6f} Hz (Period: {period_mean:.6f} ms)\n")
                            f.write(f"Frequency Std:    {freq_std:.6f} Hz\n")
                            f.write(f"Period Std:       {period_std:.6f} ms\n")
                            f.write(f"Frequency Error:  {freq_mean - frequency:.6f} Hz ({(freq_mean - frequency)/frequency*100:.3f}%)\n")
                            f.write(f"Period Min:       {np.min(period_array):.6f} ms\n")
                            f.write(f"Period Max:       {np.max(period_array):.6f} ms\n")
                            
                            # Detect outliers in period timing
                            outlier_threshold = period_mean + 3 * period_std
                            outliers = [p for p in periods if p > outlier_threshold or p < period_mean - 3 * period_std]
                            f.write(f"Outliers (>3σ):   {len(outliers)} ({len(outliers)/len(periods)*100:.2f}%)\n")
                            if outliers:
                                f.write(f"Outlier periods:  {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                            
                            f.write("\nIndividual Period Measurements:\n")
                            for idx, (period, freq) in enumerate(zip(periods, actual_frequencies)):
                                f.write(f"  Period {idx}: {period:.6f} ms ({freq:.6f} Hz, Error: {freq - frequency:.6f} Hz)\n")
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
                        
                        # Actual LED on time
                        if led_on_timings and led_off_timings and len(led_on_timings) == len(led_off_timings):
                            actual_on_times = [led_off_timings[i]['start_time'] - led_on_timings[i]['end_time'] for i in range(len(led_on_timings))]
                            actual_mean = np.mean(actual_on_times)
                            actual_std = np.std(actual_on_times)
                            logger.info(f"[STIMULATOR] ACTUAL LED ON TIME - Mean: {actual_mean:.6f} ms, Std: {actual_std:.6f} ms, Target: {pulse_width} ms, Deviation: {actual_mean - pulse_width:.6f} ms")
                            for idx, actual_on in enumerate(actual_on_times):
                                logger.info(f"[STIMULATOR] ACTUAL Pulse {idx}: {actual_on:.6f} ms (Deviation: {actual_on - pulse_width:.6f} ms)")
                    except Exception as log_error:
                        logger.error(f"[STIMULATOR] Failed to log profiling data: {log_error}")


    def _aggregate_stimulation_profiling_data(self):
        """Aggregate all stimulation profiling data from individual color files into a global summary."""
        try:
            if self._run_dir is None:
                return
            
            profile_dir = self._run_dir / "stimulation_profile"
            if not profile_dir.exists():
                return
            
            # Find all individual profiling files
            profile_files = list(profile_dir.glob("stimulation_profile_*.txt"))
            if not profile_files:
                return
            
            logger.info(f"[STIMULATOR] Aggregating {len(profile_files)} stimulation profile files...")
            
            # Parse all profile files
            all_data = {}
            for profile_file in profile_files:
                try:
                    color = self._parse_profile_file(profile_file, all_data)
                except Exception as e:
                    logger.error(f"[STIMULATOR] Failed to parse {profile_file.name}: {e}")
                    continue
            
            if not all_data:
                logger.warning("[STIMULATOR] No valid profiling data found to aggregate")
                return
            
            # Generate aggregate summary report
            summary_file = profile_dir / "stimulation_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("STIMULATION PROFILING SUMMARY - ALL COLORS\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Total Profile Files: {len(profile_files)}\n")
                f.write("=" * 80 + "\n\n")
                
                # Summary for each color
                for color, data in sorted(all_data.items()):
                    f.write(f"\n{'='*80}\n")
                    f.write(f"COLOR: {color}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    f.write(f"Frequency: {data['frequency']} Hz\n")
                    f.write(f"Target Pulse Width: {data['pulse_width']} ms\n")
                    f.write(f"Total Pulses Executed: {data['total_pulses']}\n")
                    f.write(f"Total Files: {data['file_count']}\n")
                    
                    # Show flush enabled state
                    if data['flush_enabled_files']:
                        flush_enabled_count = sum(1 for v in data['flush_enabled_files'] if v.lower() == 'true')
                        flush_disabled_count = len(data['flush_enabled_files']) - flush_enabled_count
                        f.write(f"LED Flush Enabled: {flush_enabled_count}/{data['file_count']} files")
                        if flush_disabled_count > 0:
                            f.write(f" ({flush_disabled_count} files with flush disabled)")
                        f.write("\n")
                    f.write("\n")
                    
                    # LED ON command timing
                    if data['led_on_durations']:
                        f.write("LED ON COMMAND TIMINGS:\n")
                        f.write("-" * 80 + "\n")
                        on_array = np.array(data['led_on_durations'])
                        on_mean = np.mean(on_array)
                        on_std = np.std(on_array)
                        on_min = np.min(on_array)
                        on_max = np.max(on_array)
                        
                        f.write(f"Count:            {len(on_array)}\n")
                        f.write(f"Mean:             {on_mean:.6f} ms\n")
                        f.write(f"Std Dev:          {on_std:.6f} ms\n")
                        f.write(f"Min:              {on_min:.6f} ms\n")
                        f.write(f"Max:              {on_max:.6f} ms\n")
                        
                        outlier_threshold = on_mean + 3 * on_std
                        outliers = [t for t in data['led_on_durations'] if t > outlier_threshold]
                        f.write(f"Outliers (>3σ):   {len(outliers)} ({len(outliers)/len(on_array)*100:.2f}%)\n")
                        if outliers:
                            f.write(f"Outlier values:   {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                        f.write("\n")
                    
                    # LED OFF command timing
                    if data['led_off_durations']:
                        f.write("LED OFF COMMAND TIMINGS:\n")
                        f.write("-" * 80 + "\n")
                        off_array = np.array(data['led_off_durations'])
                        off_mean = np.mean(off_array)
                        off_std = np.std(off_array)
                        off_min = np.min(off_array)
                        off_max = np.max(off_array)
                        
                        f.write(f"Count:            {len(off_array)}\n")
                        f.write(f"Mean:             {off_mean:.6f} ms\n")
                        f.write(f"Std Dev:          {off_std:.6f} ms\n")
                        f.write(f"Min:              {off_min:.6f} ms\n")
                        f.write(f"Max:              {off_max:.6f} ms\n")
                        
                        outlier_threshold = off_mean + 3 * off_std
                        outliers = [t for t in data['led_off_durations'] if t > outlier_threshold]
                        f.write(f"Outliers (>3σ):   {len(outliers)} ({len(outliers)/len(off_array)*100:.2f}%)\n")
                        if outliers:
                            f.write(f"Outlier values:   {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                        f.write("\n")
                    
                    # Actual LED on time (from Windows side)
                    if data['actual_on_times']:
                        f.write("ACTUAL LED ON TIME (from Windows side):\n")
                        f.write("-" * 80 + "\n")
                        actual_array = np.array(data['actual_on_times'])
                        actual_mean = np.mean(actual_array)
                        actual_std = np.std(actual_array)
                        actual_min = np.min(actual_array)
                        actual_max = np.max(actual_array)
                        
                        f.write(f"Count:            {len(actual_array)}\n")
                        f.write(f"Mean:             {actual_mean:.6f} ms\n")
                        f.write(f"Std Dev:          {actual_std:.6f} ms\n")
                        f.write(f"Min:              {actual_min:.6f} ms\n")
                        f.write(f"Max:              {actual_max:.6f} ms\n")
                        f.write(f"Target:           {data['pulse_width']} ms\n")
                        f.write(f"Deviation:        {actual_mean - data['pulse_width']:.6f} ms\n")
                        
                        outlier_threshold = actual_mean + 3 * actual_std
                        outliers = [t for t in data['actual_on_times'] if t > outlier_threshold]
                        f.write(f"Outliers (>3σ):   {len(outliers)} ({len(outliers)/len(actual_array)*100:.2f}%)\n")
                        if outliers:
                            f.write(f"Outlier values:   {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                        f.write("\n")
                    
                    # Frequency accuracy
                    if data['frequency_periods']:
                        f.write("FREQUENCY ACCURACY:\n")
                        f.write("-" * 80 + "\n")
                        period_array = np.array(data['frequency_periods'])
                        freq_array = 1000.0 / period_array  # Convert periods (ms) to frequencies (Hz)
                        
                        period_mean = np.mean(period_array)
                        period_std = np.std(period_array)
                        freq_mean = np.mean(freq_array)
                        freq_std = np.std(freq_array)
                        target_period_ms = 1000.0 / data['frequency']
                        
                        f.write(f"Count:            {len(period_array)}\n")
                        f.write(f"Target Frequency: {data['frequency']} Hz (Period: {target_period_ms:.6f} ms)\n")
                        f.write(f"Actual Frequency: {freq_mean:.6f} Hz (Period: {period_mean:.6f} ms)\n")
                        f.write(f"Frequency Std:    {freq_std:.6f} Hz\n")
                        f.write(f"Period Std:       {period_std:.6f} ms\n")
                        f.write(f"Frequency Error:  {freq_mean - data['frequency']:.6f} Hz ({(freq_mean - data['frequency'])/data['frequency']*100:.3f}%)\n")
                        f.write(f"Period Min:       {np.min(period_array):.6f} ms\n")
                        f.write(f"Period Max:       {np.max(period_array):.6f} ms\n")
                        
                        outlier_threshold_high = period_mean + 3 * period_std
                        outlier_threshold_low = period_mean - 3 * period_std
                        outliers = [p for p in data['frequency_periods'] if p > outlier_threshold_high or p < outlier_threshold_low]
                        f.write(f"Outliers (>3σ):   {len(outliers)} ({len(outliers)/len(period_array)*100:.2f}%)\n")
                        if outliers:
                            f.write(f"Outlier periods:  {[f'{o:.6f}' for o in sorted(outliers)]}\n")
                        f.write("\n")
                
                # Global statistics across all colors
                f.write(f"\n{'='*80}\n")
                f.write("GLOBAL STATISTICS (ALL COLORS COMBINED)\n")
                f.write(f"{'='*80}\n\n")
                
                all_led_on = []
                all_led_off = []
                all_actual = []
                all_periods = []
                
                for color, data in all_data.items():
                    all_led_on.extend(data['led_on_durations'])
                    all_led_off.extend(data['led_off_durations'])
                    all_actual.extend(data['actual_on_times'])
                    all_periods.extend(data['frequency_periods'])
                
                if all_led_on:
                    f.write("ALL LED ON COMMANDS:\n")
                    on_array = np.array(all_led_on)
                    f.write(f"  Total Count:    {len(on_array)}\n")
                    f.write(f"  Mean:           {np.mean(on_array):.6f} ms\n")
                    f.write(f"  Std Dev:        {np.std(on_array):.6f} ms\n")
                    f.write(f"  Min:            {np.min(on_array):.6f} ms\n")
                    f.write(f"  Max:            {np.max(on_array):.6f} ms\n\n")
                
                if all_led_off:
                    f.write("ALL LED OFF COMMANDS:\n")
                    off_array = np.array(all_led_off)
                    f.write(f"  Total Count:    {len(off_array)}\n")
                    f.write(f"  Mean:           {np.mean(off_array):.6f} ms\n")
                    f.write(f"  Std Dev:        {np.std(off_array):.6f} ms\n")
                    f.write(f"  Min:            {np.min(off_array):.6f} ms\n")
                    f.write(f"  Max:            {np.max(off_array):.6f} ms\n\n")
                
                if all_actual:
                    f.write("ALL ACTUAL LED ON TIMES:\n")
                    actual_array = np.array(all_actual)
                    f.write(f"  Total Count:    {len(actual_array)}\n")
                    f.write(f"  Mean:           {np.mean(actual_array):.6f} ms\n")
                    f.write(f"  Std Dev:        {np.std(actual_array):.6f} ms\n")
                    f.write(f"  Min:            {np.min(actual_array):.6f} ms\n")
                    f.write(f"  Max:            {np.max(actual_array):.6f} ms\n\n")
                
                if all_periods:
                    f.write("ALL FREQUENCY PERIODS:\n")
                    period_array = np.array(all_periods)
                    mean_period = np.mean(period_array)
                    # Calculate actual frequency from period (period in ms, so *1000 for Hz)
                    actual_freq = 1000.0 / mean_period if mean_period > 0 else 0
                    f.write(f"  Total Count:    {len(period_array)}\n")
                    f.write(f"  Mean Period:    {mean_period:.6f} ms\n")
                    f.write(f"  Actual Freq:    {actual_freq:.6f} Hz\n")
                    f.write(f"  Std Dev:        {np.std(period_array):.6f} ms\n")
                    f.write(f"  Min Period:     {np.min(period_array):.6f} ms\n")
                    f.write(f"  Max Period:     {np.max(period_array):.6f} ms\n")
            
            logger.info(f"[STIMULATOR] Aggregated profiling summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"[STIMULATOR] Failed to aggregate stimulation profiling data: {e}")


    def _parse_profile_file(self, profile_file: pathlib.Path, all_data: dict) -> str:
        """Parse a single stimulation profile file and add data to all_data dict."""
        # Extract color from filename: stimulation_profile_Red_20260128_123456_789.txt
        filename_parts = profile_file.stem.split('_')
        if len(filename_parts) < 3:
            return None
        color = filename_parts[2]
        
        # Initialize color data structure if needed
        if color not in all_data:
            all_data[color] = {
                'frequency': 0,
                'pulse_width': 0,
                'total_pulses': 0,
                'file_count': 0,
                'led_on_durations': [],
                'led_off_durations': [],
                'actual_on_times': [],
                'frequency_periods': [],
                'flush_enabled_files': [],  # Track which files had flush enabled
            }
        
        with open(profile_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse metadata and data by tracking which section we're in
        current_section = None
        for line in content.split('\n'):
            # Track which section we're in
            if 'Individual LED ON Events' in line:
                current_section = 'led_on'
            elif 'Individual LED OFF Events' in line:
                current_section = 'led_off'
            elif 'Individual Actual LED ON Times' in line:
                current_section = 'actual'
            elif 'Individual Period Measurements' in line:
                current_section = 'frequency'
            
            # Parse metadata
            if line.startswith('Frequency:'):
                all_data[color]['frequency'] = float(line.split(':')[1].strip().split()[0])
            elif line.startswith('Pulse Width:'):
                all_data[color]['pulse_width'] = float(line.split(':')[1].strip().split()[0])
            elif line.startswith('Pulses Executed:'):
                all_data[color]['total_pulses'] += int(line.split(':')[1].strip())
            elif line.startswith('LED Flush Enabled:'):
                flush_value = line.split(':')[1].strip()
                all_data[color]['flush_enabled_files'].append(flush_value)
            
            # Parse timing data based on current section
            elif 'Duration=' in line and 'Pulse' in line:
                # Extract duration from lines like: "  Pulse 0: Start=10.123 ms, End=10.456 ms, Duration=0.333 ms"
                duration_part = line.split('Duration=')[1].split('ms')[0].strip()
                duration = float(duration_part)
                
                if current_section == 'led_on':
                    all_data[color]['led_on_durations'].append(duration)
                elif current_section == 'led_off':
                    all_data[color]['led_off_durations'].append(duration)
            
            elif 'Pulse' in line and '(Target:' in line and 'Deviation:' in line:
                # Extract actual on time from lines like: "  Pulse 0: 10.123 ms (Target: 10 ms, Deviation: 0.123 ms)"
                if current_section == 'actual':
                    actual_part = line.split(':')[1].split('ms')[0].strip()
                    actual = float(actual_part)
                    all_data[color]['actual_on_times'].append(actual)
            
            elif 'Period' in line and 'Hz' in line and 'Error:' in line:
                # Extract period from lines like: "  Period 0: 333.333 ms (3.000 Hz, Error: 0.000 Hz)"
                if current_section == 'frequency':
                    period_part = line.split(':')[1].split('ms')[0].strip()
                    period = float(period_part)
                    all_data[color]['frequency_periods'].append(period)
        
        all_data[color]['file_count'] += 1
        return color
