# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import enum
import pathlib
import shutil
import time
import sys
import ctypes
import typing

import numpy as np
import cv2
import copy
import gc
import queue

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
        @staticmethod
        def unschedule(callback): pass

from modules.lumascope_api import Lumascope

import modules.image_utils as image_utils

import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations

import modules.labware_loader as labware_loader
from modules.autofocus_executor import AutofocusExecutor
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord


class SequencedCaptureRunMode(enum.Enum):
    FULL_PROTOCOL = 'full_protocol'
    SINGLE_SCAN = 'single_scan'
    SINGLE_ZSTACK = 'single_zstack'
    SINGLE_AUTOFOCUS_SCAN = 'single_autofocus_scan'
    SINGLE_AUTOFOCUS = 'single_autofocus'
from modules.video_writer import VideoWriter
from modules.sequential_io_executor import SequentialIOExecutor, IOTask
from lvp_logger import logger
from concurrent.futures import ProcessPoolExecutor
import threading

import modules.app_context as _app_ctx
from modules.settings_init import settings


class ProtocolState(enum.Enum):
    """Protocol execution state machine.

    Valid transitions:
        IDLE     -> RUNNING              (run() called)
        RUNNING  -> SCANNING             (scan started)
        RUNNING  -> COMPLETING           (all scans done, cleanup starting)
        RUNNING  -> ERROR                (unrecoverable error)
        SCANNING -> RUNNING              (scan finished, back to inter-scan wait)
        SCANNING -> COMPLETING           (abort/error during scan)
        SCANNING -> ERROR                (unrecoverable error during scan)
        COMPLETING -> IDLE               (cleanup finished)
        ERROR    -> IDLE                 (cleanup finished after error)
    """

    IDLE = "idle"
    RUNNING = "running"
    SCANNING = "scanning"
    COMPLETING = "completing"
    ERROR = "error"


# Allowed state transitions: {from_state: {set of valid to_states}}
_PROTOCOL_STATE_TRANSITIONS: dict[ProtocolState, set[ProtocolState]] = {
    ProtocolState.IDLE: {ProtocolState.RUNNING},
    ProtocolState.RUNNING: {ProtocolState.SCANNING, ProtocolState.COMPLETING, ProtocolState.ERROR},
    ProtocolState.SCANNING: {ProtocolState.RUNNING, ProtocolState.COMPLETING, ProtocolState.ERROR},
    ProtocolState.COMPLETING: {ProtocolState.IDLE},
    ProtocolState.ERROR: {ProtocolState.IDLE},
}


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
        self.camera_executor = camera_executor
        self.autofocus_io_executor = autofocus_io_executor
        self._z_ui_update_func = z_ui_update_func
        self._scan_in_progress = threading.Event()
        self._protocol_ended = threading.Event()
        self._cleanup_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._cpu_pool = cpu_pool
        self._video_write_finished = threading.Event()
        self._video_write_finished.set()
        self._stim_start_event = threading.Event()
        self._stim_stop_event = threading.Event()

        if autofocus_executor is None:
            self._autofocus_executor = AutofocusExecutor(
                scope=scope,
                camera_executor=camera_executor,
                io_executor=io_executor,
                file_io_executor=file_io_executor,
                autofocus_executor=autofocus_io_executor,
                use_kivy_clock=False,
            )
        else:
            self._autofocus_executor = autofocus_executor

        self._scope = scope
        self._run_trigger_source = None
        self._state = ProtocolState.IDLE
        self._reset_vars()
        self._grease_redistribution_done = True


    def set_scope(self, scope: Lumascope):
        self._scope = scope

    def _set_state(self, new_state: ProtocolState) -> None:
        """Transition to *new_state* with validation.

        Raises ``ValueError`` if the transition is not allowed by
        ``_PROTOCOL_STATE_TRANSITIONS``.
        """
        old_state = self._state
        if old_state == new_state:
            return  # no-op
        allowed = _PROTOCOL_STATE_TRANSITIONS.get(old_state, set())
        if new_state not in allowed:
            msg = (
                f"[{self.LOGGER_NAME}] Invalid state transition: "
                f"{old_state.value} -> {new_state.value} "
                f"(allowed: {', '.join(s.value for s in allowed)})"
            )
            logger.error(msg)
            raise ValueError(msg)
        self._state = new_state
        logger.debug(
            f"[{self.LOGGER_NAME}] State: {old_state.value} -> {new_state.value}"
        )

    @property
    def protocol_state(self) -> ProtocolState:
        """Current protocol state (read-only)."""
        return self._state

    def _reset_vars(
        self
    ):
        self._run_dir = None
        self._run_trigger_source = None
        # Keep _run_in_progress for backward compatibility (computed from _state)
        self._run_in_progress = False
        self._curr_step = 0
        self._n_scans = 0
        self._scan_count = 0
        self._scan_in_progress.clear()
        self._autofocus_count = 0
        self._auto_gain_deadline = 0.0
        self._grease_redistribution_done = True
        self._captures_taken = 0
        self._step_start_time = time.monotonic()
        self._target_x_pos = -1
        self._target_y_pos = -1
        self._target_z_pos = -1
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
    
    def get_initial_autofocus_states(self, layer_configs: dict | None = None):
        states = {}
        for layer in common_utils.get_layers():
            if layer_configs and layer in layer_configs:
                states[layer] = layer_configs[layer].get('autofocus', False)
            else:
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
        initial_autofocus_states: dict | None = None,
    ):
        with self._run_lock:
            if self._run_in_progress:
                logger.error(f"[{self.LOGGER_NAME} ] Cannot start new run, run already in progress")
                return

        # Check if file_io_executor still has pending writes
        if self.file_io_executor.is_protocol_queue_active():
            logger.error(f"[{self.LOGGER_NAME} ] Cannot start new run, file writing still in progress")
            return

        if leds_state_at_end not in ("off", "return_to_original",):
            raise ValueError(f"Unsupported value for leds_state_at_end: {leds_state_at_end}")

        if protocol.num_steps() == 0:
            logger.error(f"[PROTOCOL] Protocol has no steps. Cannot start run.")
            return

        # Pre-run validation: check positions within axis limits
        try:
            axis_limits = {}
            for axis in ('X', 'Y', 'Z'):
                try:
                    axis_limits[axis] = self._scope.get_axis_limits(axis)
                except Exception:
                    pass  # skip axis if limits unavailable
            validation_errors = protocol.validate_for_run(axis_limits=axis_limits)
            if validation_errors:
                for err in validation_errors:
                    logger.error(f"[PROTOCOL] Validation: {err}")
                logger.error(f"[PROTOCOL] Protocol has {len(validation_errors)} validation error(s). Cannot start run.")
                return
        except Exception as ex:
            logger.warning(f"[PROTOCOL] Pre-run validation failed: {ex}. Proceeding anyway.")

        try:
            if not self._scope.are_all_connected():
                logger.error(f"[PROTOCOL] Not all scope components connected. Cannot start run.")
                return
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error checking scope connection: {ex}")
            return

        
        # Not IO
        self._original_led_states = self._scope.get_led_states()
        self._original_gain = self._scope.get_gain()
        self._original_exposure = self._scope.get_exposure_time()
        if initial_autofocus_states is not None:
            self._original_autofocus_states = initial_autofocus_states
        else:
            self._original_autofocus_states = self.get_initial_autofocus_states()

        self._protocol = copy.deepcopy(protocol)
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
            logger.error(f"[{self.LOGGER_NAME} ] {result['error']}")
            return

        self._run_trigger_source = run_trigger_source
        with self._run_lock:
            self._set_state(ProtocolState.RUNNING)
            self._run_in_progress = True
        self.camera_executor.disable()
        self.protocol_executor.protocol_start()
        self._io_executor.protocol_start()
        self.file_io_executor.protocol_start()
        # Not IO
        self._scope.update_auto_gain_target_brightness(self._autogain_settings['target_brightness'])

        # Start the main run loop which manages all scan timing and execution
        self.protocol_executor.protocol_put(IOTask(action=self._run_loop))
    
    def _run_loop(self):
        """Main run loop - manages protocol execution and scan timing.
        A 'scan' is one complete iteration through all steps in the protocol.
        This loop runs until all scans are complete.
        """
        try:
            self._run_loop_inner()
        except Exception as ex:
            logger.error(f"[PROTOCOL] Unhandled exception in run loop: {ex}", exc_info=True)
        finally:
            # Safety net: ensure cleanup always runs so LEDs are turned off,
            # protocol state is reset, and resources are released even if an
            # unhandled exception occurs.  _cleanup() is idempotent (guarded
            # by _cleanup_lock and _run_in_progress check) so duplicate calls
            # from the normal path are harmless.
            self._cleanup()

    def _run_loop_inner(self):
        """Inner run loop body, called by _run_loop with crash-recovery wrapper."""
        last_maintenance_time = time.monotonic()
        last_connection_check = time.monotonic()

        while self._run_in_progress and not self._protocol_ended.is_set():
            try:
                # Periodic hardware connection check (every 30 seconds)
                now = time.monotonic()
                if now - last_connection_check > 30:
                    last_connection_check = now
                    try:
                        if not self._scope.are_all_connected():
                            logger.error("[PROTOCOL] Hardware disconnected during run — aborting protocol")
                            if self._state not in (ProtocolState.COMPLETING, ProtocolState.IDLE):
                                self._set_state(ProtocolState.ERROR)
                            self._cleanup()
                            break
                    except Exception as ex:
                        logger.warning(f"[PROTOCOL] Connection check failed: {ex}")

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
                
                # Check disk space once per scan: 10 MB/image + 500 MB/video, minimum 2 GB
                try:
                    if self._parent_dir is not None:
                        disk_usage = shutil.disk_usage(str(self._parent_dir))
                        free_mb = disk_usage.free / (1024 * 1024)
                        estimated_mb = 0
                        num_steps = self._protocol.num_steps()
                        for i in range(num_steps):
                            step = self._protocol.step(idx=i)
                            if step.get('Acquire') == 'video':
                                estimated_mb += 500
                            else:
                                estimated_mb += 10
                        required_mb = max(2048, estimated_mb)
                        if free_mb < required_mb:
                            logger.error(
                                f"[PROTOCOL] Insufficient disk space ({free_mb:.0f} MB free, "
                                f"need {required_mb:.0f} MB for {num_steps} steps) — aborting protocol"
                            )
                            self._protocol_ended.set()
                            break
                except Exception:
                    pass  # If we can't check, proceed anyway

                self._go_to_step(step_idx=self._curr_step)
                # Guard: if cleanup already ran (e.g. button spam), don't proceed
                if self._protocol_ended.is_set() or self._state == ProtocolState.IDLE:
                    break
                self._scan_in_progress.set()
                self._set_state(ProtocolState.SCANNING)
                self._auto_gain_deadline = time.monotonic() + self._autogain_settings['max_duration'].total_seconds()

                start_scan_time = datetime.datetime.now()
                # Run the scan loop - this will block until scan completes
                self._scan_loop()

                end_scan_time = datetime.datetime.now()
                scan_duration = end_scan_time - start_scan_time

                logger.info(f"Protocol scan {self._scan_count} completed in {scan_duration.total_seconds():.2f} seconds")
                
                # Scan completed - increment counter
                self._scan_count += 1
                logger.debug(f"[{self.LOGGER_NAME}] Scan {self._scan_count}/{self._n_scans} completed")
                
                if 'scan_iterate_post' in self._callbacks and self._callbacks['scan_iterate_post'] is not None:
                    Clock.schedule_once(lambda dt: self._callbacks['scan_iterate_post'](), 0)
                
                # Clear the flag so we know scan is done
                self._scan_in_progress.clear()
                if self._state == ProtocolState.SCANNING:
                    self._set_state(ProtocolState.RUNNING)

            except Exception as ex:
                logger.error(f"[Protocol] Error during run loop: {ex}", exc_info=True)
                if self._state not in (ProtocolState.COMPLETING, ProtocolState.IDLE, ProtocolState.ERROR):
                    try:
                        self._set_state(ProtocolState.ERROR)
                    except ValueError:
                        pass  # State already transitioned (e.g., concurrent cleanup)
                self._cleanup()
                break

        # If we exited the while loop (protocol_ended or run_in_progress cleared),
        # ensure cleanup runs so callbacks fire and resources are released.
        self._cleanup()

    def _scan_loop(self):
        """Scan loop - iterates through all protocol steps until scan completes.
        Returns when the scan is complete (all steps executed).
        """
        last_maintenance_time = time.monotonic()

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
                time.sleep(0.001)
                
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

        # Check if target location has not been reached yet
        # Uses is_moving() which checks in-memory axis state first (zero serial I/O
        # when all axes are IDLE), falling back to firmware only when needed.
        if self._scope.is_moving():
            if time.monotonic() - self._step_start_time > self.STEP_TIMEOUT_SECONDS:
                logger.error(f"[PROTOCOL] Step {self._curr_step} timed out waiting for motion ({self.STEP_TIMEOUT_SECONDS}s) — skipping step")
                # Force advance to next step
                num_steps = self._protocol.num_steps()
                if self._curr_step < num_steps - 1:
                    self._curr_step += 1
                    self._go_to_step(step_idx=self._curr_step)
                else:
                    self._scan_in_progress.clear()
            return
        
        if not self._grease_redistribution_done:
            return
        
        if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
            return
        
        if self._z_ui_update_func is not None:
            Clock.schedule_once(lambda dt: self._z_ui_update_func(float(step['Z'])), 0)

        # --- Pipeline timing instrumentation ---
        _t_settle = time.monotonic()
        _settle_wait_ms = (_t_settle - self._step_start_time) * 1000
        logger.debug(f"[TIMING] Step {self._curr_step} motion settle: {_settle_wait_ms:.1f}ms")

        # Set camera settings
        if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
            return

        _t_cam_start = time.monotonic()
        fut = self._io_executor.protocol_put(IOTask(
            action=self._scope.set_auto_gain,
            kwargs={
                "state": step['Auto_Gain'],
                "settings": self._autogain_settings,
            }
        ), return_future=True)
        if fut:
            fut.result(timeout=5)

        if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
            return

        _t_led_start = time.monotonic()
        logger.debug(f"[TIMING] Step {self._curr_step} camera settings: {(_t_led_start - _t_cam_start)*1000:.1f}ms")
        self._led_on(color=step['Color'], illumination=step['Illumination'], block=True)

        if not step['Auto_Gain']:
            if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
                return
            fut = self._io_executor.protocol_put(IOTask(
                action=self._scope.set_gain, args=(step['Gain'],)
            ), return_future=True)
            if fut:
                fut.result(timeout=5)
            # 2023-12-18 Instead of using only auto gain, now it's auto gain + exp. If auto gain is enabled, then don't set exposure time
            if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
                return
            fut = self._io_executor.protocol_put(IOTask(
                action=self._scope.set_exposure_time, args=(step['Exposure'],)
            ), return_future=True)
            if fut:
                fut.result(timeout=5)

        _t_led_done = time.monotonic()
        logger.debug(f"[TIMING] Step {self._curr_step} LED on: {(_t_led_done - _t_led_start)*1000:.1f}ms")

        # BF AF for fluorescence: skip AF on non-BF channels and use the last BF AF Z result
        bf_af_for_fluor = False
        try:
            bf_af_for_fluor = _app_ctx.ctx.settings.get('protocol', {}).get('bf_af_for_fluorescence', False)
        except Exception:
            pass
        if bf_af_for_fluor and step['Auto_Focus'] and step['Color'] != 'BF':
            # Use the BF autofocus Z result if available
            if self._autofocus_executor.best_focus_position() is not None:
                if self._update_z_pos_from_autofocus:
                    new_z_pos = self._autofocus_executor.best_focus_position()
                    self._protocol.modify_step_z_height(step_idx=self._curr_step, z=new_z_pos)
                logger.info(f'[Capture   ] Skipping AF on {step["Color"]} — using BF result Z={self._autofocus_executor.best_focus_position()}')
                # Skip AF and fall through to capture
                step = dict(step)
                step['Auto_Focus'] = False

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
        
        # Check if autogain has time-finished after auto-focus so that they can run in parallel
        if step['Auto_Gain'] and time.monotonic() < self._auto_gain_deadline:
            return

        # Reset the autogain deadline for next step
        self._auto_gain_deadline = time.monotonic() + self._autogain_settings['max_duration'].total_seconds()
        
        # Update the Z position with autofocus results
        if step['Auto_Focus'] and self._update_z_pos_from_autofocus:
            new_z_pos = self._autofocus_executor.best_focus_position()
            if new_z_pos is not None:
                self._protocol.modify_step_z_height(
                    step_idx=self._curr_step,
                    z=new_z_pos,
                )
            else:
                logger.warning('[Capture   ] Autofocus returned no position — keeping current Z')

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

                if step['Acquire'] == 'video':
                    self._video_write_finished.clear()

                _t_capture_start = time.monotonic()
                capture_result = self._capture(
                    save_folder=save_folder,
                    step=step,
                    scan_count=self._scan_count,
                    output_format=output_format,
                    sum_count=step["Sum"],
                )
                _t_capture_done = time.monotonic()
                logger.debug(f"[TIMING] Step {self._curr_step} capture+save: {(_t_capture_done - _t_capture_start)*1000:.1f}ms")

                self._stim_stop_event.set()


                # Protocol record creation and adding is handled in capture method


            else:
                # Normally LEDs are turned off at the end of a capture. However, when not capturing, need to manually turn
                # off LEDs (such as in autofocus scan)
                self._leds_off()

        self._autofocus_executor.reset()
        # Disable autogain when moving between steps
        if step['Auto_Gain']:
            fut = self._io_executor.protocol_put(IOTask(
                action=self._scope.set_auto_gain,
                kwargs={
                    "state": False,
                    "settings": self._autogain_settings,
                }
            ), return_future=True)
            if fut:
                fut.result(timeout=5)

        logger.debug(f"[TIMING] Step {self._curr_step} total: {(time.monotonic() - self._step_start_time)*1000:.1f}ms")

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
        with self._run_lock:
            # Derive from both legacy flag and state for safety during transition
            return self._run_in_progress or self._state in (
                ProtocolState.RUNNING, ProtocolState.SCANNING, ProtocolState.COMPLETING
            )
    

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


    STEP_TIMEOUT_SECONDS = 120  # Max time to wait for a single step (motion + capture)

    def _go_to_step(
        self,
        step_idx: int,
    ):
        self._step_start_time = time.monotonic()
        if self._protocol_ended.is_set():
            return
        
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
        fut = self._io_executor.protocol_put(IOTask(
            action=self._scope.leds_off
        ), return_future=True)
        if fut:
            fut.result(timeout=5)
        else:
            # protocol_put returned None (protocol not running) — fall back to direct call
            try:
                self._scope.leds_off()
            except Exception as ex:
                logger.warning(f"[{self.LOGGER_NAME}] Direct leds_off fallback failed: {ex}")
        if 'leds_off' in self._callbacks:
            self._callbacks['leds_off']()


    def _led_on(self, color: str, illumination: float, block: bool=True, force: bool=False):
        if self._protocol_ended.is_set() and not force:
            return

        fut = self._io_executor.protocol_put(IOTask(
            action=self._scope.led_on,
            kwargs={
                "channel": self._scope.color2ch(color),
                "mA": illumination,
                "block": block,
            },
        ), return_future=True)
        if fut:
            fut.result(timeout=5)
        # Sleep for 5 ms to ensure that LED properly turns on before next action
        time.sleep(0.005)

        if 'led_state' in self._callbacks:
            self._callbacks['led_state'](layer=color, enabled=True)


    def _cleanup(self):
        if not self._cleanup_lock.acquire(blocking=False):
            return  # Another thread is already cleaning up
        try:
            self._cleanup_inner()
        finally:
            self._cleanup_lock.release()

    def _cleanup_inner(self):
        if not self._run_in_progress:
            return

        # Transition to COMPLETING (or stay in ERROR if that's how we got here)
        if self._state not in (ProtocolState.COMPLETING, ProtocolState.ERROR, ProtocolState.IDLE):
            self._set_state(ProtocolState.COMPLETING)

        # Signal the scan/protocol loops to stop BEFORE turning off LEDs.
        # Without this, the scan loop (running in the protocol_executor thread)
        # can race with cleanup: _leds_off() turns LEDs off, then _led_on()
        # in _scan_iterate turns them back on before _protocol_ended is set.
        self._protocol_ended.set()
        self._scan_in_progress.clear()

        try:
            self._cancel_all_scheduled_events()
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error cancelling scheduled events during cleanup: {ex}")

        try:
            if self._leds_state_at_end == "off":
                self._leds_off()
            elif self._leds_state_at_end == "return_to_original":
                self._leds_off()
                for color, color_data in self._original_led_states.items():
                    if color_data['enabled']:
                        self._led_on(color=color, illumination=color_data['illumination'], block=True, force=True)
            else:
                logger.error(f"Unsupported LEDs state at end value: {self._leds_state_at_end}")
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error restoring LED states during cleanup: {ex}")

        # Always return autofocus states to initial
        try:
            for layer, layer_data in self._original_autofocus_states.items():
                # Restore via callback if available, otherwise write directly
                if 'restore_autofocus_state' in self._callbacks:
                    self._callbacks['restore_autofocus_state'](layer=layer, value=layer_data)
                else:
                    ctx = _app_ctx.ctx
                    if ctx is not None:
                        with ctx.settings_lock:
                            settings[layer]["autofocus"] = layer_data
                    else:
                        settings[layer]["autofocus"] = layer_data
                if self._callbacks.get('reset_autofocus_btns'):
                    # Updates autofocus buttons to their prior states
                    Clock.schedule_once(lambda dt: self._callbacks['reset_autofocus_btns'](), 0)
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error restoring autofocus states during cleanup: {ex}")

        # Restore camera gain and exposure to pre-protocol values
        try:
            if hasattr(self, '_original_gain') and self._original_gain >= 0:
                self._scope.set_gain(self._original_gain)
            if hasattr(self, '_original_exposure') and self._original_exposure > 0:
                self._scope.set_exposure_time(self._original_exposure)
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error restoring camera gain/exposure during cleanup: {ex}")

        try:
            if not self._disable_saving_artifacts:
                # Queue to close protocol execution record (should execute after last file/protocol record written)
                self.file_io_executor.protocol_put(IOTask(
                    action=self._protocol_execution_record.complete
                ))
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error completing protocol record during cleanup: {ex}")

        try:
            if self._return_to_position is not None:
                self._default_move(
                    px=self._return_to_position['x'],
                    py=self._return_to_position['y'],
                    z=self._return_to_position['z'],
                )
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error returning to position during cleanup: {ex}")


        self._scan_in_progress.clear()
        self._protocol_ended.set()

        self._stim_start_event.clear()
        self._stim_stop_event.set()

        self._io_executor.protocol_end()
        self.protocol_executor.protocol_end()
        self.autofocus_io_executor.protocol_end()
        self.camera_executor.enable()

        self._io_executor.clear_protocol_pending()
        self.protocol_executor.clear_protocol_pending()

        with self._run_lock:
            self._run_in_progress = False
            # Transition back to IDLE from COMPLETING or ERROR
            if self._state in (ProtocolState.COMPLETING, ProtocolState.ERROR):
                self._set_state(ProtocolState.IDLE)

        # Handle file queue completion with deferred callback
        if self.file_io_executor.is_protocol_queue_active():
            # Queue has pending work - call run_complete now for UI update,
            # but also register a deferred callback for final completion
            if 'run_complete' in self._callbacks:
                self._callbacks['run_complete'](protocol=self._protocol)

            # Register deferred callback for when queue actually drains
            if 'files_complete' in self._callbacks:
                self.file_io_executor.set_protocol_complete_callback(
                    callback=lambda: self._callbacks['files_complete'](protocol=self._protocol)
                )
            self.file_io_executor.protocol_finish_then_end()
        else:
            # No pending work - call both callbacks immediately
            if 'run_complete' in self._callbacks:
                self._callbacks['run_complete'](protocol=self._protocol)
            if 'files_complete' in self._callbacks:
                self._callbacks['files_complete'](protocol=self._protocol)
            self.file_io_executor.protocol_finish_then_end()


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
        
        if not self._run_in_progress:
            return
        
        if not self.protocol_executor.is_protocol_running():
            return
        
        is_video = True if step['Acquire'] == "video" else False
        video_as_frames = self._video_as_frames

        if not step['Auto_Gain']:
            with self._scope.update_camera_config():
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

        # Grab image and save

        sum_iteration_callback=None

        use_color = step['Color'] if step['False_Color'] else 'BF'

        if self._enable_image_saving:
            use_full_pixel_depth = self._image_capture_config['use_full_pixel_depth']

            if is_video:
                # Drain stale frames before video capture starts
                while self._scope.frame_validity.frames_until_valid() > 0:
                    self._scope.get_image(force_new_capture=True)
                    self._scope.frame_validity.count_frame()
                # Additional settle for auto-gain first frame
                time.sleep(max(step['Exposure']/1000, 0.05))
                # Disable autogain and then reenable it only for the first frame
                if step["Auto_Gain"]:
                    self._scope.set_auto_gain(state=False, settings=self._autogain_settings)
                    self._scope.auto_gain_once(
                        state=True,
                        target_brightness=self._autogain_settings['target_brightness'],
                        min_gain=self._autogain_settings['min_gain'],
                        max_gain=self._autogain_settings['max_gain'],
                    )

                duration_sec = step['Video Config']['duration']

                # Clamp the FPS to be no faster than the exposure rate
                exposure = step['Exposure']
                exposure_freq = 1.0 / (exposure / 1000)
                fps = min(exposure_freq, 40)
                
                if video_as_frames:
                    save_folder = save_folder / f"{name}"

                else:    
                    output_file_loc = save_folder / f"{name}.mp4"

                start_ts = time.time()
                stop_ts = start_ts + duration_sec
                expected_frames = fps * duration_sec
                captured_frames = 0
                seconds_per_frame = 1.0 / fps
                video_images = queue.Queue(maxsize=500)

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
                if sys.platform.startswith('win'):
                    try:
                        ctypes.windll.winmm.timeBeginPeriod(1)
                    except Exception:
                        pass
                self._stim_stop_event.clear()
                self._stim_start_event.set()

                while time.time() < stop_ts:
                    curr_time = time.time()
                    progress = (curr_time - start_ts) / duration_sec * 100
                    if "set_recording_title" in self._callbacks:
                        Clock.schedule_once(lambda dt, p=progress: self._callbacks['set_recording_title'](progress=p), 0)

                    if not self.protocol_executor.is_protocol_running():
                        self._leds_off()
                        if "reset_title" in self._callbacks:
                            Clock.schedule_once(lambda dt: self._callbacks['reset_title'](), 0)
                        return

                    # Currently only support 8-bit images for video
                    force_to_8bit = True
                    image = self._scope.get_image(force_to_8bit=force_to_8bit)

                    if isinstance(image, np.ndarray):

                        # Should never be used since forcing images to 8-bit
                        if image.dtype == np.uint16:
                            image = image_utils.convert_12bit_to_16bit(image)

                        # Note: Currently, if image is 12/16-bit, then we ignore false coloring for video captures.
                        if (image.dtype != np.uint16) and (step['False_Color']):
                            image = image_utils.add_false_color(array=image, color=use_color)

                        image = np.flip(image, 0)

                        try:
                            video_images.put_nowait((image, datetime.datetime.now()))
                        except queue.Full:
                            logger.warning(f"[Protocol-Video] Frame queue full ({video_images.maxsize}), dropping frame")
                            continue

                        captured_frames += 1

                    # Some process is slowing the video-process down (getting fewer frames than expected if delay of seconds_per_frame), so a shorter sleep time can be used
                    time.sleep(seconds_per_frame*0.9)

                if sys.platform.startswith('win'):
                    try:
                        ctypes.windll.winmm.timeEndPeriod(1)
                    except Exception:
                        pass
                self._stim_stop_event.set()
                self._stim_start_event.clear()  # Reset start event for next step

                for stim_thread in stim_threads:
                    stim_thread.join(timeout=5.0)
                    if stim_thread.is_alive():
                        logger.warning(f"[PROTOCOL] Stim thread did not exit within 5s timeout")

                if captured_frames == 0:
                    logger.warning("[PROTOCOL] Zero frames captured during video recording — skipping write")
                    self._video_write_finished.set()
                    self._leds_off()
                    return

                calculated_fps = max(1, int(captured_frames / duration_sec))

                logger.info(f"Protocol-Video] Images present in video array: {not video_images.empty()}")
                logger.info(f"Protocol-Video] Captured Frames: {captured_frames}")
                logger.info(f"Protocol-Video] Video FPS: {calculated_fps}")
                logger.info("Protocol-Video] Writing video...")

                self._leds_off()

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
                # Frame validity drains stale frames, then grabs a valid one
                captured_image = self._scope.capture_and_wait(
                    force_to_8bit=not use_full_pixel_depth,
                    all_ones_check=True,
                    timeout=datetime.timedelta(seconds=1.0),
                    sum_count=sum_count,
                    sum_delay_s=step["Exposure"]/1000,
                    sum_iteration_callback=sum_iteration_callback,
                )



                if captured_image is False:
                    logger.error(f"[PROTOCOL] Capture failed for {name} — camera inactive or frame drain failed")
                    # Still record the step with "capture_failed" so the record isn't silently missing
                    self.file_io_executor.protocol_put(IOTask(
                        action=self._write_capture,
                        kwargs={
                            "step": step,
                            "step_index": self._curr_step,
                            "scan_count": self._scan_count,
                            "capture_time": datetime.datetime.now(),
                        }
                    ))
                    self._leds_off()
                    return

                logger.info(f"Protocol Image Captured: {name}")

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


        self._leds_off()



                
    
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
        
        if self._enable_image_saving:
            if is_video:
                if "set_writing_title" in self._callbacks:
                    Clock.schedule_once(lambda dt: self._callbacks['set_writing_title'](progress=0), 0)

                try:
                    if video_as_frames:
                        frame_num = 0
                        capture_result = save_folder
                        if not save_folder.exists():
                            save_folder.mkdir(exist_ok=True, parents=True)

                        while not video_images.empty():

                            progress = frame_num / max(1, captured_frames) * 100
                            if "set_writing_title" in self._callbacks:
                                Clock.schedule_once(lambda dt, p=progress: self._callbacks['set_writing_title'](progress=p), 0)

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
                        del video_images

                    else:
                        output_file_loc = save_folder / f"{name}.mp4v"
                        video_writer = VideoWriter(
                            output_file_loc=output_file_loc,
                            fps=calculated_fps,
                            include_timestamp_overlay=True
                        )
                        try:
                            frame_num = 0
                            while not video_images.empty():
                                progress = frame_num / max(1, captured_frames) * 100
                                if "set_writing_title" in self._callbacks:
                                    Clock.schedule_once(lambda dt, p=progress: self._callbacks['set_writing_title'](progress=p), 0)

                                try:
                                    image_pair = video_images.get_nowait()
                                    video_writer.add_frame(image=image_pair[0], timestamp=image_pair[1])
                                    del image_pair
                                    video_images.task_done()
                                    frame_num += 1
                                except Exception as e:
                                    logger.error(f"Protocol-Video] FAILED TO WRITE FRAME: {e}")
                        finally:
                            video_writer.finish()
                            del video_writer

                        # Ensure queue is fully drained before deletion
                        try:
                            while not video_images.empty():
                                video_images.get_nowait()
                                video_images.task_done()
                        except Exception:
                            pass
                        del video_images

                        capture_result = output_file_loc

                finally:
                    self._video_write_finished.set()

                if "reset_title" in self._callbacks:
                    Clock.schedule_once(lambda dt: self._callbacks['reset_title'](), 0)

                logger.info("Protocol-Video] Video writing finished.")
                logger.info(f"Protocol-Video] Video saved at {capture_result}")
            
            else:
                if captured_image is False:
                    logger.warning(f"[PROTOCOL] _write_capture: captured_image is False, recording as capture_failed")
                    capture_result_filepath_name = "capture_failed"
                    self._protocol_execution_record.add_step(
                        capture_result_file_name=capture_result_filepath_name,
                        step_name=name if name else "unknown",
                        step_index=step_index,
                        scan_count=scan_count,
                        timestamp=capture_time,
                        frame_count=0,
                        duration_sec=0.0,
                    )
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

            if capture_result is None:
                capture_result_filepath_name = "unsaved"

            elif isinstance(capture_result, dict):
                capture_result_filepath_name = capture_result['metadata']['file_loc']

            elif self._separate_folder_per_channel:
                capture_result_filepath_name = pathlib.Path(step["Color"]) / capture_result.name

            else:
                capture_result_filepath_name = capture_result.name

        else:
            capture_result_filepath_name = "unsaved"

        try:
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
        except Exception as ex:
            logger.error(f"[Protocol-Writer] Failed to add step to protocol execution record: {ex}")

    

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

        try:
            period_s = 1.0 / float(frequency)
            pulse_s = float(pulse_width) / 1000.0

            # Use fast path LED toggles if available via API
            def led_on_fast():
                #logger.info(f"[STIMULATOR] {color} LED ON")
                if hasattr(self._scope, 'led_on_fast'):
                    self._scope.led_on_fast(channel=self._scope.color2ch(color=color), mA=illumination)
                else:
                    self._scope.led_on(channel=self._scope.color2ch(color=color), mA=illumination)

            def led_off_fast():
                #logger.info(f"[STIMULATOR] {color} LED OFF")
                if hasattr(self._scope, 'led_off_fast'):
                    self._scope.led_off_fast(channel=self._scope.color2ch(color=color))
                else:
                    self._scope.led_off(channel=self._scope.color2ch(color=color))

            start_epoch = time.perf_counter()
            
            start_event.wait()
            logger.info(f"[STIMULATOR] stim_start_event set for {color}")

            end_reason = "pulse_count_reached"

            pulses_executed = 0

            for i in range(pulse_count):
                if stop_event.is_set():
                    logger.info(f"[STIMULATOR] {color} stop event set, ending stimulation.")
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
                        time.sleep(0.0001)  # yield CPU instead of busy-wait

                led_on_fast()

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
                        time.sleep(0.0001)  # yield CPU instead of busy-wait

                led_off_fast()

                # Maintain period; wait until next_period_time
                while True:
                    now = time.perf_counter()
                    remaining = next_period_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        time.sleep(0.0001)  # yield CPU instead of busy-wait

        finally:
            if sys.platform.startswith('win') and time_period_set:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass
            
            logger.info(f"[STIMULATOR] {color} stimulation ended after executing {pulses_executed} pulses.")
            logger.info(f"[STIMULATOR] {color} Ended due to {end_reason}")
            # Ensure LED off at the end
            try:
                if hasattr(self._scope, 'led_off_fast'):
                    self._scope.led_off_fast(channel=self._scope.color2ch(color=color))
                else:
                    self._scope.led_off(channel=self._scope.color2ch(color=color))
            except Exception:
                pass
