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
from modules.video_capture import VideoCaptureSession, write_video

try:
    from kivy.clock import Clock
except ImportError:
    # Dummy Clock for non-Kivy environments (subprocess without kivy installed).
    class Clock:
        @staticmethod
        def schedule_once(func, timeout): func(0)
        @staticmethod
        def schedule_interval(func, interval): pass


def _schedule_ui(func, timeout=0):
    """Schedule a function on the Kivy main thread, or call directly if
    no Kivy event loop is running (e.g., in tests).
    Same signature as Clock.schedule_once — func receives dt argument.
    """
    try:
        from kivy.base import EventLoop
        if EventLoop.status == 'started':
            Clock.schedule_once(func, timeout)
            return
    except Exception:
        pass
    # No Kivy event loop — call directly (test/headless mode)
    func(0)


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
        if not self._autofocus_executor.run_in_progress():
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
                            def _show_hw_error(dt):
                                try:
                                    from ui.notification_popup import show_notification_popup
                                    show_notification_popup(title="Protocol Aborted",
                                        message="Hardware disconnected during protocol run.")
                                except Exception:
                                    pass
                            _schedule_ui(_show_hw_error)
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
                    _schedule_ui(
                        lambda dt: self._callbacks['protocol_iterate_pre'](
                            remaining_scans=remaining_scans,
                            interval=self._protocol.period()
                        )
                    )
                
                # Initialize scan variables
                self._curr_step = 0
                if 'run_scan_pre' in self._callbacks:
                    _schedule_ui(lambda dt: self._callbacks['run_scan_pre'](), 0)
                
                # Check disk space once per scan: 5 MB/image + 50 MB/video, minimum 2 GB
                # Previous estimates (10MB/image, 500MB/video) were too conservative
                # and blocked protocols on 34GB free drives.
                try:
                    if self._parent_dir is not None:
                        disk_usage = shutil.disk_usage(str(self._parent_dir))
                        free_mb = disk_usage.free / (1024 * 1024)
                        estimated_mb = 0
                        num_steps = self._protocol.num_steps()
                        for i in range(num_steps):
                            step = self._protocol.step(idx=i)
                            if step.get('Acquire') == 'video':
                                estimated_mb += 50  # MP4 compressed, ~10-50MB typical
                            else:
                                estimated_mb += 8   # 1900x1900 16-bit TIFF ~7.2 MB + metadata
                        required_mb = max(2048, estimated_mb)
                        if free_mb < required_mb:
                            msg = (f"Insufficient disk space: {free_mb:.0f} MB free, "
                                   f"need ~{required_mb:.0f} MB for {num_steps} steps.")
                            logger.error(f"[PROTOCOL] {msg} — aborting protocol")
                            def _show_disk_error(dt, m=msg):
                                try:
                                    from ui.notification_popup import show_notification_popup
                                    show_notification_popup(title="Protocol Aborted", message=m)
                                except Exception:
                                    pass
                            _schedule_ui(_show_disk_error)
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
                    _schedule_ui(lambda dt: self._callbacks['scan_iterate_post'](), 0)
                
                # Clear the flag so we know scan is done
                self._scan_in_progress.clear()
                if self._state == ProtocolState.SCANNING:
                    self._set_state(ProtocolState.RUNNING)

            except Exception as ex:
                logger.error(f"[Protocol] Error during run loop: {ex}", exc_info=True)
                err_msg = str(ex)
                def _show_run_error(dt, m=err_msg):
                    try:
                        from ui.notification_popup import show_notification_popup
                        show_notification_popup(title="Protocol Error", message=m)
                    except Exception:
                        pass
                _schedule_ui(_show_run_error)
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
                scan_err = str(ex)
                def _show_scan_error(dt, m=scan_err):
                    try:
                        from ui.notification_popup import show_notification_popup
                        show_notification_popup(title="Protocol Scan Error", message=m)
                    except Exception:
                        pass
                _schedule_ui(_show_scan_error)
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
                timeout_msg = f"Step {self._curr_step} timed out waiting for motion ({self.STEP_TIMEOUT_SECONDS}s)."
                logger.error(f"[PROTOCOL] {timeout_msg} — transitioning to ERROR state")
                def _show_timeout_error(dt, m=timeout_msg):
                    try:
                        from ui.notification_popup import show_notification_popup
                        show_notification_popup(title="Protocol Error — Motion Timeout", message=m)
                    except Exception:
                        pass
                _schedule_ui(_show_timeout_error)
                self._scan_in_progress.clear()
                try:
                    self._set_state(ProtocolState.ERROR)
                except ValueError:
                    pass  # Already in a terminal state
            return
        
        if not self._grease_redistribution_done:
            return
        
        if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
            return
        
        if self._z_ui_update_func is not None:
            _schedule_ui(lambda dt: self._z_ui_update_func(float(step['Z'])))

        # --- Pipeline timing instrumentation ---
        _t_settle = time.monotonic()
        _settle_wait_ms = (_t_settle - self._step_start_time) * 1000
        logger.debug(f"[TIMING] Step {self._curr_step} motion settle: {_settle_wait_ms:.1f}ms")

        # Set camera settings
        if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
            return

        _t_cam_start = time.monotonic()
        fut = self._io_executor.protocol_put(IOTask(
            action=self._scope.apply_layer_camera_settings,
            kwargs={
                'gain': step['Gain'],
                'exposure_ms': step['Exposure'],
                'auto_gain': step['Auto_Gain'],
                'auto_gain_settings': self._autogain_settings if step['Auto_Gain'] else None,
            }
        ), return_future=True)
        if fut:
            fut.result(timeout=5)

        if self._protocol_ended.is_set() or not self._scan_in_progress.is_set():
            return

        _t_led_start = time.monotonic()
        logger.debug(f"[TIMING] Step {self._curr_step} camera settings: {(_t_led_start - _t_cam_start)*1000:.1f}ms")
        self._led_on(color=step['Color'], illumination=step['Illumination'], block=True)

        _t_led_done = time.monotonic()
        logger.debug(f"[TIMING] Step {self._curr_step} LED on: {(_t_led_done - _t_led_start)*1000:.1f}ms")

        # BF AF for fluorescence: skip AF on non-BF channels and use the last BF AF Z result
        bf_af_for_fluor = False
        try:
            bf_af_for_fluor = _app_ctx.ctx.settings.get('protocol', {}).get('bf_af_for_fluorescence', False)
        except Exception:
            pass
        if bf_af_for_fluor and step['Color'] != 'BF':
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
                _schedule_ui(lambda dt: self._callbacks['autofocus_in_progress'](), 0)

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
            _schedule_ui(lambda dt: self._callbacks['autofocus_complete'](), 0)

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

                # Protocol record creation and adding is handled in capture method


            else:
                # Normally LEDs are turned off at the end of a capture. However, when not capturing, need to manually turn
                # off LEDs (such as in autofocus scan)
                self._leds_off()

        if not self._autofocus_executor.run_in_progress():
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
                _schedule_ui(lambda dt: self._callbacks['update_step_number'](self._curr_step+1), 0)
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
            if 'move_position' in self._callbacks:
                _schedule_ui(lambda dt: self._callbacks['move_position']('X'), 0)

            self._scope.move_absolute_position('Y', sy)
            self._target_y_pos = sy
            if 'move_position' in self._callbacks:
                _schedule_ui(lambda dt: self._callbacks['move_position']('Y'), 0)

            if z is not None:
                self._scope.move_absolute_position('Z', z)
                self._target_z_pos = z
                if 'move_position' in self._callbacks:
                    _schedule_ui(lambda dt: self._callbacks['move_position']('Z'), 0)


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
        _t_start = time.monotonic()
        z_orig = self._scope.get_current_position(axis=axis)
        self._scope.move_absolute_position(
            axis=axis,
            pos=0,
            wait_until_complete=True,
            overshoot_enabled=True
        )

        if 'move_position' in self._callbacks:
            _schedule_ui(lambda dt, a=axis: self._callbacks['move_position'](a))

        self._scope.move_absolute_position(
            axis=axis,
            pos=z_orig,
            wait_until_complete=True,
            overshoot_enabled=True
        )

        if 'move_position' in self._callbacks:
            _schedule_ui(lambda dt, a=axis: self._callbacks['move_position'](a))

        elapsed = time.monotonic() - _t_start
        if elapsed > 30:
            logger.warning(f"[PROTOCOL] Grease redistribution took {elapsed:.1f}s (> 30s threshold)")
        else:
            logger.debug(f"[PROTOCOL] Grease redistribution completed in {elapsed:.1f}s")

        self._grease_redistribution_done = True


    def _cancel_all_scheduled_events(self):
        """Cancel any remaining scheduled events. 
        Note: With the loop-based approach, most work happens in executor threads,
        so there's less to unschedule than before.
        """
        try:
            if self._protocol_iterator is not None:
                Clock.unschedule(self._protocol_iterator)
            if self._scan_iterator is not None:
                Clock.unschedule(self._scan_iterator)
        except Exception:
            pass  # Safe to ignore — iterators may not be Clock-scheduled


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
            _schedule_ui(lambda dt: self._callbacks['leds_off'](), 0)


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
            _schedule_ui(lambda dt, c=color: self._callbacks['led_state'](layer=c, enabled=True))


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
                # Restore original LED states directly — don't turn off first
                # to avoid a visible flash (off then on) during autofocus
                any_restored = False
                for color, color_data in self._original_led_states.items():
                    if color_data['enabled']:
                        self._led_on(color=color, illumination=color_data['illumination'], block=True, force=True)
                        any_restored = True
                if not any_restored:
                    # No LEDs were originally on — turn everything off
                    self._leds_off()
            else:
                logger.error(f"Unsupported LEDs state at end value: {self._leds_state_at_end}")
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error restoring LED states during cleanup: {ex}")
        logger.info(f"[{self.LOGGER_NAME}] Cleanup: LED/camera restore complete")

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
                    _schedule_ui(lambda dt: self._callbacks['reset_autofocus_btns'](), 0)
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
                logger.info(f"[{self.LOGGER_NAME}] Cleanup: returning to position x={self._return_to_position['x']}, y={self._return_to_position['y']}, z={self._return_to_position['z']}")
                self._default_move(
                    px=self._return_to_position['x'],
                    py=self._return_to_position['y'],
                    z=self._return_to_position['z'],
                )
                logger.info(f"[{self.LOGGER_NAME}] Cleanup: return-to-position move issued")
        except Exception as ex:
            logger.error(f"[PROTOCOL] Error returning to position during cleanup: {ex}")


        self._scan_in_progress.clear()
        self._protocol_ended.set()

        self._io_executor.protocol_end()
        self.protocol_executor.protocol_end()
        self.autofocus_io_executor.protocol_end()
        self.camera_executor.enable()
        logger.info(f"[{self.LOGGER_NAME}] Cleanup: protocol_end called on all executors")

        self._io_executor.clear_protocol_pending()
        self.protocol_executor.clear_protocol_pending()

        with self._run_lock:
            self._run_in_progress = False
            # Transition back to IDLE from COMPLETING or ERROR
            if self._state in (ProtocolState.COMPLETING, ProtocolState.ERROR):
                self._set_state(ProtocolState.IDLE)

        # Handle file queue completion with deferred callback
        _file_queue_active = self.file_io_executor.is_protocol_queue_active()
        logger.info(f"[{self.LOGGER_NAME}] Cleanup: file queue active={_file_queue_active}")
        if _file_queue_active:
            # Queue has pending work - call run_complete now for UI update,
            # but also register a deferred callback for final completion
            if 'run_complete' in self._callbacks:
                _schedule_ui(lambda dt: self._callbacks['run_complete'](protocol=self._protocol), 0)

            # Register deferred callback for when queue actually drains
            if 'files_complete' in self._callbacks:
                self.file_io_executor.set_protocol_complete_callback(
                    callback=lambda: _schedule_ui(lambda dt: self._callbacks['files_complete'](protocol=self._protocol), 0)
                )
            self.file_io_executor.protocol_finish_then_end()
            logger.info(f"[{self.LOGGER_NAME}] Cleanup: callbacks scheduled (run_complete now, files_complete deferred)")
        else:
            # No pending work - call both callbacks immediately
            if 'run_complete' in self._callbacks:
                _schedule_ui(lambda dt: self._callbacks['run_complete'](protocol=self._protocol), 0)
            if 'files_complete' in self._callbacks:
                _schedule_ui(lambda dt: self._callbacks['files_complete'](protocol=self._protocol), 0)
            self.file_io_executor.protocol_finish_then_end()
            logger.info(f"[{self.LOGGER_NAME}] Cleanup: callbacks scheduled (run_complete + files_complete immediate)")


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
            obj_info = self._scope.get_objective_info(objective_id=step["Objective"])
            if obj_info is not None:
                objective_short_name = obj_info.get('short_name')
            else:
                logger.warning(f"[PROTOCOL] Turret available but no objective info for ID '{step['Objective']}' — using None for filename")
                objective_short_name = None
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

        # In engineering mode, include turret position in filename
        turret_pos = None
        if self._scope.engineering_mode and self._scope.has_turret():
            try:
                turret_pos = int(self._scope.get_current_position('T'))
            except Exception:
                pass

        name = common_utils.generate_default_step_name(
            well_label=step['Well'],
            color=step['Color'],
            z_height_idx=step['Z-Slice'],
            scan_count=scan_count,
            custom_name_prefix=combined_prefix,
            objective_short_name=objective_short_name,
            tile_label=step['Tile'],
            video=is_video,
            turret_position=turret_pos,
        )
        # Ensure the filename base has no invalid path characters
        try:
            name = Protocol.sanitize_step_name(input=name)
        except Exception:
            pass

        # Illuminate
        if self._scope.led_connected:
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
                session = VideoCaptureSession(
                    scope=self._scope,
                    step=step,
                    autogain_settings=self._autogain_settings,
                    is_protocol_running_fn=self.protocol_executor.is_protocol_running,
                    callbacks=self._callbacks,
                    leds_off_fn=self._leds_off,
                )
                video_result = session.capture()

                if video_result is None:
                    # Cancelled or zero frames — skip write
                    self._video_write_finished.set()
                    self._leds_off()
                    return

                self._leds_off()

                self.file_io_executor.protocol_put(IOTask(
                    action=self._write_capture,
                    kwargs={
                        "is_video": is_video,
                        "video_as_frames": video_as_frames,
                        "video_result": video_result,
                        "save_folder": save_folder,
                        "use_color": use_color,
                        "name": name,
                        "output_format": output_format,
                        "step": step,
                        "captured_image": None,
                        "step_index": self._curr_step,
                        "scan_count": self._scan_count,
                        "capture_time": datetime.datetime.now(),
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
                        "save_folder": save_folder,
                        "use_color": use_color,
                        "name": name,
                        "output_format": output_format,
                        "step": step,
                        "captured_image": captured_image,
                        "step_index": self._curr_step,
                        "scan_count": self._scan_count,
                        "capture_time": datetime.datetime.now(),
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
                       video_result=None,
                       save_folder=None,
                       use_color=None,
                       name=None,
                       output_format=None,
                       step=None,
                       captured_image=None,
                       step_index=None,
                       scan_count=None,
                       capture_time=None,
                       ):

        if self._enable_image_saving:
            if is_video:
                try:
                    capture_result = write_video(
                        result=video_result,
                        save_folder=save_folder,
                        name=name,
                        video_as_frames=video_as_frames,
                        step=step,
                        callbacks=self._callbacks,
                    )
                finally:
                    self._video_write_finished.set()

                captured_frames = video_result.captured_frames
                duration_sec = video_result.duration_sec

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

    

