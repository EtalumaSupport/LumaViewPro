# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import pathlib
import time
import typing

from modules.protocol_state_machine import (
    ProtocolState,
    SequencedCaptureRunMode,
    PROTOCOL_STATE_TRANSITIONS,
    validate_transition,
)
from modules.protocol_callbacks import ProtocolCallbacks
from modules.protocol_image_writer import ProtocolImageWriter
from modules.protocol_cleanup import run_cleanup
from modules.protocol_step_executor import ProtocolStepExecutor
from modules.protocol_run_loop import ProtocolRunLoop

from modules.kivy_utils import schedule_ui as _schedule_ui


from modules.lumascope_api import Lumascope

import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations

import modules.labware_loader as labware_loader
from modules.autofocus_executor import AutofocusExecutor
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord

from modules.sequential_io_executor import SequentialIOExecutor, IOTask
from lvp_logger import logger
from concurrent.futures import ProcessPoolExecutor
import threading

import modules.app_context as _app_ctx
from modules.settings_init import settings


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
    STEP_TIMEOUT_SECONDS = 120  # Max time to wait for a single step (motion + capture)

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
        self._run_in_progress_event = threading.Event()  # GIL-free safe replacement for _run_in_progress bool
        self._cleanup_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._cpu_pool = cpu_pool
        self._video_write_finished = threading.Event()
        self._video_write_finished.set()
        self._grease_redistribution_event = threading.Event()
        self._grease_redistribution_event.set()

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
        self._protocol_state_lock = threading.Lock()
        self._state = ProtocolState.IDLE
        self._reset_vars()
        self._step_executor = ProtocolStepExecutor(self)
        self._run_loop_executor = ProtocolRunLoop(self)


    def set_scope(self, scope: Lumascope):
        self._scope = scope

    def _set_state(self, new_state: ProtocolState) -> None:
        """Transition to *new_state* with validation. Thread-safe.

        Raises ``ValueError`` if the transition is not allowed by
        ``PROTOCOL_STATE_TRANSITIONS``.
        """
        with self._protocol_state_lock:
            if self._state == new_state:
                return  # no-op
            validate_transition(self._state, new_state, self.LOGGER_NAME)
            self._state = new_state

    @property
    def protocol_state(self) -> ProtocolState:
        """Current protocol state (read-only). Thread-safe."""
        with self._protocol_state_lock:
            return self._state

    def _reset_vars(
        self
    ):
        self._run_dir = None
        self._run_trigger_source = None
        self._run_in_progress_event.clear()
        self._curr_step = 0
        self._n_scans = 0
        self._scan_count = 0
        self._scan_in_progress.clear()
        self._autofocus_count = 0
        self._auto_gain_deadline = 0.0
        self._grease_redistribution_event.set()
        self._captures_taken = 0
        self._protocol_execution_record = None
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
        if not self._run_in_progress_event.is_set():
            return
        
        self._cleanup()


    def protocol_interval(self):
        return self._protocol.period()
    
    def get_initial_autofocus_states(self, layer_configs: dict | None = None):
        states = {}
        ctx = _app_ctx.ctx
        for layer in common_utils.get_layers():
            if layer_configs and layer in layer_configs:
                states[layer] = layer_configs[layer].get('autofocus', False)
            else:
                with ctx.settings_lock:
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
            if self._run_in_progress_event.is_set():
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

        # Lightweight copy — shares read-only loaders, copies only the mutable
        # steps DataFrame (which AF modifies via modify_step_z_height). Much
        # cheaper than deepcopy for large protocols (M14).
        self._protocol = protocol.copy_for_execution()
        self._run_mode = run_mode
        self._sequence_name = sequence_name
        self._parent_dir = parent_dir
        self._image_capture_config = image_capture_config
        self._enable_image_saving = enable_image_saving
        self._separate_folder_per_channel = separate_folder_per_channel
        # Immutable after assignment — do not mutate from protocol thread (M4 GIL-free safety)
        self._autogain_settings = autogain_settings
        self._callbacks = ProtocolCallbacks.from_dict(callbacks) if isinstance(callbacks, dict) else (callbacks or ProtocolCallbacks())
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

        self._image_writer = ProtocolImageWriter(
            scope=self._scope,
            callbacks=self._callbacks,
            protocol_ended=self._protocol_ended,
            video_write_finished=self._video_write_finished,
            file_io_executor=self.file_io_executor,
            protocol_executor=self.protocol_executor,
            execution_record=self._protocol_execution_record,
            leds_off_fn=self._step_executor.leds_off,
            led_on_fn=self._step_executor.led_on,
            is_run_in_progress_fn=lambda: self._run_in_progress_event.is_set(),
        )

        self._run_trigger_source = run_trigger_source
        with self._run_lock:
            self._set_state(ProtocolState.RUNNING)
            self._run_in_progress_event.set()
        self.camera_executor.disable()
        self.protocol_executor.protocol_start()
        self._io_executor.protocol_start()
        self.file_io_executor.protocol_start()
        # Not IO
        self._scope.update_auto_gain_target_brightness(self._autogain_settings['target_brightness'])

        # Start the main run loop which manages all scan timing and execution
        self.protocol_executor.protocol_put(IOTask(action=self._run_loop_executor.run_loop))
    
    def run_in_progress(self) -> bool:
        with self._run_lock:
            # Derive from both legacy flag and state for safety during transition
            return self._run_in_progress_event.is_set() or self._state in (
                ProtocolState.RUNNING, ProtocolState.SCANNING, ProtocolState.COMPLETING
            )
    

    def run_trigger_source(self) -> str:
        return self._run_trigger_source
    
    
    def _cancel_all_scheduled_events(self):
        """Cancel any remaining scheduled events. 
        Note: With the loop-based approach, most work happens in executor threads,
        so there's less to unschedule than before.
        """
        # Legacy Clock.unschedule calls removed — with the loop-based
        # architecture, iterators run on executor threads, not Kivy Clock.
        self._protocol_iterator = None
        self._scan_iterator = None


    def _cleanup(self):
        if not self._cleanup_lock.acquire(blocking=False):
            return  # Another thread is already cleaning up
        try:
            self._cleanup_inner()
        finally:
            self._cleanup_lock.release()

    def _cleanup_inner(self):
        if not self._run_in_progress_event.is_set():
            return

        run_cleanup(
            get_state_fn=lambda: self._state,
            set_state_fn=self._set_state,
            run_lock=self._run_lock,
            protocol_ended=self._protocol_ended,
            scan_in_progress=self._scan_in_progress,
            leds_state_at_end=self._leds_state_at_end,
            original_led_states=self._original_led_states,
            original_autofocus_states=self._original_autofocus_states,
            original_gain=getattr(self, '_original_gain', -1),
            original_exposure=getattr(self, '_original_exposure', 0),
            return_to_position=self._return_to_position,
            disable_saving_artifacts=self._disable_saving_artifacts,
            protocol=self._protocol,
            protocol_execution_record=self._protocol_execution_record,
            scope=self._scope,
            callbacks=self._callbacks,
            leds_off_fn=self._step_executor.leds_off,
            led_on_fn=self._step_executor.led_on,
            default_move_fn=self._step_executor.default_move,
            cancel_scheduled_events_fn=self._cancel_all_scheduled_events,
            io_executor=self._io_executor,
            protocol_executor=self.protocol_executor,
            autofocus_io_executor=self.autofocus_io_executor,
            file_io_executor=self.file_io_executor,
            camera_executor=self.camera_executor,
            set_run_in_progress_fn=lambda v: self._run_in_progress_event.set() if v else self._run_in_progress_event.clear(),
            logger_name=self.LOGGER_NAME,
        )


