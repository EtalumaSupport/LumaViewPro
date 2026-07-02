# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import copy
import datetime
import pathlib
import time
import typing

from modules.protocol_state_machine import (
    ProtocolState,
    SequencedCaptureRunMode,
    validate_transition,
)
from modules.protocol_callbacks import ProtocolCallbacks
from modules.protocol_image_writer import ProtocolImageWriter
from modules.protocol_cleanup import run_cleanup
from modules.protocol_step_runner import ProtocolStepRunner
from modules.protocol_run_loop import ProtocolRunLoop

from modules.lumascope_api import Lumascope

import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
import modules.coord_transformations as coord_transformations
import modules.image_mode as image_mode

import modules.labware_loader as labware_loader
from modules.autofocus_runner import AutofocusRunner
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord

from modules.sequential_io_executor import SequentialIOExecutor
from lvp_logger import logger
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


class SequencedCaptureRunner:
    LOGGER_NAME = 'SeqCapExec'
    # Max time for ONE continuous stage motion to complete. The timer
    # starts when motion is first observed in flight and resets whenever
    # the stage reports idle, so in-step autofocus time never counts
    # against it. The longest legitimate single move (full-travel XY)
    # finishes well inside this bound; a stage still "moving" past it is
    # stalled hardware, not a slow move. Per PERFORMANCE_BUDGETS.md row
    # protocol_motion_timeout_s.
    MOTION_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        scope: Lumascope,
        stage_offset: dict,
        io_executor: SequentialIOExecutor,
        protocol_thread,
        file_io_executor: SequentialIOExecutor,
        camera_executor: SequentialIOExecutor,
        autofocus_thread,
        autofocus_runner: AutofocusRunner | None = None,
        z_ui_update_func: typing.Callable | None = None,
    ):
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()
        self._wellplate_loader = labware_loader.WellPlateLoader()
        # Hold stage_offset by reference so UI edits between runs are visible
        # to the next run; _snapshot_run_state takes a deepcopy at run() start
        # so an in-flight protocol's coordinate transforms are immune to
        # mid-run mutations of ctx.settings['stage_offset'].
        self._stage_offset_source = stage_offset
        self._stage_offset = stage_offset
        self._io_executor = io_executor
        self.protocol_thread = protocol_thread
        self.file_io_executor = file_io_executor
        self.camera_executor = camera_executor
        self.autofocus_thread = autofocus_thread
        self._z_ui_update_func = z_ui_update_func
        self._scan_in_progress = threading.Event()
        # Abort signal. Owned by protocol_thread; SCE holds a reference
        # assigned in run() from protocol_thread.aborted. Tests that
        # construct SCE without a real protocol_thread can still read
        # this Event because it defaults to a local Event before run().
        self._aborted: threading.Event = threading.Event()
        self._run_in_progress_event = (
            threading.Event()
        )  # GIL-free safe replacement for _run_in_progress bool
        # The loaded protocol exists from construction so a runner that has
        # never started (or refused to start) answers getters with None
        # instead of raising AttributeError from inside a UI handler.
        # (_run_dir gets the same treatment via _reset_vars below.)
        self._protocol = None
        self._cleanup_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._grease_redistribution_event = threading.Event()
        self._grease_redistribution_event.set()

        if autofocus_runner is None:
            # Headless / test fallback. Caller may construct a private
            # AFE that bypasses the AutofocusThread for unit tests that
            # only need the protocol state machine.
            self._autofocus_runner = AutofocusRunner(
                scope=scope,
                camera_executor=camera_executor,
                io_executor=io_executor,
                file_io_executor=file_io_executor,
            )
        else:
            self._autofocus_runner = autofocus_runner

        self._scope = scope
        self._run_trigger_source = None
        # LED lease held for the duration of a scan -- acquired at run
        # start, passed to AF steps as the parent lease, released in
        # cleanup. None outside a run.
        self._led_lease = None
        self._protocol_state_lock = threading.Lock()
        self._state = ProtocolState.IDLE
        # Defensive default so attribute access before the first run()
        # (e.g. from a test that drives scan_iterate directly) returns
        # a no-op callbacks object instead of AttributeError.
        self._callbacks = ProtocolCallbacks()
        self._reset_vars()
        self._step_executor = ProtocolStepRunner(self)
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

    def _reset_scan_state(self) -> None:
        """Reset the per-scan state at each scan start.

        One place for the fields that must start fresh for every scan, so the
        run loop calls this rather than open-coding the resets. Does NOT touch
        _grease_redistribution_event: that gate is owned by the grease task
        itself (it always set()s on completion-or-failure, and the enqueue path
        set()s when the task never runs), so re-setting it here would race a
        grease move still in flight when the period is zero (continuous mode).
        """
        self._curr_step = 0
        # Per-step AF state pointer; None means AF has not been kicked off for
        # the current step. Set by scan_iterate when AF starts; cleared at step
        # transition and at scan start.
        self._af_future = None
        # One-shot latch: the resolved AF future is consumed exactly once, even
        # if the stage is still settling on the polls that follow. Travels with
        # _af_future (reset wherever the pointer is cleared).
        self._af_result_consumed = False

    def _reset_vars(self):
        self._run_dir = None
        self._run_trigger_source = None
        self._run_in_progress_event.clear()
        self._reset_scan_state()
        # _n_scans and _scan_count are the cross-thread progress pair, read
        # together under _protocol_state_lock by progress_snapshot(). Zero them
        # under the same lock so a concurrent remaining_scans() poll during run
        # re-init cannot observe a half-reset pair (n_scans already 0 while
        # scan_count still holds the prior run's value -> negative remaining).
        with self._protocol_state_lock:
            self._n_scans = 0
            self._scan_count = 0
        self._scan_in_progress.clear()
        self._autofocus_count = 0
        # Tracks the curr_step value for which Auto_Gain was already
        # armed (apply_layer_camera_settings ... auto_gain=True fired
        # in scan_iterate). -1 means "no AG armed yet this scan."
        # Reset at each scan start in protocol_run_loop so each scan
        # arms once per step.
        self._auto_gain_armed_step = -1
        self._grease_redistribution_event.set()
        self._captures_taken = 0
        self._protocol_execution_record = None
        self._step_start_time = time.monotonic()
        self._motion_wait_start = None
        self._target_x_pos = -1
        self._target_y_pos = -1
        self._target_z_pos = -1
        # _aborted is owned by protocol_thread; cleared there when a new
        # run is enqueued via run_protocol(). Do not clear here -- doing
        # so would race a concurrent abort() request that fired between
        # the abort and the next run kickoff.

    @staticmethod
    def _calculate_num_scans(
        protocol: Protocol,
        run_mode: SequencedCaptureRunMode,
        max_scans: int | None,
    ) -> int:
        if run_mode in (SequencedCaptureRunMode.FULL_PROTOCOL,):
            # Protocol.from_file permits period==0 as a "valid single-scan
            # marker"; protocol_time_estimator handles it. Treating it as
            # 1 scan here matches that contract -- otherwise a valid TSV
            # silently no-ops on Start with a ZeroDivisionError logged
            # but no popup.
            period_s = protocol.period()
            if period_s == 0:
                n_scans = 1
            else:
                n_scans = int(protocol.duration() / period_s)

            if max_scans is not None:
                n_scans = min(n_scans, max_scans)
        else:
            n_scans = max_scans

        return n_scans

    def progress_snapshot(self) -> tuple[int, int]:
        """Atomic (n_scans, scan_count) for cross-thread readers.

        scan_count is advanced on the protocol worker under
        _protocol_state_lock; reading it together with n_scans under the same
        lock hands the UI a consistent pair, so a torn read can't report a
        remaining count where one half updated between the two field reads.
        """
        with self._protocol_state_lock:
            return self._n_scans, self._scan_count

    def num_scans(self) -> int:
        with self._protocol_state_lock:
            return self._n_scans

    def scan_count(self) -> int:
        with self._protocol_state_lock:
            return self._scan_count

    def remaining_scans(self) -> int:
        n_scans, scan_count = self.progress_snapshot()
        return n_scans - scan_count

    def advance_scan_count(self) -> int:
        """Increment the completed-scan counter and return the new value.

        The only site that ADVANCES scan_count (the run-init reset to zero in
        _reset_vars is the other writer; both hold _protocol_state_lock). The
        counter's lock is owned here rather than reached into from the run loop.
        Called only on the protocol worker at scan completion.
        """
        with self._protocol_state_lock:
            self._scan_count += 1
            return self._scan_count

    def _init_for_new_scan(self, max_scans: int) -> bool:
        self._reset_vars()
        n_scans = self._calculate_num_scans(
            protocol=self._protocol,
            run_mode=self._run_mode,
            max_scans=max_scans,
        )
        with self._protocol_state_lock:
            self._n_scans = n_scans

        # Scan-interval pacing uses a monotonic clock, not wall time: a DST
        # change, an NTP step, or a backward clock adjustment must not stretch,
        # shrink, or stall a multi-day timelapse's inter-scan wait. Wall-clock
        # timestamps for filenames and records are taken separately where needed.
        self._start_t = time.monotonic()

        if self._disable_saving_artifacts:
            return {'status': True, 'data': None, 'error': None}

        try:
            self._parent_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            err_str = f'Unable to save data to {self._parent_dir!s}. Please select an accessible capture location.'
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
            err_str = f'Unable to initialize sequenced run directory: {ex}'
            return {'status': False, 'data': None, 'error': err_str}

        return {'status': True, 'data': None, 'error': None}

    def run_dir(self):
        return self._run_dir

    def _create_run_dir(self):
        # Directory name uses second-resolution timestamps. Runs started
        # within the same wall-clock second collide; retry with _001,
        # _002, ... up to 999 so user-visible "directory exists" errors
        # only fire on the impossibly-rare case of a thousand collisions.
        now = datetime.datetime.now()
        base_time_string = now.strftime('%Y%m%d_%H%M%S')
        candidates = [base_time_string] + [f'{base_time_string}_{i:03d}' for i in range(1, 1000)]
        for candidate in candidates:
            self._run_dir = self._parent_dir / candidate
            try:
                self._run_dir.mkdir(exist_ok=False)
                return {
                    'status': True,
                    'data': None,
                    'error': None,
                }
            except FileExistsError:
                continue
            except FileNotFoundError:
                err_str = f'Unable to save data to {self._run_dir!s}. Please select an accessible capture location.'
                return {
                    'status': False,
                    'data': None,
                    'error': err_str,
                }

        err_str = (
            f'Unable to save data to {self._run_dir!s}: '
            f'exhausted 1000 collision suffixes within the same second. '
            f'Please wait a moment and retry.'
        )
        return {
            'status': False,
            'data': None,
            'error': err_str,
        }

    def _initialize_run_dir(self):
        if self._sequence_name in (None, ''):
            self._sequence_name = 'unsaved_protocol'

        protocol_filename = self._sequence_name
        if not protocol_filename.endswith('.tsv'):
            protocol_filename += '.tsv'

        protocol_file_loc = self._run_dir / protocol_filename
        self._protocol.to_file(file_path=protocol_file_loc)

        protocol_record_file_loc = self._run_dir / ProtocolExecutionRecord.DEFAULT_FILENAME
        self._protocol_execution_record = ProtocolExecutionRecord(
            outfile=protocol_record_file_loc,
            protocol_file_loc=protocol_filename,
        )

        return True

    def reset(self):
        """Signal an in-flight run to unwind. Non-blocking for the caller.

        Hardware cleanup (queued LED-off, camera restore, multi-second
        return-to-position moves) runs on the protocol thread via the run
        loop's finally-block -- never on the caller. A UI abort lands here
        on the Kivy main thread; running cleanup inline froze the GUI for
        the full duration of the queued futures (seconds typical, minutes
        with wedged hardware). Callers that must wait for the teardown to
        finish (app shutdown) use wait_for_run_idle().
        """
        if not self._run_in_progress_event.is_set():
            return

        # Signal abort before any cleanup runs hardware. Without this, an
        # abort tears down LEDs / camera / position while the protocol
        # thread is still mid-step.
        self.protocol_thread.abort()

        if self.protocol_thread.is_running:
            # The run loop notices the abort within one tick and its
            # finally-block calls _cleanup() on the protocol thread.
            return

        # No live run loop to unwind (dispatch failed, or the thread died
        # before its cleanup). Last-resort inline cleanup so run state is
        # not orphaned; _cleanup is idempotent if the loop raced us here.
        logger.warning(
            f'[{self.LOGGER_NAME}] reset(): run flagged in-progress but the '
            'protocol thread is not running -- running cleanup inline on the '
            'calling thread as a fallback'
        )
        self._cleanup()

    def wait_for_run_idle(self, timeout_s: float) -> bool:
        """Block until the run (including its cleanup) has fully unwound.

        For callers that need the teardown complete before proceeding --
        app shutdown tears down the executors right after aborting, and
        cleanup still has hardware work queued on them.

        Args:
            timeout_s: Maximum seconds to wait.

        Returns:
            bool: True when the run is idle; False if the timeout expired
                with cleanup still in flight.
        """
        deadline = time.monotonic() + timeout_s
        while self._run_in_progress_event.is_set():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def protocol_interval(self):
        return self._protocol.period()

    def _snapshot_run_state(self) -> None:
        """Snapshot mutable settings dicts into private copies for this run.

        The live ctx.settings['stage_offset'] dict is shared by reference; a
        deepcopy here gives each run a private value so a UI mutation (or
        future programmatic edit) mid-protocol doesn't change coordinate
        transforms partway through. The next run() re-snapshots from the
        source so between-run edits are still picked up.
        """
        self._stage_offset = copy.deepcopy(self._stage_offset_source)

    def get_initial_autofocus_states(self, layer_configs: dict | None = None):
        states = {}
        ctx = _app_ctx.ctx
        for layer in common_utils.get_layers():
            if layer_configs and layer in layer_configs:
                states[layer] = layer_configs[layer].get('autofocus', False)
            else:
                with ctx.settings_lock:
                    states[layer] = settings[layer]['autofocus']
        return states

    def _acquire_led_lease_for_run(self):
        """Acquire the run's LED lease, recovering from a stranded one.

        A refused acquire means a prior run's lease was never released -- its
        thread died mid-run without cleanup. Nothing legitimately owns the LEDs
        when a fresh run begins (live-UI control is unleased; autofocus only
        nests under a running protocol), so the stranded lease is dropped and the
        lease re-acquired. Without this the run holds no lease, every STEP_LIGHT
        apply no-ops, and the whole acquisition saves dark frames.
        """
        lease = self._scope.illumination.acquire_led_lease('protocol')
        if lease is None:
            logger.warning(
                f'[{self.LOGGER_NAME} ] LED lease refused at run start (a prior '
                'run stranded it); resetting leases and re-acquiring so the run '
                'owns illumination'
            )
            self._scope.illumination.reset_led_leases()
            lease = self._scope.illumination.acquire_led_lease('protocol')
            if lease is None:
                logger.error(
                    f'[{self.LOGGER_NAME} ] LED lease still refused after reset; '
                    'the run will not illuminate'
                )
        return lease

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
        leds_state_at_end: str = 'off',
        video_as_frames: bool = False,
        initial_autofocus_states: dict | None = None,
        keep_led_between_steps: bool = False,
    ) -> bool:
        """Start a sequenced run; return whether it actually started.

        Returns:
            bool: True when the run was dispatched to the protocol
                thread. False when the run was REFUSED (already running,
                files still writing, empty protocol, validation errors,
                hardware not connected) -- each refusal notifies the user
                itself. Callers must gate any started-run follow-up
                (run_dir(), remaining_scans(), protocol_interval()) on
                this result: a refused run loads no protocol and creates
                no run directory, so those getters answer for the
                PREVIOUS run or return None.
        """
        with self._run_lock:
            if self._run_in_progress_event.is_set():
                logger.error(f'[{self.LOGGER_NAME} ] Cannot start new run, run already in progress')
                from modules.notification_center import notifications

                notifications.warning(
                    'Protocol', 'Already Running', 'A protocol run is already in progress.'
                )
                return False

        # Check if file_io_executor still has pending writes
        if self.file_io_executor.is_protocol_queue_active():
            logger.error(
                f'[{self.LOGGER_NAME} ] Cannot start new run, file writing still in progress'
            )
            from modules.notification_center import notifications

            notifications.warning(
                'Protocol',
                'Files Still Writing',
                "Previous run's files are still being written. Please wait.",
            )
            return False

        if leds_state_at_end not in (
            'off',
            'return_to_original',
        ):
            raise ValueError(f'Unsupported value for leds_state_at_end: {leds_state_at_end}')

        if protocol.num_steps() == 0:
            logger.error('[PROTOCOL] Protocol has no steps. Cannot start run.')
            from modules.notification_center import notifications

            notifications.warning(
                'Protocol',
                'No Steps',
                'Protocol has no steps. Add at least one step before running.',
            )
            return False

        # Pre-run validation: check positions within axis limits
        try:
            axis_limits = {}
            for axis in self._scope.capabilities.axes:
                # get_axis_limits returns None for axes without
                # software-enforced bounds (T axis is the canonical
                # case). Skip those -- validate_for_run only checks
                # axes present in the dict.
                limits = self._scope.motion.get_axis_limits(axis)
                if limits is not None:
                    axis_limits[axis] = limits
            validation_errors = protocol.validate_for_run(
                axis_limits=axis_limits, stage_offset=self._stage_offset
            )
            if validation_errors:
                for err in validation_errors:
                    logger.error(f'[PROTOCOL] Validation: {err}')
                logger.error(
                    f'[PROTOCOL] Protocol has {len(validation_errors)} validation error(s). Cannot start run.'
                )
                from modules.notification_center import notifications

                err_summary = '\n'.join(f'  - {err}' for err in validation_errors[:5])
                if len(validation_errors) > 5:
                    err_summary += f'\n  ... and {len(validation_errors) - 5} more (see log)'
                notifications.error(
                    'Protocol',
                    'Validation failed',
                    f'Protocol has {len(validation_errors)} validation error(s):\n{err_summary}',
                )
                return False
        except Exception as ex:
            # validate_for_run raised before producing a validation_errors
            # list -- e.g. labware loader OS error, missing objectives.json,
            # pandas exception inside the steps DataFrame. Without the
            # popup + return the run proceeded past validation and hit
            # hardware mid-run with bad coordinates. Mirrors the
            # are_all_connected exception handling below.
            logger.error(f'[PROTOCOL] Pre-run validation could not run: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Protocol',
                'Cannot validate protocol',
                f'Pre-run validation could not run: {type(ex).__name__}: {ex}. '
                f'Check the labware + objectives configuration and try again.',
            )
            return False

        try:
            if not self._scope.are_all_connected():
                logger.error('[PROTOCOL] Not all scope components connected. Cannot start run.')
                from modules.notification_center import notifications

                notifications.error(
                    'Protocol',
                    'Hardware Disconnected',
                    'Not all hardware components are connected. Check connections and try again.',
                )
                return False
        except Exception as ex:
            logger.error(f'[PROTOCOL] Error checking scope connection: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Protocol',
                'Cannot verify hardware state',
                f'Could not check hardware connection status: {type(ex).__name__}: {ex}. '
                f'Reconnect the scope and try again.',
            )
            return False

        # Snapshot stage_offset so mid-run mutations to
        # ctx.settings['stage_offset'] don't change the in-flight coordinate
        # transforms partway through a multi-day soak.
        self._snapshot_run_state()

        # Acquire the LED lease for the whole scan so live UI illumination
        # changes cannot disturb a running protocol's channels. AF steps nest a
        # child under it. A refused acquire recovers a stranded lease from a
        # hard-killed prior run, so the run always ends up owning illumination
        # (else every STEP_LIGHT apply no-ops and the run captures dark).
        self._led_lease = self._acquire_led_lease_for_run()

        # Snapshot hardware state for restoration after protocol
        self._original_led_states = self._scope.illumination.get_led_states()
        self._saved_camera_state = self._scope.imaging.save_camera_state('protocol')
        if initial_autofocus_states is not None:
            self._original_autofocus_states = initial_autofocus_states
        else:
            self._original_autofocus_states = self.get_initial_autofocus_states()

        # Lightweight copy -- shares read-only loaders, copies only the mutable
        # steps DataFrame (which AF modifies via modify_step_z_height). Much
        # cheaper than deepcopy for large protocols (M14).
        self._protocol = protocol.copy_for_execution()
        self._run_mode = run_mode
        self._sequence_name = sequence_name
        self._parent_dir = parent_dir
        self._image_capture_config = image_capture_config
        self._enable_image_saving = enable_image_saving
        self._separate_folder_per_channel = separate_folder_per_channel
        # Snapshot at run() entry so mid-run UI mutations of the autogain
        # settings dict (target_brightness, max_duration, min/max_gain_db)
        # do not leak into the in-flight scan. Mirrors the save-encoding
        # snapshot pattern below.
        self._autogain_settings = (
            copy.deepcopy(autogain_settings) if autogain_settings is not None else {}
        )
        self._callbacks = (
            ProtocolCallbacks.from_dict(callbacks)
            if isinstance(callbacks, dict)
            else (callbacks or ProtocolCallbacks())
        )
        self._return_to_position = return_to_position
        self._disable_saving_artifacts = disable_saving_artifacts
        self._save_autofocus_data = save_autofocus_data
        self._update_z_pos_from_autofocus = update_z_pos_from_autofocus
        self._leds_state_at_end = leds_state_at_end
        self._keep_led_between_steps = keep_led_between_steps
        self._video_as_frames = video_as_frames
        # No AFE.reset() here -- AFE.run()'s own _reset_state() on
        # entry handles stale state, and self._af_future is reset at
        # scan start in protocol_run_loop. An external reset() here
        # would race with AFE.run() on the AF thread.

        self._scan_iterate_running = False
        self._protocol_iterator = None
        self._scan_iterator = None

        if self._parent_dir is None:
            self._disable_saving_artifacts = True

        self._cancel_all_scheduled_events()
        result = self._init_for_new_scan(max_scans=max_scans)
        if not result['status']:
            logger.error(f'[{self.LOGGER_NAME} ] {result["error"]}')
            return False

        ctx = _app_ctx.ctx
        stim_profiling = (
            ctx.settings.get('profiling', {}).get('stim_profiling', False)
            if ctx is not None
            else False
        )
        # PIW-3: read once per run under settings_lock to avoid per-save lock acquires
        # in image_utils.write_tiff. Mid-run UI changes intentionally do not retro-affect
        # an in-flight protocol -- saves use the value as of run-start.
        # bf_af_for_fluorescence shares the same snapshot lane so mid-run
        # toggles do not produce inconsistent AF behavior across steps
        # within one scan; protocol_step_runner reads p._bf_af_for_fluorescence.
        if ctx is not None:
            with ctx.settings_lock:
                save_encoding = config_helpers.get_image_capture_config_from_settings(ctx.settings)[
                    'save_encoding'
                ]
                self._bf_af_for_fluorescence = ctx.settings.get('protocol', {}).get(
                    'bf_af_for_fluorescence', False
                )
        else:
            save_encoding = image_mode.SAVE_ENCODING_8BIT
            self._bf_af_for_fluorescence = False

        # Borrow protocol_thread's abort Event as SCE's _aborted reference.
        # Cross-thread readers (protocol_step_runner, protocol_run_loop)
        # consult self._aborted.is_set() each tick. PIW receives a callable
        # bound to protocol_thread.abort so its capture-failure / disk-fail
        # paths abort the run.
        self._aborted = self.protocol_thread.aborted
        self._image_writer = ProtocolImageWriter(
            scope=self._scope,
            callbacks=self._callbacks,
            aborted=self._aborted,
            file_io_executor=self.file_io_executor,
            abort_fn=self.protocol_thread.abort,
            execution_record=self._protocol_execution_record,
            leds_off_fn=self._step_executor.leds_off,
            is_run_in_progress_fn=lambda: self._run_in_progress_event.is_set(),
            stim_profiling=stim_profiling,
            run_dir=self._run_dir,
            save_encoding=save_encoding,
        )

        self._run_trigger_source = run_trigger_source
        with self._run_lock:
            self._set_state(ProtocolState.RUNNING)
            self._run_in_progress_event.set()
        # The unattended scan starts here: suppress non-fatal popups (no one is
        # watching a running protocol); fatal faults still surface. Cleared on
        # every cleanup path in _cleanup_inner.
        from modules.notification_center import notifications

        notifications.set_protocol_running(True)
        self.camera_executor.disable()
        self._io_executor.protocol_start()
        self.file_io_executor.protocol_start()
        # Not IO
        self._scope.imaging.update_auto_gain_target_brightness(
            self._autogain_settings['target_brightness']
        )

        # Dispatch the main run loop onto protocol_thread. The returned
        # Future is fire-and-forget here -- completion is signalled via
        # _run_in_progress_event clearing inside _cleanup. run_protocol
        # also clears _aborted under its state lock atomically with
        # publishing the new Future, mirroring the AutofocusThread fix.
        self.protocol_thread.run_protocol(self._run_loop_executor.run_loop)
        return True

    def run_in_progress(self) -> bool:
        with self._run_lock:
            # Derive from both legacy flag and state for safety during transition
            return self._run_in_progress_event.is_set() or self._state in (
                ProtocolState.RUNNING,
                ProtocolState.SCANNING,
                ProtocolState.COMPLETING,
            )

    def run_trigger_source(self) -> str:
        return self._run_trigger_source

    def current_step_color(self) -> str | None:
        """Return the Color of the currently-executing protocol step.

        Returns None when no protocol is running, or when the step
        index / protocol cannot be resolved (early init, race during
        teardown). Callers gate on the None to fall back to UI state.
        """
        if not self.run_in_progress() or self._protocol is None:
            return None
        try:
            return self._protocol.step(idx=self._curr_step)['Color']
        except Exception:
            return None

    def _cancel_all_scheduled_events(self):
        """Cancel any remaining scheduled events.
        Note: With the loop-based approach, most work happens in executor threads,
        so there's less to unschedule than before.
        """
        # Legacy Clock.unschedule calls removed -- with the loop-based
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

    def _release_scan_led_lease(self):
        """Release the scan's LED lease (idempotent), leaving the LEDs as-is.

        leave_on: the run's end-state is set by run_cleanup's RUN_END
        transition (or, on the early-abort path, left untouched), so the
        release itself must not turn anything off. Releasing also drops any
        stranded autofocus child lease if an abort unwound out of order, so
        the next run can acquire. getattr so a stub driving _cleanup_inner
        directly need not set the slot.
        """
        led_lease = getattr(self, '_led_lease', None)
        if led_lease is not None:
            led_lease.release(leave_on=True)
            self._led_lease = None

    def _cleanup_inner(self):
        from modules.notification_center import notifications

        try:
            # Restore popups: the unattended-protocol suppression ends here, on
            # every cleanup path (normal end and abort).
            notifications.set_protocol_running(False)

            if not self._run_in_progress_event.is_set():
                # run-in-progress was already cleared, so run_cleanup (which
                # ends the executors' protocol-mode and drives the RUN_END LED
                # transition) will not run here. Guarantee the io + file
                # executors still leave protocol-mode -- an abort that cleared
                # the run flag without ending them would otherwise wedge their
                # worker on protocol_queue.get and starve normal file ops.
                # Idempotent: a no-op when not in protocol-mode.
                self._io_executor.end_protocol_mode()
                self.file_io_executor.end_protocol_mode()
                return

            run_cleanup(
                get_state_fn=lambda: self._state,
                set_state_fn=self._set_state,
                run_lock=self._run_lock,
                scan_in_progress=self._scan_in_progress,
                leds_state_at_end=self._leds_state_at_end,
                original_led_states=self._original_led_states,
                original_autofocus_states=self._original_autofocus_states,
                saved_camera_state=getattr(self, '_saved_camera_state', None),
                return_to_position=self._return_to_position,
                disable_saving_artifacts=self._disable_saving_artifacts,
                protocol=self._protocol,
                protocol_execution_record=self._protocol_execution_record,
                scope=self._scope,
                callbacks=self._callbacks,
                apply_led_transition_fn=self._step_executor.apply_led_transition,
                default_move_fn=self._step_executor.default_move,
                cancel_scheduled_events_fn=self._cancel_all_scheduled_events,
                io_executor=self._io_executor,
                autofocus_thread=self.autofocus_thread,
                file_io_executor=self.file_io_executor,
                camera_executor=self.camera_executor,
                set_run_in_progress_fn=lambda v: (
                    self._run_in_progress_event.set() if v else self._run_in_progress_event.clear()
                ),
                logger_name=self.LOGGER_NAME,
            )
        finally:
            # Release on every path -- early-return, normal end, or an
            # exception mid-cleanup -- so the lease can never leak and lock out
            # the next run. After run_cleanup, not before: apply(RUN_END) runs
            # inside it and the authority refuses a released lease, so the lease
            # stays held through it; this release still runs once it returns.
            self._release_scan_led_lease()
