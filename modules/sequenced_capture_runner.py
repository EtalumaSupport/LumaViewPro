# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import copy
import dataclasses
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
from modules.protocol_image_writer import ProtocolImageWriter, WRITE_STALL_FATAL_S
from modules.protocol_cleanup import run_cleanup
from modules.protocol_step_runner import ProtocolStepRunner
from modules.protocol_run_loop import ProtocolRunLoop

from modules.lumascope_api import Lumascope

import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations
import modules.image_mode as image_mode

import modules.labware_loader as labware_loader
from modules.activity_claim import ActivityClaim
from modules.autofocus_runner import AutofocusRunner
from modules.exceptions import ProtocolRunRefusedError
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord

from modules.sequential_io_executor import SequentialIOExecutor
from lvp_logger import logger
import threading

import modules.app_context as _app_ctx
import modules.stack_builder as stack_builder
from modules.settings_init import settings

# How often the post-run hyperstack waiter re-checks the protocol file
# queue. The build must not start until every per-step file has flushed
# (a stack built mid-flush would silently miss planes), and queue-idle is
# a poll-only signal.
_HYPERSTACK_QUEUE_POLL_S = 0.5


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


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """Everything a sequenced run needs, validated and computed up front.

    Built exclusively by SequencedCaptureRunner.prepare(), which performs
    every refusal check before constructing the plan. Because the plan is
    the only way to call start(), a caller physically cannot commit
    run-is-underway state (events, buttons, motion locks) before the run
    has passed every gate. The held protocol is prepare()'s private
    execution copy and the dicts are snapshots, so mid-run UI mutations
    cannot leak into an in-flight run.
    """

    protocol: Protocol
    run_mode: SequencedCaptureRunMode
    run_trigger_source: str
    sequence_name: str
    image_capture_config: image_mode.ImageCaptureConfig
    autogain_settings: dict
    callbacks: ProtocolCallbacks
    n_scans: int | None
    parent_dir: pathlib.Path | None
    enable_image_saving: bool
    separate_folder_per_channel: bool
    disable_saving_artifacts: bool
    save_autofocus_data: bool
    update_z_pos_from_autofocus: bool
    leds_state_at_end: str
    video_as_frames: bool
    initial_autofocus_states: dict | None
    keep_led_between_steps: bool
    return_to_position: dict | None
    stage_offset: dict


# Constructor sentinel: distinguishes "omitted -- build a local loader"
# from an explicit (possibly None) session-owned handle.
_BUILD_LOCALLY = object()


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
        activity_claim: ActivityClaim | None = None,
        coordinate_transformer=_BUILD_LOCALLY,
        wellplate_loader=_BUILD_LOCALLY,
    ):
        # The composing session passes its own loaders -- it owns the
        # GUARDED construction, where a corrupt labware/coordinate
        # config disables one feature with a notification instead of
        # killing the whole composition (a session-passed None stays
        # None and surfaces at use). Only a bare runner nobody composed
        # builds its own.
        self._coordinate_transformer = (
            coord_transformations.CoordinateTransformer()
            if coordinate_transformer is _BUILD_LOCALLY
            else coordinate_transformer
        )
        self._wellplate_loader = (
            labware_loader.WellPlateLoader()
            if wellplate_loader is _BUILD_LOCALLY
            else wellplate_loader
        )
        # Hold stage_offset by reference so UI edits between runs are visible
        # to the next run; prepare() takes a deepcopy into the RunPlan so an
        # in-flight protocol's coordinate transforms are immune to mid-run
        # mutations of ctx.settings['stage_offset'].
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
        # assigned in start() from protocol_thread.aborted. Tests that
        # construct SCE without a real protocol_thread can still read
        # this Event because it defaults to a local Event before start().
        self._aborted: threading.Event = threading.Event()
        self._run_in_progress_event = (
            threading.Event()
        )  # GIL-free safe replacement for _run_in_progress bool
        # Monotonic per-start() counter that scopes each run's LED-lease
        # liveness probe to ITS run: the Event above is shared across
        # runs, so without the generation a stale lease would probe live
        # again the moment the next run sets it.
        self._run_generation = 0
        # The loaded protocol exists from construction so a runner that has
        # never started (or refused to start) answers getters with None
        # instead of raising AttributeError from inside a UI handler.
        # (_run_dir gets the same treatment via _reset_vars below.)
        self._protocol = None
        self._cleanup_lock = threading.Lock()
        self._run_lock = threading.Lock()
        # Session-tier exclusivity: a protocol run and a video recording
        # can never run concurrently, arbitrated by one compare-and-claim
        # both acquire. Production callers (the GUI composition root and
        # ScopeSession) inject the session's claim; the private fallback
        # exists so a bare runner keeps the refusal semantics locally.
        self._activity_claim = activity_claim if activity_claim is not None else ActivityClaim()
        self._activity_claim_held = False
        self._grease_redistribution_event = threading.Event()
        self._grease_redistribution_event.set()

        # May be None on a bare session: an AF-bearing step then fails
        # loudly at its producer site rather than running against a
        # half-wired private AFE nobody composed.
        self._autofocus_runner = autofocus_runner

        self._scope = scope
        self._run_trigger_source = None
        # LED lease held for the duration of a scan -- acquired at run
        # start, passed to AF steps as the parent lease, released in
        # cleanup. None outside a run.
        self._led_lease = None
        self._protocol_state_lock = threading.Lock()
        self._state = ProtocolState.IDLE
        # Defensive default so attribute access before the first start()
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
        # Monotonic time of this scan's first completed capture; None until
        # it lands. The run loop re-anchors scan 1's timelapse period here
        # (the first ACQUISITION) so run setup + initial motion/AF cannot
        # shorten the first interval. Per-scan coupled data, not a latch.
        self._scan_first_capture_t = None
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
        self._image_writer = None
        self._run_in_progress_event.clear()
        # Fresh object per run, never a shared Event cleared in place: queued
        # write tasks keep draining after a run ends, and a drain task hitting
        # a fatal fault (disk floor) would set a SHARED flag after the next
        # run's clear -- fatal-branding and force-darkening the wrong run. A
        # late set on the old run's object lands dead instead.
        self._fatal_abort_event = threading.Event()
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
        self._cleanup(run_status='aborted')

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

    @property
    def video_drain_busy(self) -> bool:
        """True while a video step's write drain or finish outlives the run.

        The app-close gate reads this: a run can end (or abort) while a
        video drain tail is still writing final artifacts, and a silent
        close in that window eats the tail.
        """
        writer = self._image_writer
        return writer is not None and writer.video_busy

    @property
    def video_pending_writes(self) -> int:
        """Frames across the run's video steps not yet on disk."""
        writer = self._image_writer
        return writer.video_pending_writes if writer is not None else 0

    def discard_video_pending(self) -> None:
        """Drop the run's unwritten video backlog loudly (app-close discard)."""
        writer = self._image_writer
        if writer is not None:
            writer.discard_video_pending()

    def protocol_interval(self):
        # None before the first run: a status poller may ask before any
        # protocol is loaded, and an AttributeError from a getter is a
        # crash in a UI handler, not an answer.
        return self._protocol.period() if self._protocol is not None else None

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
        """Acquire the run's LED lease; a live holder fails the run.

        The illumination API arbitrates contention on the resource: a
        provably-dead holder (a hard-killed prior run) is reclaimed with
        evidence logged and the acquire succeeds, so a fresh run still
        recovers from a stranded lease. A LIVE holder (an interactive
        autofocus sweep, a future standalone recording) refuses us -- and
        a refused run must refuse itself rather than steal authority
        mid-sweep and leave the holder scanning dark. Runs inside
        start()'s committed phase, so the raise unwinds as an
        immediately-failed run with a notification naming the holder.
        """
        # Generation-scoped probe: the runner's in-progress Event is shared
        # across runs, so a stale lease from a hard-killed prior run would
        # probe True the moment the RETRYING run sets the event -- the stale
        # holder would vouch for itself with the new run's own liveness.
        # Binding the probe to this run's generation makes the prior run's
        # lease provably dead as soon as a newer run starts.
        generation = self._run_generation
        try:
            lease = self._scope.illumination.acquire_led_lease(
                'protocol',
                alive=lambda: (
                    self._run_in_progress_event.is_set() and self._run_generation == generation
                ),
            )
        except ValueError as ex:
            # The probe answered False at acquire time: an abort cleared the
            # in-progress event between start()'s commit and this acquire.
            # Surface it in user language, not probe mechanics.
            raise RuntimeError('The run was stopped while it was starting.') from ex
        if lease is None:
            holder = self._scope.illumination.led_lease_owner
            holder_desc = f'Another operation ({holder})' if holder else 'Another operation'
            raise RuntimeError(
                f'{holder_desc} is controlling the microscope illumination. '
                'Stop it or let it finish, then start the run.'
            )
        return lease

    def _refuse(
        self, reason: str, title: str, message: str, severity: str = 'warning'
    ) -> typing.NoReturn:
        """Log, notify once, and raise the typed refusal.

        The single funnel every refusal gate routes through, so a refusal
        is always exactly one log line + one user notification + one typed
        exception -- callers reconcile their own state without re-notifying.
        """
        logger.error(f'[{self.LOGGER_NAME} ] Run refused ({reason}): {message}')
        from modules.notification_center import notifications

        notify = notifications.error if severity == 'error' else notifications.warning
        notify('Protocol', title, message)
        raise ProtocolRunRefusedError(reason=reason, title=title, message=message)

    def prepare(
        self,
        protocol: Protocol,
        run_trigger_source: str,
        run_mode: SequencedCaptureRunMode,
        sequence_name: str,
        image_capture_config: image_mode.ImageCaptureConfig,
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
    ) -> RunPlan:
        """Validate a run request and build its immutable RunPlan.

        Mutates no runner state, touches no hardware, and writes nothing
        to disk: a refused prepare is observationally a no-op, and every
        getter (run_dir(), num_scans(), run_trigger_source()) still
        answers for the previous run. Callers commit their own
        "a run is now underway" state (events, buttons, motion locks)
        only between a successful prepare() and start().

        Returns:
            RunPlan: The validated plan to pass to start().

        Raises:
            ProtocolRunRefusedError: The run cannot start (already
                running, files still writing, empty protocol, validation
                errors, hardware not connected). The user has already
                been notified once when this raises.
            ValueError: leds_state_at_end is not a supported literal --
                a programming error at the call site, not a refusal.
            TypeError: image_capture_config is not an ImageCaptureConfig
                -- same class of call-site programming error.
        """
        with self._run_lock:
            if self._run_in_progress_event.is_set():
                self._refuse(
                    reason='already_running',
                    title='Already Running',
                    message='A protocol run is already in progress.',
                )

        if self.file_io_executor.is_protocol_queue_active():
            # Module layer must not popup-with-buttons, so the refusal only
            # NAMES the stalled-vs-draining difference; the recovery action
            # itself lives with the UI gate helper and the Session method.
            if self.file_io_executor.protocol_drain_stalled(WRITE_STALL_FATAL_S):
                self._refuse(
                    reason='files_writing_stalled',
                    title='File Writer Stalled',
                    message=(
                        "Previous run's file writer has stopped making "
                        f'progress ({self.file_io_executor.describe_running_task()}). '
                        'Recover it (discard unsaved images) before starting '
                        'a new run.'
                    ),
                )
            self._refuse(
                reason='files_writing',
                title='Files Still Writing',
                message="Previous run's files are still being written. Please wait.",
            )

        # A live interactive autofocus owns the Z axis and the LED lease;
        # starting a run under it would contest Z motion and steal
        # illumination mid-sweep (dark AF frames, garbage focus). An AF
        # enqueued AFTER this check but before start()'s lease acquire
        # still loses the lease race and aborts itself loudly -- the
        # inversion (run wins over an earlier-clicked AF) is a
        # milliseconds-wide window that closes for good when AF acquires
        # its lease at enqueue time instead of on the worker.
        if self.autofocus_thread is not None and bool(self.autofocus_thread.is_running):
            self._refuse(
                reason='autofocus_running',
                title='Autofocus Running',
                message=(
                    'Autofocus is still running. Stop it or let it finish, then start the run.'
                ),
            )

        if leds_state_at_end not in (
            'off',
            'return_to_original',
        ):
            raise ValueError(f'Unsupported value for leds_state_at_end: {leds_state_at_end}')

        # A wrong-shaped config (e.g. a legacy dict) must fail at this
        # boundary, not as an AttributeError on the protocol thread after
        # hardware has already moved to the first step.
        if not isinstance(image_capture_config, image_mode.ImageCaptureConfig):
            raise TypeError(
                'image_capture_config must be an ImageCaptureConfig (build one '
                'with ImageCaptureConfig.from_image_mode); got '
                f'{type(image_capture_config).__name__}'
            )

        if protocol.num_steps() == 0:
            self._refuse(
                reason='empty_protocol',
                title='No Steps',
                message='Protocol has no steps. Add at least one step before running.',
            )

        # Snapshot stage_offset BEFORE validation so the pre-run travel
        # check and the run's coordinate transforms use the same offset.
        # Validating against a stale prior-run snapshot could pass a step
        # the fresh offset places beyond the axis limit (or refuse one
        # that would actually run fine). The deepcopy also makes the run
        # immune to mid-run mutations of ctx.settings['stage_offset']
        # partway through a multi-day soak.
        stage_offset = copy.deepcopy(self._stage_offset_source)

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
                axis_limits=axis_limits, stage_offset=stage_offset
            )
        except Exception as ex:
            # validate_for_run raised before producing a validation_errors
            # list -- e.g. labware loader OS error, missing objectives.json,
            # pandas exception inside the steps DataFrame. Without a
            # refusal the run would proceed past validation and hit
            # hardware mid-run with bad coordinates.
            logger.error(f'[PROTOCOL] Pre-run validation could not run: {ex}')
            self._refuse(
                reason='validation_crashed',
                title='Cannot validate protocol',
                message=(
                    f'Pre-run validation could not run: {type(ex).__name__}: {ex}. '
                    f'Check the labware + objectives configuration and try again.'
                ),
                severity='error',
            )
        if validation_errors:
            for err in validation_errors:
                logger.error(f'[PROTOCOL] Validation: {err}')
            err_summary = '\n'.join(f'  - {err}' for err in validation_errors[:5])
            if len(validation_errors) > 5:
                err_summary += f'\n  ... and {len(validation_errors) - 5} more (see log)'
            self._refuse(
                reason='validation_failed',
                title='Validation failed',
                message=(
                    f'Protocol has {len(validation_errors)} validation error(s):\n{err_summary}'
                ),
                severity='error',
            )

        try:
            all_connected = self._scope.are_all_connected()
        except Exception as ex:
            logger.error(f'[PROTOCOL] Error checking scope connection: {ex}')
            self._refuse(
                reason='hardware_state_unknown',
                title='Cannot verify hardware state',
                message=(
                    f'Could not check hardware connection status: {type(ex).__name__}: {ex}. '
                    f'Reconnect the scope and try again.'
                ),
                severity='error',
            )
        if not all_connected:
            self._refuse(
                reason='hardware_disconnected',
                title='Hardware Disconnected',
                message=(
                    'Not all hardware components are connected. Check connections and try again.'
                ),
                severity='error',
            )

        # Lightweight copy -- shares read-only loaders, copies only the
        # mutable steps DataFrame (which AF modifies via
        # modify_step_z_height). Much cheaper than deepcopy for large
        # protocols.
        execution_protocol = protocol.copy_for_execution()

        if parent_dir is None:
            disable_saving_artifacts = True

        return RunPlan(
            protocol=execution_protocol,
            run_mode=run_mode,
            run_trigger_source=run_trigger_source,
            sequence_name=sequence_name,
            # Frozen value object -- immutable by construction, so the plan
            # holds a true snapshot without copying.
            image_capture_config=image_capture_config,
            # Snapshot so mid-run UI mutations of the autogain settings
            # dict (target_brightness, max_duration, min/max_gain_db) do
            # not leak into the in-flight scan.
            autogain_settings=(
                copy.deepcopy(autogain_settings) if autogain_settings is not None else {}
            ),
            callbacks=(
                ProtocolCallbacks.from_dict(callbacks)
                if isinstance(callbacks, dict)
                else (callbacks or ProtocolCallbacks())
            ),
            n_scans=self._calculate_num_scans(
                protocol=execution_protocol,
                run_mode=run_mode,
                max_scans=max_scans,
            ),
            parent_dir=parent_dir,
            enable_image_saving=enable_image_saving,
            separate_folder_per_channel=separate_folder_per_channel,
            disable_saving_artifacts=disable_saving_artifacts,
            save_autofocus_data=save_autofocus_data,
            update_z_pos_from_autofocus=update_z_pos_from_autofocus,
            leds_state_at_end=leds_state_at_end,
            video_as_frames=video_as_frames,
            initial_autofocus_states=copy.deepcopy(initial_autofocus_states),
            keep_led_between_steps=keep_led_between_steps,
            return_to_position=return_to_position,
            stage_offset=stage_offset,
        )

    def start(self, plan: RunPlan) -> None:
        """Commit to the prepared run and dispatch it.

        The commitment point: once entered, the run's terminal callback
        (run_complete) fires exactly once on every path -- normal
        completion, abort, or a setup failure, which unwinds through the
        same cleanup as a mid-run failure (with status 'failed_at_start').
        There is no path on which a caller waits forever.

        The exceptions are the pre-commitment refusals: when another
        run started between this plan's prepare() and its start(), or
        an exclusive activity (a video recording) holds the session's
        activity claim, the typed refusal raises here BEFORE any
        commitment. Treating those as a failed run instead would fire
        this plan's completion callbacks while the other, live activity
        is mid-flight -- clearing running-state the live activity still
        owns.

        Raises:
            ProtocolRunRefusedError: reason 'already_running' for the
                prepare-to-start race, or 'exclusive_activity_running'
                when the session's activity claim is held (e.g. a video
                recording in progress).
        """
        # Gate and commit under ONE lock hold: releasing between the
        # already-running check and the event set would let two
        # concurrently-prepared plans both pass the gate and interleave
        # their field writes onto the same runner.
        with self._run_lock:
            if self._run_in_progress_event.is_set():
                self._refuse(
                    reason='already_running',
                    title='Already Running',
                    message='A protocol run is already in progress.',
                )

            if not self._activity_claim.try_claim('protocol'):
                holder = self._activity_claim.owner
                if holder == 'recording':
                    title = 'Recording Active'
                    message = (
                        'A video recording is in progress. Stop it or let it '
                        'finish, then start the run.'
                    )
                else:
                    title = 'Another Activity Running'
                    message = (
                        'Another exclusive activity is using the microscope. '
                        'Let it finish, then start the run.'
                    )
                self._refuse(
                    reason='exclusive_activity_running',
                    title=title,
                    message=message,
                )
            self._activity_claim_held = True

            self._reset_vars()
            self._run_generation += 1
            self._protocol = plan.protocol
            self._run_mode = plan.run_mode
            self._sequence_name = plan.sequence_name
            self._parent_dir = plan.parent_dir
            self._image_capture_config = plan.image_capture_config
            self._enable_image_saving = plan.enable_image_saving
            self._separate_folder_per_channel = plan.separate_folder_per_channel
            self._autogain_settings = plan.autogain_settings
            self._callbacks = plan.callbacks
            self._return_to_position = plan.return_to_position
            self._disable_saving_artifacts = plan.disable_saving_artifacts
            self._save_autofocus_data = plan.save_autofocus_data
            self._update_z_pos_from_autofocus = plan.update_z_pos_from_autofocus
            self._leds_state_at_end = plan.leds_state_at_end
            self._keep_led_between_steps = plan.keep_led_between_steps
            self._video_as_frames = plan.video_as_frames
            self._stage_offset = plan.stage_offset
            self._run_trigger_source = plan.run_trigger_source
            # Failure-safe defaults: a setup failure below unwinds through
            # the normal run cleanup, which reads these; a prior run's stale
            # snapshots must not leak into that unwind.
            self._original_led_states = None
            self._saved_camera_state = None
            self._original_autofocus_states = plan.initial_autofocus_states
            # No AFE.reset() here -- AFE.run()'s own _reset_state() on
            # entry handles stale state, and self._af_future is reset at
            # scan start in protocol_run_loop. An external reset() here
            # would race with AFE.run() on the AF thread.

            self._scan_iterate_running = False
            self._protocol_iterator = None
            self._scan_iterator = None
            self._cancel_all_scheduled_events()

            with self._protocol_state_lock:
                self._n_scans = plan.n_scans
            # Scan-interval pacing uses a monotonic clock, not wall time: a
            # DST change, an NTP step, or a backward clock adjustment must
            # not stretch, shrink, or stall a multi-day timelapse's
            # inter-scan wait. Wall-clock timestamps for filenames and
            # records are taken separately where needed.
            self._start_t = time.monotonic()

            self._set_state(ProtocolState.RUNNING)
            self._run_in_progress_event.set()

        try:
            # The unattended scan starts here: suppress non-fatal popups
            # (no one is watching a running protocol); fatal faults still
            # surface. Cleared on every cleanup path in _cleanup_inner.
            from modules.notification_center import notifications

            notifications.set_protocol_running(True)

            self._setup_run_dir()

            # The LED lease covers the whole scan so live UI illumination
            # changes cannot disturb a running protocol's channels. AF steps
            # nest a child under it. The illumination API reclaims a
            # provably-dead prior holder at acquire; a LIVE holder refuses
            # us and this run fails itself rather than steal authority
            # (else the holder scans dark, or every STEP_LIGHT apply
            # no-ops and the whole acquisition captures dark).
            self._led_lease = self._acquire_led_lease_for_run()

            # Snapshot hardware state for restoration after protocol
            self._original_led_states = self._scope.illumination.get_led_states()
            self._saved_camera_state = self._scope.imaging.save_camera_state('protocol')
            if self._original_autofocus_states is None:
                self._original_autofocus_states = self.get_initial_autofocus_states()

            ctx = _app_ctx.ctx
            # bf_af_for_fluorescence is snapshotted once per run under
            # settings_lock so mid-run toggles do not produce inconsistent AF
            # behavior across steps within one scan; protocol_step_runner
            # reads p._bf_af_for_fluorescence.
            if ctx is not None:
                with ctx.settings_lock:
                    self._bf_af_for_fluorescence = ctx.settings.get('protocol', {}).get(
                        'bf_af_for_fluorescence', False
                    )
                    # Snapshot once per run so a mid-run toggle cannot make
                    # some video steps stamped and others clean; overlay-on
                    # is the shipped default.
                    self._timestamp_overlay = ctx.settings.get('video', {}).get(
                        'timestamp_overlay', True
                    )
                    # Same snapshot discipline for the global rate cap: the
                    # recording rate and the disk sizing must read one
                    # per-run value, never live settings mid-run.
                    self._video_max_fps = ctx.settings.get('video', {}).get('max_fps', 0)
            else:
                self._bf_af_for_fluorescence = False
                self._timestamp_overlay = True
                self._video_max_fps = 0

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
                fatal_abort_event=self._fatal_abort_event,
                execution_record=self._protocol_execution_record,
                leds_off_fn=self._step_executor.leds_off,
                is_run_in_progress_fn=lambda: self._run_in_progress_event.is_set(),
                image_capture_config=self._image_capture_config,
                timestamp_overlay=self._timestamp_overlay,
                video_max_fps=self._video_max_fps,
            )

            self.camera_executor.disable()
            self._io_executor.protocol_start()
            self.file_io_executor.protocol_start()
            # Not IO
            self._scope.imaging.update_auto_gain_target_brightness(
                self._autogain_settings['target_brightness']
            )

            # Dispatch the main run loop onto protocol_thread. Completion is
            # signalled via _run_in_progress_event clearing inside _cleanup.
            # run_protocol also clears _aborted under its state lock
            # atomically with publishing the new Future, mirroring the
            # AutofocusThread fix.
            dispatch_future = self.protocol_thread.run_protocol(self._run_loop_executor.run_loop)
            # A dispatch refusal is synchronous: run_protocol seals the
            # returned Future with its error BEFORE returning, while a
            # genuinely dispatched run loop leaves it unresolved for the
            # run's whole duration. A done Future here therefore means the
            # loop will never execute -- raise so the failed-at-start unwind
            # runs instead of the runner sitting committed forever.
            if dispatch_future.done() and dispatch_future.exception() is not None:
                raise dispatch_future.exception()
        except Exception as exc:
            self._fail_run_at_start(exc)

    def _setup_run_dir(self) -> None:
        """Create and initialize the run directory; raise on failure.

        Runs inside start()'s committed phase: a failure here unwinds as
        an immediately-failed run (terminal callback fires), never as a
        refusal -- the same class of event as the capture disk vanishing
        mid-scan.
        """
        if self._disable_saving_artifacts:
            return

        try:
            self._parent_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            raise RuntimeError(
                f'Unable to save data to {self._parent_dir!s}. '
                'Please select an accessible capture location.'
            ) from None

        result = self._create_run_dir()
        if not result['status']:
            raise RuntimeError(result['error'])

        try:
            self._initialize_run_dir()
        except Exception as ex:
            raise RuntimeError(f'Unable to initialize sequenced run directory: {ex}') from ex

    def _fail_run_at_start(self, exc: Exception) -> None:
        """Unwind a run that failed during start()'s setup phase.

        Routes the failure through the normal run cleanup so the terminal
        run_complete callback fires (status 'failed_at_start') and the
        executors leave protocol-mode.
        """
        logger.error(f'[{self.LOGGER_NAME} ] Run failed during start: {exc}', exc_info=True)
        run_dir = self._run_dir
        if run_dir is not None:
            # A just-created EMPTY directory is noise from a run that never
            # produced anything and is removed; a non-empty one holds
            # forensic evidence of a real failed run and is kept, like any
            # mid-run abort's.
            try:
                run_dir.rmdir()
            except OSError as rm_ex:
                logger.debug(f'[{self.LOGGER_NAME} ] Failed-start run dir kept: {rm_ex}')
        # A failed start has no usable run directory; answering with the
        # (possibly just-deleted) path would send callers' started-run
        # follow-ups (last-save-folder shortcuts) to a dead location.
        self._run_dir = None
        self._cleanup(run_status='failed_at_start')
        # Notify AFTER cleanup: start() enabled the protocol-running popup
        # suppression, which drops this non-fatal error until cleanup's
        # set_protocol_running(False) restores popups.
        from modules.notification_center import notifications

        notifications.error(
            'Protocol',
            'Run failed to start',
            'The run could not start. See the log for details.',
        )

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

    def _cleanup(self, run_status: str):
        """Unwind the run; run_status names the terminal outcome.

        run_status ('completed', 'aborted', 'failed', 'failed_at_start')
        is REQUIRED so every cleanup site states the truth it knows --
        a defaulted value would let an abort or failure silently report
        itself as a normal completion to run_complete subscribers.
        """
        if not self._cleanup_lock.acquire(blocking=False):
            return  # Another thread is already cleaning up
        try:
            self._cleanup_inner(run_status=run_status)
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

    def _release_activity_claim(self):
        """Release the run's exclusivity claim (idempotent).

        The held flag flips first so a re-entrant cleanup cannot release
        twice; the claim itself raises on a mismatched release, keeping
        any double-release loud instead of silently freeing a claim a
        newer activity now holds.
        """
        if self._activity_claim_held:
            self._activity_claim_held = False
            self._activity_claim.release('protocol')

    def _start_hyperstack_build(self) -> threading.Thread | None:
        """Kick off the post-run per-well hyperstack build, when configured.

        Runs from cleanup for every capturing run mode (an autofocus scan
        captures nothing to stack). The build waits for the protocol file
        queue to drain first -- the per-step TIFFs are its input, and a
        stack built mid-flush would silently miss planes -- then builds
        from the run's own config snapshot, never the live UI, so a
        headless / L2 run triggers exactly like a GUI run.

        Returns:
            The build thread, or None when this run does not build.
        """
        if self._run_mode is SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN:
            return None
        config = self._image_capture_config
        if config is None or config.output_format_sequenced != image_mode.OUTPUT_FORMAT_HYPERSTACK:
            return None
        run_dir = self._run_dir
        if run_dir is None:
            return None
        has_turret = self._scope.capabilities.has_turret

        def _wait_and_build():
            while self.file_io_executor.is_protocol_queue_active():
                time.sleep(_HYPERSTACK_QUEUE_POLL_S)
            stack_builder.build_hyperstacks_for_run(run_dir=run_dir, has_turret=has_turret)

        thread = threading.Thread(target=_wait_and_build, name='hyperstack-build', daemon=True)
        thread.start()
        return thread

    def _cleanup_inner(self, run_status: str):
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

            # A video step's drain tail writes on its own thread; its
            # execution-record row must land before the record reconciles
            # inside run_cleanup, so wait it out here (bounded).
            writer = self._image_writer
            if writer is not None and writer.video_busy:
                logger.info('[Protocol] Waiting for video write drain before run cleanup')
                writer.wait_for_video_drains()

            # Read once, pass a bool: cleanup's fatal decision must not flip
            # mid-cleanup if a new run's _reset_vars replaces the Event object
            # after the run flag clears.
            run_cleanup(
                get_state_fn=lambda: self._state,
                set_state_fn=self._set_state,
                run_lock=self._run_lock,
                scan_in_progress=self._scan_in_progress,
                fatal_abort=self._fatal_abort_event.is_set(),
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
                run_status=run_status,
            )
            # After run_cleanup: the stack loader reads the execution
            # record, which reconciles inside it.
            self._start_hyperstack_build()
        finally:
            # Release on every path -- early-return, normal end, or an
            # exception mid-cleanup -- so the lease can never leak and lock out
            # the next run. After run_cleanup, not before: apply(RUN_END) runs
            # inside it and the authority refuses a released lease, so the lease
            # stays held through it; this release still runs once it returns.
            self._release_scan_led_lease()
            # The activity claim releases on the same every-path guarantee:
            # a leaked claim would refuse every future run AND recording.
            self._release_activity_claim()
