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

import modules.coord_transformations as coord_transformations
import modules.image_mode as image_mode

import modules.labware_loader as labware_loader
from modules.activity_claim import ActivityClaim
from modules.autofocus_runner import AutofocusRunner
from modules.exceptions import ProtocolRunRefusedError, RunStartError
from modules.protocol import Protocol
import modules.path_utils as path_utils
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.run_outcome import EndingLatch, PendingRunOutcome, RunEnding

from modules.sequential_io_executor import SequentialIOExecutor
from lvp_logger import logger
import threading

import modules.stack_builder as stack_builder
from modules.config_helpers import AutofocusSnapshot

# How often the post-run hyperstack waiter re-checks the protocol file
# queue. The build must not start until every per-step file has flushed
# (a stack built mid-flush would silently miss planes), and queue-idle is
# a poll-only signal.
_HYPERSTACK_QUEUE_POLL_S = 0.5

# How long a composite merge waits for the run's own frames to reach disk
# before giving up and reporting a typed timeout. Bounded because a wedged
# writer must not hold an L2 caller open forever; generous because the
# alternative -- merging a directory that is still filling -- silently
# produces a composite missing a channel.
_MERGE_DRAIN_BOUND_S = 600.0


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


# Run kinds whose user is waiting in front of the scope, keyed by the
# RunPlan.run_trigger_source the entry point supplies.
#
# The Autofocus button is the only one today: it takes seconds and saves
# nothing, so its failures belong on screen. Every other kind this runner
# drives is a batch that runs for minutes and writes files, where a modal
# would stall the run in front of an empty chair and transient faults
# would pile up.
#
# Deliberately NOT a classification of all nine trigger sources. Several
# of them are already described elsewhere in the tree in terms that
# disagree with each other, and settling that is its own change with its
# own evidence. This set answers one question: raise popups, or log them.
_ATTENDED_RUN_TRIGGERS = frozenset({'autofocus'})


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
    autofocus_snapshot: AutofocusSnapshot
    keep_led_between_steps: bool
    return_to_position: dict | None
    stage_offset: dict
    # The settings-derived run values, resolved once by the caller that
    # owns the settings (config_helpers.get_sequenced_run_settings) and
    # frozen here so a mid-run toggle cannot make some steps of one scan
    # behave differently from others. The engine never reads settings for
    # these itself: a headless run gets exactly what its caller resolved.
    bf_af_for_fluorescence: bool
    timestamp_overlay: bool
    video_max_fps: int
    # Flat per-class AG/AE exposure ceilings ({'fluorescence': 150.0, ...}).
    ag_ae_max_exposure_ms: dict
    # Whether the run names its files with the turret position. Stated by
    # the caller from the store its world owns -- the GUI's live flag, or
    # the mode a headless session was built in -- so the engine never
    # reaches for the GUI's context to learn it.
    engineering_mode: bool
    # Per-layer blend thresholds the post-run merge needs, snapshotted by
    # the caller that has the settings. Carried on the plan rather than
    # read at merge time: the merge runs on a worker thread after the run,
    # where reading live settings is both unavailable headless and a
    # different value than the run was configured with.
    composite_thresholds_percent: dict | None = None


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
        grease move still in flight between back-to-back scans.
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
        # The step whose z-stack group has already been placed around its found
        # focus and moved to. -1 means none yet this scan. Keyed on the step
        # index because the corrective move must be issued exactly once: the
        # placement itself is idempotent, but re-issuing its move on every
        # settle poll would starve the step of its capture. Every scan focuses
        # afresh, so this clears with the rest of the per-scan state.
        self._focus_placed_step = -1

    def _reset_vars(self):
        self._run_dir = None
        self._tiling_configs_file_loc = None
        self._run_trigger_source = None
        self._image_writer = None
        # Nulled, not replaced: the next run's start() builds a fresh one
        # under the run lock. A run that never started must leave nothing
        # for a caller to wait on.
        self._run_outcome = None
        self._run_in_progress_event.clear()
        # Fresh object per run, never a shared Event cleared in place: queued
        # write tasks keep draining after a run ends, and a drain task hitting
        # a fatal fault (disk floor) would set a SHARED flag after the next
        # run's clear -- fatal-branding and force-darkening the wrong run. A
        # late set on the old run's object lands dead instead.
        self._fatal_abort_event = threading.Event()
        # The ending record shares that lifetime for the same reason: a late
        # fault from a drained writer must land on the run it belongs to, not
        # brand the successor with a cause that was never its own.
        self._ending = EndingLatch()
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
            # Period 0 is the loader's single-scan marker (Manual Z-Stack
            # and single-shot capture write it), and a duration shorter
            # than one period still runs its first scan: the run loop
            # captures scan 1 at t=0 before any period elapses. Both are
            # timedeltas, compared in seconds -- a timedelta is never equal
            # to the integer 0 -- and divided as timedeltas, which is exact
            # where float seconds would drop a scan on some ratios.
            period = protocol.period()
            duration = protocol.duration()
            if period.total_seconds() == 0:
                n_scans = 1
                single_scan_reason = 'the capture period is 0'
            else:
                n_scans = max(1, int(duration / period))
                single_scan_reason = (
                    f'the duration ({duration}) is shorter than the capture period ({period})'
                    if duration < period
                    else None
                )
            if single_scan_reason is not None:
                from modules.notification_center import notifications

                notifications.notice(
                    'Protocol',
                    'Single Scan',
                    f'This run performs a single scan because {single_scan_reason}.',
                )

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
        # Naming is this runner's; reserving the name is not. The
        # same-second collision retry lives in path_utils with the reason
        # it cannot be a check-then-create, and one copy of it means a
        # capture-location failure is diagnosed in one place.
        now = datetime.datetime.now()
        base_time_string = now.strftime('%Y%m%d_%H%M%S')
        try:
            self._run_dir = path_utils.allocate_directory(self._parent_dir / base_time_string)
        except path_utils.CaptureLocationError as exc:
            return {
                'status': False,
                'data': None,
                'error': str(exc),
            }
        return {
            'status': True,
            'data': None,
            'error': None,
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

    def reset(self, requester: str) -> None:
        """Unwind the run this caller owns. Non-blocking for the caller.

        ``requester`` is the caller's run_trigger_source, required rather
        than defaulted: a default would let a new call site tear a run
        down without ever saying who it was, which is the hole a stale UI
        toggle used to destroy a scan it never started. A caller that
        does not own the live run is refused and the run keeps going.

        Hardware cleanup (queued LED-off, camera restore, multi-second
        return-to-position moves) runs on the protocol thread via the run
        loop's finally-block -- never on the caller. A UI abort lands here
        on the Kivy main thread; running cleanup inline froze the GUI for
        the full duration of the queued futures (seconds typical, minutes
        with wedged hardware). Callers that must wait for the teardown to
        finish (app shutdown) use wait_for_run_idle().

        Raises:
            ProtocolRunRefusedError: reason 'not_run_owner' -- a
                different trigger owns the live run. Logged and notified
                once before it is raised, like every other refusal.
        """
        with self._run_lock:
            # No live run means no owner to be wrong about: a stop with
            # nothing to stop is a no-op, never a refusal.
            if not self._run_in_progress_event.is_set():
                return

            holder = self._run_trigger_source
            if requester != holder:
                self._refuse(
                    reason='not_run_owner',
                    title='Run In Progress',
                    message=(
                        f'A {holder} run is using the microscope. Stop it from the '
                        'control that started it, or let it finish.'
                    ),
                    holder='protocol',
                    holder_trigger=holder,
                )

            # Recorded only past the guards above: a Stop that was refused
            # or had nothing to stop ended no run, and must not leave a
            # reason behind for the next one to report.
            ending = RunEnding('aborted', 'stopped', 'Protocol Stopped', f'Stopped by {requester}')
            self._ending.set_if_unset(ending)
            needs_inline_cleanup = self._signal_abort_locked()

        if needs_inline_cleanup:
            self._cleanup(ending)

    def force_reset(self, reason: str) -> None:
        """Unwind the live run whoever owns it -- app shutdown only.

        Exists so the shutdown path does not have to impersonate the
        run's owner to get past reset()'s guard. A named method rather
        than a privileged requester string: a string meaning "skip the
        check" is guessable by callers that should not have it, and
        invisible to a grep for the override's users.
        """
        with self._run_lock:
            if not self._run_in_progress_event.is_set():
                return

            logger.warning(
                f'[{self.LOGGER_NAME}] force_reset({reason}): tearing down the '
                f'{self._run_trigger_source} run without an owner check'
            )
            ending = RunEnding('aborted', 'force_reset', 'Protocol Stopped', reason)
            self._ending.set_if_unset(ending)
            needs_inline_cleanup = self._signal_abort_locked()

        if needs_inline_cleanup:
            self._cleanup(ending)

    def _signal_abort_locked(self) -> bool:
        """Signal the run loop to unwind. The caller holds _run_lock.

        Returns True when no live run loop will run the cleanup, so the
        caller must run it inline -- and OUTSIDE the lock, because
        _cleanup reaches run_cleanup(run_lock=...) which takes the same
        lock again. Holding it across that call self-deadlocks.
        """
        # Signal abort before any cleanup runs hardware. Without this, an
        # abort tears down LEDs / camera / position while the protocol
        # thread is still mid-step.
        self.protocol_thread.abort()

        if self.protocol_thread.is_running:
            # The run loop notices the abort within one tick and its
            # finally-block calls _cleanup() on the protocol thread.
            return False

        # No live run loop to unwind (dispatch failed, or the thread died
        # before its cleanup). Last-resort inline cleanup so run state is
        # not orphaned; _cleanup is idempotent if the loop raced us here.
        logger.warning(
            f'[{self.LOGGER_NAME}] run flagged in-progress but the protocol '
            'thread is not running -- running cleanup inline on the calling '
            'thread as a fallback'
        )
        return True

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

    def _refuse_exclusive_activity(self, holder: str | None) -> None:
        """Refuse this run because an exclusive activity holds the session claim.

        Shared by the prepare-time look and the start-time claim so the two
        phases cannot describe the same holder in two different ways.
        """
        if holder == 'recording':
            title = 'Recording Active'
            message = (
                'A video recording is in progress. Stop it or let it finish, then start the run.'
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
            holder=holder,
            holder_trigger=(self._run_trigger_source if holder == 'protocol' else None),
        )

    def _acquire_led_lease_for_run(self):
        """Acquire the run's LED lease, or None when a live owner holds it.

        The illumination API arbitrates contention on the resource: a
        provably-dead holder (a hard-killed prior run) is reclaimed with
        evidence logged and the acquire succeeds, so a fresh run still
        recovers from a stranded lease. A LIVE holder (an interactive
        autofocus sweep, a future standalone recording) refuses us -- and
        a refused run must refuse itself rather than steal authority
        mid-sweep and leave the holder scanning dark.

        Runs inside start()'s gate-and-commit lock, BEFORE the run
        commits, so None becomes a refusal like every other: nothing
        committed, no terminal callback, no run directory on disk. The
        caller owns that translation, because the caller is what holds
        the claim this must release before it refuses.

        Returns None rather than raising on contention; a raise here is
        an acquire-time fault, not a busy holder.
        """
        # The claim, not the run flag, is this run's in-flight fact at
        # acquire time. The flag is not set until the end of the locked
        # block below, so a probe reading it would answer False here and
        # the acquire would reject its own caller. The claim is taken
        # immediately above and released only after the lease is released,
        # so it brackets the lease's whole life.
        #
        # Generation-scoped as well, because the claim is one object reused
        # across runs: a stale lease from a hard-killed prior run would
        # otherwise vouch for itself with the RETRYING run's claim. Binding
        # the probe to this run's generation makes the prior run's lease
        # provably dead as soon as a newer run starts.
        generation = self._run_generation
        return self._scope.illumination.acquire_led_lease(
            'protocol',
            alive=lambda: self._activity_claim_held and self._run_generation == generation,
        )

    def _refuse(
        self,
        reason: str,
        title: str,
        message: str,
        severity: str = 'warning',
        holder: 'str | None' = None,
        holder_trigger: 'str | None' = None,
    ) -> typing.NoReturn:
        """Log, notify once, and raise the typed refusal.

        The single funnel every refusal gate routes through, so a refusal
        is always exactly one log line + one user notification + one typed
        exception -- callers reconcile their own state without re-notifying.
        """
        logger.error(f'[{self.LOGGER_NAME} ] Run refused ({reason}): {message}')
        from modules.notification_center import REFUSAL_OPERATION_KEY, notifications

        notify = notifications.error if severity == 'error' else notifications.warning
        notify(
            'Protocol',
            title,
            message,
            solicited=True,
            operation_key=REFUSAL_OPERATION_KEY,
        )
        raise ProtocolRunRefusedError(
            reason=reason,
            title=title,
            message=message,
            holder=holder,
            holder_trigger=holder_trigger,
        )

    def prepare(
        self,
        protocol: Protocol,
        run_trigger_source: str,
        run_mode: SequencedCaptureRunMode,
        sequence_name: str,
        image_capture_config: image_mode.ImageCaptureConfig,
        autogain_settings: dict,
        *,
        autofocus_snapshot: AutofocusSnapshot,
        parent_dir: pathlib.Path | None = None,
        enable_image_saving: bool = True,
        separate_folder_per_channel: bool = False,
        callbacks: dict[str, typing.Callable] | None = None,
        max_scans: int | None = None,
        return_to_position: dict | None = None,
        disable_saving_artifacts: bool = False,
        save_autofocus_data: bool = False,
        # Reachable ONLY from the autofocus-scan run mode, whose completion
        # harvests the focused Z column back into the user's protocol. It is
        # absent from get_sequenced_run_settings, so no Run button and no L2
        # caller can turn it on: in an ordinary run it is always False and the
        # write-backs it guards never fire. Anything a normal run must do with
        # a found focus therefore cannot be gated on this.
        update_z_pos_from_autofocus: bool = False,
        leds_state_at_end: str = 'off',
        video_as_frames: bool = False,
        keep_led_between_steps: bool = False,
        bf_af_for_fluorescence: bool = False,
        timestamp_overlay: bool = True,
        video_max_fps: int = 0,
        ag_ae_max_exposure_ms: dict | None = None,
        composite_thresholds_percent: dict | None = None,
        engineering_mode: bool = False,
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
                    holder='protocol',
                    holder_trigger=self._run_trigger_source,
                )

        # A foreign exclusive activity (a video recording) is the durable,
        # user-actionable reason a run cannot start, and start()'s claim is
        # the only thing that used to see it -- so a prepare() refused for
        # the transient file drain below reported "files still writing,
        # please wait" while the recording was the real blocker and waiting
        # could never clear it. Look before the transient gates so the
        # refusal names what the user has to act on. Only a FOREIGN holder
        # is read here: a 'protocol' holder is this run subsystem's own
        # claim, which already_running and the file-drain gates describe
        # with better messages. The claim is still TAKEN in start() under
        # the run lock -- prepare() stays a no-op, and an activity that
        # begins after this look is caught there.
        activity_holder = self._activity_claim.owner
        if activity_holder is not None and activity_holder != 'protocol':
            self._refuse_exclusive_activity(activity_holder)

        if self.file_io_executor.is_protocol_queue_active():
            # Module layer must not popup-with-buttons, so the refusal only
            # NAMES the stalled-vs-draining difference; the recovery action
            # itself lives with the UI gate helper and the Session method.
            if self.file_io_executor.protocol_drain_stalled(WRITE_STALL_FATAL_S):
                self._refuse(
                    holder_trigger=self._run_trigger_source,
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
                holder_trigger=self._run_trigger_source,
            )

        # Nearly vestigial now that a standalone autofocus is itself a
        # run (already_running fires first) -- kept deliberately for the
        # abort-tail window where the AF thread is still winding down
        # after the run flag clears.
        # A live interactive autofocus owns the Z axis and the LED lease;
        # starting a run under it would contest Z motion and steal
        # illumination mid-sweep (dark AF frames, garbage focus). An AF
        # enqueued AFTER this check but before start()'s lease acquire
        # loses the lease race and is now REFUSED there rather than
        # aborting itself loudly -- 'illumination_held', naming the
        # holder. That race is this gate's shadow: because this check
        # fires first and covers every clickable case, the lease refusal
        # is reachable only through that milliseconds-wide inversion (or
        # a future non-AF holder), which is why it is pinned by tests and
        # not by a sim scenario. The window closes for good when AF
        # acquires its lease at enqueue time instead of on the worker.
        in_flight_sweep = (
            self.autofocus_thread.in_flight_sweep if self.autofocus_thread is not None else None
        )
        if in_flight_sweep is not None:
            self._refuse(
                reason='autofocus_running',
                title='Autofocus Running',
                message=(
                    f'An autofocus sweep from the {in_flight_sweep.run_trigger_source} run '
                    'is still running. Stop it or let it finish, then start the run.'
                ),
                holder='autofocus',
                holder_trigger=in_flight_sweep.run_trigger_source,
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
            # The states freeze with the plan; the restorer is a function,
            # which deepcopy leaves as the same object, so it keeps writing
            # the live session dict at cleanup.
            autofocus_snapshot=copy.deepcopy(autofocus_snapshot),
            keep_led_between_steps=keep_led_between_steps,
            return_to_position=return_to_position,
            stage_offset=stage_offset,
            bf_af_for_fluorescence=bf_af_for_fluorescence,
            timestamp_overlay=timestamp_overlay,
            video_max_fps=video_max_fps,
            ag_ae_max_exposure_ms=copy.deepcopy(ag_ae_max_exposure_ms or {}),
            composite_thresholds_percent=composite_thresholds_percent,
            engineering_mode=engineering_mode,
        )

    def _take_auto_gain_arm_for_run(self) -> None:
        """Hold a live-view auto-gain arm for the run's duration.

        The snapshot just taken recorded the arm and the cleanup restore
        re-arms it. Left standing, an auto-gain-off protocol's first
        capture locks it and, because a live-view arm resumes after a
        capture, re-arms it -- every capture of a run the user set to
        manual then runs auto and pays the settle -- and a protocol's
        first autofocus step scans at the live view's values instead of
        the step's. The manual autofocus one-shot keeps the arm: it
        focuses the field the user is watching, live arm included, and
        its own lock scans at what that arm achieved.
        """
        arm = self._saved_camera_state.get('auto_gain_arm')
        if arm is None or self._run_mode is SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN:
            return
        self._scope.imaging._set_auto_gain_impl(False, dict(arm.settings))

    def start(self, plan: RunPlan) -> 'PendingRunOutcome':
        """Commit to the prepared run and dispatch it.

        The commitment point: once entered, the run's terminal callback
        (run_complete) fires exactly once on every path -- normal
        completion, abort, or a setup failure, which unwinds through the
        same cleanup as a mid-run failure (with status 'failed_at_start').
        There is no path on which a caller waits forever.

        The exceptions are the pre-commitment refusals: when another
        run started between this plan's prepare() and its start(), an
        exclusive activity (a video recording) holds the session's
        activity claim, or a live owner holds the illumination lease, the
        typed refusal raises here BEFORE any commitment. Treating those
        as a failed run instead would fire this plan's completion
        callbacks while the other, live activity is mid-flight --
        clearing running-state the live activity still owns.

        Returns:
            This run's outcome. Already resolved for every run kind that
            has no merge, so a caller always gets an answer rather than
            the bound.

        Raises:
            ProtocolRunRefusedError: reason 'already_running' for the
                prepare-to-start race, 'exclusive_activity_running' when
                the session's activity claim is held (e.g. a video
                recording in progress), or 'illumination_held' when a
                live owner (an autofocus sweep) holds the LED lease;
                'holder' names it.
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
                    holder='protocol',
                    holder_trigger=self._run_trigger_source,
                )

            if not self._activity_claim.try_claim('protocol'):
                self._refuse_exclusive_activity(self._activity_claim.owner)
            self._activity_claim_held = True

            # Bumped before the acquire below, because that acquire's
            # liveness probe compares against this value: a generation
            # captured before the bump would make the live run's own probe
            # answer False for its entire life, and a later contender would
            # read this run as stranded and reclaim its lease mid-scan.
            self._run_generation += 1

            # The LED lease covers the whole scan so live UI illumination
            # changes cannot disturb a running protocol's channels; AF steps
            # nest a child under it. Acquired HERE, before the first state
            # write, so a live holder is a refusal rather than a run that
            # committed and then failed itself -- the holder keeps authority
            # and this caller gets the same nothing-committed contract every
            # other refusal gives.
            #
            # The except covers ANY exit, not just the None one: the probe
            # path and a stale holder's own probe both run inside the
            # acquire, so a raise there would leave the claim held for the
            # life of the process and refuse every future run and recording.
            try:
                lease = self._acquire_led_lease_for_run()
            except BaseException:
                self._release_activity_claim()
                raise
            if lease is None:
                holder = self._scope.illumination.led_lease_owner
                holder_desc = f'Another operation ({holder})' if holder else 'Another operation'
                self._release_activity_claim()
                self._refuse(
                    reason='illumination_held',
                    title='Illumination In Use',
                    message=(
                        f'{holder_desc} is controlling the microscope illumination. '
                        'Stop it or let it finish, then start the run.'
                    ),
                    holder=holder,
                    holder_trigger=None,
                )
            self._led_lease = lease

            self._reset_vars()
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
            self._bf_af_for_fluorescence = plan.bf_af_for_fluorescence
            self._timestamp_overlay = plan.timestamp_overlay
            self._video_max_fps = plan.video_max_fps
            self._ag_ae_max_exposure_ms = plan.ag_ae_max_exposure_ms
            self._engineering_mode = plan.engineering_mode
            self._stage_offset = plan.stage_offset
            self._composite_thresholds_percent = plan.composite_thresholds_percent
            self._run_trigger_source = plan.run_trigger_source
            # Failure-safe defaults: a setup failure below unwinds through
            # the normal run cleanup, which reads these; a prior run's stale
            # snapshots must not leak into that unwind.
            self._original_led_states = None
            self._saved_camera_state = None
            self._autofocus_snapshot = plan.autofocus_snapshot
            # The autofocus result belongs to the run that produced it, so
            # this run drops the previous one's before it can be mistaken
            # for this run's answer. Placed here, ahead of the failure
            # window below, so a run that fails at start also reports with
            # no result rather than with its predecessor's.
            #
            # Not a full AFE.reset(): that wipes _params, which AFE.run()
            # reads on the AF thread. Clearing only the result is safe
            # because run() writes it and never reads it. self._af_future
            # is reset separately at scan start in protocol_run_loop.
            if self._autofocus_runner is not None:
                self._autofocus_runner.clear_result()

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

            # Created here, inside the gate-and-commit lock and after both
            # refusals: a refused start leaves no outcome object at all, so a
            # caller that never started a run cannot wait on one. It must
            # exist BEFORE the run flag is set, because setting the flag is
            # what lets reset() drive cleanup into a finally that reads it.
            #
            # Bound to a local as well, and the local is what start() returns:
            # a caller then holds the outcome belonging to the run it actually
            # started. Reading the attribute back afterwards is a race -- a run
            # that fails at start releases the activity claim synchronously, so
            # a rival can commit in between and the caller waits on the rival's
            # run instead of its own.
            outcome = PendingRunOutcome()
            self._run_outcome = outcome

            self._set_state(ProtocolState.RUNNING)
            self._run_in_progress_event.set()

        try:
            # Declare whether anyone is watching, so non-fatal popups are
            # suppressed for a batch nobody is in front of and delivered for
            # an operation the user is waiting on. Cleared on every cleanup
            # path in _cleanup_inner.
            #
            # Passing an unconditional True here is what silenced the
            # Autofocus button's own failure popup: the button runs through
            # this runner, so the run suppressed the very message it existed
            # to produce, ~0.5s before cleanup lowered the flag again.
            from modules.notification_center import notifications

            notifications.set_unattended_run(plan.run_trigger_source not in _ATTENDED_RUN_TRIGGERS)

            # Resolved once here, before anything touches the disk, so a
            # scope with no registered source path fails the run at start
            # with no run directory left behind -- rather than raising
            # later on the post-run daemon thread, where nothing is
            # watching. Both post-run steps take the value captured here.
            self._tiling_configs_file_loc = self._scope.protocols.tiling_configs_path()

            self._setup_run_dir()

            # Snapshot hardware state for restoration after protocol
            self._original_led_states = self._scope.illumination.get_led_states()
            self._saved_camera_state = self._scope.imaging.save_camera_state('protocol')
            self._take_auto_gain_arm_for_run()

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
                ending=self._ending,
                execution_record=self._protocol_execution_record,
                leds_off_fn=self._step_executor.leds_off,
                is_run_in_progress_fn=lambda: self._run_in_progress_event.is_set(),
                image_capture_config=self._image_capture_config,
                timestamp_overlay=self._timestamp_overlay,
                video_max_fps=self._video_max_fps,
                engineering_mode=self._engineering_mode,
            )

            self.camera_executor.disable()
            self._io_executor.protocol_start()
            self.file_io_executor.protocol_start()
            # Not IO. The impl, not the dispatcher: the camera lane was
            # disabled two lines up, so the public form would refuse the
            # run's own bring-up write.
            self._scope.imaging._update_auto_gain_target_brightness_impl(
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
                raise RunStartError(
                    'dispatch_refused',
                    'Run failed to start',
                    str(dispatch_future.exception()),
                ) from dispatch_future.exception()
        except Exception as exc:
            self._fail_run_at_start(exc)

        # Reached on the failed-at-start path too, where the unwind has
        # already resolved the outcome: the caller waits and is told at once.
        return outcome

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
            raise RunStartError(
                'capture_location_unusable',
                'Run failed to start',
                f'Unable to save data to {self._parent_dir!s}. '
                'Please select an accessible capture location.',
            ) from None

        result = self._create_run_dir()
        if not result['status']:
            # The allocator's own sentence: every failure it reports is about
            # the capture location, so it shares that code.
            raise RunStartError('capture_location_unusable', 'Run failed to start', result['error'])

        try:
            self._initialize_run_dir()
        except Exception as ex:
            # The exception text stays in the log. A message field is read by
            # a popup and serialised by a remote caller; a raw traceback
            # string is neither a sentence nor safe to put in front of them.
            raise RunStartError(
                'run_dir_init_failed',
                'Run failed to start',
                'The run folder could not be initialized. See the log for details.',
            ) from ex

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
        if isinstance(exc, RunStartError):
            ending = RunEnding('failed_at_start', exc.reason, exc.title, exc.message)
        else:
            # Anything else reaching here is not an L1 sentence -- a serial
            # fault from the camera-state save, say. The user gets the one
            # thing that is true and actionable; the log has the rest.
            ending = RunEnding(
                'failed_at_start',
                'start_failed',
                'Run failed to start',
                'The run could not start. See the log for details.',
            )
        self._ending.set_if_unset(ending)
        self._cleanup(ending)
        # Notify AFTER cleanup: on an unattended run start() enabled the popup
        # suppression, which drops this non-fatal error until cleanup's
        # set_unattended_run(False) restores popups.
        from modules.notification_center import notifications

        notifications.error('Protocol', ending.title, ending.message)

    def abort_run_fatal(self, reason: str, title: str, message: str) -> None:
        """End the run now: abort, mark it dark, record the cause, darken.

        The one way the run loop and the step runner reach the fatal-abort
        funnel. The funnel lives on the image writer because the writer owns
        the per-run flag and record it sets; routing through here keeps its
        callers out of a peer's privates and gives them one shape to call.

        The writer is built before the run is dispatched and is replaced only
        by the next run's reset, so it is never absent while a run is live.
        """
        self._image_writer._abort_run_fatal(reason, 'Protocol', title, message)

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

    def _cleanup(self, ending: RunEnding):
        """Unwind the run; ending names the terminal outcome and its cause.

        ending is REQUIRED so every cleanup site states the truth it
        knows -- a defaulted value would let an abort or failure silently
        report itself as a normal completion to run_complete subscribers.
        It is what this site believes; a fault that recorded its own
        cause into the run's ending latch outranks it, and cleanup
        resolves the two in one read below.
        """
        if not self._cleanup_lock.acquire(blocking=False):
            return  # Another thread is already cleaning up
        try:
            self._cleanup_inner(ending)
        finally:
            self._cleanup_lock.release()

    def _release_scan_led_lease(self):
        """Release the scan's LED lease (idempotent), leaving the LEDs as-is.

        leave_on: the run's end-state is set by run_cleanup's RUN_END
        transition when it applies -- and when it does not, _cleanup_inner
        forces all channels dark before this release -- so the release
        itself must not turn anything off. Releasing also drops any
        stranded autofocus child lease if an abort unwound out of order, so
        the next run can acquire. getattr so a stub driving _cleanup_inner
        directly need not set the slot.
        """
        led_lease = getattr(self, '_led_lease', None)
        if led_lease is not None:
            led_lease.release(leave_on=True)
            self._led_lease = None

    def _settle_run_outcome(self, ending: RunEnding) -> None:
        """Arm the merge on a completed composite; settle every other ending.

        The ending is carried into the outcome rather than restated here:
        the status and reason a caller reads are the ones whatever ended
        the run recorded, and this method decides only whether a merge is
        still owed. Only a completed composite is, so every other run
        settles immediately -- a caller waiting on a scan's outcome gets
        an answer rather than the bound.

        Never raises: it runs inside cleanup's finally ahead of the
        activity-claim release, and a raise here would leak the claim and
        refuse every future run and recording.
        """
        outcome = getattr(self, '_run_outcome', None)
        if outcome is None:
            return
        try:
            if (
                ending.status != 'completed'
                or self._run_mode is not SequencedCaptureRunMode.SINGLE_COMPOSITE
            ):
                outcome.resolve_if_pending(ending)
                return
            if self._start_composite_merge(outcome, ending) is None:
                # Either something already settled the run, or the merge
                # declined to start and said why. Both leave the outcome
                # resolved; neither leaves it armed with nothing coming.
                outcome.resolve_if_pending(ending, 'merge_not_started')
        except Exception:
            logger.error(
                f'[{self.LOGGER_NAME}] Failed to settle the run outcome; '
                'resolving it so no caller waits on a run that ended',
                exc_info=True,
            )
            outcome.force_resolve('cleanup_error', fallback=ending)

    def run_outcome(self) -> 'PendingRunOutcome | None':
        """This run's outcome, or None when no run has started."""
        return getattr(self, '_run_outcome', None)

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
        tiling_configs_file_loc = self._tiling_configs_file_loc

        def _wait_for_queue() -> bool:
            while self.file_io_executor.is_protocol_queue_active():
                time.sleep(_HYPERSTACK_QUEUE_POLL_S)
            return True

        return self._spawn_post_run_step(
            name='hyperstack-build',
            wait_fn=_wait_for_queue,
            build_fn=lambda: stack_builder.build_hyperstacks_for_run(
                run_dir=run_dir,
                has_turret=has_turret,
                tiling_configs_file_loc=tiling_configs_file_loc,
            ),
        )

    def _start_composite_merge(
        self, outcome: PendingRunOutcome, ending: RunEnding
    ) -> threading.Thread | None:
        """Merge this run's per-channel frames, then settle the outcome.

        Runs only for a composite run that reached 'completed'. Every exit
        settles the outcome through the arming token -- success, a merge
        that produced nothing, a raise, or the write bound expiring -- so a
        caller blocked on the result is always released with a real answer
        rather than a timeout it has to interpret.

        The run's objects are captured BY VALUE here, at arming: the next
        run's start() nulls these fields, and a merge still running would
        otherwise follow them onto the successor run's directory. The
        ending goes in at the same moment and for the same reason: the
        merge thread reports only what the merge produced, and would have
        no honest way to restate how the run itself ended.
        """
        token = outcome.arm(ending)
        if token is None:
            return None

        run_dir = self._run_dir
        writer = self._image_writer
        thresholds = self._composite_thresholds_percent
        output_format = (
            self._image_capture_config.output_format_sequenced
            if self._image_capture_config is not None
            else image_mode.OUTPUT_FORMAT_TIFF
        )
        has_turret = self._scope.capabilities.has_turret
        tiling_configs_file_loc = self._tiling_configs_file_loc

        def _fail(reason: str, detail: str) -> None:
            # The one place a merge failure becomes visible: one log line,
            # one notification, one resolved outcome -- the shape a run
            # refusal uses, so a caller waiting on the outcome never
            # re-notifies. Success is silent by design: the saved folder
            # is the record, and the button handed the UI back at run end.
            from modules.notification_center import notifications

            logger.error(f'[{self.LOGGER_NAME}] Composite merge failed ({reason}): {detail}')
            notifications.error('Protocol', 'Composite Failed', detail)
            outcome.resolve(token, merged=False, artifact_path=None, merge_reason=reason)

        # Decline-to-start is TOTAL: anything that makes a merge impossible
        # settles here and now, rather than leaving the outcome armed with
        # nothing on its way to resolve it.
        if run_dir is None:
            _fail('no_run_dir', 'The run has no directory to merge from.')
            return None
        if thresholds is None:
            _fail('no_composite_config', 'The run carries no composite thresholds.')
            return None

        def _merge():
            try:
                from modules.composite_generation import CompositeGeneration

                # The loader is told the run owns the surface: its own
                # unattended-batch notices would otherwise open a modal at
                # start and another at the end of every composite.
                result = CompositeGeneration(has_turret=has_turret).load_folder(
                    path=run_dir,
                    tiling_configs_file_loc=tiling_configs_file_loc,
                    output_format=output_format,
                    brightness_thresholds_percent=thresholds,
                    announce=False,
                )
            except Exception as ex:
                logger.error(f'[{self.LOGGER_NAME}] Composite merge raised', exc_info=True)
                _fail('merge_error', f'The merge failed with {type(ex).__name__}: {ex}')
                return
            paths = result.get('artifact_paths') or []
            if result.get('status') and paths:
                logger.info(f'[{self.LOGGER_NAME}] Composite saved: {paths[0]}')
                outcome.resolve(token, merged=True, artifact_path=paths[0], merge_reason='')
            elif result.get('status'):
                _fail('merge_failed', 'The merge finished without producing a composite file.')
            else:
                _fail(
                    result.get('reason') or 'merge_failed',
                    result.get('message') or 'See lumaviewpro.log for details.',
                )

        return self._spawn_post_run_step(
            name='composite-merge',
            wait_fn=lambda: writer is None or writer.wait_for_still_writes(_MERGE_DRAIN_BOUND_S),
            build_fn=_merge,
            on_wait_expired=lambda: _fail(
                'merge_timeout',
                f"The run's frames did not finish writing within "
                f'{_MERGE_DRAIN_BOUND_S:.0f} s; nothing was merged.',
            ),
        )

    def _spawn_post_run_step(
        self,
        *,
        name: str,
        wait_fn,
        build_fn,
        on_wait_expired=None,
    ) -> threading.Thread:
        """Run a post-run build on a daemon thread once the run's files land.

        The one owner of the wait-then-build shape both post-run steps need
        -- the per-well stack build and the composite merge. Each supplies
        its own wait, because they wait on different things: the stack build
        polls the whole protocol file queue and never gives up, while the
        merge waits on its own run's write count under a bound. Sharing the
        thread lifecycle rather than the wait keeps one place responsible
        for the daemon flag and the thread name a stall report prints.

        wait_fn returns False when its bound expires, in which case the
        build does NOT run and on_wait_expired says so instead: building
        from a directory that is still filling produces a silently
        incomplete artifact.
        """

        def _wait_and_build():
            if wait_fn():
                build_fn()
            elif on_wait_expired is not None:
                on_wait_expired()

        thread = threading.Thread(target=_wait_and_build, name=name, daemon=True)
        thread.start()
        return thread

    def _cleanup_inner(self, ending: RunEnding):
        from modules.notification_center import notifications

        # Restore popups: the unattended-run suppression ends here, on
        # every cleanup path (normal end and abort). Unconditional -- an
        # attended run never raised it, and lowering it twice is harmless,
        # where missing one lowering mutes popups for the whole session.
        notifications.set_unattended_run(False)

        if not self._run_in_progress_event.is_set():
            # run-in-progress was already cleared, so run_cleanup (which
            # ends the executors' protocol-mode and drives the RUN_END LED
            # transition) will not run here. Guarantee the io + file
            # executors still leave protocol-mode -- an abort that cleared
            # the run flag without ending them would otherwise wedge their
            # worker on protocol_queue.get and starve normal file ops.
            # Idempotent: a no-op when not in protocol-mode.
            #
            # Returns ahead of the try below, so this pass settles no
            # outcome and releases nothing: it does not own the run. The
            # releases are keyed on runner-lifetime state, so a pass
            # arriving after the owner's release could otherwise hand away
            # a claim a SUCCESSOR run had already taken.
            self._io_executor.end_protocol_mode()
            self.file_io_executor.end_protocol_mode()
            return

        led_end_state_applied = False
        try:
            # A video step's drain tail writes on its own thread; its
            # execution-record row must land before the record reconciles
            # inside run_cleanup, so wait it out here (bounded).
            writer = self._image_writer
            if writer is not None and writer.video_busy:
                logger.info('[Protocol] Waiting for video write drain before run cleanup')
                writer.wait_for_video_drains()

            # One read, here, after the last lane cleanup waits on has
            # drained -- so a fault that lands during that drain is still
            # the ending, not a word chosen before the last fact arrived.
            # Read once and passed down: cleanup's decisions must not flip
            # mid-cleanup if a new run's _reset_vars replaces these objects
            # after the run flag clears. The latch outranks the caller's
            # word because the latch is what the site that ended the run
            # wrote, while the word is what the loop knew on its way out.
            latched = self._ending.get()
            forced_dark = self._fatal_abort_event.is_set()
            ending = latched or ending
            led_end_state_applied = run_cleanup(
                get_state_fn=lambda: self._state,
                set_state_fn=self._set_state,
                run_lock=self._run_lock,
                scan_in_progress=self._scan_in_progress,
                forced_dark=forced_dark,
                leds_state_at_end=self._leds_state_at_end,
                original_led_states=self._original_led_states,
                autofocus_snapshot=self._autofocus_snapshot,
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
                ending=ending,
            )
            # After run_cleanup: the stack loader reads the execution
            # record, which reconciles inside it.
            self._start_hyperstack_build()
        finally:
            if not led_end_state_applied and getattr(self, '_led_lease', None) is not None:
                # The run's LED end-state was never decided (the RUN_END
                # transition failed, was cancelled, or cleanup raised
                # before reaching it), and the release below deliberately
                # leaves LEDs as-is. Owner-blind off, because the lit
                # channel may be recorded to an autofocus child or to no
                # owner at all, so an owner-scoped darken can miss it.
                # Gated on still holding the lease: a cleanup that never
                # owned the run's LEDs (double cleanup, early return)
                # must not darken a prior cleanup's restored end-state.
                try:
                    self._scope.illumination._leds_off_impl()
                    logger.warning(
                        f'[{self.LOGGER_NAME}] Cleanup: LED end-state undecided; '
                        'forced all channels dark before lease release'
                    )
                except Exception:
                    logger.error(
                        f'[{self.LOGGER_NAME}] Cleanup: forced LED extinguish failed',
                        exc_info=True,
                    )
            # Settle (or arm) the run's merge outcome before the releases
            # below. Cleanup runs TWICE on a normal run -- the loop's
            # 'completed' call, then the safety net's 'failed' call whose
            # early return sits inside the try -- so this must be
            # arm-or-first-wins, never a plain assignment: the second pass
            # carries a contradictory status and would otherwise report
            # 'failed' over a real merge result on every successful run.
            # Non-raising by construction, because the claim release below
            # has to run whatever happens here; a raise would leak the claim
            # and refuse every future run.
            self._settle_run_outcome(ending)
            # Release on every path -- early-return, normal end, or an
            # exception mid-cleanup -- so the lease can never leak and lock out
            # the next run. After run_cleanup, not before: apply(RUN_END) runs
            # inside it and the authority refuses a released lease, so the lease
            # stays held through it; this release still runs once it returns.
            self._release_scan_led_lease()
            # The activity claim releases on the same every-path guarantee:
            # a leaked claim would refuse every future run AND recording.
            self._release_activity_claim()
