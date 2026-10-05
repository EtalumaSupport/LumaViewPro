# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import copy
import dataclasses
import datetime
import functools
import pathlib
import time
import typing

from modules.protocol_state_machine import (
    ProtocolState,
    SequencedCaptureRunMode,
    validate_transition,
)
from modules.protocol_callbacks import ProtocolCallbacks
from modules.protocol_image_writer import ProtocolImageWriter, RunWriteBatch, WRITE_STALL_FATAL_S
from modules.protocol_cleanup import RunCompleteNotice, run_cleanup, schedule_files_complete
from modules.protocol_step_runner import ProtocolStepRunner
from modules.protocol_run_loop import ProtocolRunLoop

from modules.lumascope_api import AxisState, Lumascope

import modules.image_mode as image_mode

from modules.activity_claim import ActivityClaim, ActivityHolder, BorrowedClaim, Taking, acting
from modules.autofocus_runner import AutofocusRunner
from modules.exceptions import (
    CameraSettingRejected,
    CompositeFailedError,
    FileWriterStalledError,
    ProtocolRunRefusedError,
    RecordIncompleteError,
    Remedy,
    RunAlreadyEndedError,
    RunCheckFailedError,
    RunFailedToStartError,
    RunImagesNotSavedError,
    RunIncompleteError,
    RunStartError,
    SingleScanNotice,
    describe_unknown_positions,
)
from modules.protocol import Protocol
import modules.path_utils as path_utils
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.run_outcome import (
    FILES_LOST_ENDINGS,
    EndingLatch,
    PendingRunOutcome,
    RunEnding,
    RunOutcome,
)

from modules.scheduler import Scheduler
from modules.sequential_io_executor import SequentialIOExecutor
from lvp_logger import logger
import threading

import modules.stack_builder as stack_builder
from modules.config_helpers import COMPOSITE_MIN_CHANNELS, AutofocusSnapshot

# How often the run loop re-asks whether the camera lane has gone idle
# before the run takes the camera; a still's grab is tens to hundreds of
# milliseconds, so this bounds how late the run starts after it.
_CAMERA_LANE_POLL_S = 0.01

# How long a post-run build -- the composite merge, the hyperstack build --
# waits for the run's own images to reach disk before giving up and
# reporting a typed timeout. Bounded because a wedged writer must not hold an
# L2 caller open forever; generous because the alternative -- building from
# a directory that is still filling -- silently produces an artifact missing
# a channel or a plane.
_POST_RUN_WRITES_WAIT_S = 600.0


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

T = typing.TypeVar('T')


class RunHandle:
    """One run, as the caller that started it holds it: watch it, wait for it, stop it.

    What every start returns, and the only thing a caller needs about its
    run: whether it is live, whether a Stop has been accepted, its folder,
    its progress, its outcome, and its Stop. Every answer is about THIS
    run -- a handle never answers for whichever run happens to be live --
    and the handle cannot write the run's outcome: the outcome it waits
    on is the engine's, and only the engine settles it.
    """

    def __init__(self, engine: 'SequencedCaptureRunner', outcome: PendingRunOutcome) -> None:
        self._engine = engine
        self._pending = outcome
        # Written once, by the start() that made this handle, after the
        # run's setup: None for a run that saves nothing, or that never
        # started.
        self._run_dir: pathlib.Path | None = None

    def wait(self, timeout_s: float | None) -> 'RunOutcome | None':
        """How this run ended, once it no longer holds the scope.

        Blocks for the outcome, then, inside the same bound, for the run's
        teardown to finish, so a caller woken here can start the next run,
        move the stage or write a setting without a second wait. A new run
        can still be refused while this run's files finish writing; that
        refusal is the drain's.

        Returns:
            The run's outcome, or None when the bound passes first.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        outcome = self._pending.wait(timeout_s=timeout_s)
        if outcome is None:
            return None
        while self.is_live:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.02)
        return outcome

    def stop(self) -> None:
        """Stop this run. Only asks: the run ends on its own thread.

        Raises:
            RunAlreadyEndedError: no run is live.
            ProtocolRunRefusedError: reason 'run_not_live' -- another run
                is live and this one is not it.
        """
        self._engine.reset(self)

    @property
    def is_live(self) -> bool:
        return self._engine.is_live_run(self)

    @property
    def is_stopping(self) -> bool:
        """Live, and a Stop of it accepted: True until its teardown finishes."""
        return self._engine.is_stopping(self)

    @property
    def is_last_run(self) -> bool:
        """Whether this is the engine's most recent run, live or finished."""
        return self._engine.run_outcome() is self

    @property
    def run_dir(self) -> pathlib.Path | None:
        return self._run_dir

    @property
    def step_number(self) -> int | None:
        """The step executing now, counted from 1; None once this run is not live."""
        return self._engine.live_run_value(self, self._engine._step_number)

    @property
    def num_steps(self) -> int | None:
        """This run's step count; None once this run is not live.

        The run's own count: a caller that started the run through a member
        never holds the protocol the member built, so a count it made from
        settings could disagree with the run.
        """
        return self._engine.live_run_value(self, self._engine._num_steps)

    @property
    def remaining_scans(self) -> int | None:
        return self._engine.live_run_value(self, self._engine.remaining_scans)

    @property
    def interval(self) -> datetime.timedelta | None:
        """This run's scan period; None once it is not live."""
        return self._engine.live_run_value(self, self._engine.protocol_interval)


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
    # The caller's own protocol, not a copy: the one object a completed
    # autofocus scan writes its focus into.
    write_focus_to: Protocol | None
    leds_state_at_end: str
    video_as_frames: bool
    autofocus_snapshot: AutofocusSnapshot
    keep_led_between_steps: bool
    return_to_position: dict | None
    # None only on a scope never initialized, which the run gate admits only
    # when it has no X/Y stage to convert plate positions for.
    stage_offset: dict | None
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
    # A claim the run acts under instead of taking the session's: a
    # diagnostic's, lent so the run can neither be refused by the activity
    # it runs inside nor release that activity's claim when it ends. None
    # for a run that takes the scope for itself.
    borrowed_claim: BorrowedClaim | None = None


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
        protocol_thread,
        file_io_executor: SequentialIOExecutor,
        autofocus_thread,
        activity_claim: ActivityClaim,
        autofocus_runner: AutofocusRunner | None = None,
        z_ui_update_func: typing.Callable | None = None,
        on_run_idle: typing.Callable[[], None] | None = None,
    ):
        # Told once a run's cleanup has put the runner back to IDLE. The
        # claim releases just before IDLE, so a listener woken by the claim
        # alone can still read the run as live; this is the edge after
        # which is_live_run says it has ended.
        self._on_run_idle = on_run_idle
        # Stateless, so the run keeps its own. The labware catalogue is the
        # scope's, read where a run needs a plate.
        # The offset a run converts plate positions with; prepare() snapshots
        # the scope's into the RunPlan, and start() sets it from there.
        self._stage_offset: dict | None = None
        self.protocol_thread = protocol_thread
        self.file_io_executor = file_io_executor
        # The last run's write batch: created with each run and kept until
        # the next one starts, so the files a finished run still owes the
        # disk are answered per run -- by prepare's refusal, the Session and
        # the post-run builds -- never by the shared lane.
        self._write_batch: RunWriteBatch | None = None
        # The run kind of the run in hand; None before the first one.
        self._run_mode: SequencedCaptureRunMode | None = None
        self.autofocus_thread = autofocus_thread
        self._z_ui_update_func = z_ui_update_func
        self._scan_in_progress = threading.Event()
        # Abort signal. Owned by protocol_thread; SCE holds a reference
        # assigned in start() from protocol_thread.aborted. Tests that
        # construct SCE without a real protocol_thread can still read
        # this Event because it defaults to a local Event before start().
        self._aborted: threading.Event = threading.Event()
        # The loaded protocol exists from construction so a runner that has
        # never started (or refused to start) answers getters with None
        # instead of raising AttributeError from inside a UI handler.
        # (_run_dir gets the same treatment via _reset_vars below.)
        self._protocol = None
        self._run_lock = threading.Lock()
        # The live run's dispatched loop, None until start() dispatches it:
        # what tells force_reset whether a loop that will unwind the run
        # ever existed. Written and read under _run_lock.
        self._run_loop_future = None
        # Session-tier exclusivity: a protocol run and a video recording
        # can never run concurrently, arbitrated by one compare-and-claim
        # both acquire. Required, so no runner exists with a claim of its
        # own that nothing else contends for.
        self._activity_claim = activity_claim
        # The taking this runner holds, or None; only it releases the claim.
        self._held_claim: Taking | None = None
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
        # The session's scheduler and the stalled-writer check on it, armed
        # at bring-up (start_file_writer_check); None until then.
        self._file_writer_check_scheduler = None
        self._file_writer_check_handle = None
        self._protocol_state_lock = threading.Lock()
        self._state = ProtocolState.IDLE
        # Defensive default so attribute access before the first start()
        # (e.g. from a test that drives scan_iterate directly) returns
        # a no-op callbacks object instead of AttributeError.
        self._callbacks = ProtocolCallbacks()
        self._reset_vars()
        self._step_executor = ProtocolStepRunner(self)
        self._run_loop_executor = ProtocolRunLoop(self)

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
        self._run_handle = None
        self._write_focus_to = None
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
                single_scan = True
            else:
                n_scans = max(1, int(duration / period))
                single_scan = duration < period
            if single_scan:
                from modules.notification_center import notifications

                notifications.report_outcome(
                    SingleScanNotice(period=period, duration=duration),
                    solicited=False,
                    category='Protocol',
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

    def reset(self, run: 'RunHandle | None') -> None:
        """Stop *run*, the object its start() returned. Non-blocking for the caller.

        Anyone may stop the live run, and the stop names the run rather
        than the caller: who asks is self-declared and protects nothing,
        while a stale toggle naming an OLD run is the hole that once
        destroyed a scan it never started. A stop naming a run that is not
        the live one never touches the live one: while another run is live
        it is refused (logged and notified); when nothing is live it is a
        Stop that arrived after its run ended, raised but not notified.

        A Stop only asks: it records the run's ending and signals it, and
        the thread that owns the run tears it down -- the run loop's
        finally, or start() itself for a run stopped before its loop was
        dispatched. Never on the caller: a UI Stop lands on the Kivy main
        thread, and a teardown there froze the GUI for the duration of the
        queued hardware work (seconds typical, minutes with wedged
        hardware); a teardown on the Stop's thread while start() was still
        setting the run up left both lanes in run mode under an idle
        runner. Callers that must wait for the teardown to finish (app
        shutdown) use wait_for_run_idle().

        Raises:
            RunAlreadyEndedError: no run is live.
            ProtocolRunRefusedError: reason 'run_not_live' -- another run
                is live and *run* is not it.
        """
        with self._run_lock:
            if not self._is_run_live():
                raise RunAlreadyEndedError('That run has already ended; no run is live.')
            if not self._is_live_run_locked(run):
                holder = self._run_trigger_source
                self._refuse(
                    reason='run_not_live',
                    title='Run Already Ended',
                    message=(
                        f'That run has already ended. {self._the_run_holding_the_scope(holder)} '
                        'is using the microscope now; stop it from its own control.'
                    ),
                    holder='protocol',
                    holder_trigger=holder,
                )

            # Recorded only past the guard above: a Stop that was refused
            # ended no run, and must not leave a reason behind for the next
            # one to report.
            self._ending.set_if_unset(
                RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped')
            )
            self.protocol_thread.abort()

    def force_reset(self, reason: str) -> None:
        """Unwind the live run without naming it -- app shutdown only.

        Exists because the shutdown path holds no run's handle and must
        stop whatever is live. A named method rather than a special
        handle value: a value meaning "whatever is live" would be
        reachable by callers that should not have it, and invisible to a
        grep for the override's users.

        Like a Stop it records and signals, with one difference: a run
        whose loop has finished without unwinding it is unwound here, on
        the shutdown's thread. App close is the last chance to release the
        scope, and nothing else will. A run whose loop was never
        dispatched is not this method's: start() is still setting it up
        and unwinds it itself when it sees the ending.
        """
        with self._run_lock:
            if not self._is_run_live():
                return

            logger.warning(
                f'[{self.LOGGER_NAME}] force_reset({reason}): tearing down the '
                f'{self._run_trigger_source} run without an owner check'
            )
            ending = RunEnding('aborted', 'force_reset', 'Protocol Stopped', reason)
            self._ending.set_if_unset(ending)
            self.protocol_thread.abort()
            run = self._run_handle
            loop = self._run_loop_future
            loop_ended_without_unwinding = loop is not None and loop.done()

        if loop_ended_without_unwinding:
            logger.warning(
                f'[{self.LOGGER_NAME}] force_reset({reason}): the run loop ended '
                'without unwinding the run -- unwinding it on the calling thread'
            )
            self._cleanup(ending, run)

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
        while self._is_run_live():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    @property
    def _io_executor(self) -> SequentialIOExecutor:
        """The scope's IO lane. Read from the scope, never held as a copy."""
        return self._scope.io_lane()

    @property
    def camera_executor(self) -> SequentialIOExecutor:
        """The scope's CAMERA lane. Read from the scope, never held as a copy."""
        return self._scope.camera_lane()

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

    def _num_steps(self) -> int | None:
        return self._protocol.num_steps() if self._protocol is not None else None

    def _step_number(self) -> int:
        # One int, written only by the step runner as it advances; the
        # step's colour is read from the same index.
        return self._curr_step + 1

    @staticmethod
    def _the_run_holding_the_scope(holder_trigger: 'str | None') -> str:
        """Name the run that holds the scope, for a refusal to put in a sentence.

        One phrasing, one home. Every start refusal is handed the holder
        already; before this they each printed a literal instead, so a
        user turned away from a Z-stack by their own protocol read "a
        protocol run is already in progress" and had to guess which
        control to go back to. The stop refusal in reset() has always
        named it; this is that sentence's other half.

        Falls back to the indefinite form rather than printing None: a
        trigger is absent only where no run holds the scope, and a
        sentence a user reads must still parse.
        """
        if not holder_trigger:
            return 'A run'
        return f'The {holder_trigger} run'

    def _refuse_already_running(self) -> typing.NoReturn:
        """Refuse a start because a run already holds the scope.

        Shared by prepare()'s look and start()'s commit, which asked the
        same question and printed the same literal in two places -- two
        phases with one answer between them.
        """
        holder_trigger = self._run_trigger_source
        self._refuse(
            reason='already_running',
            title='Already Running',
            message=(
                f'{self._the_run_holding_the_scope(holder_trigger)} is using the microscope. '
                'Stop it from the control that started it, or let it finish.'
            ),
            holder='protocol',
            holder_trigger=holder_trigger,
        )

    def _refuse_foreign_holder(self, claim: 'ActivityClaim | BorrowedClaim') -> None:
        """Refuse when an activity other than a run holds the scope.

        Shared by prepare()'s look and start()'s commit, so the two read the
        same holder the same way. A 'protocol' holder is left to the
        already-running and file-drain gates, which describe a run better.
        """
        holder = claim.blocking_holder
        if holder is not None and holder.kind != 'protocol':
            self._refuse_exclusive_activity(holder)

    def _refuse_exclusive_activity(self, holder: 'ActivityHolder | None') -> None:
        """Refuse this run because an exclusive activity holds the session claim.

        Shared by the prepare-time look and the start-time claim so the two
        phases cannot describe the same holder in two different ways. The
        holder is the claim's own snapshot, so the kind and the run behind
        it are one fact rather than two reads that can disagree.
        """
        kind = holder.kind if holder is not None else None
        if kind == 'recording':
            title = 'Recording Active'
            message = (
                'A video recording is in progress. Stop it or let it finish, then start the run.'
            )
        else:
            title = 'Another Activity Running'
            # The kind, not the word "another": an activity the user cannot
            # name is one they cannot go and stop.
            named = f'A {kind} activity' if kind else 'Another exclusive activity'
            message = f'{named} is using the microscope. Let it finish, then start the run.'
        self._refuse(
            reason='exclusive_activity_running',
            title=title,
            message=message,
            holder=kind,
            holder_trigger=(holder.run_trigger_source if holder is not None else None),
        )

    def _refuse(
        self,
        reason: str,
        title: str,
        message: str,
        holder: 'str | None' = None,
        holder_trigger: 'str | None' = None,
        remedy: Remedy | None = None,
    ) -> typing.NoReturn:
        """Report once, and raise, the typed refusal.

        The single funnel every refusal gate routes through. The one
        reporter logs and shows it -- one WARNING line naming the reason,
        one warning, answered to whoever asked even during a run -- and the
        same exception is raised, so a caller that reports it again changes
        nothing and a caller that reconciles its own state tells no one.
        """
        from modules.notification_center import notifications

        refusal = ProtocolRunRefusedError(
            reason=reason,
            title=title,
            message=message,
            holder=holder,
            holder_trigger=holder_trigger,
            remedy=remedy,
        )
        notifications.report_outcome(refusal, solicited=True, category='Protocol')
        raise refusal

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
        # Set ONLY by ProtocolRunner.run_autofocus_all_steps, whose completed
        # scan writes the focused Z column into this protocol. It is absent
        # from get_sequenced_run_settings, so no Run button turns it on: in an
        # ordinary run it is None and the write-backs it guards never fire.
        # Anything a normal run must do with a found focus therefore cannot
        # be gated on this.
        write_focus_to: Protocol | None = None,
        leds_state_at_end: str = 'off',
        video_as_frames: bool = False,
        keep_led_between_steps: bool = False,
        bf_af_for_fluorescence: bool = False,
        timestamp_overlay: bool = True,
        video_max_fps: int = 0,
        ag_ae_max_exposure_ms: dict | None = None,
        composite_thresholds_percent: dict | None = None,
        engineering_mode: bool = False,
        borrowed_claim: BorrowedClaim | None = None,
    ) -> RunPlan:
        """Validate a run request and build its immutable RunPlan.

        Mutates no runner state, touches no hardware, and writes nothing
        to disk: a refused prepare is observationally a no-op. The
        record getters (run_dir(), num_scans()) still answer for the
        previous run; run_trigger_source() answers for whoever holds
        the scope, which a refused prepare did not change either. Callers commit their own
        "a run is now underway" state (events, buttons, motion locks)
        only between a successful prepare() and start().

        Returns:
            RunPlan: The validated plan to pass to start().

        Raises:
            ProtocolRunRefusedError: The run cannot start (already
                running, files still writing, empty protocol, validation
                errors, hardware not connected, an axis position not
                known). The user has already been notified once when
                this raises.
            ValueError: leds_state_at_end is not a supported literal --
                a programming error at the call site, not a refusal.
            TypeError: image_capture_config is not an ImageCaptureConfig
                -- same class of call-site programming error.
        """
        # A foreign exclusive activity (a video recording, a diagnostic) is
        # the durable, user-actionable reason a run cannot start, and
        # start()'s claim is the only thing that used to see it -- so a
        # prepare() refused for the transient file drain below reported
        # "files still writing, please wait" while the recording was the
        # real blocker and waiting could never clear it. Look before every
        # other gate so the refusal names what the user has to act on --
        # including before already-running: a run inside a diagnostic is
        # live under the diagnostic's claim, and "a run is using the
        # microscope, stop it from the control that started it" names a run
        # the user never started and has no control for. Only a FOREIGN
        # holder is read here: a 'protocol' holder is this run subsystem's
        # own claim, which already_running and the file-drain gates describe
        # with better messages. A run that borrows a claim asks the borrow,
        # which does not count its own lender as in the way. The claim is
        # still TAKEN in start() under the run lock -- prepare() stays a
        # no-op, and an activity that begins after this look is caught there.
        claim = borrowed_claim if borrowed_claim is not None else self._activity_claim
        self._refuse_foreign_holder(claim)

        with self._run_lock:
            if self._is_run_live():
                self._refuse_already_running()

        batch = self._write_batch
        if batch is not None and batch.draining:
            # A stalled writer will not finish on its own, so its refusal
            # carries the recovery that answers it, by name: the Session
            # owns recovery, and this layer cannot reach it.
            if batch.stalled(WRITE_STALL_FATAL_S):
                unsaved = batch.pending
                stalled = FileWriterStalledError.stalled_sentence(
                    self._the_run_holding_the_scope(self._run_trigger_source),
                    batch.describe_stuck_write(),
                )
                self._refuse(
                    holder_trigger=self._run_trigger_source,
                    reason='files_writing_stalled',
                    title=FileWriterStalledError.title,
                    message=(
                        f'{stalled} Recover the file writer before starting a new run: '
                        f'{FileWriterStalledError.cost_sentence(unsaved)}'
                    ),
                    remedy=FileWriterStalledError.recovery(unsaved),
                )
            self._refuse(
                reason='files_writing',
                title='Files Still Writing',
                message=(
                    f'{self._the_run_holding_the_scope(self._run_trigger_source)} is still '
                    'writing its files. Please wait.'
                ),
                holder_trigger=self._run_trigger_source,
            )

        # Nearly vestigial now that a standalone autofocus is itself a
        # run (already_running fires first) -- kept deliberately for the
        # abort-tail window where the AF thread is still winding down
        # after the run flag clears.
        # A winding-down autofocus still drives the Z axis and its LED
        # restore; starting a run under it would contest Z motion and
        # illumination (dark AF frames, garbage focus).
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

        # Ahead of the empty-protocol gate: a composite with no channel set
        # to capture is an empty protocol too, and "turn on another channel"
        # is the answer its user can act on, where "add a step" is not.
        if run_mode is SequencedCaptureRunMode.SINGLE_COMPOSITE:
            channels = protocol.steps()['Color'].nunique() if protocol.num_steps() else 0
            if channels < COMPOSITE_MIN_CHANNELS:
                set_to_capture = (
                    'no channel is'
                    if channels == 0
                    else f'only {channels} {"is" if channels == 1 else "are"}'
                )
                self._refuse(
                    reason='composite_needs_two_channels',
                    title='Not Enough Channels',
                    message=(
                        f'A composite combines at least {COMPOSITE_MIN_CHANNELS} channels, '
                        f'but {set_to_capture} set to capture an image. Turn on another '
                        'channel and try again.'
                    ),
                )

        if protocol.num_steps() == 0:
            self._refuse(
                reason='empty_protocol',
                title='No Steps',
                message='Protocol has no steps. Add at least one step before running.',
            )

        # Pre-run validation: the steps are well-formed enough to run
        try:
            validation_errors = protocol.validate_for_run(
                objective_helper=self._scope.objective_helper,
                wellplate_loader=self._scope.wellplate_loader,
                led_max_ma=self._scope.capabilities.led_max_ma,
            )
        except Exception as ex:
            # validate_for_run raised before producing a validation_errors
            # list -- e.g. a pandas exception inside the steps DataFrame.
            # Without this the
            # run would proceed past validation and hit hardware mid-run with
            # bad coordinates.
            raise RunCheckFailedError(
                reason='validation_crashed',
                title='Cannot validate protocol',
                message=(
                    f'Pre-run validation could not run: {type(ex).__name__}: {ex}. '
                    f'Check the labware + objectives configuration and try again.'
                ),
            ) from ex
        if validation_errors:
            for err in validation_errors:
                logger.warning(f'[PROTOCOL] Validation: {err}')
            err_summary = '\n'.join(f'  - {err}' for err in validation_errors[:5])
            if len(validation_errors) > 5:
                err_summary += f'\n  ... and {len(validation_errors) - 5} more (see log)'
            self._refuse(
                reason='validation_failed',
                title='Validation failed',
                message=(
                    f'Protocol has {len(validation_errors)} validation error(s):\n{err_summary}'
                ),
            )

        # A protocol names glass and the turret carries glass; when they
        # disagree the run does not fail, it COMPLETES and writes files that
        # lie about themselves -- the filename is built from the step's
        # objective and the metadata is read from the turret's, so the name
        # says 4x over an image taken through the 20x that was already in
        # the light path.
        #
        # Below the validation above, and that ordering is load-bearing: an
        # id that names nothing in the catalogue, or a blank cell from a
        # hand-edited protocol file, is not a turret mismatch. Judged before
        # validation it was answered with "assign the missing objectives",
        # sending the user to mount glass that does not exist. Validation
        # names the real defect first, so the ids reaching here are ones it
        # accepted and the only question left is which slot holds them.
        #
        # One hole, and it is not this gate's to close: the objective half of
        # that validation is skipped outright when the catalogue fails to
        # load, so a scope with an unreadable objectives.json still arrives
        # here with unvetted ids and gets the turret's message for them.
        #
        # The rule itself belongs to the protocol-construction API, which
        # owns it for the load and the step navigation too. Asked here
        # rather than restated: a run refusing on a rule of its own is how
        # the load and the navigation came to disagree with it.
        self._scope.protocols.refuse_unaddressable_objectives(
            protocol.steps()['Objective'].to_list()
        )
        # A step on a layer this scope lacks cannot be taken as the step
        # says. The load refuses it on the same rule; a protocol built in
        # memory meets it only here.
        self._scope.protocols.refuse_absent_layers(protocol)

        try:
            all_connected = self._scope.are_all_connected()
        except Exception as ex:
            raise RunCheckFailedError(
                reason='hardware_state_unknown',
                title='Cannot verify hardware state',
                message=(
                    f'Could not check hardware connection status: {type(ex).__name__}: {ex}. '
                    f'Reconnect the scope and try again.'
                ),
            ) from ex
        if not all_connected:
            self._refuse(
                reason='hardware_disconnected',
                title='Hardware Disconnected',
                message=(
                    'Not all hardware components are connected. Check connections and try again.'
                ),
            )

        # After the connection gate, so a motorized scope whose board fell
        # off is told it is disconnected rather than that it cannot move;
        # after the objective gate, so a protocol for glass this scope cannot
        # put in the light path is told that first. The offset is the scope's,
        # the one the gate judges X/Y with, snapshotted so the run converts
        # with the offset it was admitted at.
        self._scope.protocols.refuse_unreachable_positions(protocol.steps(), protocol.labware())
        stage_offset = copy.deepcopy(self._scope.runtime_state.get_stage_offset())
        self._scope.protocols.refuse_camera_values_out_of_range(protocol.steps())

        # Every run mode moves every axis this scope has, and each move on
        # an axis whose position is not known is refused -- so a run
        # admitted here would fail every scan and end in a disconnect's
        # words. After the connection gate: a board that dropped reports
        # its axes unknown as a consequence, and the disconnect is the
        # cause the user must act on.
        unknown_axes = self._scope.motion.axes_without_position()
        if unknown_axes:
            waiting = all(state == AxisState.HOMING for state in unknown_axes.values())
            self._refuse(
                reason='position_unknown',
                title='Scope Not Homed',
                message=(
                    f'Cannot start the run: {describe_unknown_positions(unknown_axes)}. '
                    + (
                        'Wait for the home to finish, then start the run.'
                        if waiting
                        else 'Home the scope, then start the run.'
                    )
                ),
            )

        # The last gate, and the only one about where the run SAVES rather
        # than about the instrument. Without it a bad save location is
        # discovered after the run has committed, moved the stage and taken
        # images -- a failed run for a request that could have been turned
        # away. It is last so that it cannot reorder any refusal that was
        # already reachable.
        #
        # parent_dir None means the run writes nowhere at all (a
        # non-engineering standalone autofocus), so it has no location to
        # be unusable. A parent_dir with artifacts suppressed is NOT that
        # case: the autofocus characterization data still lands there.
        if parent_dir is not None:
            location_problem = path_utils.capture_location_problem(parent_dir)
            if location_problem is not None:
                self._refuse(
                    # Kept a literal at the raise: the refusal-vocabulary
                    # census collects reason= only when it is one, so a
                    # reason hoisted into a local becomes invisible to the
                    # guard that makes the vocabulary a contract.
                    reason='capture_location_unusable',
                    title='Save Location Unusable',
                    message=(
                        f'Cannot save this run to {parent_dir}: {location_problem}. '
                        'Reconnect the drive or choose an accessible save '
                        'location, then try again.'
                    ),
                )

        # Lightweight copy -- shares read-only loaders, copies only the
        # mutable steps DataFrame (which AF modifies via
        # modify_step_z_height). Much cheaper than deepcopy for large
        # protocols.
        execution_protocol = protocol.copy_for_execution()

        if parent_dir is None:
            disable_saving_artifacts = True

        return RunPlan(
            borrowed_claim=borrowed_claim,
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
            write_focus_to=write_focus_to,
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
        try:
            self._scope.imaging.set_auto_gain(False, dict(arm.settings))
        except CameraSettingRejected as rejected:
            # The run goes on with the camera as it is; the refusal ends its
            # flight here.
            from modules.notification_center import notifications

            notifications.report_outcome(rejected, solicited=False, category='Camera')

    def _take_camera(self) -> 'RunEnding | None':
        """Wait for the camera lane to finish what it holds, then make the camera this run's.

        start() puts the lane in protocol mode for the run, and the lane
        refuses what was queued there before; but a command already
        running (a still, a gain write, a settings widget's task) finishes
        on the lane's worker. Taking the camera before it has finished let
        the run's first LED land under a still's grab and its snapshot
        read the camera mid-command. So every read and write of the camera
        the run makes comes after the lane is idle, snapshots included: a
        snapshot taken
        before the lane's last command records the state that command was
        about to replace, and the restore at the end would hand that stale
        state back.

        Runs on the protocol thread, so the wait never holds the caller
        that clicked. Returns None once the camera is taken, or when the
        run is stopping -- a Stop during the wait -- so a stopped run
        never writes the camera or takes a snapshot nothing will restore;
        the loop's own tail ends the stopped run, and cleanup finds no
        snapshot. An
        ending when the lane's in-flight task is stuck past the threshold
        that already calls the file lane wedged.
        """
        while self.camera_executor.is_busy():
            if self._aborted.is_set() or not self._is_run_live():
                return None
            if self.camera_executor.in_flight_task_stalled(WRITE_STALL_FATAL_S):
                return RunEnding(
                    'failed',
                    'camera_lane_stalled',
                    'Camera Busy',
                    'A camera command did not finish, so the run did not start. '
                    'Restart LumaViewPro if the camera stays busy.',
                )
            time.sleep(_CAMERA_LANE_POLL_S)
        if not self._is_run_live():
            return None
        self._original_led_states = self._scope.illumination.get_led_states()
        self._saved_camera_state = self._scope.imaging.save_camera_state('protocol')
        # No target brightness is written here: every step that arms
        # auto-gain, and every one-shot, hands the camera its target in the
        # same write.
        self._take_auto_gain_arm_for_run()
        return None

    def start(self, plan: RunPlan) -> RunHandle:
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
            This run's handle. Its outcome is already resolved for every
            run kind that has no merge, so a caller always gets an answer
            rather than the bound.

        Raises:
            ProtocolRunRefusedError: reason 'already_running' for the
                prepare-to-start race, 'exclusive_activity_running' when
                the session's activity claim is held (e.g. a video
                recording in progress); 'holder' names it.
        """
        # The session's claim, or the one this run was lent. Taken and
        # released the same way either way: a borrowed taking's release
        # leaves the lender's claim held, so neither release site below can
        # end the activity this run runs inside.
        claim = plan.borrowed_claim if plan.borrowed_claim is not None else self._activity_claim
        # Gate and commit under ONE lock hold: releasing between the
        # already-running check and the event set would let two
        # concurrently-prepared plans both pass the gate and interleave
        # their field writes onto the same runner.
        with self._run_lock:
            self._refuse_foreign_holder(claim)
            if self._is_run_live():
                self._refuse_already_running()

            # The claim carries WHICH run holds the scope, written by the
            # same call that takes it: the holder question has one store,
            # and it is the one that already knows whether anything holds
            # the scope at all.
            held = claim.try_claim('protocol', run_trigger_source=plan.run_trigger_source)
            if held is None:
                self._refuse_exclusive_activity(claim.holder)
            self._held_claim = held

            # The LED lease covers the whole scan so live UI illumination
            # changes cannot disturb a running protocol's channels; AF steps
            # nest a child under it. It is taken under the claim just taken,
            # which is released only after the lease, so the claim brackets
            # the lease's whole life; a lease stranded by a hard-killed prior
            # run was taken under that run's taking, which no longer holds,
            # and the acquire reclaims it.
            #
            # A raise from the acquire must release the claim, or the claim
            # stays held for the life of the process and refuses every future
            # run and recording.
            try:
                self._led_lease = self._scope.illumination.acquire_led_lease('protocol', claim=held)
            except BaseException:
                self._release_activity_claim()
                raise

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
            self._write_focus_to = plan.write_focus_to
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
            # exist BEFORE the run leaves IDLE, because leaving IDLE is
            # what lets reset() drive cleanup into a finally that reads it.
            #
            # Bound to a local as well, and the local is what start() returns:
            # a caller then holds the outcome belonging to the run it actually
            # started. Reading the attribute back afterwards is a race -- a run
            # that fails at start releases the activity claim synchronously, so
            # a rival can commit in between and the caller waits on the rival's
            # run instead of its own.
            outcome = PendingRunOutcome()
            # Not written until cleanup writes it, so a run settled without
            # reaching that write -- a shutdown, a failed start -- says so.
            if plan.write_focus_to is not None:
                outcome.record_focus_written(False)
            self._run_outcome = outcome
            # What the caller holds: the run's identity for every question
            # and Stop, published with the outcome it waits on.
            handle = RunHandle(self, outcome)
            self._run_handle = handle
            # The run's writes, created with its outcome and for the same
            # reasons: after the refusals, so a refused start leaves the live
            # run's batch in place; before anything can fail, so every run
            # that reaches cleanup has a batch to close.
            self._write_batch = RunWriteBatch(self.file_io_executor)

            self._run_loop_future = None
            self._set_state(ProtocolState.RUNNING)

        # How a Stop accepted during the setup below ended the run; None
        # while no Stop has been seen, and for every dispatched run.
        stopped = None
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
                write_batch=self._write_batch,
                abort_fn=self.protocol_thread.abort,
                fatal_abort_event=self._fatal_abort_event,
                ending=self._ending,
                execution_record=self._protocol_execution_record,
                leds_off_fn=self._step_executor.leds_off,
                is_run_in_progress_fn=self._is_run_live,
                image_capture_config=self._image_capture_config,
                timestamp_overlay=self._timestamp_overlay,
                video_max_fps=self._video_max_fps,
                engineering_mode=self._engineering_mode,
                run_claim=self._held_claim.lend(),
                labware=self._scope.wellplate_loader.get_plate(plate_key=self._protocol.labware()),
                captures_asked=(
                    plan.n_scans * self._protocol.num_steps()
                    if plan.enable_image_saving and not plan.disable_saving_artifacts
                    else 0
                ),
            )

            # From here each lane serves only the run's queue, and only work
            # under the run's taking enters it; what the camera lane already
            # holds finishes on its worker, and the run loop's first act
            # waits for it before the run reads the camera.
            self.camera_executor.protocol_start(self._held_claim)
            self._io_executor.protocol_start(self._held_claim)

            # Dispatch the main run loop onto protocol_thread. Completion is
            # signalled by the run phase returning to IDLE inside _cleanup.
            # run_protocol also clears _aborted under its state lock
            # atomically with publishing the new Future, mirroring the
            # AutofocusThread fix -- so a Stop accepted during the setup
            # above, whose signal reached no loop, is read here from the
            # run's ending instead. Read and dispatched under the run lock
            # every Stop takes, so no Stop falls between the two: one
            # before is seen here, one after reaches the dispatched loop.
            with self._run_lock:
                stopped = self._ending.get()
                if stopped is None:
                    dispatch_future = self.protocol_thread.run_protocol(
                        functools.partial(self._run_loop_under_claim, handle)
                    )
                    self._run_loop_future = dispatch_future
            # A dispatch refusal is synchronous: run_protocol seals the
            # returned Future with its error BEFORE returning, while a
            # genuinely dispatched run loop leaves it unresolved for the
            # run's whole duration. A done Future here therefore means the
            # loop will never execute -- raise so the failed-at-start unwind
            # runs instead of the runner sitting committed forever.
            if (
                stopped is None
                and dispatch_future.done()
                and dispatch_future.exception() is not None
            ):
                raise RunStartError(
                    'dispatch_refused',
                    'Run failed to start',
                    str(dispatch_future.exception()),
                ) from dispatch_future.exception()
        except Exception as exc:
            self._fail_run_at_start(exc, handle)
        else:
            if stopped is not None:
                self._unwind_undispatched_run(stopped, handle)

        # The run's folder, as its setup left it: None when it saves
        # nothing, and when a failed or stopped start unwound it.
        handle._run_dir = self._run_dir

        # Reached on the failed-at-start and stopped-before-dispatch paths too, where the unwind has
        # already resolved the outcome: the caller waits and is told at once.
        return handle

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

    def _fail_run_at_start(self, exc: Exception, run: RunHandle) -> None:
        """Unwind a run that failed during start()'s setup phase.

        Routes the failure through the normal run cleanup so the terminal
        run_complete callback fires (status 'failed_at_start') and the
        executors leave protocol-mode.
        """
        logger.error(f'[{self.LOGGER_NAME} ] Run failed during start: {exc}', exc_info=True)
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
        self._unwind_undispatched_run(ending, run)
        # Notify AFTER cleanup: on an unattended run start() enabled the popup
        # suppression, which drops this non-fatal error until cleanup's
        # set_unattended_run(False) restores popups.
        from modules.notification_center import notifications

        notifications.report_outcome(
            RunFailedToStartError(reason=ending.reason, title=ending.title, message=ending.message),
            solicited=False,
            category='Protocol',
        )

    def _unwind_undispatched_run(self, ending: RunEnding, run: RunHandle) -> None:
        """Unwind a run whose loop was never dispatched, on start()'s thread.

        Two ways to get here: the setup failed, or a Stop was accepted
        while it ran. Either way no loop exists to unwind the run, so
        start() does it, after the setup has finished or failed -- never
        in the middle of it.
        """
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
        # A run that never started has no usable run directory; answering
        # with the (possibly just-deleted) path would send callers'
        # started-run follow-ups (last-save-folder shortcuts) to a dead
        # location.
        self._run_dir = None
        self._cleanup(ending, run)

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

    def _is_run_live(self) -> bool:
        """Is a run happening, in any phase? The one predicate.

        Read straight off the state machine, which is the only store of
        the run's phase. ERROR counts as live deliberately: it is
        written on three in-run paths and cleanup holds it through the
        whole teardown, so an answer that excluded it would tell a
        shutdown the run was already idle and make reset() and
        force_reset() silently return with a run still unwinding.

        Lock-free, like the flag it replaces. Callers whose decision
        must not race a start take _run_lock around their own read
        (prepare(), start(), run_in_progress()); the run loop and the
        step path poll it and tolerate a stale tick.
        """
        return self._state is not ProtocolState.IDLE

    def run_in_progress(self) -> bool:
        with self._run_lock:
            return self._is_run_live()

    def run_trigger_source(self) -> 'str | None':
        """The trigger of the run HOLDING the scope; None when none does.

        Answered off the session claim, which is taken and released with
        the run, so this cannot outlive the run it names. The runner's
        private field is a different thing -- the plan's copy, read
        inside the run as the run's own parameter and by the file-drain
        refusals as the just-finished run's.
        """
        run = self._activity_claim.run_holder
        return run.run_trigger_source if run is not None else None

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

    def _cleanup(self, ending: RunEnding, run: RunHandle):
        """Unwind *run*; ending names the terminal outcome and its cause.

        ending is REQUIRED so every cleanup site states the truth it
        knows -- a defaulted value would let an abort or failure silently
        report itself as a normal completion to run_complete subscribers.
        It is what this site believes; a fault that recorded its own
        cause into the run's ending latch outranks it, and cleanup
        resolves the two in one read below.

        Called once per run, by the thread that owns it: the run loop's
        finally; start(), for a run whose loop was never dispatched; or
        force_reset, for a run whose loop ended without unwinding it. No
        two of them can reach the same run.
        """
        try:
            # Cleanup runs on whichever thread ended the run -- the protocol
            # thread, a stop pressed in the GUI, a script's reset -- and its
            # restores and return moves are the run's own writes, so it acts
            # under the run's taking wherever it runs. A cleanup with no
            # taking left keeps the thread's own.
            held = self._held_claim
            if held is None:
                self._cleanup_inner(ending, run)
            else:
                with acting(held):
                    self._cleanup_inner(ending, run)
        finally:
            # After IDLE: a listener that reads is_live_run, or starts the
            # next run, sees the run ended.
            if self._on_run_idle is not None:
                self._on_run_idle()

    def _run_loop_under_claim(self, run: RunHandle) -> None:
        """The run loop, on the protocol thread, acting under the run's taking.

        Every move, LED and camera task the run submits is stamped with the
        taking its thread acts under, and a lane refuses one that is not the
        holder's while the scope is held -- this is what makes the run's own
        work its own.
        """
        with acting(self._held_claim):
            self._run_loop_executor.run_loop(run)

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

    def _write_focus(self, ending: RunEnding, run: RunHandle) -> None:
        """Write a completed autofocus scan's focus into the caller's protocol.

        Here, on the run's thread before run_complete is sent, because the
        run still holds the scope: the GUI's protocol edits stay disabled
        until the claim is released, so no edit lands between the scan and
        the write. A scan that did not complete focused only some of its
        steps, so it writes none. A write that fails is reported once and
        does not raise: cleanup must still give the scope back.
        """
        target = self._write_focus_to
        if target is None or ending.status != 'completed':
            return
        try:
            target.adopt_focus_from(self._protocol)
        except Exception as failed:
            from modules.notification_center import notifications

            notifications.report_outcome(failed, solicited=False, category='Protocol')
            return
        run._pending.record_focus_written(True)

    def _account_for_captures(self, ending: RunEnding) -> RunEnding:
        """Record what the run captured, and say 'incomplete' when it fell short.

        Only a run that would end 'completed' becomes 'incomplete': a stop
        stays 'aborted' and a fault stays 'failed', because what ended the
        run is the first thing a caller needs and the tally is on the
        outcome either way. Decided here, where the ending is read once, so
        run_complete, the run-end log and the outcome all carry one word.

        The person is told here, once: the unattended mute is already down.
        A composite is told by its merge instead, which knows whether the
        channels it has made a composite, so its one report says both.
        """
        writer = getattr(self, '_image_writer', None)
        outcome = getattr(self, '_run_outcome', None)
        if writer is None:
            return ending
        tally = writer.capture_tally
        if outcome is not None:
            outcome.record_captures(tally)
        if ending.status != 'completed' or tally.missing <= 0:
            return ending
        incomplete = RunIncompleteError(
            asked=tally.asked,
            captured=tally.captured,
            failed_steps=[failed.step_name for failed in tally.failed],
        )
        if self._run_mode is not SequencedCaptureRunMode.SINGLE_COMPOSITE:
            from modules.notification_center import notifications

            notifications.report_outcome(incomplete, solicited=False, category='Protocol')
        return RunEnding('incomplete', 'captures_failed', incomplete.title, str(incomplete))

    def _settle_run_outcome(self, ending: RunEnding) -> None:
        """Arm the merge on a composite that reached its end; settle every other ending.

        The ending is carried into the outcome rather than restated here:
        the status and reason a caller reads are the ones whatever ended
        the run recorded, and this method decides only whether a merge is
        still owed. Only a composite that ended 'completed' or 'incomplete'
        is, for the channels it captured, so every other run
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
            # Carry what this run's autofocus actually wrote onto the
            # outcome before any path settles it, so the answer is the
            # same whichever one gets there. Sourced from the sweep
            # rather than from the plan's request or from the results
            # folder: both say what was ASKED FOR, and a caller that
            # cannot tell a delivered file from a requested one has to
            # go looking on disk to find out -- which is the whole
            # reason this field exists. The sweep clears it per run, so
            # a run whose autofocus never fired reads None.
            if self._autofocus_runner is not None:
                af_path = self._autofocus_runner.saved_data_path()
                outcome.record_autofocus_data(str(af_path) if af_path is not None else None)
                # Only a standalone autofocus run has one focus to report;
                # the sweep clears its result per run, so a sweep that chose
                # none reads None here rather than an earlier run's focus.
                # An autofocus scan of every step runs in the same mode and
                # is told apart by the protocol it writes its focus into:
                # its answer is that protocol's Z column, not the last
                # step's focus.
                if (
                    self._run_mode is SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN
                    and self._write_focus_to is None
                ):
                    outcome.record_autofocus_focus(self._autofocus_runner.best_focus_position())
            if (
                ending.status not in ('completed', 'incomplete')
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

    def run_outcome(self) -> 'RunHandle | None':
        """The live or last run's handle, or None when no run has started."""
        return getattr(self, '_run_handle', None)

    def settle_unfinished_run(self, merge_reason: str, *, fallback: RunEnding) -> None:
        """Settle the last run's outcome now, for a teardown that will not wait for it.

        The session's shutdown, which tears the lanes down without waiting
        for a merge still waiting on this run's writes. A run that already
        recorded its own ending keeps it; *fallback* is used only when the
        run never reached cleanup.
        """
        outcome = getattr(self, '_run_outcome', None)
        if outcome is not None:
            outcome.settle_unfinished(merge_reason, fallback=fallback)

    def write_batch(self) -> RunWriteBatch | None:
        """The live or last run's writes, or None when no run has started."""
        return self._write_batch

    _FILE_WRITER_CHECK_INTERVAL_S = 1.0

    def start_file_writer_check(self, scheduler: Scheduler) -> None:
        """Watch a finished run's file writer, and report it once when it stalls.

        Nobody waits on a run's files after the run ends, so a writer stuck
        on one would be heard of only when the next run was refused. The
        check runs on the session's scheduler, in every host, and reports
        ``FileWriterStalledError`` unsolicited, carrying the recovery as its
        remedy, once for each stuck write.

        Internal scheduling -- the session arms it at bring-up, and the
        session's scheduler stops with the session; not part of the L2 API
        surface.

        Args:
            scheduler: The session's scheduler (``schedule_interval`` /
                ``unschedule``).
        """
        if self._file_writer_check_handle is not None:
            self._file_writer_check_scheduler.unschedule(self._file_writer_check_handle)
        self._file_writer_check_scheduler = scheduler
        self._file_writer_check_handle = scheduler.schedule_interval(
            self._check_file_writer, self._FILE_WRITER_CHECK_INTERVAL_S
        )

    def _check_file_writer(self, _dt: float = 0) -> None:
        batch = self._write_batch
        # A live run's own writes answer for a stuck writer; the watch is for
        # the drain nobody is waiting on.
        if batch is None or not batch.draining or not batch.stall_to_report(WRITE_STALL_FATAL_S):
            return
        from modules.notification_center import notifications

        stalled = FileWriterStalledError(
            self._the_run_holding_the_scope(self._run_trigger_source),
            batch.describe_stuck_write(),
            batch.pending,
        )
        notifications.report_outcome(stalled, solicited=False, category='Protocol')

    def is_live_run(self, run: 'RunHandle | None') -> bool:
        """Whether *run* -- the object a start() returned -- is the live run.

        What a stop control asks to decide that a click means Stop: the
        answer is the engine's, so a widget never compares triggers.
        """
        with self._run_lock:
            return self._is_live_run_locked(run)

    def is_stopping(self, run: 'RunHandle | None') -> bool:
        """Whether *run* is live and a Stop of it has been accepted.

        A stopped run stays live until its teardown finishes -- the LEDs,
        the camera and the lanes are put back first -- so "live" alone
        cannot tell a stop control whether to say the run is running or
        stopping. The answer is the run's own recorded ending: someone asked
        for it to end, and it has not finished ending yet.
        """
        with self._run_lock:
            if not self._is_live_run_locked(run):
                return False
            ending = self._ending.get()
            return ending is not None and ending.status == 'aborted'

    def held_by_other(self, run: 'RunHandle | None') -> bool:
        """Whether the scope is held by anything but *run* -- the object a start() returned.

        What a run control greys on: anything else holding the scope --
        another run, a recording, a diagnostic -- greys it, and its own run
        leaves it live as that run's Stop. None asks whether anything holds
        it at all. Decided in one read under the run lock, so the
        answer never pairs one run's liveness with another's hold.
        """
        with self._run_lock:
            if self._activity_claim.holder is None:
                return False
            return not self._is_live_run_locked(run)

    def live_run_value(self, run: RunHandle, read: typing.Callable[[], T]) -> T | None:
        """*read*'s answer while *run* is the live run; None once it is not.

        One hold of the run lock across the liveness read and *read*, so a
        handle never answers with a successor's value: a run cannot end and
        another start while the lock is held. *read* must not take the run
        lock itself.
        """
        with self._run_lock:
            if not self._is_live_run_locked(run):
                return None
            return read()

    def _is_live_run_locked(self, run: 'RunHandle | None') -> bool:
        return run is not None and run is self._run_handle and self._is_run_live()

    def _release_activity_claim(self):
        """Release the run's exclusivity claim (idempotent).

        The held taking is cleared first so a re-entrant cleanup cannot
        release twice; the claim itself raises on a release by a taking
        that no longer holds it, keeping any double-release loud instead
        of silently freeing a claim a newer activity now holds.
        """
        held, self._held_claim = self._held_claim, None
        if held is not None:
            held.release()

    def _start_hyperstack_build(self) -> threading.Thread | None:
        """Kick off the post-run per-well hyperstack build, when configured.

        Runs from cleanup for every capturing run mode (an autofocus scan
        captures nothing to stack). The build waits for THIS run's images
        to land first -- the per-step TIFFs are its input, and a stack built
        mid-flush would silently miss planes -- then builds from the run's
        own config snapshot, never the live UI, so a headless / L2 run
        triggers exactly like a GUI run. The batch is captured by value: a
        successor run replaces the runner's.

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
        write_batch = self._write_batch

        return self._spawn_post_run_step(
            name='hyperstack-build',
            build_fn=lambda: stack_builder.build_hyperstacks_for_run(
                run_dir=run_dir,
                has_turret=has_turret,
                tiling_configs_file_loc=tiling_configs_file_loc,
                wait_for_images=lambda: write_batch.wait_until_written(_POST_RUN_WRITES_WAIT_S),
                save_encoding=config.save_encoding,
            ),
        )

    def _start_composite_merge(
        self, outcome: PendingRunOutcome, ending: RunEnding
    ) -> threading.Thread | None:
        """Merge this run's per-channel frames, then settle the outcome.

        Runs only for a composite run that ended 'completed' or
        'incomplete'. Every exit
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
        write_batch = self._write_batch
        thresholds = self._composite_thresholds_percent
        output_format = (
            self._image_capture_config.output_format_sequenced
            if self._image_capture_config is not None
            else image_mode.OUTPUT_FORMAT_TIFF
        )
        has_turret = self._scope.capabilities.has_turret
        tiling_configs_file_loc = self._tiling_configs_file_loc
        # A composite missing a channel is told by the merge, once, beside
        # what the channels it has made: the run's shortfall alone when the
        # merge succeeds, inside the merge failure when it does not.
        incomplete = None
        if ending.status == 'incomplete':
            tally = self._image_writer.capture_tally
            incomplete = RunIncompleteError(
                asked=tally.asked,
                captured=tally.captured,
                failed_steps=[failed.step_name for failed in tally.failed],
            )

        def _fail(reason: str, detail: str, cause: BaseException | None = None) -> None:
            # The one place a merge failure becomes visible: one report,
            # one resolved outcome -- the shape a run
            # refusal uses, so a caller waiting on the outcome never
            # re-notifies. Success is silent by design: the saved folder
            # is the record, and the button handed the UI back at run end.
            # A merge that raised is the failure's cause, so the one record
            # carries its traceback.
            from modules.notification_center import notifications

            if incomplete is not None:
                detail = f'{incomplete} {detail}'
            failure = CompositeFailedError(detail, reason)
            failure.__cause__ = cause
            notifications.report_outcome(failure, solicited=False, category='Protocol')
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

                write_batch.wait_until_written(_POST_RUN_WRITES_WAIT_S)
                result = CompositeGeneration(has_turret=has_turret).load_folder(
                    path=run_dir,
                    tiling_configs_file_loc=tiling_configs_file_loc,
                    output_format=output_format,
                    brightness_thresholds_percent=thresholds,
                )
            except Exception as ex:
                # A typed outcome says why in its own reason and words; only
                # an exception that carries neither is named by its class.
                reason = getattr(ex, 'reason', None)
                if reason:
                    _fail(reason, str(ex), cause=ex)
                else:
                    _fail(
                        'merge_error', f'The merge failed with {type(ex).__name__}: {ex}', cause=ex
                    )
                return
            paths = result.get('artifact_paths') or []
            if paths:
                logger.info(f'[{self.LOGGER_NAME}] Composite saved: {paths[0]}')
                outcome.resolve(token, merged=True, artifact_path=paths[0], merge_reason='')
                if incomplete is not None:
                    from modules.notification_center import notifications

                    notifications.report_outcome(incomplete, solicited=False, category='Protocol')
            else:
                _fail('merge_failed', 'The merge finished without producing a composite file.')

        return self._spawn_post_run_step(name='composite-merge', build_fn=_merge)

    def _spawn_post_run_step(self, *, name: str, build_fn) -> threading.Thread:
        """Run a post-run build on a daemon thread.

        The one owner of the thread both post-run steps need -- the per-well
        stack build and the composite merge -- so one place is responsible
        for the daemon flag and the thread name a stall report prints. Each
        build begins by waiting for its run's images to land, inside its own
        outcome handling, so a wait that expires or finds images not written
        is reported in that build's words.
        """
        thread = threading.Thread(target=build_fn, name=name, daemon=True)
        thread.start()
        return thread

    def _close_run_writes(
        self, write_batch: RunWriteBatch, run_complete: RunCompleteNotice
    ) -> None:
        """Close the run's writes, and set what follows the last one landing.

        Once every image the run captured is on disk -- or given up on by a
        writer recovery or a shutdown -- the execution record completes and
        reconciles, run_complete goes out if cleanup never reached it, then
        files_complete with the outcome, then the Session hears the drain
        end. Everything is captured by value: by then a successor run may
        own this runner's fields.

        Never raises: it runs in cleanup's finally ahead of the releases, and
        a raise there would leak the claim and refuse every future run.
        """
        from modules.notification_center import notifications

        record = None if self._disable_saving_artifacts else self._protocol_execution_record
        callbacks = self._callbacks
        protocol = self._protocol
        run_dir = self._run_dir
        on_run_state = self._on_run_idle
        # A composite's merge waits on these same files and says when they
        # are not all there, in the one report its run makes.
        merge_reports_files = (
            self._run_mode is SequencedCaptureRunMode.SINGLE_COMPOSITE
            and run_complete.ending.status in ('completed', 'incomplete')
        )
        already_told = run_complete.ending.reason in FILES_LOST_ENDINGS

        def _complete_record() -> None:
            try:
                record.complete()
            except RecordIncompleteError as short:
                notifications.report_outcome(
                    short, solicited=False, category='Protocol', log_only=already_told
                )

        def _files_written(outcome: str) -> None:
            # One line per run, when its last write lands: a run's images
            # keep landing after the run ends, and this is the one place
            # that knows they are all in.
            reason = write_batch.not_written_reason
            # A run that saves no images never makes a folder, and a failed
            # start removes the one it made; the batch can still carry
            # autofocus data, so the counts stay.
            where = f'in {run_dir}' if run_dir is not None else 'with no run folder'
            line = (
                f"[{self.LOGGER_NAME}] The run's files are {outcome} {where}: "
                f'{write_batch.written} written, {write_batch.not_written} not written'
                + (f' ({reason})' if reason else '')
            )
            if outcome == 'written':
                logger.info(line)
            else:
                logger.warning(line)
            # Each on its own: a raise in one must not skip the rest -- a
            # caller never told its files are done, or a Session reading the
            # drain as live for good. No caller waits here, so a raise is
            # reported where it stops.
            actions = []
            if write_batch.not_written and not merge_reports_files:
                # The person's one report of a lost image, made when the count
                # is final; logged only when the ending's popup told them.
                lost = RunImagesNotSavedError(
                    written=write_batch.written,
                    not_written=write_batch.not_written,
                    reason=reason,
                )
                actions.append(
                    lambda: notifications.report_outcome(
                        lost, solicited=False, category='Protocol', log_only=already_told
                    )
                )
            if record is not None:
                actions.append(_complete_record)
            actions.append(run_complete.send)
            actions.append(
                lambda: schedule_files_complete(
                    callbacks, protocol=protocol, run_dir=run_dir, files=outcome
                )
            )
            # The drain's end is a run-state change: the Session re-reads
            # its levels, as it does when the run itself goes idle.
            if on_run_state is not None:
                actions.append(on_run_state)
            for action in actions:
                try:
                    action()
                except Exception as ex:
                    notifications.report_outcome(ex, solicited=False, category='Protocol')

        try:
            write_batch.close(_files_written)
        except Exception as ex:
            notifications.report_outcome(ex, solicited=False, category='Protocol')

    def _cleanup_inner(self, ending: RunEnding, run: RunHandle):
        from modules.notification_center import notifications

        # Restore popups: the unattended-run suppression ends here, on
        # every cleanup path (normal end and abort). Unconditional -- an
        # attended run never raised it, and lowering it twice is harmless,
        # where missing one lowering mutes popups for the whole session.
        notifications.set_unattended_run(False)

        led_end_state_applied = False
        # This run's writes and its one run_complete, read while the run is
        # still this runner's.
        write_batch = self._write_batch
        run_complete = None
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
            ending = self._account_for_captures(latched or ending)
            self._write_focus(ending, run)
            run_complete = RunCompleteNotice(
                self._callbacks, protocol=self._protocol, ending=ending, run_dir=self._run_dir
            )
            led_end_state_applied = run_cleanup(
                get_state_fn=lambda: self._state,
                set_state_fn=self._set_state,
                scan_in_progress=self._scan_in_progress,
                forced_dark=forced_dark,
                leds_state_at_end=self._leds_state_at_end,
                original_led_states=self._original_led_states,
                autofocus_snapshot=self._autofocus_snapshot,
                saved_camera_state=getattr(self, '_saved_camera_state', None),
                return_to_position=self._return_to_position,
                scope=self._scope,
                callbacks=self._callbacks,
                apply_led_transition_fn=self._step_executor.apply_led_transition,
                default_move_fn=self._step_executor.default_move,
                cancel_scheduled_events_fn=self._cancel_all_scheduled_events,
                autofocus_thread=self.autofocus_thread,
                write_batch=write_batch,
                run_complete=run_complete,
                logger_name=self.LOGGER_NAME,
                ending=ending,
                record_cleanup_failures=run._pending.record_cleanup_failures,
            )
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
                    self._scope.illumination.force_off()
                    logger.warning(
                        f'[{self.LOGGER_NAME}] Cleanup: LED end-state undecided; '
                        'forced all channels dark before lease release'
                    )
                except Exception:
                    logger.error(
                        f'[{self.LOGGER_NAME}] Cleanup: forced LED extinguish failed',
                        exc_info=True,
                    )
            # Close the run's writes on every path out, before the outcome
            # settles and before the run ends: a caller released by the
            # outcome, and a next run's prepare(), read this batch, and must
            # find it closed and still draining rather than open and not yet
            # asked.
            self._close_run_writes(
                write_batch,
                run_complete
                or RunCompleteNotice(
                    self._callbacks,
                    protocol=self._protocol,
                    ending=self._ending.get() or ending,
                    run_dir=self._run_dir,
                ),
            )
            # Settle (or arm) the run's merge outcome before the releases
            # below, while the fields it reads are still this run's: once
            # the run is IDLE a successor's start() replaces them.
            # Non-raising by construction, because the claim release below
            # has to run whatever happens here; a raise would leak the claim
            # and refuse every future run.
            self._settle_run_outcome(ending)
            # run_cleanup ends both lanes' run modes; a cleanup that raised
            # before it did would leave their workers serving a queue no run
            # fills, starving every later move and camera task. Ended here,
            # on every path and before the run ends, because no later pass
            # may: once the run is IDLE its lanes may be a successor's.
            # A no-op on a lane already out of its run mode.
            self.camera_executor.end_protocol_mode()
            self._io_executor.end_protocol_mode()
            # Release on every path -- early-return, normal end, or an
            # exception mid-cleanup -- so the lease can never leak and lock out
            # the next run. After run_cleanup, not before: apply(RUN_END) runs
            # inside it and the authority refuses a released lease, so the lease
            # stays held through it; this release still runs once it returns.
            self._release_scan_led_lease()
            # The activity claim releases on the same every-path guarantee:
            # a leaked claim would refuse every future run AND recording.
            self._release_activity_claim()
            # The run ENDS here, last, and only here: this is the store
            # prepare() and start() read, so while it says non-IDLE the
            # next run is refused rather than admitted onto resources
            # this cleanup is still handing back. A raise anywhere above
            # still reaches this line -- which is the whole point, since
            # a run phase that outlives its run is a lockout: the next
            # start takes the claim and the lease, then dies on the
            # illegal transition before the try that would unwind them.
            self._set_state(ProtocolState.IDLE)
