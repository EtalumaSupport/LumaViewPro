# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Video recording engine: select -> queue -> write-final-continuously.

One engine records video for every caller (manual record, protocol video
steps, and future REST/SDK clients). Per delivered camera frame the
engine decides once (cadence selection), enqueues the kept frame on an
in-RAM queue, and the recording's writer lanes (as many as its caller
names) write each frame immediately as its final artifact -- there is no
scratch buffer and no finalize pass. The queue is a shock absorber holding
the writers' current lag: drain speed decides WHEN files finish. Its one
bound is the PC's memory: while frames wait beyond what the writers hold
and the memory left is under ``MEMORY_AVAILABLE_MARGIN_BYTES``, each new
frame is skipped, counted, so the backlog never pushes the PC into its
pagefile or out of commit; the recording goes on.

The engine sits below the UI: no Kivy imports, callers inject the
dispatcher, writer edge, exclusivity claim, and time source at
construction. ``RecordingConfig`` is an immutable snapshot taken at
record start -- the engine never re-reads live settings mid-recording.

What can go wrong, and where it is told. The engine reports nothing
itself: it records what failed in its result, and the caller that
finishes the recording reports it, because only that caller knows whether
the recording belongs to a run the failure must also end.

- FATAL, aborts the recording: writer-lane death (a lane thread dies).
  The escape rides the result as ``writer_failure``, and ``aborted``
  reads it; the other lanes are ended and the recording finalizes at the
  last one's exit.
- Non-fatal, recording continues, counted: a single frame's write failure
  (costs exactly that frame), short delivery (the camera delivered fewer
  frames than the configured rate promised), a frame skipped for memory,
  a frame the camera numbered and the recording never received. These
  land in the manifest and the result.
- Non-fatal, the MANIFEST write itself failing: it cannot land in the
  manifest, and the manifest is the sole carrier of channel color and
  measured rate, so its error rides the result as ``manifest_failure``
  for the caller to report.
- A start refusal (exclusive activity already running) raises
  ``RecordingRefusedError`` directly to the refused caller.
"""

import ctypes
import enum
import json
import pathlib
import queue
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import psutil

from _lumascope.lib import profile_trace
from _lumascope.lvp_logger import logger
from _lumascope.modules.api_surface import api_fields
from _lumascope.modules.activity_claim import (
    ActivityClaim,
    BorrowedClaim,
    ControlSession,
    FalsifyingChangeInFlightError,
    Taking,
    acting,
    the_holder_named,
)
from _lumascope.modules.exceptions import CaptureFailedError, RecordingRefusedError
from _lumascope.modules.lumascope_api.frame_record import ClockMapping, TimestampSource
from _lumascope.modules.live_work import WorkKind
from _lumascope.modules.video_cadence import CadenceSelector, StallWatch, frame_budget

if TYPE_CHECKING:
    from _lumascope.modules.lumascope_api.imaging import ImagingAPI


MANIFEST_FILENAME = 'recording_manifest.json'


class PreRollEnd(enum.StrEnum):
    """What ended a recording's wait before its first frame."""

    READY = 'ready'  # every change is in force; the recording may start
    STOPPED = 'stopped'  # the caller's stop was asked for
    DEAD_FEED = 'dead_feed'  # no frame arrived for the stall bound
    TIMED_OUT = 'timed_out'  # the time limit passed with a change not in force


def wait_for_changes_in_force(
    imaging: 'ImagingAPI',
    *,
    clock: Callable[[], float],
    stop_requested: Callable[[], bool] | None = None,
    stall_threshold_s: float | None = None,
    limit_s: float | None = None,
) -> tuple[PreRollEnd, ClockMapping | None]:
    """Take frames until every change recorded is in force; say what ended
    the wait, and on READY latch the recording's one clock mapping.

    A recording's pre-roll, manual or a run's video step: its first kept
    frame must count for every change, by the condition a capture drains
    on (``ImagingAPI.changes_in_force``). Each grab counts its frame, so a
    change's count is met by the frames taken here even with live view
    off. The stop is checked first on every pass, so a stop
    always wins; a dead feed is zero frame ARRIVALS for the stall bound,
    not a change that stays owed while frames arrive, which would blame the
    camera for a settle fault. ``clock`` is in seconds, and the stall
    bound and the limit run on it.

    The mapping is latched once every change is in force, and the
    recording times every frame and reads its start through it. Latched
    before the loop instead, it would age by the wait, and the camera's
    clock drifts against the host's over a wait with no time limit.

    Args:
        imaging: The scope's imaging API.
        clock: The caller's monotonic clock, in seconds.
        stop_requested: Ends the wait as STOPPED when it answers True, or
            None for a wait nobody stops.
        stall_threshold_s: Seconds without an arrival that make the feed
            dead, or None for no stall bound.
        limit_s: Seconds the whole wait may take, or None for no limit.

    Returns:
        tuple[PreRollEnd, ClockMapping | None]: READY with the recording's
        mapping (None for a camera without a latch, whose frames state
        their arrival); STOPPED, DEAD_FEED or TIMED_OUT with None.

    Raises:
        CameraReadFailedError: The camera did not answer the latch of its clock.
        HardwareCommandRefusedError: ``'not_connected'``, the camera left
            during the wait.
    """
    watch = StallWatch(stall_threshold_s) if stall_threshold_s is not None else None
    deadline = clock() + limit_s if limit_s is not None else None
    arrivals = 0
    while not imaging.changes_in_force():
        if stop_requested is not None and stop_requested():
            return PreRollEnd.STOPPED, None
        now = clock()
        if watch is not None and watch.stalled(arrivals, now):
            return PreRollEnd.DEAD_FEED, None
        if deadline is not None and now >= deadline:
            return PreRollEnd.TIMED_OUT, None
        # A grab that got no frame is no arrival, for the stall watch.
        try:
            imaging.capture_image()
        except CaptureFailedError:
            continue
        arrivals += 1
    return PreRollEnd.READY, imaging.map_camera_clock()


# End reason for a recording whose caller failed to finish starting it.
# No manifest is published for one: nothing was recorded, and a manifest
# describing a recording that never ran pollutes the shared output folder
# and misleads anyone reading the tree afterwards.
END_REASON_START_FAILED = 'start_failed'

# How long saving may run on past a recording's end before the person is
# told, once it finishes, that every frame was kept and how long saving took.
# PERFORMANCE_BUDGETS.md row recording_save_wait_notice_s.
RECORDING_SAVE_WAIT_NOTICE_S = 10.0

# The memory a recording's backlog must leave the PC: physical memory
# available and, on Windows, the memory it can still commit. Under it, with
# frames waiting beyond what the writers hold, new frames are skipped. 1 GiB
# is above the +-0.5 GB the Mac's available figure swings at idle; on the
# 2026-11-12 recording bench (an 8 GB PC, frames at the camera's top rate)
# available fell under it 11 s before the camera's first lost frame and 13 s
# before the pagefile began to grow.
MEMORY_AVAILABLE_MARGIN_BYTES = 1 << 30


@dataclass(frozen=True)
class MemoryHeadroom:
    """What memory the PC has left, as one reading.

    Attributes:
        physical_available_bytes: Physical memory available without paging.
        commit_headroom_bytes: On Windows, what can still be committed before
            the commit limit (the pagefile at its present size included);
            None elsewhere, where there is no commit limit of that kind.
    """

    physical_available_bytes: int
    commit_headroom_bytes: int | None

    @property
    def short(self) -> bool:
        """True when either figure is under ``MEMORY_AVAILABLE_MARGIN_BYTES``."""
        return self.physical_available_bytes < MEMORY_AVAILABLE_MARGIN_BYTES or (
            self.commit_headroom_bytes is not None
            and self.commit_headroom_bytes < MEMORY_AVAILABLE_MARGIN_BYTES
        )


class _MemoryStatusEx(ctypes.Structure):
    """Win32 ``MEMORYSTATUSEX``."""

    _fields_ = (
        ('dwLength', ctypes.c_ulong),
        ('dwMemoryLoad', ctypes.c_ulong),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
    )


def read_memory_headroom() -> MemoryHeadroom:
    """Read the PC's memory left: both figures on Windows, physical elsewhere.

    On Windows both come from one ``GlobalMemoryStatusEx`` call, the one
    reader of both: psutil publishes the physical figure from the same call
    and drops the commit headroom (``ullAvailPageFile``). macOS and Linux
    have no commit limit of Windows' kind, so only psutil's physical figure
    is read there; on macOS it counts inactive memory, so the margin bounds
    the backlog without promising nothing is compressed or swapped.

    Raises:
        OSError: ``GlobalMemoryStatusEx`` failed.
    """
    if sys.platform == 'win32':
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return MemoryHeadroom(
            physical_available_bytes=status.ullAvailPhys,
            commit_headroom_bytes=status.ullAvailPageFile,
        )
    return MemoryHeadroom(
        physical_available_bytes=psutil.virtual_memory().available,
        commit_headroom_bytes=None,
    )


# Queue sentinel closing a writer lane's drain loop. One is enqueued per lane
# when selection closes, so every lane exits once the frames ahead of it are
# written.
_END_OF_RECORDING = object()


@dataclass(frozen=True)
class RecordingConfig:
    """Immutable per-recording snapshot; baked at record start.

    Attributes:
        fps: The rate limit selection samples at, in frames per second,
            or None to keep every frame the camera delivers.
        duration_s: Maximum recording duration in seconds; Stop may end
            the recording earlier.
        width: Frame width in pixels. Any resolution is legal; frames are
            never assumed square.
        height: Frame height in pixels.
        bit_depth: Capture bit depth (8 or the camera's native depth).
        output_dir: Directory receiving the per-frame artifacts and the
            manifest.
        filename_template: Caller-supplied naming template so each caller
            keeps its existing filename tokens.
        timestamp_overlay: The user's burn-timestamps-into-video choice,
            snapshotted with the rest of the config. REQUIRED so no
            caller can silently decide it; the write edge consumes it.
        timestamp_source: What every kept frame's time is
            (``TimestampSource``), set once by the controller as it starts:
            the camera's frame start when it took a clock mapping, the
            host's arrival when the camera has no latch. The manifest's
            ``timestamp_grade`` states it.
        manifest_extra: Caller-supplied static metadata (provenance,
            camera identity, channel color) merged into the manifest.
            Engine-measured fields always win on key collision -- the
            engine is the authority on measured truth.
        manifest_filename: Manifest name inside ``output_dir``, or None to
            name it after the file this recording actually writes. Callers
            whose artifacts share a folder across recordings (the flat MP4
            leg) pass None: the writer is the only authority on its own
            name, because it renames itself on collision, and a manifest
            named any other way describes a file it did not measure.
            Per-recording folders keep the default.
    """

    fps: float | None
    duration_s: float
    width: int
    height: int
    bit_depth: int
    output_dir: pathlib.Path
    filename_template: str
    timestamp_overlay: bool
    timestamp_source: TimestampSource
    manifest_extra: dict | None = None
    manifest_filename: str | None = MANIFEST_FILENAME

    @property
    def frame_budget(self) -> int | None:
        """Exact frame capacity: ``ceil(fps * duration_s)``, no truncation.

        None with no rate limit: without a rate there is no count to
        derive, and the recording ends at ``duration_s`` in time instead.
        """
        if self.fps is None:
            return None
        return frame_budget(self.fps, self.duration_s)


@api_fields('end_reason', 'save_wait_s')
@dataclass(frozen=True)
class RecordingFinished:
    """What a manual recording's operation answers once it is done.

    Attributes:
        end_reason: Why frame selection ended ('user_stop',
            'duration_elapsed', 'frame_budget_filled', 'camera_stalled',
            ...), the manifest's ``end_reason``; None when the recording
            never got as far as recording.
        save_wait_s: Seconds saving ran on after the recording ended; None
            when its backlog was discarded or it has no measured result.
    """

    end_reason: str | None
    save_wait_s: float | None


@dataclass(frozen=True)
class FrameLosses:
    """The frames one recording, or a run's video steps together, lost, by cause.

    Attributes:
        dropped: Selected frames that could not be written: write failures
            and the MP4 encoder's drops.
        selected: Frames selected.
        unstamped: Frames that came without the camera's timestamp and were
            left out.
        skipped: Frames skipped because frames waited to be saved while the
            PC's memory ran short.
        lost_at_camera: Frames the camera numbered that never reached the
            recording; None when no recording's camera numbers its frames.
    """

    dropped: int = 0
    selected: int = 0
    unstamped: int = 0
    skipped: int = 0
    lost_at_camera: int | None = None

    def __bool__(self) -> bool:
        """Whether any frame was lost."""
        return bool(self.dropped or self.unstamped or self.skipped or self.lost_at_camera)

    def __add__(self, other: 'FrameLosses') -> 'FrameLosses':
        """Both counts together; lost at the camera is None only when both are."""
        if self.lost_at_camera is None and other.lost_at_camera is None:
            lost_at_camera = None
        else:
            lost_at_camera = (self.lost_at_camera or 0) + (other.lost_at_camera or 0)
        return FrameLosses(
            dropped=self.dropped + other.dropped,
            selected=self.selected + other.selected,
            unstamped=self.unstamped + other.unstamped,
            skipped=self.skipped + other.skipped,
            lost_at_camera=lost_at_camera,
        )


@dataclass(frozen=True)
class RecordingResult:
    """Measured truth of a finished recording (the manifest's source).

    Every quantity is measured from the capture that actually happened,
    never echoed from the configuration.

    Attributes:
        frames_selected: Frames the cadence selector kept.
        frames_written: Frames whose final artifact landed on disk.
        write_failures: Frames lost to per-frame write errors (each cost
            exactly that frame; the recording continued).
        frames_unstamped: Frames that arrived without the camera's
            timestamp in a recording whose times are the camera's, and so
            were never offered to the selector (``note_unstamped``).
        frames_skipped: Frames the selector kept a slot for and the
            recording skipped, because frames waited beyond its writers while
            the PC's memory left was under the margin.
        frames_lost_at_camera: Frames the camera numbered that never reached
            the recording, from the gaps in its ``FrameID``; None for a
            camera that numbers none, whose losses are not known.
        writer_failure: What escaped the writer lane and killed it, or
            None; ``aborted`` is read from it, so the two cannot disagree.
        manifest_failure: The manifest write's error, or None. Needed
            because ``manifest_path`` is also None when there was nothing
            to describe, which is not a failure.
        configured_fps: The snapshot rate limit, for comparison against
            measured; None when every delivered frame was kept.
        frame_budget: The frames the limit and the duration allow,
            ``ceil(fps x duration)``; None with no limit. A camera slower
            than the limit ends at the duration as ``duration_elapsed``
            with fewer frames than this, so the count beside it is what
            says how much it delivered.
        measured_fps: Rate computed from real frame timestamps.
        measured_duration_s: First-to-last-frame span in seconds.
        timestamp_grade: What every kept frame's time is: the config's
            ``timestamp_source``.
        frame_timestamps_s: Per-frame capture timestamps in seconds, in
            frame order.
        manifest_path: The written manifest file, or None if aborted
            before the manifest landed.
        end_reason: Why frame SELECTION ended -- the caller's word at
            the stop that closed it ('user_stop', 'duration_elapsed',
            'camera_stalled', ...), 'frame_budget_filled' when the
            budget did, 'discarded' on a discard-first close. A support
            bundle needs this to tell a user stop from a camera death,
            which the frame counts alone cannot.
        save_wait_s: Seconds from selection closing to the last queued
            frame written: how long saving ran on after the recording
            ended. None when the backlog was discarded.
    """

    frames_selected: int
    frames_written: int
    write_failures: int
    frames_unstamped: int
    frames_skipped: int
    frames_lost_at_camera: int | None
    writer_failure: BaseException | None
    manifest_failure: OSError | None
    configured_fps: float | None
    frame_budget: int | None
    measured_fps: float
    measured_duration_s: float
    timestamp_grade: TimestampSource
    frame_timestamps_s: tuple
    manifest_path: pathlib.Path | None
    end_reason: str
    save_wait_s: float | None

    @property
    def aborted(self) -> bool:
        """True when the recording died fatally (writer-lane death)."""
        return self.writer_failure is not None

    def summary(self) -> str:
        """The recording's counts and rate in one line, for its host's log.

        With a limit the line names the budget, so a camera slower than
        the limit reads as the frames it delivered of those it was allowed.
        """
        budget = '' if self.frame_budget is None else f' of a {self.frame_budget}-frame budget'
        saved = (
            '' if self.save_wait_s is None else f', saved {self.save_wait_s:.1f} s after it ended'
        )
        lost = (
            'not known' if self.frames_lost_at_camera is None else str(self.frames_lost_at_camera)
        )
        return (
            f'{self.frames_written} written{budget}, {self.write_failures} failed, '
            f'{self.frames_skipped} skipped for memory, {lost} lost at the camera, '
            f'measured {self.measured_fps:.2f} fps over {self.measured_duration_s:.2f} s{saved}'
        )

    def losses(self, writer_dropped: int) -> FrameLosses:
        """The frames the camera sent that the recording lost, by cause.

        Lost to a write failure, to the caller's own writer
        (``writer_dropped``, the MP4 encoder's count, known only once the
        encoder has closed), to a missing camera timestamp, to a skip for
        memory, or at the camera.
        """
        return FrameLosses(
            dropped=self.write_failures + writer_dropped,
            selected=self.frames_selected,
            unstamped=self.frames_unstamped,
            skipped=self.frames_skipped,
            lost_at_camera=self.frames_lost_at_camera,
        )

    def lost_frames(self, writer_dropped: int) -> bool:
        """Whether any frame the camera sent was lost (``losses``).

        A camera that numbers no frames states no loss there, so its
        recording is judged by the rest.
        """
        return bool(self.losses(writer_dropped))

    def kept_every_frame(self, writer_dropped: int) -> bool:
        """Whether the recording kept and saved every frame it was sent.

        True only when the backlog was saved, not discarded, the writer did
        not die, and no frame was lost (``lost_frames``).
        """
        return (
            self.save_wait_s is not None
            and not self.aborted
            and not self.lost_frames(writer_dropped)
        )

    def save_wait_to_tell(self, writer_dropped: int) -> float | None:
        """The save wait the person is told of, or None when there is none to tell.

        Told only when it is longer than ``RECORDING_SAVE_WAIT_NOTICE_S`` and
        every frame was kept (``kept_every_frame``): the notice says so, and
        each loss is told in its own words.
        """
        if (
            not self.kept_every_frame(writer_dropped)
            or self.save_wait_s <= RECORDING_SAVE_WAIT_NOTICE_S
        ):
            return None
        return self.save_wait_s


@dataclass(eq=False)
class _Take:
    """One start's recording: its queue, its lanes, what they count, its claim.

    Made at each start and handed to that start's lanes, so a lane that
    outlives its recording -- one still inside a write when a discard
    finalized it -- finishes into this recording and never reads or counts
    into the next one's. The engine's lock guards every field.
    """

    # The taking held for this recording, exactly while it holds the claim.
    # Consuming it and releasing are one step, so it is both the guard and
    # the credential: a second arrival finds None and releases nothing, and
    # the claim raises on a taking that no longer holds it.
    held_claim: Taking | None
    writers: int
    lane_queue: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)
    # Set while selection is closed, so a caller waiting on the recording's
    # end wakes at it rather than at its own next check.
    selection_closed: threading.Event = field(default_factory=threading.Event)
    drained: threading.Event = field(default_factory=threading.Event)
    config: RecordingConfig | None = None
    selector: CadenceSelector | None = None
    start_ts: float = 0.0
    lanes_live: int = 0
    # A start that failed partway: its own unwind waits out the lanes it
    # started and releases the claim, so their exit finalizes nothing.
    start_failed: bool = False
    discarding: bool = False
    pending: int = 0
    frames_selected: int = 0
    frames_written: int = 0
    # The artifact this recording actually produced, as reported by the
    # write edge. A writer may not take the name it was asked for -- an MP4
    # writer renames on collision -- so a manifest named from the REQUESTED
    # name can end up describing a different recording's file.
    last_written_path: pathlib.Path | None = None
    write_failures: int = 0
    frames_unstamped: int = 0
    frames_skipped: int = 0
    # None until a delivered frame states its FrameID: a camera that numbers
    # none has losses that are not known, which 0 would claim are none.
    frames_lost_at_camera: int | None = None
    last_frame_id: int | None = None
    timestamps: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    writer_failure: BaseException | None = None
    manifest_failure: OSError | None = None
    result: RecordingResult | None = None
    end_reason: str = ''
    # The steady clock when selection closed, which the save wait is
    # measured from.
    selection_closed_ns: int | None = None


class VideoRecordingEngine:
    """The one video-capture engine beneath every caller.

    Composition (all injected; the engine owns no global state):

    Args:
        write_frame: Writer edge invoked on a writer lane once per kept
            frame: ``write_frame(image, timestamp_s, frame_number, config,
            chunks, fact) -> pathlib.Path``. ``chunks`` is the frame's
            camera chunk metadata (or None) and ``fact`` is whatever the
            caller recorded about the scope when the frame arrived (or
            None) -- both travel WITH the frame, because the write runs
            later, behind the backlog, and a write-time read would
            describe a different moment. The engine reads neither.
            Raising costs exactly that frame. With more than one lane,
            frames are written concurrently and may finish out of order;
            each keeps the frame number it was enqueued under.
        writers: The writer lanes each recording drains on: the caller's
            count for what it writes (several for per-frame files, one for
            an encoder that takes frames in order).
        claim: The session's exclusivity claim, which ``start`` takes and
            refuses when an exclusive activity already holds it -- or,
            for a recording inside a run, the run's claim lent to it,
            which ``start`` acts under and the recording's end leaves
            held.
        clock: Time source returning seconds; injectable so cadence and
            duration behavior is testable without wall-clock sleeps. The
            start of a recording whose frames have no clock mapping.
        steady_clock: The host's steady clock in nanoseconds
            (``host_clock.steady_ns``), on which a recording with a clock
            mapping reads its start through that mapping.
    """

    def __init__(
        self,
        *,
        write_frame: Callable[..., pathlib.Path],
        writers: int,
        claim: ActivityClaim | BorrowedClaim,
        clock: Callable[[], float],
        steady_clock: Callable[[], int],
    ):
        self._write_frame = write_frame
        self._writers = writers
        self._claim = claim
        self._clock = clock
        self._steady_clock = steady_clock
        # One lock covers selection state and counters. ingest_frame runs
        # on the camera ingest thread, stop()/start() on callers' threads,
        # and the writer lanes decrement the pending count -- all under
        # this lock, held only for cheap bookkeeping (never across a
        # write or a queue wait).
        self._lock = threading.Lock()
        # The latest start's recording; None before the first.
        self._take: _Take | None = None

    @property
    def is_recording(self) -> bool:
        """True from a successful ``start`` until selection closes."""
        take = self._take
        return take is not None and not take.selection_closed.is_set()

    def wait_for_selection_close(self, timeout: float | None = None) -> bool:
        """Block until selection closes; True once it has, False on timeout."""
        take = self._take
        return take is None or take.selection_closed.wait(timeout)

    @property
    def is_draining(self) -> bool:
        """True while the writer lanes still hold queued frames."""
        take = self._take
        return take is not None and not take.drained.is_set()

    @property
    def pending_writes(self) -> int:
        """Frames enqueued but not yet written (the writers' current lag)."""
        take = self._take
        return 0 if take is None else take.pending

    @property
    def end_reason(self) -> str:
        """Why selection closed, the result's ``end_reason``; read once it has closed."""
        take = self._take
        return '' if take is None else take.end_reason

    @property
    def frames_selected(self) -> int:
        """Frames the cadence selector has kept so far."""
        take = self._take
        return 0 if take is None else take.frames_selected

    def start(self, make_config: Callable[[], tuple[RecordingConfig, ClockMapping | None]]) -> None:
        """Open selection for one recording.

        Atomically acquires the exclusivity claim; exactly one of two
        concurrent starts can win. ``make_config`` is called once the claim
        is held: what the file will claim -- frame size, binning, pixel
        size, objective -- is read only when no write that would change it
        can start or still be running. It returns the config with the clock
        mapping the caller times every frame through, or None for a camera
        without one, whose frames are timed on the host's wall clock.

        The start is read on the clock the frames are: through the mapping,
        as a frame's time is, so the start and every frame share its one
        wall read and its rounding, and a step of the wall clock, or its
        coarse read, between the latch and the start cannot move the start
        against the frames. Without a mapping it is a wall read.

        Each start makes its own queue and lanes: a lane of an earlier
        recording, still inside a write after a discard, shares nothing
        with this one.

        Raises:
            RecordingRefusedError: When an exclusive activity (protocol
                run or another recording) already holds the claim, a write
                that would falsify the recording is running, or this engine
                is already recording or draining.
            ControlSessionRefusedError: ``'control_session_held'``, a
                control session is held by someone else.
            Anything ``make_config`` raises, with the claim released.
        """
        start_failure: BaseException | None = None
        with self._lock:
            if self.is_recording or self.is_draining:
                raise RecordingRefusedError(
                    reason='recording_active',
                    message='A recording is already in progress. Stop it, then record again.',
                )
            try:
                held = self._claim.try_claim(WorkKind.RECORDING)
            except FalsifyingChangeInFlightError:
                raise RecordingRefusedError(
                    reason='falsifying_change_in_flight',
                    message=(
                        'The microscope is changing the objective or the camera frame. '
                        'Start the recording when it finishes.'
                    ),
                ) from None
            if held is None:
                # Busy-with-what comes off the claim this just failed to
                # take: the activity that holds it names itself and, when
                # it is a run, which run.
                holder = self._claim.holder
                raise RecordingRefusedError(
                    reason='exclusive_activity_running',
                    message=(
                        f'{the_holder_named(holder)} is using the microscope. '
                        'Let it finish, then start the recording.'
                    ),
                    holder=holder.kind if holder is not None else None,
                    holder_trigger=(holder.run_started_by if holder is not None else None),
                )
            take = _Take(held_claim=held, writers=self._writers)
            # Idle until its config is built: a reader during make_config
            # sees no recording and no drain, as before any start.
            take.selection_closed.set()
            take.drained.set()
            self._take = take
            started: list[threading.Thread] = []
            try:
                config, mapping = make_config()
                take.config = config
                if mapping is not None:
                    start_ts = mapping.wall_time_at(
                        mapping.camera_ticks_at(self._steady_clock())
                    ).timestamp()
                else:
                    start_ts = self._clock()
                take.selector = CadenceSelector(
                    fps=config.fps, max_frames=config.frame_budget, start_ts=start_ts
                )
                take.start_ts = start_ts
                take.selection_closed.clear()
                take.drained.clear()
                for lane_number in range(self._writers):
                    lane = threading.Thread(
                        target=self._drain_under,
                        args=(take,),
                        name=f'VideoRecordingWriter-{lane_number}',
                        daemon=True,
                    )
                    lane.start()
                    started.append(lane)
                    take.lanes_live += 1
            except BaseException as ex:
                # BaseException is right here and nowhere else: this frame
                # took the claim, and no caller has a reference to this
                # engine yet, so nothing else can ever free it. The lanes
                # already started each get an end marker and are waited out
                # below, outside the lock their exit takes, before the
                # claim goes: a later start must find none of them running.
                start_failure = ex
                take.start_failed = True
                take.selection_closed.set()
                for _ in started:
                    take.lane_queue.put(_END_OF_RECORDING)
            else:
                # The claim's grant was heard before the recording was live;
                # this is the edge at which it is.
                take.held_claim.claim.announce()
        if start_failure is not None:
            for lane in started:
                lane.join()
            # Restore the idle pairing rather than only freeing the claim: a
            # half-started engine that still reads as recording or draining
            # refuses its own next start, so the failure would outlive the
            # call that caused it.
            with self._lock:
                self._release_claim_locked(take)
                take.drained.set()
            raise start_failure

    def ingest_frame(
        self,
        image: Any,
        timestamp_s: float,
        chunks: Mapping[str, int | float] | None = None,
        *,
        fact: Any,
    ) -> None:
        """Offer one delivered camera frame: select + enqueue only.

        Runs on the camera ingest thread; must stay cheap. A frame that
        takes a slot is enqueued, unless frames already wait beyond what
        the writers hold and the PC's memory left is under
        ``MEMORY_AVAILABLE_MARGIN_BYTES``: then it is skipped, counted, so
        writer lag never pushes the backlog into the pagefile or runs the
        PC out of commit. Its slot is spent, so the cadence does not catch
        up in a flood when the backlog shrinks, and the recording goes on.
        The memory is read only while frames wait, so a recording that
        keeps up reads it only when a write stalls. Frame numbers derive
        from enqueue order (contiguous ordinals), so holes are
        unrepresentable, and a frame keeps its number whichever lane
        writes it.

        A gap in the frames' ``FrameID`` is counted as frames the camera
        sent that never reached the recording, before the cadence
        selects, so a frame it passes over is no loss.

        ``fact`` is required, with no default, so a caller that records
        frames cannot forget to say what was true when this one arrived;
        a caller with nothing to record passes None and says so.
        """
        take = self._take
        if take is None:
            return
        with (
            profile_trace.timer(
                'video_ingest_trace.csv',
                'ts_ms,duration_ms,selected,pending',
                lambda: [take.frames_selected, take.pending],
            ),
            self._lock,
        ):
            if take.selection_closed.is_set():
                return
            config = take.config
            # With no limit there is no budget, so the duration on the
            # frames' clock is the boundary. With a rate the boundary is the
            # budget's last slot (below), so a frame past the duration still
            # fills a slot the budget owes.
            if config.fps is None and timestamp_s - take.start_ts >= config.duration_s:
                self._close_selection_locked(take, 'duration_elapsed')
                return
            self._count_camera_frame_locked(take, chunks)
            if not take.selector.slot_open(timestamp_s):
                return
            take.selector.reserve()
            # pending counts queued and in-flight frames, so more than the
            # writers is at least one frame no lane has taken.
            if take.pending > take.writers and read_memory_headroom().short:
                take.frames_skipped += 1
            else:
                frame_number = take.frames_selected
                take.frames_selected += 1
                take.pending += 1
                take.timestamps.append(timestamp_s)
                take.chunks.append(chunks)
                # Enqueue the delivered array as-is: no copy (pypylon's
                # GetArray already returns an owned array) and no flip --
                # orientation and contiguity are the write edge's business,
                # never paid per-frame in the callback. Chunk metadata and
                # the caller's fact ride the queue with their frame so
                # identity, pixels and the moment never separate.
                take.lane_queue.put((image, timestamp_s, frame_number, chunks, fact))
            # A skipped last slot ends the recording as a kept one does.
            if take.selector.at_capacity:
                self._close_selection_locked(take, 'frame_budget_filled')
            elif (
                config.fps is not None
                and timestamp_s >= take.start_ts + config.frame_budget / config.fps
            ):
                # A camera below the rate never fills its budget. Its first
                # frame at or past the last slot's deadline has taken the
                # slot it was owed, and the recording ends there on its
                # frames' clock: a caller's own clock ending it would race
                # the last frame on its way.
                self._close_selection_locked(take, 'duration_elapsed')

    def note_unstamped(self, chunks: Mapping[str, int | float] | None) -> None:
        """Count one delivered frame that came without the camera's timestamp.

        In a recording whose times are the camera's frame starts, a frame
        without its tick has no time on that clock: stamped at its arrival
        it would fall behind its neighbours and invert the timeline. It is
        never offered to the selector; it is counted, so the result and the
        manifest say how many the camera sent that way. Its ``chunks`` are
        the frame's camera chunk metadata (or None): it reached the
        recording, so its ``FrameID`` is no gap.
        """
        take = self._take
        if take is None:
            return
        with self._lock:
            if not take.selection_closed.is_set():
                take.frames_unstamped += 1
                self._count_camera_frame_locked(take, chunks)

    def _count_camera_frame_locked(
        self, take: _Take, chunks: Mapping[str, int | float] | None
    ) -> None:
        """Count the camera's frames this delivered frame's ``FrameID`` skipped over.

        A camera that states no ``FrameID`` leaves the count None. A number
        that does not increase is the camera's numbering restarting: the
        frames lost across it cannot be known, so the restart is logged and
        counting goes on from the new number.
        """
        frame_id = None if chunks is None else chunks.get('FrameID')
        if frame_id is None:
            return
        frame_id = int(frame_id)
        last, take.last_frame_id = take.last_frame_id, frame_id
        if last is None:
            take.frames_lost_at_camera = 0
        elif frame_id > last:
            take.frames_lost_at_camera += frame_id - last - 1
        else:
            logger.warning(
                f"[VideoEngine] The camera's FrameID went from {last} to {frame_id}: its "
                'numbering restarted, so frames lost across the restart are not counted'
            )

    def stop(self, reason: str, *, taken_for: ControlSession | None = None) -> bool:
        """Close selection promptly (within one selection decision).

        Drain continues: frames already queued keep writing until the
        lanes are empty. Stopping does not require the writers to have
        kept up -- they never do.

        Args:
            reason: Why selection is ending, recorded in the manifest as
                ``end_reason``. Required: every stop has a cause, and a
                defaulted value here would let a new caller silently
                file its recordings under the wrong one.
            taken_for: Stop only while this recording holds a claim taken
                for that control session -- the session's take back, which
                must not stop a recording begun after it.

        Returns:
            Whether this stop closed selection, or found it closed: False
            only when ``taken_for`` names a session this recording was not
            taken for, and then nothing changed.
        """
        take = self._take
        with self._lock:
            if taken_for is not None:
                held = None if take is None else take.held_claim
                if held is None or held.control is not taken_for:
                    return False
            if take is not None:
                self._close_selection_locked(take, reason)
            return True

    def wait_for_drain(self, timeout: float | None = None) -> bool:
        """Block until every queued frame is written; True when drained."""
        take = self._take
        return take is None or take.drained.wait(timeout)

    def discard_pending(self) -> None:
        """Drop the queued backlog loudly (explicit user choice at close).

        The discarded count lands in the manifest and the result as a
        short delivery; discard is never silent. The recording finalizes
        here, without waiting for its lanes: a lane wedged inside a write
        never exits, and this is the path that still frees the claim. A
        write in flight on another lane may then land after the manifest
        counted it unwritten -- one per lane at most.
        """
        take = self._take
        if take is None:
            return
        with self._lock:
            self._close_selection_locked(take, 'discarded')
            take.discarding = True
            discarded = self._drop_backlog_locked(take)
            # A lane may reach get() after the drop emptied the queue
            # (including the close's end markers); re-post one per lane so
            # every lane wakes and exits instead of parking forever.
            for _ in range(take.writers):
                take.lane_queue.put(_END_OF_RECORDING)
        logger.warning(
            f'[VideoEngine] Discarded {discarded} unwritten frames at user request; '
            'the manifest records the short delivery'
        )
        # Runs on the caller's thread, not a lane's: a writer wedged inside
        # write_frame never lets its lane exit.
        self._finalize(take)

    def result(self) -> RecordingResult:
        """Measured truth of the finished recording; valid after drain."""
        take = self._take
        result = None if take is None else take.result
        if result is None:
            raise RuntimeError('result() before the recording finished draining')
        return result

    def _release_claim_locked(self, take: _Take) -> None:
        """Release the recording's claim at most once; the caller holds the lock.

        Consuming the token and releasing are a single step, so this is safe
        to call from every end path without any caller needing to know
        whether another one got there first.
        """
        held, take.held_claim = take.held_claim, None
        if held is not None:
            held.release()

    def _close_selection_locked(self, take: _Take, reason: str) -> None:
        """Close selection exactly once; the caller holds the lock.

        First close names the end reason: a later stop or discard on an
        already-closed recording must not rewrite why it ended. One end
        marker is posted per lane, behind the frames already queued.
        """
        if take.selection_closed.is_set():
            return
        take.selection_closed_ns = self._steady_clock()
        take.end_reason = reason
        for _ in range(take.writers):
            take.lane_queue.put(_END_OF_RECORDING)
        # Set once the end reason is written, so a waiter it wakes reads it.
        take.selection_closed.set()
        # Live to draining, with the claim still held: no grant or release
        # marks it. Selection is open only while the recording holds its
        # taking, so the taking is here to announce through.
        take.held_claim.claim.announce()

    def _drop_backlog_locked(self, take: _Take) -> int:
        """Take every queued frame off the queue unwritten; the count dropped."""
        dropped = 0
        while True:
            try:
                item = take.lane_queue.get_nowait()
            except queue.Empty:
                break
            if item is not _END_OF_RECORDING:
                dropped += 1
        take.pending -= dropped
        return dropped

    def _drain_under(self, take: _Take) -> None:
        """Run one writer lane under the recording's taking.

        Each write is the recording's own work, so it acts under the taking
        the recording holds: a fault a write meets that ends the activity
        (a protocol step's disk floor reaching the run's fatal funnel) acts
        as that activity, whose off is the holder's alone.
        """
        with acting(take.held_claim):
            self._drain_loop(take)

    def _drain_loop(self, take: _Take) -> None:
        """Writer lane: pop each frame and write it as its final artifact.

        A single frame's write failure costs exactly that frame (counted,
        never fatal). Any other escape -- including SystemExit from a
        dying dependency -- is writer-lane death: the recording aborts
        loudly, the backlog is dropped and every other lane is told to
        exit. The recording finalizes at its last lane's exit.
        """
        try:
            while True:
                item = take.lane_queue.get()
                if item is _END_OF_RECORDING:
                    break
                image, timestamp_s, frame_number, chunks, fact = item
                try:
                    with profile_trace.timer(
                        'video_write_trace.csv',
                        'ts_ms,duration_ms,frame_number,pending',
                        lambda n=frame_number: [n, take.pending],
                    ):
                        written_path = self._write_frame(
                            image, timestamp_s, frame_number, take.config, chunks, fact
                        )
                except Exception as ex:
                    with self._lock:
                        take.write_failures += 1
                    logger.error(
                        f'[VideoEngine] Frame {frame_number} write failed and is lost '
                        f'({ex}); the recording continues'
                    )
                except BaseException as ex:
                    self._abort_from_lane_death(take, ex)
                    return
                else:
                    with self._lock:
                        take.frames_written += 1
                        take.last_written_path = written_path
                finally:
                    with self._lock:
                        take.pending -= 1
                if take.discarding:
                    return
        except BaseException:
            # An exception escaping a thread target reaches
            # threading.excepthook, not this module's logger, so the failure
            # this lane exists to report would otherwise leave no trace in
            # the app log. The other lanes are told to exit, since this one
            # will not drain its share.
            logger.critical('[VideoEngine] writer lane exited abnormally', exc_info=True)
            with self._lock:
                self._close_selection_locked(take, 'lane_exited')
                for _ in range(take.writers):
                    take.lane_queue.put(_END_OF_RECORDING)
            raise
        finally:
            self._lane_exited(take)

    def _abort_from_lane_death(self, take: _Take, ex: BaseException) -> None:
        """Record a lane's death and end the recording's other lanes.

        Selection closes, the queued backlog is dropped, and one end marker
        is posted per lane whether or not selection had already closed, so
        every lane still running exits after its write in flight.
        """
        with self._lock:
            if take.writer_failure is None:
                take.writer_failure = ex
            self._close_selection_locked(take, 'aborted')
            self._drop_backlog_locked(take)
            for _ in range(take.writers):
                take.lane_queue.put(_END_OF_RECORDING)

    def _lane_exited(self, take: _Take) -> None:
        """Count one lane out; the last to exit finalizes the recording."""
        with self._lock:
            take.lanes_live -= 1
            last = take.lanes_live == 0
        if last and not take.start_failed:
            self._finalize(take)

    def _finalize(self, take: _Take) -> None:
        """Compute measured truth, write the manifest, release the claim.

        Runs exactly once per recording: the last lane's exit and a
        caller's discard_pending() both route here, and whichever comes
        second is a no-op. The claim is released even when the body raises,
        because a held claim outlives this engine and blocks every later
        recording and protocol run.
        """
        with self._lock:
            if take.held_claim is None:
                return
            # Every ordinary end path has already closed selection to post
            # the markers that end the lanes; this names the end only when
            # none did.
            self._close_selection_locked(
                take, 'aborted' if take.writer_failure is not None else 'lane_exited'
            )
            try:
                self._finalize_locked(take)
            finally:
                # Release BEFORE signalling drained: a caller woken by
                # wait_for_drain must observe the claim already free.
                self._release_claim_locked(take)
                take.drained.set()

    def _finalize_locked(self, take: _Take) -> None:
        """Measure the finished recording and record it; the caller holds the lock."""
        # Measured to where the queue drained, before the manifest is written,
        # so the manifest carries it.
        save_wait_s = (
            None
            if take.discarding or take.selection_closed_ns is None
            else (self._steady_clock() - take.selection_closed_ns) / 1e9
        )
        timestamps = tuple(take.timestamps)
        if len(timestamps) >= 2:
            span = timestamps[-1] - timestamps[0]
            measured_duration = span
            measured_fps = (len(timestamps) - 1) / span if span > 0 else 0.0
        else:
            measured_duration = 0.0
            measured_fps = 0.0
        grade = take.config.timestamp_source
        manifest_path = None
        manifest_name = self._manifest_name(take)
        # An aborted recording has no trustworthy measured truth to publish,
        # a failed start has nothing to describe at all, and a recording that
        # produced no artifact has nothing to attach a manifest to; all get a
        # result in memory but no manifest on disk.
        if (
            take.writer_failure is None
            and take.end_reason != END_REASON_START_FAILED
            and manifest_name is not None
        ):
            manifest_path = self._write_manifest(
                take,
                name=manifest_name,
                measured_fps=measured_fps,
                measured_duration=measured_duration,
                grade=grade,
                timestamps=timestamps,
                save_wait_s=save_wait_s,
            )
        take.result = RecordingResult(
            frames_selected=take.frames_selected,
            frames_written=take.frames_written,
            write_failures=take.write_failures,
            frames_unstamped=take.frames_unstamped,
            frames_skipped=take.frames_skipped,
            frames_lost_at_camera=take.frames_lost_at_camera,
            writer_failure=take.writer_failure,
            manifest_failure=take.manifest_failure,
            configured_fps=take.config.fps,
            frame_budget=take.config.frame_budget,
            measured_fps=measured_fps,
            measured_duration_s=measured_duration,
            timestamp_grade=grade,
            frame_timestamps_s=timestamps,
            manifest_path=manifest_path,
            end_reason=take.end_reason,
            save_wait_s=save_wait_s,
        )

    def _manifest_name(self, take: _Take) -> str | None:
        """The manifest's filename, or None when nothing exists to name it after.

        A caller whose artifacts share a folder across recordings leaves the
        name unpinned, and the manifest takes the name of the file this
        recording actually wrote. That file is the only authority: an MP4
        writer renames itself on collision, so two recordings starting in one
        second produce two differently-named videos, and manifests built from
        the requested name would collapse onto one -- overwriting each other
        and describing the wrong video.

        None means the recording wrote nothing, so there is no artifact for a
        manifest to describe.
        """
        pinned = take.config.manifest_filename
        if pinned is not None:
            return pinned
        if take.last_written_path is None:
            return None
        return f'{take.last_written_path.stem}_manifest.json'

    def _write_manifest(
        self,
        take: _Take,
        name: str,
        measured_fps: float,
        measured_duration: float,
        grade: TimestampSource,
        timestamps: tuple,
        save_wait_s: float | None,
    ) -> pathlib.Path | None:
        # Caller metadata first, engine truth second: on any key collision
        # the engine's measured fields win -- a caller cannot overwrite
        # measured truth with configuration echoes.
        manifest = dict(take.config.manifest_extra or {})
        manifest.update(
            {
                'manifest_version': 1,
                'frames_selected': take.frames_selected,
                'frames_written': take.frames_written,
                'write_failures': take.write_failures,
                'frames_unstamped': take.frames_unstamped,
                'frames_skipped': take.frames_skipped,
                'frames_lost_at_camera': take.frames_lost_at_camera,
                'end_reason': take.end_reason,
                'timestamp_grade': str(grade),
                'configured_fps': take.config.fps,
                'frame_budget': take.config.frame_budget,
                'measured_fps': measured_fps,
                'measured_duration_s': measured_duration,
                'save_wait_s': save_wait_s,
                # Chunk dicts are stored verbatim: their keys are the
                # camera driver's vocabulary, which the engine does not
                # interpret -- downstream readers do.
                'frame_index': [
                    {
                        'i': i,
                        'ts_s': ts,
                        'chunks': take.chunks[i] if i < len(take.chunks) else None,
                    }
                    for i, ts in enumerate(timestamps)
                ],
            }
        )
        path = pathlib.Path(take.config.output_dir) / name
        try:
            path.write_text(json.dumps(manifest, indent=2))
        except OSError as ex:
            # The frames on disk are the artifact; a manifest write
            # failure degrades reporting, never the recording. But the
            # manifest is the SOLE carrier of the recording's channel
            # color and measured rate, so the loss must be loud: without
            # it every later build of these frames silently plays
            # grayscale at a default rate. The caller reports it.
            take.manifest_failure = ex
            return None
        return path
