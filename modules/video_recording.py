# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Video recording engine: select -> queue -> write-final-continuously.

One engine records video for every caller (manual record, protocol video
steps, and future REST/SDK clients). Per delivered camera frame the
engine decides once (cadence selection), enqueues the kept frame on an
unbounded in-RAM queue, and a dedicated writer lane writes each frame
immediately as its final artifact -- there is no scratch buffer and no
finalize pass. The queue is a shock absorber holding only the writer's
current lag: drain speed decides WHEN files finish, never WHETHER.
Capture never drops a frame because the writer is behind; the only bound
is RAM.

The engine sits below the UI: no Kivy imports, callers inject the
dispatcher, writer edge, exclusivity claim, and time source at
construction. ``RecordingConfig`` is an immutable snapshot taken at
record start -- the engine never re-reads live settings mid-recording.

Fatality classification (the notification policy's teeth; a misclassified
event fails silent):

- FATAL, aborts the recording: writer-lane death (the lane thread dies or
  wedges past recovery). Surfaced at critical severity, which reaches
  listeners through the protocol notification mute.
- Non-fatal, recording continues, counted and reported: a single frame's
  write failure (costs exactly that frame), short delivery (the camera
  delivered fewer frames than the configured rate promised), frame drops.
  These land in the manifest and the end-of-run report, never a popup
  mid-run.
- Non-fatal, surfaced at warning severity: the MANIFEST write itself
  failing -- it cannot land in the manifest, and it is the sole carrier
  of channel color and measured rate, so it goes through the notify sink
  (the protocol mute keeps it log-only mid-run; manual gets the popup).
- A start refusal (exclusive activity already running) raises
  ``RecordingRefusedError`` directly to the refused caller, outside the
  mute's scope.
"""

import json
import pathlib
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from lib import profile_trace
from lvp_logger import logger
from modules.exceptions import RecordingRefusedError
from modules.video_cadence import CadenceSelector, frame_budget


MANIFEST_FILENAME = 'recording_manifest.json'

# End reason for a recording whose caller failed to finish starting it.
# No manifest is published for one: nothing was recorded, and a manifest
# describing a recording that never ran pollutes the shared output folder
# and misleads anyone reading the tree afterwards.
END_REASON_START_FAILED = 'start_failed'

# Queue sentinel closing the writer lane's drain loop. Enqueued exactly
# once per recording, when selection closes.
_END_OF_RECORDING = object()


class ExclusivityClaim(Protocol):
    """The session-owned compare-and-claim handle the engine acquires.

    Exactly one exclusive activity (a protocol run XOR a recording) may
    hold the claim; ``try_claim`` is atomic -- two concurrent claimants
    cannot both win.
    """

    def try_claim(self, owner: str) -> bool:
        """Atomically claim for ``owner``; False if another owner holds it."""
        ...

    def release(self, owner: str) -> None:
        """Release ``owner``'s claim. Releasing an unheld claim is an error."""
        ...


@dataclass(frozen=True)
class RecordingConfig:
    """Immutable per-recording snapshot; baked at record start.

    Attributes:
        fps: Effective recording rate in frames per second (post-clamp).
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

    fps: float
    duration_s: float
    width: int
    height: int
    bit_depth: int
    output_dir: pathlib.Path
    filename_template: str
    timestamp_overlay: bool
    manifest_extra: dict | None = None
    manifest_filename: str | None = MANIFEST_FILENAME

    @property
    def frame_budget(self) -> int:
        """Exact frame capacity: ``ceil(fps * duration_s)``, no truncation."""
        return frame_budget(self.fps, self.duration_s)


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
        short_delivery: True when the camera delivered fewer frames than
            the configured rate promised over the measured duration.
        aborted: True when the recording died fatally (writer-lane death)
            or was discarded before drain completed.
        abort_reason: Human-readable cause when ``aborted``; empty string
            otherwise.
        configured_fps: The snapshot rate, for comparison against measured.
        measured_fps: Rate computed from real frame timestamps.
        measured_duration_s: First-to-last-frame span in seconds.
        timestamp_grade: ``'camera'`` when hardware chunk timestamps
            stamped the frames, ``'host'`` when the host clock had to.
        frame_timestamps_s: Per-frame capture timestamps in seconds, in
            frame order.
        manifest_path: The written manifest file, or None if aborted
            before the manifest landed.
        end_reason: Why frame SELECTION ended -- the caller's word at
            the stop that closed it ('user_stop', 'duration_elapsed',
            'camera_stalled', ...), 'frame_budget_filled' when the
            budget did, 'discarded' on a discard-first close. A support
            bundle needs this to tell a user stop from a camera death;
            short_delivery alone cannot.
    """

    frames_selected: int
    frames_written: int
    write_failures: int
    short_delivery: bool
    aborted: bool
    abort_reason: str
    configured_fps: float
    measured_fps: float
    measured_duration_s: float
    timestamp_grade: str
    frame_timestamps_s: tuple
    manifest_path: pathlib.Path | None
    end_reason: str


class VideoRecordingEngine:
    """The one video-capture engine beneath every caller.

    Composition (all injected; the engine owns no global state):

    Args:
        write_frame: Writer edge invoked on the writer lane once per kept
            frame: ``write_frame(image, timestamp_s, frame_number, config,
            chunks) -> pathlib.Path``. ``chunks`` is the frame's camera
            chunk metadata (or None) -- frame identity travels WITH the
            frame so the write edge never re-derives it. Raising costs
            exactly that frame.
        claim: The session-owned exclusivity claim handle; ``start``
            acquires it and refuses when an exclusive activity already
            holds it.
        clock: Time source returning seconds; injectable so cadence and
            duration behavior is testable without wall-clock sleeps.
        notify: Optional notification sink for the fatality classification
            above; None means log-only.
    """

    def __init__(
        self,
        *,
        write_frame: Callable[..., pathlib.Path],
        claim: ExclusivityClaim,
        clock: Callable[[], float],
        notify: Any = None,
        run_trigger_lookup: 'Callable[[], str | None] | None' = None,
    ):
        self._write_frame = write_frame
        self._claim = claim
        self._clock = clock
        self._notify = notify
        # Busy-with-what for the claim refusal below: when a run holds
        # the claim, the refusal names the run's trigger. Kind stays the
        # runner's job -- the claim carries only the owner.
        self._run_trigger_lookup = run_trigger_lookup
        # One lock covers selection state and counters. ingest_frame runs
        # on the camera ingest thread, stop()/start() on callers' threads,
        # and the writer lane decrements the pending count -- all under
        # this lock, held only for cheap bookkeeping (never across a
        # write or a queue wait).
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._drained = threading.Event()
        self._drained.set()
        # Holds the claim's owner string exactly while this engine holds the
        # claim. Consuming it and releasing are one step, so the token is
        # both the guard and the argument: a second arrival cannot release a
        # claim it does not hold, and release() raises on a non-owner.
        self._claim_owner: str | None = None
        self._config: RecordingConfig | None = None
        self._selector: CadenceSelector | None = None
        self._writer_thread: threading.Thread | None = None
        self._recording = False
        self._selection_closed = True
        self._discarding = False
        self._pending = 0
        self._frames_selected = 0
        self._frames_written = 0
        # The artifact this recording actually produced, as reported by the
        # write edge. A writer may not take the name it was asked for -- an
        # MP4 writer renames on collision -- so a manifest named from the
        # REQUESTED name can end up describing a different recording's file.
        self._last_written_path: pathlib.Path | None = None
        self._write_failures = 0
        self._timestamps: list[float] = []
        self._chunks: list = []
        self._all_frames_carried_chunks = True
        self._aborted = False
        self._abort_reason = ''
        self._result: RecordingResult | None = None

    @property
    def is_recording(self) -> bool:
        """True from a successful ``start`` until selection closes."""
        return self._recording

    @property
    def is_draining(self) -> bool:
        """True while the writer lane still holds queued frames."""
        return not self._drained.is_set()

    @property
    def pending_writes(self) -> int:
        """Frames enqueued but not yet written (the writer's current lag)."""
        return self._pending

    @property
    def frames_selected(self) -> int:
        """Frames the cadence selector has kept so far."""
        return self._frames_selected

    def start(self, config: RecordingConfig) -> None:
        """Open selection for one recording.

        Atomically acquires the exclusivity claim; exactly one of two
        concurrent starts can win.

        Raises:
            RecordingRefusedError: When an exclusive activity (protocol
                run or another recording) already holds the claim, or
                this engine is already recording or draining.
        """
        with self._lock:
            if self._recording or not self._drained.is_set():
                raise RecordingRefusedError(
                    reason='recording_active',
                    title='Recording Active',
                    message='A recording is already in progress. Stop it, then record again.',
                )
            if not self._claim.try_claim('recording'):
                holder = self._claim.owner
                holder_trigger = None
                if holder == 'protocol' and self._run_trigger_lookup is not None:
                    holder_trigger = self._run_trigger_lookup()
                raise RecordingRefusedError(
                    reason='exclusive_activity_running',
                    title='Another Activity Running',
                    message=(
                        'Another exclusive activity is using the microscope. '
                        'Let it finish, then start the recording.'
                    ),
                    holder=holder,
                    holder_trigger=holder_trigger,
                )
            self._claim_owner = 'recording'
            try:
                self._config = config
                start_ts = self._clock()
                self._selector = CadenceSelector(
                    fps=config.fps, max_frames=config.frame_budget, start_ts=start_ts
                )
                self._start_ts = start_ts
                self._recording = True
                self._selection_closed = False
                self._discarding = False
                self._drained.clear()
                self._pending = 0
                self._frames_selected = 0
                self._frames_written = 0
                self._write_failures = 0
                self._timestamps = []
                self._chunks = []
                self._all_frames_carried_chunks = True
                self._aborted = False
                self._abort_reason = ''
                self._end_reason = ''
                self._result = None
                self._writer_thread = threading.Thread(
                    target=self._drain_loop, name='VideoRecordingWriter', daemon=True
                )
                self._writer_thread.start()
            except BaseException:
                # BaseException is right here and nowhere else: this frame
                # took the claim, and no caller has a reference to this
                # engine yet, so nothing else can ever free it.
                #
                # Restore the idle pairing __init__ establishes rather than
                # only freeing the claim: a half-started engine that still
                # reads as recording refuses its own next start, so the
                # failure would outlive the call that caused it.
                self._recording = False
                self._selection_closed = True
                self._release_claim_locked()
                self._drained.set()
                raise

    def ingest_frame(self, image: Any, timestamp_s: float, chunks: Any = None) -> None:
        """Offer one delivered camera frame: select + enqueue only.

        Runs on the camera ingest thread; must stay cheap. A kept frame
        is enqueued unconditionally -- writer lag never causes a
        capture-side drop. Frame numbers derive from enqueue order
        (contiguous ordinals), so holes are unrepresentable.
        """
        with (
            profile_trace.timer(
                'video_ingest_trace.csv',
                'ts_ms,duration_ms,selected,pending',
                lambda: [self._frames_selected, self._pending],
            ),
            self._lock,
        ):
            if not self._recording:
                return
            # No separate duration cutoff: the frame budget
            # (ceil(fps * duration)) IS the duration boundary, and the
            # selector's catch-up semantics require late frames to
            # claim outstanding slots -- an independent wall-clock
            # close would truncate exactly that catch-up.
            if not self._selector.slot_open(timestamp_s):
                return
            self._selector.reserve()
            frame_number = self._frames_selected
            self._frames_selected += 1
            self._pending += 1
            self._timestamps.append(timestamp_s)
            self._chunks.append(chunks)
            if chunks is None:
                self._all_frames_carried_chunks = False
            # Enqueue the delivered array as-is: no copy (pypylon's
            # GetArray already returns an owned array) and no flip --
            # orientation and contiguity are the write edge's business,
            # never paid per-frame in the callback. Chunk metadata rides
            # the queue with its frame so identity and pixels never
            # separate.
            self._queue.put((image, timestamp_s, frame_number, chunks))
            if self._selector.at_capacity:
                self._close_selection_locked('frame_budget_filled')

    def stop(self, reason: str) -> None:
        """Close selection promptly (within one selection decision).

        Drain continues: frames already queued keep writing until the
        lane is empty. Stopping does not require the writer to have kept
        up -- it never does.

        Args:
            reason: Why selection is ending, recorded in the manifest as
                ``end_reason``. Required: every stop has a cause, and a
                defaulted value here would let a new caller silently
                file its recordings under the wrong one.
        """
        with self._lock:
            self._close_selection_locked(reason)

    def wait_for_drain(self, timeout: float | None = None) -> bool:
        """Block until every queued frame is written; True when drained."""
        return self._drained.wait(timeout)

    def discard_pending(self) -> None:
        """Drop the queued backlog loudly (explicit user choice at close).

        The discarded count lands in the manifest and the result as a
        short delivery; discard is never silent.
        """
        with self._lock:
            self._close_selection_locked('discarded')
            self._discarding = True
        discarded = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _END_OF_RECORDING:
                continue
            discarded += 1
        with self._lock:
            self._pending -= discarded
        # The lane may reach get() after this drain emptied the queue
        # (including the close sentinel); re-post it so the lane always
        # wakes and exits instead of parking forever.
        self._queue.put(_END_OF_RECORDING)
        logger.warning(
            f'[VideoEngine] Discarded {discarded} unwritten frames at user request; '
            'the manifest records the short delivery'
        )
        # Runs on the caller's thread, not the lane's: a writer wedged inside
        # write_frame never lets the lane exit, and this is the path that
        # still frees the claim and lets the application close.
        self._finalize()

    def result(self) -> RecordingResult:
        """Measured truth of the finished recording; valid after drain."""
        result = self._result
        if result is None:
            raise RuntimeError('result() before the recording finished draining')
        return result

    def _release_claim_locked(self) -> None:
        """Release the exclusivity claim at most once; the caller holds the lock.

        Consuming the token and releasing are a single step, so this is safe
        to call from every end path without any caller needing to know
        whether another one got there first.
        """
        owner, self._claim_owner = self._claim_owner, None
        if owner is not None:
            self._claim.release(owner)

    def _close_selection_locked(self, reason: str) -> None:
        """Close selection exactly once; the caller holds the lock.

        First close names the end reason: a later stop or discard on an
        already-closed recording must not rewrite why it ended.
        """
        if self._selection_closed:
            return
        self._selection_closed = True
        self._end_reason = reason
        self._recording = False
        self._queue.put(_END_OF_RECORDING)

    def _drain_loop(self) -> None:
        """Writer lane: pop each frame and write it as its final artifact.

        A single frame's write failure costs exactly that frame (counted,
        never fatal). Any other escape -- including SystemExit from a
        dying dependency -- is writer-lane death: the recording aborts
        loudly and the backlog is discarded.
        """
        try:
            while True:
                item = self._queue.get()
                if item is _END_OF_RECORDING:
                    break
                image, timestamp_s, frame_number, chunks = item
                try:
                    with profile_trace.timer(
                        'video_write_trace.csv',
                        'ts_ms,duration_ms,frame_number,pending',
                        lambda n=frame_number: [n, self._pending],
                    ):
                        written_path = self._write_frame(
                            image, timestamp_s, frame_number, self._config, chunks
                        )
                except Exception as ex:
                    with self._lock:
                        self._write_failures += 1
                    logger.error(
                        f'[VideoEngine] Frame {frame_number} write failed and is lost '
                        f'({ex}); the recording continues'
                    )
                except BaseException as ex:
                    self._abort_from_lane_death(ex)
                    return
                else:
                    with self._lock:
                        self._frames_written += 1
                        self._last_written_path = written_path
                finally:
                    with self._lock:
                        self._pending -= 1
                if self._discarding:
                    return
        except BaseException:
            # An exception escaping a thread target reaches
            # threading.excepthook, not this module's logger, so the failure
            # this lane exists to report would otherwise leave no trace in
            # the app log.
            logger.critical('[VideoEngine] writer lane exited abnormally', exc_info=True)
            raise
        finally:
            # Every lane exit ends the recording, including the abort path
            # and an escape from the abort path's own notification sink.
            self._finalize()

    def _abort_from_lane_death(self, ex: BaseException) -> None:
        reason = f'writer lane died: {ex}'
        with self._lock:
            self._aborted = True
            self._abort_reason = reason
        logger.critical(f'[VideoEngine] {reason} -- recording aborted')
        if self._notify is not None:
            self._notify.critical(
                'Recording',
                'Recording Failed',
                'The video writer stopped working and the recording was aborted. '
                'Frames already written are on disk; check the log for the cause.',
            )

    def _finalize(self) -> None:
        """Compute measured truth, write the manifest, release the claim.

        Runs exactly once per recording: the writer lane's exit and a
        caller's discard_pending() both route here, and whichever comes
        second is a no-op. The claim is released even when the body raises,
        because a held claim outlives this engine and blocks every later
        recording and protocol run.
        """
        with self._lock:
            if self._claim_owner is None:
                return
            # Only an abnormal lane exit reaches here with selection still
            # open: every ordinary end path closes it to post the sentinel
            # that wakes the lane in the first place.
            self._close_selection_locked('aborted' if self._aborted else 'lane_exited')
            try:
                self._finalize_locked()
            finally:
                # Release BEFORE signalling drained: a caller woken by
                # wait_for_drain must observe the claim already free.
                self._release_claim_locked()
                self._drained.set()

    def _finalize_locked(self) -> None:
        """Measure the finished recording and record it; the caller holds the lock."""
        timestamps = tuple(self._timestamps)
        if len(timestamps) >= 2:
            span = timestamps[-1] - timestamps[0]
            measured_duration = span
            measured_fps = (len(timestamps) - 1) / span if span > 0 else 0.0
        else:
            measured_duration = 0.0
            measured_fps = 0.0
        grade = 'camera' if timestamps and self._all_frames_carried_chunks else 'host'
        short_delivery = self._frames_written < self._config.frame_budget
        manifest_path = None
        manifest_name = self._manifest_name()
        # An aborted recording has no trustworthy measured truth to publish,
        # a failed start has nothing to describe at all, and a recording that
        # produced no artifact has nothing to attach a manifest to; all get a
        # result in memory but no manifest on disk.
        if (
            not self._aborted
            and self._end_reason != END_REASON_START_FAILED
            and manifest_name is not None
        ):
            manifest_path = self._write_manifest(
                name=manifest_name,
                measured_fps=measured_fps,
                measured_duration=measured_duration,
                grade=grade,
                short_delivery=short_delivery,
                timestamps=timestamps,
            )
        self._result = RecordingResult(
            frames_selected=self._frames_selected,
            frames_written=self._frames_written,
            write_failures=self._write_failures,
            short_delivery=short_delivery,
            aborted=self._aborted,
            abort_reason=self._abort_reason,
            configured_fps=self._config.fps,
            measured_fps=measured_fps,
            measured_duration_s=measured_duration,
            timestamp_grade=grade,
            frame_timestamps_s=timestamps,
            manifest_path=manifest_path,
            end_reason=self._end_reason,
        )

    def _manifest_name(self) -> str | None:
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
        pinned = self._config.manifest_filename
        if pinned is not None:
            return pinned
        if self._last_written_path is None:
            return None
        return f'{self._last_written_path.stem}_manifest.json'

    def _write_manifest(
        self,
        name: str,
        measured_fps: float,
        measured_duration: float,
        grade: str,
        short_delivery: bool,
        timestamps: tuple,
    ) -> pathlib.Path | None:
        # Caller metadata first, engine truth second: on any key collision
        # the engine's measured fields win -- a caller cannot overwrite
        # measured truth with configuration echoes.
        manifest = dict(self._config.manifest_extra or {})
        manifest.update(
            {
                'manifest_version': 1,
                'frames_selected': self._frames_selected,
                'frames_written': self._frames_written,
                'write_failures': self._write_failures,
                'short_delivery': short_delivery,
                'end_reason': self._end_reason,
                'timestamp_grade': grade,
                'configured_fps': self._config.fps,
                'measured_fps': measured_fps,
                'measured_duration_s': measured_duration,
                # Chunk dicts are stored verbatim: their keys are the
                # camera driver's vocabulary, which the engine does not
                # interpret -- downstream readers do.
                'frame_index': [
                    {
                        'i': i,
                        'ts_s': ts,
                        'chunks': self._chunks[i] if i < len(self._chunks) else None,
                    }
                    for i, ts in enumerate(timestamps)
                ],
            }
        )
        path = pathlib.Path(self._config.output_dir) / name
        try:
            path.write_text(json.dumps(manifest, indent=2))
        except OSError as ex:
            # The frames on disk are the artifact; a manifest write
            # failure degrades reporting, never the recording. But the
            # manifest is the SOLE carrier of the recording's channel
            # color and measured rate, so the loss must be loud: without
            # it every later build of these frames silently plays
            # grayscale at a default rate.
            logger.error(f'[VideoEngine] Manifest write failed ({ex}); frames are unaffected')
            if self._notify is not None:
                self._notify.warning(
                    'Video Recording',
                    'Recording details not saved',
                    'The video frames are safe on disk, but the recording details '
                    'file could not be written. Videos built from this recording '
                    'may be grayscale and use a default frame rate; check disk '
                    'space and the log.',
                )
            return None
        return path
