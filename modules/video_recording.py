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
        manifest_filename: Manifest name inside ``output_dir``. Callers
            whose artifacts share a folder across recordings (the flat
            MP4 leg) name it per recording so manifests never overwrite
            each other; per-recording folders keep the default.
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
    manifest_filename: str = MANIFEST_FILENAME

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
    ):
        self._write_frame = write_frame
        self._claim = claim
        self._clock = clock
        self._notify = notify
        # One lock covers selection state and counters. ingest_frame runs
        # on the camera ingest thread, stop()/start() on callers' threads,
        # and the writer lane decrements the pending count -- all under
        # this lock, held only for cheap bookkeeping (never across a
        # write or a queue wait).
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._drained = threading.Event()
        self._drained.set()
        self._config: RecordingConfig | None = None
        self._selector: CadenceSelector | None = None
        self._writer_thread: threading.Thread | None = None
        self._recording = False
        self._selection_closed = True
        self._discarding = False
        self._pending = 0
        self._frames_selected = 0
        self._frames_written = 0
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
                raise RecordingRefusedError(
                    reason='exclusive_activity_running',
                    title='Another Activity Running',
                    message=(
                        'Another exclusive activity is using the microscope. '
                        'Let it finish, then start the recording.'
                    ),
                )
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
            self._result = None
            self._writer_thread = threading.Thread(
                target=self._drain_loop, name='VideoRecordingWriter', daemon=True
            )
            self._writer_thread.start()

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
                self._close_selection_locked()

    def stop(self) -> None:
        """Close selection promptly (within one selection decision).

        Drain continues: frames already queued keep writing until the
        lane is empty. Stopping does not require the writer to have kept
        up -- it never does.
        """
        with self._lock:
            self._close_selection_locked()

    def wait_for_drain(self, timeout: float | None = None) -> bool:
        """Block until every queued frame is written; True when drained."""
        return self._drained.wait(timeout)

    def discard_pending(self) -> None:
        """Drop the queued backlog loudly (explicit user choice at close).

        The discarded count lands in the manifest and the result as a
        short delivery; discard is never silent.
        """
        with self._lock:
            self._close_selection_locked()
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
        self._finalize(aborted=False)

    def result(self) -> RecordingResult:
        """Measured truth of the finished recording; valid after drain."""
        result = self._result
        if result is None:
            raise RuntimeError('result() before the recording finished draining')
        return result

    def _close_selection_locked(self) -> None:
        """Close selection exactly once; the caller holds the lock."""
        if self._selection_closed:
            return
        self._selection_closed = True
        self._recording = False
        self._queue.put(_END_OF_RECORDING)

    def _drain_loop(self) -> None:
        """Writer lane: pop each frame and write it as its final artifact.

        A single frame's write failure costs exactly that frame (counted,
        never fatal). Any other escape -- including SystemExit from a
        dying dependency -- is writer-lane death: the recording aborts
        loudly and the backlog is discarded.
        """
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
                    self._write_frame(image, timestamp_s, frame_number, self._config, chunks)
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
            finally:
                with self._lock:
                    self._pending -= 1
            if self._discarding:
                return
        self._finalize(aborted=False)

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
        self._finalize(aborted=True)

    def _finalize(self, aborted: bool) -> None:
        """Compute measured truth, write the manifest, release the claim.

        Runs exactly once per recording: both the lane's normal drain end
        and a caller's discard_pending() route here, and whichever comes
        second is a no-op.
        """
        with self._lock:
            if self._result is not None:
                return
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
            if not aborted:
                manifest_path = self._write_manifest(
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
            )
            # Release BEFORE signalling drained: a caller woken by
            # wait_for_drain must observe the claim already free.
            self._claim.release('recording')
            self._drained.set()

    def _write_manifest(
        self,
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
        path = pathlib.Path(self._config.output_dir) / self._config.manifest_filename
        try:
            path.write_text(json.dumps(manifest, indent=2))
        except OSError as ex:
            # The frames on disk are the artifact; a manifest write
            # failure degrades reporting, never the recording.
            logger.error(f'[VideoEngine] Manifest write failed ({ex}); frames are unaffected')
            return None
        return path
