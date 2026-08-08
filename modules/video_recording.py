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

This module is currently a signatures-only skeleton: the contract tests
in tests/test_video_recording_contract.py define the behavior; the
implementation follows them.
"""

import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from modules.video_cadence import frame_budget


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
    """

    fps: float
    duration_s: float
    width: int
    height: int
    bit_depth: int
    output_dir: pathlib.Path
    filename_template: str

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
            frame: ``write_frame(image, timestamp_s, frame_number, config)
            -> pathlib.Path``. Raising costs exactly that frame.
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
        raise NotImplementedError('VideoRecordingEngine is a contract skeleton')

    @property
    def is_recording(self) -> bool:
        """True from a successful ``start`` until selection closes."""
        raise NotImplementedError

    @property
    def is_draining(self) -> bool:
        """True while the writer lane still holds queued frames."""
        raise NotImplementedError

    @property
    def pending_writes(self) -> int:
        """Frames enqueued but not yet written (the writer's current lag)."""
        raise NotImplementedError

    def start(self, config: RecordingConfig) -> None:
        """Open selection for one recording.

        Atomically acquires the exclusivity claim; exactly one of two
        concurrent starts can win.

        Raises:
            RecordingRefusedError: When an exclusive activity (protocol
                run or another recording) already holds the claim, or
                this engine is already recording or draining.
        """
        raise NotImplementedError

    def ingest_frame(self, image: Any, timestamp_s: float, chunks: Any = None) -> None:
        """Offer one delivered camera frame: select + enqueue only.

        Runs on the camera ingest thread; must stay cheap. A kept frame
        is enqueued unconditionally -- writer lag never causes a
        capture-side drop. Frame numbers derive from enqueue order
        (contiguous ordinals), so holes are unrepresentable.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Close selection promptly (within one selection decision).

        Drain continues: frames already queued keep writing until the
        lane is empty. Stopping does not require the writer to have kept
        up -- it never does.
        """
        raise NotImplementedError

    def wait_for_drain(self, timeout: float | None = None) -> bool:
        """Block until every queued frame is written; True when drained."""
        raise NotImplementedError

    def discard_pending(self) -> None:
        """Drop the queued backlog loudly (explicit user choice at close).

        The discarded count lands in the manifest and the result as a
        short delivery; discard is never silent.
        """
        raise NotImplementedError

    def result(self) -> RecordingResult:
        """Measured truth of the finished recording; valid after drain."""
        raise NotImplementedError
