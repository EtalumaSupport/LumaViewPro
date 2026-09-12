# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Frame validity tracking for hardware state changes.

When hardware state changes (LED on/off, gain/exposure change, motor movement),
frames from the camera may not yet reflect the new state due to:
  1. Camera pipeline latency (2-3 frames to flush)
  2. Physical hardware settle time (motor moves take seconds)

Frame validity is the SINGLE source of truth for capture readiness. No capture
should proceed until frame_validity confirms all pending state changes have
settled. This includes both camera pipeline flush AND physical completion.

For camera-only sources (LED, gain, exposure): settled = frame count met.
For motion sources (xy_move, z_move, turret): settled = frame count met AND
axis has physically stopped moving (via settle callback).

A frame only counts toward a change it could actually show: each pending
source records which frame the camera had already delivered when its
register was written, and a frame credits it only if it arrived later.
Without that, a frame the live preview grabbed before the write but
counted after it would retire part of the wait it was never subject to.

Usage:
    fv = FrameValidity(lambda: camera.frames_delivered)
    fv.set_settle_check(my_axis_check_fn)  # Register motion completion callback
    fv.invalidate('z_move')                # Z axis started moving
    fv.frames_until_valid()                # Returns >0 (motion not complete)
    # ... frames are grabbed by live view, counter increments ...
    fv.frames_until_valid()                # Still >0 if Z still moving
    # ... Z arrives at target, settle check returns True ...
    fv.frames_until_valid()                # Returns 0 -- next frame is valid

Autofocus can exclude Z motion from validity checks since a slightly
defocused frame still produces a valid focus score:
    fv.is_valid_for(exclude_sources=('z_move',))
"""

import dataclasses
import threading
import time
from typing import ClassVar

from lib import profile_trace


@dataclasses.dataclass
class _PendingSource:
    """One unsettled state change: how many more frames it needs, and which
    frame the camera had already delivered when the write went out.

    That ordinal is what makes the count honest. A frame already in the
    camera's pipeline when the register was written cannot show the new
    state, so crediting it would retire the skip count against frames that
    predate the change -- and the capture then returns a frame from before
    the write.

    An ordinal rather than a clock reading, because the comparison has to
    survive the clock moving. Wall time runs backwards across a DST
    fall-back, an NTP correction or a host resume, and a settle count
    measured against a timestamp that jumped forward never completes: every
    later frame looks older than the write, the source never clears, and
    captures fail out for the whole window. Frame numbers only go up.
    """

    remaining: int
    at_seq: int


class FrameValidity:
    """Tracks frame validity after hardware state changes.

    Each hardware state change source has a configurable number of frames
    that must be skipped before the camera output reflects the new state.
    Motion sources additionally require physical completion (axis stopped).
    """

    DEFAULT_SKIP_FRAMES = 2

    # Per-source skip frame counts (camera pipeline flush).
    # Default skip counts -- overridden by per-camera measured values
    # from data/camera_timing/<model>.json via load_camera_timing().
    SKIP_FRAMES: ClassVar[dict] = {
        'led': 2,  # LED on/off or current change (measured: 2 on a2A3536)
        'gain': 2,  # Camera gain change (measured: 2 on a2A3536)
        'exposure': 3,  # Camera exposure time change (measured: 3 on a2A3536)
        'xy_move': 2,  # X or Y axis movement
        'z_move': 2,  # Z axis movement (autofocus may exclude this)
        'turret': 2,  # Turret rotation
        # Frames for hardware continuous auto-gain to settle against the lit
        # scene after arming. Like led/gain/exposure this is an instrumentation
        # settling count, not a content measurement: the LED must be lit before
        # arming so AG settles on the real scene, not a dark frame. At 10 a dim
        # brightfield scene (low LED current, gain starting at 0) often had not
        # converged when the frame was grabbed -- the capture came out dark.
        # Doubled to give AG more frames to ramp; the [AG CONVERGE] diagnostic
        # measures the real convergence need per camera for the eventual
        # camera_timing/<model>.json value (or a brightness-converge gate).
        'auto_gain': 20,
        # Geometry / format changes restart the grab engine or realloc the
        # camera buffer; the pipeline needs frames to flush the old
        # geometry before a capture reflects the new one. Conservative
        # defaults pending per-camera bench measurement into
        # camera_timing/<model>.json, like the others.
        'pixel_format': 3,
        'frame_size': 3,
        'binning': 3,
    }

    # Sources that require physical hardware completion in addition to frame count.
    MOTION_SOURCES = frozenset({'xy_move', 'z_move', 'turret'})

    # Sources carrying per-frame chunk metadata, used to REJECT a frame at
    # capture time -- never to accept one. A chunk reports the register value
    # in force when the camera TAGGED the frame, not the light integrated
    # into it, so a chunk that AGREES with the target proves nothing about
    # the pixels; a chunk that DISAGREES proves the frame predates the write.
    # Only the disagreement direction is sound. LED has no chunk equivalent;
    # motion is firmware-gated via _settle_check_fn.
    CHUNK_VALIDATABLE_SOURCES = frozenset({'gain', 'exposure'})

    # Maps our source names to the chunk_data dict keys used by camera
    # drivers (chunk_data uses the genicam attribute symbolic names).
    CHUNK_KEY_FOR_SOURCE: ClassVar[dict] = {
        'gain': 'Gain',
        'exposure': 'ExposureTime',
    }

    # Float-tolerance for chunk-match equality. ChunkExposureTime is in
    # microseconds (the API converts ms -> us when calling set_target);
    # ChunkGain is in dB. Values measured by sweeping set values across
    # the supported range and reading back ChunkGain / ChunkExposureTime
    # on multiple Basler USB3 cameras (a2A3536-31umBAS ace 2; daA3840-45um
    # dart with firmware 1.1.0 and 2.6.0): observed deltas were bit-
    # identical across hardware and firmware -- quantization happens at
    # the Pylon SDK / genicam nodemap layer, not in camera firmware.
    # Gain round-trip error peaked at ~5e-5 dB (float epsilon), exposure was
    # bit-exact in microseconds. Tolerances set ~20x above observed max
    # for safety across future firmware revisions.
    DEFAULT_CHUNK_TOLERANCE: ClassVar[dict] = {
        'gain': 0.001,  # dB
        'exposure': 2.0,  # microseconds
    }

    def __init__(self, frames_delivered):
        """
        Args:
            frames_delivered: zero-argument callable returning how many frames
                the camera has delivered so far. REQUIRED -- without it a write
                cannot be placed in the frame stream, every frame would credit
                every pending source, and that is precisely the defect this
                class exists to prevent. Passing a callable rather than the
                driver keeps the lookup late: the composition root assigns the
                camera driver after this object is built.
        """
        self._lock = threading.Lock()
        self._frames_delivered = frames_delivered
        self._frame_counter = 0
        self._pending = {}  # source -> _PendingSource
        self._settle_check_fn = None  # Optional: (source) -> bool
        self._target_values = {}  # source -> requested value (for chunk-match)
        self._last_counted_seq = None  # ordinal of the last counted frame
        # Monotone per-source invalidation history. Unlike _pending, entries
        # are never consumed by frames -- count_frame and reset() leave this
        # map untouched -- so a capture can snapshot it before its grab and
        # compare after to detect an invalidation that frames have already
        # settled. Pending-state snapshots cannot do that: a poller frame
        # plus the capture's own final grab can consume a 2-skip source's
        # entry exactly, leaving pending/frames_until_valid bit-identical
        # around a real mid-window invalidation.
        self._invalidation_counts = {}  # source -> total invalidate() calls

    def set_settle_check(self, fn):
        """Register a callback that checks if a source has physically settled.

        Args:
            fn: callable(source: str) -> bool. Returns True if the hardware
                for this source has physically completed its state change.
                For motion sources, this typically checks axis state == IDLE.
                For non-motion sources, should return True.

        Called during validity checks for MOTION_SOURCES. Without this
        callback, motion sources settle based on frame count only (legacy
        behavior, incorrect for long moves).
        """
        self._settle_check_fn = fn

    def invalidate(self, source: str) -> None:
        """Record that hardware state changed and frames need to settle.

        Args:
            source: What changed ('led', 'gain', 'exposure', 'auto_gain',
                    'xy_move', 'z_move', 'turret'). Unknown sources use
                    DEFAULT_SKIP_FRAMES.
        """
        skip = self.SKIP_FRAMES.get(source, self.DEFAULT_SKIP_FRAMES)
        with self._lock:
            # Read INSIDE the lock: two threads invalidating the same source
            # can otherwise store the earlier ordinal for the later write,
            # which is this whole class of bug reintroduced by a race.
            self._pending[source] = _PendingSource(remaining=skip, at_seq=self._frames_delivered())
            self._invalidation_counts[source] = self._invalidation_counts.get(source, 0) + 1
            counter = self._frame_counter
        if profile_trace.ENABLE_PROFILE_TRACE:
            profile_trace.trace(
                'frame_validity_trace.csv',
                'ts_ms,event,source,frame_counter,target_frame,pending_count',
                [
                    int(time.time() * 1000),
                    'invalidate',
                    source,
                    counter,
                    counter + skip,
                    len(self._pending),
                ],
                recording_id=profile_trace.NO_RECORDING,
            )

    def count_frame(self, frame_seq: int) -> None:
        """Record that a frame was grabbed from the camera.

        Call this after every successful camera grab (grab() or grab_new_capture()).
        A frame credits only the sources whose write it POSTDATES; a source
        clears once its skip count has been met that way. Motion sources
        clear only when both that count AND the settle check pass.

        Args:
            frame_seq: The frame's arrival ordinal, from the grab that
                produced it. Required: it is both the frame's identity and
                the evidence that the frame is newer than a given write.
                Multiple consumers poll the same buffered frame concurrently
                (live preview, histogram, capture drains); without identity
                dedupe those polls retire the skip counts with zero new
                frames, and a capture can then accept a frame exposed under
                the previous gain/exposure/LED state.
        """
        with self._lock:
            if frame_seq == self._last_counted_seq:
                return
            self._last_counted_seq = frame_seq
            self._frame_counter += 1
            settled = []
            for source, pending in self._pending.items():
                if frame_seq > pending.at_seq:
                    pending.remaining -= 1
                if self._is_source_settled_unlocked(source, pending):
                    settled.append(source)
            for s in settled:
                del self._pending[s]
            counter = self._frame_counter
            pending_count = len(self._pending)
        if profile_trace.ENABLE_PROFILE_TRACE and settled:
            profile_trace.trace(
                'frame_validity_trace.csv',
                'ts_ms,event,source,frame_counter,target_frame,pending_count',
                [
                    int(time.time() * 1000),
                    'settled',
                    '+'.join(settled),
                    counter,
                    counter,
                    pending_count,
                ],
                recording_id=profile_trace.NO_RECORDING,
            )

    def _is_source_settled_unlocked(self, source: str, pending: '_PendingSource') -> bool:
        """Check if a source has settled. Must be called with _lock held."""
        if pending.remaining > 0:
            return False
        # Motion sources also require physical completion
        if source in self.MOTION_SOURCES and self._settle_check_fn is not None:
            return self._settle_check_fn(source)
        return True

    def set_target(self, source: str, value: float | None) -> None:
        """Record the requested value for a chunk-validatable source.

        The API layer (Lumascope.set_gain_db / set_exposure_ms) calls this
        after invalidate() so the capture path can compare a returned
        frame's chunk against what was asked for and REJECT a frame whose
        chunk disagrees. The target never clears a source: settling is by
        frame count alone.

        Args:
            source: Source name (e.g. 'gain', 'exposure'). Sources outside
                CHUNK_VALIDATABLE_SOURCES are accepted but never consulted.
            value: Target value to compare chunks against. None clears any
                prior target.
        """
        with self._lock:
            if value is None:
                self._target_values.pop(source, None)
            else:
                self._target_values[source] = float(value)

    def target(self, source: str) -> float | None:
        """Return the recorded target value for a source, or None if unset.

        Capture paths use this to decide whether a returned frame can be
        chunk-verified: a set target plus a present chunk key means the
        frame must match; no target (e.g. hardware auto-gain owns the
        value) means chunk verification does not apply.

        Args:
            source: Source name (e.g. 'gain', 'exposure').
        """
        with self._lock:
            return self._target_values.get(source)

    def chunk_match(
        self, source: str, chunk_value: float | None, tolerance: float | None = None
    ) -> bool:
        """Public float-tolerant equality between a chunk value and the recorded target.

        Used by the capture path's stale-frame rejection, and by tests.

        Args:
            source: Source name.
            chunk_value: Observed value from chunk metadata. May be None.
            tolerance: Optional tolerance override; defaults to DEFAULT_CHUNK_TOLERANCE.
        """
        if chunk_value is None:
            return False
        with self._lock:
            target = self._target_values.get(source)
        if target is None:
            return False
        if tolerance is None:
            tolerance = self.DEFAULT_CHUNK_TOLERANCE.get(source, 0.0)
        return abs(float(chunk_value) - target) <= tolerance

    @property
    def is_valid(self) -> bool:
        """True if all pending state changes have settled."""
        with self._lock:
            return all(self._is_source_settled_unlocked(s, p) for s, p in self._pending.items())

    def is_valid_for(self, exclude_sources: tuple = ()) -> bool:
        """True if valid, ignoring specified sources.

        Useful for autofocus which can accept frames during Z motion:
            fv.is_valid_for(exclude_sources=('z_move',))
        """
        with self._lock:
            return all(
                self._is_source_settled_unlocked(s, p)
                for s, p in self._pending.items()
                if s not in exclude_sources
            )

    def frames_until_valid(self, exclude_sources: tuple = ()) -> int:
        """Number of frames that must be grabbed before the next valid frame.

        Returns 0 if already valid. For motion sources that have met the frame
        count but are still physically moving, returns 1 (keep draining).
        """
        with self._lock:
            max_remaining = 0
            for source, pending in self._pending.items():
                if source in exclude_sources:
                    continue
                if pending.remaining > 0:
                    max_remaining = max(max_remaining, pending.remaining)
                elif source in self.MOTION_SOURCES and self._settle_check_fn is not None:  # noqa: SIM102
                    # Frame count met but axis still moving -- keep draining
                    if not self._settle_check_fn(source):
                        max_remaining = max(max_remaining, 1)
            return max(0, max_remaining)

    @property
    def pending_sources(self) -> dict:
        """Current pending sources and the frames each still needs (for debugging)."""
        with self._lock:
            return {s: p.remaining for s, p in self._pending.items()}

    @property
    def frame_counter(self) -> int:
        """Current frame counter value (for debugging)."""
        with self._lock:
            return self._frame_counter

    @property
    def invalidation_counts(self) -> dict:
        """Per-source count of every invalidate() call, monotone for the
        instance's lifetime.

        Snapshot before a grab and compare (!=) after it to detect a
        mid-window invalidation regardless of whether frames have since
        settled it. Compare full dicts, not shared keys only: a source's
        first-ever invalidation adds a key the snapshot lacks.
        """
        with self._lock:
            return dict(self._invalidation_counts)

    def unsettled_motion_sources(self, exclude_sources: tuple = ()) -> tuple:
        """Pending motion sources whose settle-check says still moving.

        The capture deadline suspends while this is non-empty: motion has
        an authoritative completion signal, and a frame-budget clock must
        not out-vote it. Calling the settle-check under the lock follows
        the ordering frames_until_valid already established (validity lock,
        then the axis-state lock inside the callback).
        """
        with self._lock:
            if self._settle_check_fn is None:
                return ()
            return tuple(
                s
                for s in self._pending
                if s in self.MOTION_SOURCES
                and s not in exclude_sources
                and not self._settle_check_fn(s)
            )

    def load_camera_timing(self, config: dict):
        """Override SKIP_FRAMES from measured per-camera timing config.

        Args:
            config: dict with 'skip_frames' key mapping source names to
                    measured frame counts. Only sources present in the config
                    are overridden; others keep their defaults.

        Typically called after camera connects with data loaded from
        data/camera_timing/<model>.json.
        """
        measured = config.get('skip_frames', {})
        for source, count in measured.items():
            if isinstance(count, int) and count >= 0:
                self.SKIP_FRAMES[source] = count

    def reset(self) -> None:
        """Clear all pending invalidations and reset frame counter.

        Deliberately leaves invalidation_counts untouched: the counts are
        a monotone history, not pending state, and a capture comparing
        snapshots across a reset must still see any invalidation the reset
        would otherwise erase. Clearing them here would let an
        invalidate-then-reset sequence hide a real state change from an
        in-flight capture.
        """
        with self._lock:
            self._pending.clear()
            self._frame_counter = 0
            self._target_values.clear()
            self._last_counted_seq = None
