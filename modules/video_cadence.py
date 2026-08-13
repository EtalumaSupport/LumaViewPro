# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Frame-cadence selection, frame-budget, and delivery-liveness arithmetic
for video recording.

A recording keeps at most one camera frame per cadence slot (slots are
1/fps seconds wide) and never more than its frame budget. The camera
free-runs; unselected frames are skipped, so cadence enforcement is
sampling, not loss. The liveness half answers the inverse question:
when has the free-running source stopped delivering entirely, so the
loop consuming it must terminate instead of hanging or lying.
"""

import math


# Interim per-camera delivery bound in frames per second: what the
# capture chain (sensor readout + USB bandwidth) actually sustains at
# typical resolutions, used to bound uncapped recording rates so a frame
# budget never derives from a bare 1/exposure the camera cannot deliver.
# Resolution-dependent in reality; the per-camera characterization map
# replaces this single value when it lands.
INTERIM_DELIVERY_BOUND_FPS = 40


def effective_recording_fps(requested_fps: float, global_max_fps: float) -> float:
    """The one rate authority: clamp a requested recording rate.

    CLAMP semantics, not fallback: ``min(requested, global max if
    nonzero, interim delivery bound)``. The global "Video max FPS" limit
    applies over a present per-caller request; a global value of 0 means
    uncapped, in which case the delivery bound alone keeps the frame
    budget at a rate the camera can actually deliver.

    Args:
        requested_fps: The caller's rate -- a protocol step's Video
            Config fps, or the manual path's exposure-derived rate.
        global_max_fps: The user's global cap (``video.max_fps``); 0
            means uncapped.

    Returns:
        The effective recording rate in frames per second.
    """
    fps = min(requested_fps, INTERIM_DELIVERY_BOUND_FPS)
    if global_max_fps > 0:
        fps = min(fps, global_max_fps)
    return fps


# Feed-death detection bounds; PERFORMANCE_BUDGETS.md rows
# recording_feed_stall_s / prologue_feed_stall_s. The floor covers OS
# scheduling jitter and unmeasured acquisition-restart gaps (a settings
# write can force a driver Stop+realloc+Start); the multiple tolerates
# delivery an order of magnitude slower than the slowest LEGITIMATE
# frame interval (worst observed configured-vs-measured ratio is 1.6x).
STALL_FLOOR_S = 5.0
STALL_INTERVAL_MULTIPLE = 10


def stall_threshold_s(effective_fps: float, exposure_s: float) -> float:
    """Seconds of zero frame progress after which a recording feed is dead.

    The base is the slowest LEGITIMATE frame interval: a recording's fps
    is user-set and decoupled from delivery, and a long exposure gaps
    frames longer than any fps-derived interval (exposures run to 10 s
    on some cameras) -- so the base is ``max(1/fps, exposure)``. For an
    exposure-derived rate the two terms coincide. This detects DEATH,
    not degradation: a trickling feed below the configured rate ends at
    the wall cap as an honest short delivery, by design.

    Raises:
        ValueError: If ``effective_fps`` is not positive or
            ``exposure_s`` is negative.
    """
    if effective_fps <= 0:
        raise ValueError(f'effective_fps must be positive, got {effective_fps}')
    if exposure_s < 0:
        raise ValueError(f'exposure_s must be >= 0, got {exposure_s}')
    return max(STALL_FLOOR_S, STALL_INTERVAL_MULTIPLE * max(1.0 / effective_fps, exposure_s))


def prologue_stall_threshold_s(exposure_s: float) -> float:
    """Feed-death bound for the pre-recording validity drain.

    The drain consumes the camera's free-run stream, whose interval is
    governed by exposure alone -- no recording fps applies yet.

    Raises:
        ValueError: If ``exposure_s`` is negative.
    """
    if exposure_s < 0:
        raise ValueError(f'exposure_s must be >= 0, got {exposure_s}')
    return max(STALL_FLOOR_S, STALL_INTERVAL_MULTIPLE * exposure_s)


class StallWatch:
    """Fires when an observed progress value stops changing for too long.

    The one silent-feed-death detector for loops that consume camera
    frames: the caller polls with its progress signal (a frame counter,
    an arrival count) and its own clock; ANY change in the value resets
    the timer, and the anchor starts at the first call so "no progress
    ever" also fires at the threshold. Loop-local by design -- the
    consuming loop owns the response (abort, refuse, notify), so the
    watch carries no callbacks, no thread, and no shared state.
    """

    def __init__(self, threshold_s: float):
        if threshold_s <= 0:
            raise ValueError(f'threshold_s must be positive, got {threshold_s}')
        self._threshold_s = threshold_s
        self._last_value = None
        self._last_change_ts: float | None = None

    def stalled(self, progress_value, now: float) -> bool:
        """True when ``progress_value`` has not changed for the threshold."""
        if self._last_change_ts is None or progress_value != self._last_value:
            self._last_value = progress_value
            self._last_change_ts = now
            return False
        return (now - self._last_change_ts) >= self._threshold_s


def frame_budget(fps: float, duration_s: float) -> int:
    """Number of frame slots a recording of ``duration_s`` at ``fps`` can fill.

    This exact capacity sizes recording buffers, pre-flight disk checks,
    and capacity-stop conditions; it is never an estimate.

    Args:
        fps: Target recording rate in frames per second. Must be positive.
        duration_s: Recording duration in seconds. Must be positive.

    Returns:
        ``ceil(fps * duration_s)``.

    Raises:
        ValueError: If ``fps`` or ``duration_s`` is not positive.
    """
    if fps <= 0:
        raise ValueError(f'fps must be positive, got {fps}')
    if duration_s <= 0:
        raise ValueError(f'duration_s must be positive, got {duration_s}')
    return math.ceil(fps * duration_s)


class CadenceSelector:
    """Slot-cadence frame selector: keeps one frame per 1/fps window.

    The frame source free-runs; per delivered frame the selector answers
    "does this frame claim the next save slot?". Slot deadlines advance
    by exactly one interval per reservation and are never re-anchored to
    the wall clock, so after a delivery stall the following frames are
    selected back-to-back until the slot schedule catches up to real
    time -- a stall costs latency, never slots.

    Not thread-safe: callers serialize ``slot_open`` / ``reserve``. The
    two-step shape (query, then commit) is deliberate -- it lets a caller
    interpose a drop-before-reserve backpressure probe between the
    decision and the reservation without the selector knowing about it.
    """

    def __init__(self, fps: float, max_frames: int, start_ts: float):
        """Set up selection at ``fps`` starting from ``start_ts``.

        Args:
            fps: Selection rate in frames per second. Must be positive.
            max_frames: Capacity in slots; selection closes once all are
                reserved.
            start_ts: Recording start time (``time.time()`` scale). The
                first slot deadline is one interval after this.

        Raises:
            ValueError: If ``fps`` is not positive or ``max_frames`` is
                negative.
        """
        if fps <= 0:
            raise ValueError(f'fps must be positive, got {fps}')
        if max_frames < 0:
            raise ValueError(f'max_frames must be >= 0, got {max_frames}')
        self._interval_s = 1.0 / fps
        self._next_slot_ts = start_ts + self._interval_s
        self._max_frames = max_frames
        self._reserved = 0

    @property
    def reserved_count(self) -> int:
        """Slots reserved so far."""
        return self._reserved

    @property
    def at_capacity(self) -> bool:
        """True once every slot is reserved; selection is closed."""
        return self._reserved >= self._max_frames

    def slot_open(self, now: float) -> bool:
        """Whether a frame delivered at ``now`` claims the next slot.

        Pure query -- ``reserve`` commits the claim.

        The deadline and the timestamp are independently-rounded float
        representations of physical instants (the deadline is an
        accumulated sum of the rounded interval; the timestamp carries
        its own rounding), so when both encode the SAME instant -- a
        camera delivering at exactly the configured cadence -- either
        may land a few ulps below the other and exact ``>=`` would
        silently lose that slot (the recording's final frame, at the
        boundary). Equality is therefore judged with a closeness bound
        scaled to a millionth of the slot width: representation error
        never approaches it, and no two distinct real frames can sit
        within a millionth of a slot of the same deadline.
        """
        if self.at_capacity:
            return False
        return now >= self._next_slot_ts or math.isclose(
            now, self._next_slot_ts, rel_tol=0.0, abs_tol=self._interval_s * 1e-6
        )

    def reserve(self) -> int:
        """Commit the pending slot claim and return the reserved slot index.

        Advances the slot deadline by one interval (never re-anchored to
        the wall clock -- that is what makes catch-up-after-stall work).

        Raises:
            RuntimeError: If called at capacity; callers gate on
                ``slot_open`` first.
        """
        if self.at_capacity:
            raise RuntimeError(f'reserve() past capacity ({self._max_frames} slots)')
        slot_index = self._reserved
        self._reserved += 1
        self._next_slot_ts += self._interval_s
        return slot_index
