# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Frame-cadence selection and frame-budget arithmetic for video recording.

A recording keeps at most one camera frame per cadence slot (slots are
1/fps seconds wide) and never more than its frame budget. The camera
free-runs; unselected frames are skipped, so cadence enforcement is
sampling, not loss.
"""

import math


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
        """
        return not self.at_capacity and now >= self._next_slot_ts

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
