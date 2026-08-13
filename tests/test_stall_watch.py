# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Delivery-liveness primitives: StallWatch and the stall thresholds.

The watch is the one detector for loops that must terminate when a
camera feed silently dies; the threshold formulas anchor on the slowest
LEGITIMATE frame interval (max of the fps slot and the exposure) so a
long-exposure recording cannot false-trip, and detect death only --
trickle degradation is the wall cap's and the metrics surface's job.
"""

import pytest

from modules.video_cadence import (
    STALL_FLOOR_S,
    STALL_INTERVAL_MULTIPLE,
    StallWatch,
    prologue_stall_threshold_s,
    stall_threshold_s,
)


class TestStallWatch:
    def test_no_progress_ever_fires_at_threshold(self):
        watch = StallWatch(threshold_s=5.0)
        assert not watch.stalled(0, now=100.0)
        assert not watch.stalled(0, now=104.9)
        assert watch.stalled(0, now=105.0)

    def test_any_change_resets_the_timer(self):
        watch = StallWatch(threshold_s=5.0)
        assert not watch.stalled(0, now=100.0)
        assert not watch.stalled(1, now=104.9)
        assert not watch.stalled(1, now=109.8)
        assert watch.stalled(1, now=109.9)

    def test_nonmonotonic_change_still_counts_as_progress(self):
        # The prologue's signal can move in either direction; only
        # constancy means stalled.
        watch = StallWatch(threshold_s=5.0)
        assert not watch.stalled(3, now=100.0)
        assert not watch.stalled(2, now=104.0)
        assert not watch.stalled(2, now=108.9)
        assert watch.stalled(2, now=109.1)

    def test_nonpositive_threshold_refused(self):
        with pytest.raises(ValueError, match='threshold_s'):
            StallWatch(threshold_s=0.0)


class TestStallThreshold:
    def test_floor_governs_at_high_fps_and_short_exposure(self):
        assert stall_threshold_s(30.0, 0.03) == STALL_FLOOR_S

    def test_fps_term_governs_at_low_fps(self):
        # 0.5 fps, short exposure: 10 x 2 s slots.
        assert stall_threshold_s(0.5, 0.03) == STALL_INTERVAL_MULTIPLE * 2.0

    def test_exposure_term_governs_when_exposure_exceeds_the_slot(self):
        # The protocol regime: user-set fps with a long exposure. A 6 s
        # exposure at 2 fps legitimately gaps 6 s between frames; the
        # threshold must scale with the exposure, not the fps slot.
        assert stall_threshold_s(2.0, 6.0) == STALL_INTERVAL_MULTIPLE * 6.0

    def test_prologue_threshold_scales_with_exposure_only(self):
        assert prologue_stall_threshold_s(0.03) == STALL_FLOOR_S
        assert prologue_stall_threshold_s(6.0) == STALL_INTERVAL_MULTIPLE * 6.0

    def test_invalid_inputs_refused(self):
        with pytest.raises(ValueError, match='effective_fps'):
            stall_threshold_s(0.0, 0.03)
        with pytest.raises(ValueError, match='exposure_s'):
            stall_threshold_s(30.0, -1.0)
        with pytest.raises(ValueError, match='exposure_s'):
            prologue_stall_threshold_s(-0.1)
