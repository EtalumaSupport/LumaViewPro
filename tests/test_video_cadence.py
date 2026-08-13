# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Characterization tests for the manual-video cadence selector and the
frame-budget arithmetic (modules/video_cadence.py).

These pin the semantics the video-capture engine KEEPS through the port:
slot cadence (one frame per 1/fps window, sampling not loss) and
catch-up-after-stall (slot deadlines advance one interval per
reservation and are never re-anchored to the wall clock, so a delivery
stall costs latency, never slots).

Deliberately NOT pinned: the manual sink's drop-before-reserve
backpressure probe (camera_executor.admit_live_frame), which the UI
interposes between slot_open and reserve. That leg is writer-lag
coupling that dies with the old memmap sink at the manual cutover; the
engine's queue admits unconditionally.
"""

import itertools
import math
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modules.video_cadence import CadenceSelector, frame_budget

settings.register_profile('ci', derandomize=True)
if os.environ.get('CI'):
    settings.load_profile('ci')


START = 1000.0


def run_selection(selector, arrival_times):
    """Feed timestamps through the selector; return (arrival, slot) picks."""
    picks = []
    for t in arrival_times:
        if selector.slot_open(t):
            picks.append((t, selector.reserve()))
    return picks


class TestFrameBudget:
    def test_matches_production_sizing_formula(self):
        # The manual record path sizes its buffer with this exact product;
        # representative values: exposure-derived fps at the 300 s default
        # duration, and a fractional-fps case where ceil matters.
        assert frame_budget(5.0, 300) == 1500
        assert frame_budget(33.333333, 30) == math.ceil(33.333333 * 30)
        assert frame_budget(0.5, 3) == 2

    def test_rejects_nonpositive_fps(self):
        with pytest.raises(ValueError):
            frame_budget(0, 10)
        with pytest.raises(ValueError):
            frame_budget(-5, 10)

    def test_rejects_nonpositive_duration(self):
        with pytest.raises(ValueError):
            frame_budget(5, 0)
        with pytest.raises(ValueError):
            frame_budget(5, -1)

    @given(
        fps=st.integers(min_value=1, max_value=120),
        duration_s=st.integers(min_value=1, max_value=3600),
    )
    def test_integer_products_are_exact(self, fps, duration_s):
        assert frame_budget(fps, duration_s) == fps * duration_s

    @given(
        fps=st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
        duration_s=st.floats(min_value=0.01, max_value=3600, allow_nan=False, allow_infinity=False),
    )
    def test_budget_never_truncates(self, fps, duration_s):
        # The budget is a capacity, never an estimate: it must cover the
        # full fps x duration product (no truncation at any duration).
        budget = frame_budget(fps, duration_s)
        assert budget >= fps * duration_s
        assert budget - fps * duration_s < 1.0 + 1e-9

    @given(
        fps=st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
        d1=st.floats(min_value=0.01, max_value=1800, allow_nan=False, allow_infinity=False),
        d2=st.floats(min_value=0.01, max_value=1800, allow_nan=False, allow_infinity=False),
    )
    def test_budget_monotonic_in_duration(self, fps, d1, d2):
        lo, hi = sorted([d1, d2])
        assert frame_budget(fps, lo) <= frame_budget(fps, hi)


class TestCadenceSelectorConstruction:
    def test_rejects_nonpositive_fps(self):
        with pytest.raises(ValueError):
            CadenceSelector(fps=0, max_frames=10, start_ts=START)
        with pytest.raises(ValueError):
            CadenceSelector(fps=-1, max_frames=10, start_ts=START)

    def test_rejects_negative_capacity(self):
        with pytest.raises(ValueError):
            CadenceSelector(fps=5, max_frames=-1, start_ts=START)

    def test_zero_capacity_is_closed_from_birth(self):
        s = CadenceSelector(fps=5, max_frames=0, start_ts=START)
        assert s.at_capacity
        assert not s.slot_open(START + 100)


class TestSlotCadence:
    def test_first_slot_deadline_is_one_interval_after_start(self):
        # A frame arriving at the start instant is NOT selected; the first
        # slot opens one full interval later.
        s = CadenceSelector(fps=5, max_frames=10, start_ts=START)
        assert not s.slot_open(START)
        assert not s.slot_open(START + 0.199)
        assert s.slot_open(START + 0.2)

    def test_fast_delivery_selects_one_frame_per_slot(self):
        # Camera free-runs at ~100 fps; a 5 fps cadence keeps exactly one
        # frame per 0.2 s window across a 2 s run. Delivery runs one step
        # past the window: deadlines accumulate by repeated float addition
        # of the interval, so the last deadline sits an epsilon past 2.0 s.
        s = CadenceSelector(fps=5, max_frames=100, start_ts=START)
        arrivals = [START + i * 0.01 for i in range(1, 203)]
        picks = run_selection(s, arrivals)
        assert len(picks) == 10
        pick_times = [t for t, _ in picks]
        gaps = [b - a for a, b in itertools.pairwise(pick_times)]
        assert all(0.19 < g < 0.21 for g in gaps)

    def test_slot_indices_are_sequential(self):
        s = CadenceSelector(fps=10, max_frames=50, start_ts=START)
        arrivals = [START + i * 0.02 for i in range(1, 101)]
        picks = run_selection(s, arrivals)
        assert [slot for _, slot in picks] == list(range(len(picks)))

    def test_capacity_closes_selection(self):
        s = CadenceSelector(fps=100, max_frames=3, start_ts=START)
        arrivals = [START + i * 0.01 for i in range(1, 100)]
        picks = run_selection(s, arrivals)
        assert len(picks) == 3
        assert s.at_capacity
        assert not s.slot_open(START + 1000)

    def test_reserve_past_capacity_raises(self):
        s = CadenceSelector(fps=5, max_frames=1, start_ts=START)
        assert s.slot_open(START + 0.2)
        s.reserve()
        with pytest.raises(RuntimeError):
            s.reserve()

    def test_reserved_count_tracks_reservations(self):
        s = CadenceSelector(fps=5, max_frames=10, start_ts=START)
        assert s.reserved_count == 0
        s.reserve()
        s.reserve()
        assert s.reserved_count == 2


class TestCatchUpAfterStall:
    def test_stall_is_repaid_back_to_back(self):
        # One frame claims slot 0, then delivery stalls for 1.8 s. The slot
        # schedule keeps its cadence, so the frames after the stall are
        # selected consecutively (no cadence gap) until the schedule
        # catches back up to the wall clock.
        s = CadenceSelector(fps=5, max_frames=100, start_ts=START)
        assert s.slot_open(START + 0.2)
        s.reserve()

        post_stall = [START + 2.0 + i * 0.01 for i in range(0, 40)]
        picks = run_selection(s, post_stall)

        # Deadlines 0.4, 0.6, ..., 2.0 are all overdue at the first
        # post-stall frame: nine consecutive frames are selected.
        consecutive = [t for t, _ in picks[:9]]
        assert consecutive == post_stall[:9]
        # After catch-up, cadence resumes: the next pick waits for the
        # 2.2 s deadline instead of being consecutive.
        assert len(picks) == 10
        assert abs(picks[9][0] - (START + 2.2)) < 0.011

    def test_deadline_advances_from_schedule_not_from_now(self):
        # A late frame reserves against an overdue deadline; the next
        # deadline is one interval after the OLD deadline, not one
        # interval after the late frame's arrival.
        s = CadenceSelector(fps=5, max_frames=10, start_ts=START)
        late = START + 1.0
        assert s.slot_open(late)
        s.reserve()
        # If the deadline had re-anchored to the late arrival, the next
        # slot would not open until late + 0.2; catch-up requires it to
        # already be open.
        assert s.slot_open(late + 0.001)

    def test_stall_never_costs_slots(self):
        # Total selections across a stalled run equal what an unstalled
        # run of the same duration yields, capacity permitting.
        fps, duration = 5, 4.0
        smooth = CadenceSelector(fps=fps, max_frames=100, start_ts=START)
        smooth_picks = run_selection(
            smooth, [START + i * 0.01 for i in range(1, int(duration * 100) + 1)]
        )

        stalled = CadenceSelector(fps=fps, max_frames=100, start_ts=START)
        # Delivery gap from 0.5 s to 3.0 s mid-run.
        arrivals = [
            START + t * 0.01 for t in range(1, int(duration * 100) + 1) if not (50 < t * 1.0 < 300)
        ]
        stalled_picks = run_selection(stalled, arrivals)
        assert len(stalled_picks) == len(smooth_picks)


class TestSelectorProperties:
    @given(
        fps=st.floats(min_value=0.5, max_value=120, allow_nan=False, allow_infinity=False),
        max_frames=st.integers(min_value=0, max_value=500),
        deltas=st.lists(
            st.floats(min_value=0.0001, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=300,
        ),
    )
    def test_invariants_hold_for_arbitrary_arrivals(self, fps, max_frames, deltas):
        s = CadenceSelector(fps=fps, max_frames=max_frames, start_ts=START)
        arrivals = []
        t = START
        for d in deltas:
            t += d
            arrivals.append(t)
        picks = run_selection(s, arrivals)

        # Capacity is never exceeded and indices are sequential.
        assert s.reserved_count <= max_frames
        assert [slot for _, slot in picks] == list(range(len(picks)))

        # Selection can never outrun elapsed slots: each reservation
        # consumes one 1/fps window of the schedule.
        elapsed = arrivals[-1] - START
        assert s.reserved_count <= math.floor(elapsed * fps) + 1

    @given(
        fps=st.floats(min_value=0.5, max_value=120, allow_nan=False, allow_infinity=False),
        deltas=st.lists(
            st.floats(min_value=0.0001, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=200,
        ),
    )
    def test_selection_is_deterministic(self, fps, deltas):
        arrivals = []
        t = START
        for d in deltas:
            t += d
            arrivals.append(t)
        first = run_selection(CadenceSelector(fps=fps, max_frames=1000, start_ts=START), arrivals)
        second = run_selection(CadenceSelector(fps=fps, max_frames=1000, start_ts=START), arrivals)
        assert first == second

    @given(
        fps=st.floats(min_value=1.0, max_value=50, allow_nan=False, allow_infinity=False),
        n_slots=st.integers(min_value=1, max_value=100),
    )
    def test_dense_delivery_fills_every_elapsed_slot(self, fps, n_slots):
        # With delivery much faster than the cadence, every slot in the
        # elapsed window is claimed (within one slot of float tolerance).
        interval = 1.0 / fps
        duration = n_slots * interval
        step = interval / 8
        s = CadenceSelector(fps=fps, max_frames=n_slots + 5, start_ts=START)
        arrivals = []
        t = START
        while t <= START + duration + step:
            t += step
            arrivals.append(t)
        picks = run_selection(s, arrivals)
        assert abs(len(picks) - n_slots) <= 1
