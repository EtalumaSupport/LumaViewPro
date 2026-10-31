# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""capture_still delivers while the camera streams.

A connected camera that is not grabbing is refused ``camera_not_streaming``
and a stalled stream raises ``no_frame`` (both pinned in
``test_a_capture_that_gets_no_frame_raises_why.py``); these pin that the
streaming path still delivers, a sum included.
"""

from __future__ import annotations

import dataclasses
import datetime


def test_capture_still_succeeds_while_streaming(sim_scope):
    # The precondition guard must not block the normal grabbing path: a
    # streaming sim camera still delivers a frame.
    assert sim_scope.imaging.is_streaming()
    result = sim_scope.imaging.capture_still(timeout_s=2.0).image
    assert result is not None


def test_a_summed_capture_survives_a_backwards_clock_step(sim_scope):
    """A sum orders its frames by arrival ordinal, not by wall clock.

    Wall time runs backwards across a DST fall-back, an NTP correction or a
    host resume. A sum that ordered on a clock rejected every frame stamped
    earlier than its predecessor -- which, after a backwards step, is every
    frame that follows -- and burned its timeout to return nothing, losing a
    protocol capture. Frame ordinals only go up, so the sum completes.
    """
    driver = sim_scope.imaging._driver
    real_grab_new_capture = driver.grab_new_capture
    stamped = {'n': 0}
    origin = datetime.datetime.now()

    def grab_with_a_clock_that_runs_backwards(timeout_s):
        frame = real_grab_new_capture(timeout_s)
        if frame is None:
            return None
        stamped['n'] += 1
        # Every frame is stamped EARLIER than the one before it. The ordinal
        # the driver minted is passed through untouched.
        return dataclasses.replace(
            frame, timestamp=origin - datetime.timedelta(seconds=stamped['n'])
        )

    driver.grab_new_capture = grab_with_a_clock_that_runs_backwards
    try:
        result = sim_scope.imaging.capture_still(sum_count=3, timeout_s=3.0).image
    finally:
        driver.grab_new_capture = real_grab_new_capture

    assert result is not None, (
        'a summed capture must not be lost because the host clock stepped backwards between frames'
    )
    assert stamped['n'] >= 3, (
        f'the sum should have grabbed at least its three frames, saw {stamped["n"]}'
    )
