# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""capture_and_wait names the not-grabbing precondition instead of timing out.

A camera that is connected (active) but not grabbing -- a bare scope that never
called start_streaming, or a feed deliberately halted by stop_streaming --
cannot deliver a frame. Before, capture_and_wait entered the drain loop and
limped to a falsy return after burning the grab timeout, with a symptom
indistinguishable from a real grab failure. Now it returns the not-ready
sentinel immediately and logs a distinct cause.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

from modules.lumascope_api import imaging as imaging_module


def test_capture_and_wait_returns_none_and_names_cause_when_not_grabbing(sim_scope):
    # sim_scope starts streaming; stop it so the camera stays connected
    # (active) but is no longer grabbing.
    sim_scope.imaging.stop_streaming()
    assert not sim_scope.imaging.is_streaming()

    # The module logger is mocked in the test env, so assert on the warning
    # call rather than a captured record.
    with patch.object(imaging_module, 'logger') as mock_logger:
        result = sim_scope.imaging.capture_and_wait(timeout_s=1.0)

    assert result is None
    warned = ' '.join(str(c).lower() for c in mock_logger.warning.call_args_list)
    assert 'no active grab' in warned, (
        'not-grabbing capture must warn a distinct cause, not fail silently'
    )


def test_capture_and_wait_succeeds_while_streaming(sim_scope):
    # The precondition guard must not block the normal grabbing path: a
    # streaming sim camera still delivers a frame.
    assert sim_scope.imaging.is_streaming()
    result = sim_scope.imaging.capture_and_wait(timeout_s=2.0)
    assert result is not None


def test_capture_and_wait_returns_none_when_drain_stalls(sim_scope):
    # A live feed whose drain cannot complete (grab timeouts while frame
    # validity still wants frames) is the stalled-feed failure mode. The
    # contract's failure sentinel is None -- a bool here slips every
    # `is None` caller check, so the stills leg skipped its capture
    # strike (and reset the accumulated counter) on exactly this mode.
    assert sim_scope.imaging.is_streaming()
    with (
        patch.object(sim_scope.imaging.frame_validity, 'frames_until_valid', return_value=1),
        patch.object(
            sim_scope.imaging._driver, 'grab_new_capture', return_value=(False, None, None)
        ),
        patch.object(imaging_module, 'logger'),
    ):
        result = sim_scope.imaging.capture_and_wait(timeout_s=1.0)

    assert result is None, 'stalled-feed drain failure must return the None sentinel'


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
        status, _ts, seq = real_grab_new_capture(timeout_s)
        stamped['n'] += 1
        # Every frame is stamped EARLIER than the one before it. The ordinal
        # the driver minted is passed through untouched.
        return status, origin - datetime.timedelta(seconds=stamped['n']), seq

    driver.grab_new_capture = grab_with_a_clock_that_runs_backwards
    try:
        result = sim_scope.imaging.capture_and_wait(sum_count=3, timeout_s=3.0)
    finally:
        driver.grab_new_capture = real_grab_new_capture

    assert result is not None, (
        'a summed capture must not be lost because the host clock stepped backwards between frames'
    )
    assert stamped['n'] >= 3, (
        f'the sum should have grabbed at least its three frames, saw {stamped["n"]}'
    )
