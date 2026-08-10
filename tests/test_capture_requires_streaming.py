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
        result = sim_scope.imaging.capture_and_wait(dark_floor_check=False, timeout_s=1.0)

    assert result is None
    warned = ' '.join(str(c).lower() for c in mock_logger.warning.call_args_list)
    assert 'no active grab' in warned, (
        'not-grabbing capture must warn a distinct cause, not fail silently'
    )


def test_capture_and_wait_succeeds_while_streaming(sim_scope):
    # The precondition guard must not block the normal grabbing path: a
    # streaming sim camera still delivers a frame.
    assert sim_scope.imaging.is_streaming()
    result = sim_scope.imaging.capture_and_wait(dark_floor_check=False, timeout_s=2.0)
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
        patch.object(sim_scope.imaging._driver, 'grab_new_capture', return_value=(False, None)),
        patch.object(imaging_module, 'logger'),
    ):
        result = sim_scope.imaging.capture_and_wait(dark_floor_check=False, timeout_s=1.0)

    assert result is None, 'stalled-feed drain failure must return the None sentinel'
