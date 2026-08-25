# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The capture honors invalidation across its whole window, in bounded time.

Two defect shapes, both demonstrated red before the fix:

Stale window: `capture_and_wait` drained validity and then grabbed, but a
state change landing after the drain -- during the expectation derivation
or the grab itself -- was invisible: the capture returned a frame (or a
rejection) that predated what the caller had just commanded. Recorded
red: one `_get_image_impl` call with the stale dark-floor expectation and
no re-check telemetry at all.

Unbounded drain: the drain loop had no time bound, so a sustained
invalidation stream (a 10 Hz illumination-slider drag at >= 150 ms
exposure) held the capture for the stream's whole duration -- 25+ s
observed, released only when the stream stopped.

The fix pairs a monotone invalidation-count snapshot/compare around the
grab window (re-drain + re-derive + re-grab on any mid-window change)
with a drain-and-recheck deadline frozen at entry (loud, distinctly
logged None when invalidation outruns the budget). Illumination is REAL
in every test here -- the derivation and the invalidation both run end to
end over the simulated LED board; stubbing either would let an inert
derivation pass (the defect shape that killed two plan drafts).
"""

import threading

import numpy as np
import pytest
from unittest.mock import patch

import modules.lumascope_api.imaging as imaging_module
from modules.lumascope_api import Lumascope

_DARK = np.full((8, 8), 6, dtype=np.uint8)  # max 2.4% of full scale -- no signal


@pytest.fixture
def live_scope():
    """A full simulated scope: real IlluminationAPI over SimulatedLEDBoard,
    streaming camera, no executors (dispatch runs inline on the calling
    thread, so an injected LED write invalidates synchronously)."""
    scope = Lumascope(simulate=True)
    scope._led_driver.set_timing_mode('fast')
    scope._motion_driver.set_timing_mode('fast')
    scope._camera_driver.set_timing_mode('fast')
    scope._camera_driver.load_cycle_images()
    scope.imaging.start_streaming()
    yield scope
    scope.imaging.stop_streaming()
    scope.disconnect()


def _inject_during_first_grab(imaging, action):
    """Wrap _get_image_impl so `action` fires inside the first grab window,
    after the drain settled and the expectation was derived -- the exact
    seam the pre-fix capture could not see. Returns the recorded
    dark_floor_check flag per call."""
    flags = []
    armed = {'on': True}
    orig = imaging._get_image_impl

    def wrapped(**kw):
        flags.append(kw.get('dark_floor_check'))
        if armed['on']:
            armed['on'] = False
            action()
        return orig(**kw)

    imaging._get_image_impl = wrapped
    return flags, orig


class TestMidWindowInvalidationHonored:
    def test_stale_lit_change_triggers_recheck_and_rederivation(self, live_scope):
        """LED commanded ON during the grab window: the capture must
        detect the dirtied window, re-drain, re-derive the dark-floor
        expectation (off -> on), and grab again under the new state."""
        flags, orig = _inject_during_first_grab(
            live_scope.imaging, lambda: live_scope.illumination.led_on('BF', 100)
        )
        try:
            out = live_scope.imaging._capture_and_wait_impl(timeout_s=1.0)
        finally:
            live_scope.imaging._get_image_impl = orig

        assert out is not None
        assert len(flags) == 2, 'the dirtied window must be re-run, not returned'
        assert flags == [False, True], (
            'the dark-floor expectation must be re-derived from the NEW '
            f'commanded state; saw {flags}'
        )
        info = live_scope.imaging._last_capture_info
        assert info['rechecks'] >= 1

    def test_dirtied_window_rejection_recovers(self, live_scope, monkeypatch):
        """The other stale direction: LEDs commanded OFF during a lit
        capture whose frames are black. Pre-fix the dark-floor retry burns
        its budget and fails the capture on a state the caller no longer
        commands; post-fix the compare sees the dirtied window, the
        expectation re-derives to dark-by-design, and the frame is
        accepted."""
        live_scope.illumination.led_on('BF', 100)
        monkeypatch.setattr(live_scope._camera_driver, 'get_array', lambda: _DARK)
        flags, orig = _inject_during_first_grab(
            live_scope.imaging, lambda: live_scope.illumination.led_off('BF')
        )
        try:
            out = live_scope.imaging._capture_and_wait_impl(timeout_s=0.3)
        finally:
            live_scope.imaging._get_image_impl = orig

        assert out is not None, 'a rejection caused by the very change must recover'
        assert flags == [True, False], f'the expectation must re-derive lit -> dark; saw {flags}'
        assert live_scope.imaging._last_capture_info['rechecks'] >= 1

    def test_excluded_source_never_retriggers(self, live_scope):
        """Autofocus excludes z_move from validity; a z_move invalidation
        mid-window must not re-run its capture (the sweep IS the motion)."""
        flags, orig = _inject_during_first_grab(
            live_scope.imaging,
            lambda: live_scope.imaging.frame_validity.invalidate('z_move'),
        )
        try:
            out = live_scope.imaging._capture_and_wait_impl(
                timeout_s=1.0, exclude_sources=('z_move',)
            )
        finally:
            live_scope.imaging._get_image_impl = orig

        assert out is not None
        assert len(flags) == 1
        assert live_scope.imaging._last_capture_info['rechecks'] == 0

    def test_clean_window_failure_still_propagates(self, live_scope, monkeypatch):
        """A genuine rejection with NO mid-window change must stay a loud
        None -- the recovery path must not retry failures it cannot cure."""
        live_scope.illumination.led_on('BF', 100)
        monkeypatch.setattr(live_scope._camera_driver, 'get_array', lambda: _DARK)

        out = live_scope.imaging._capture_and_wait_impl(timeout_s=0.3)

        assert out is None
        assert live_scope.imaging._last_capture_info['rechecks'] == 0


class TestDrainDeadline:
    def test_sustained_invalidation_returns_loud_none_in_bounded_time(
        self, live_scope, monkeypatch
    ):
        """The live-lock pin: invalidation arriving at least as fast as
        frames drain kept the pre-fix capture spinning for the stream's
        whole duration (25+ s recorded, bounded only by the harness
        watchdog). Post-fix the frozen budget expires and the capture
        returns None with the deadline named distinctly in the log --
        never mislabeled as a grab failure."""
        fv = live_scope.imaging.frame_validity
        real_get_array = live_scope._camera_driver.get_array

        def streaming_interference():
            fv.invalidate('led')  # each drained frame is answered by another change
            return real_get_array()

        monkeypatch.setattr(live_scope._camera_driver, 'get_array', streaming_interference)
        fv.invalidate('led')

        with patch.object(imaging_module, 'logger') as mock_logger:
            out = live_scope.imaging._capture_and_wait_impl(timeout_s=0.5)

        assert out is None
        info = live_scope.imaging._last_capture_info
        assert info.get('deadline_expired') is True
        assert info['active_s'] <= info['deadline_s'] + 2.0, (
            'expiry must land near the budget, not an open-ended hold'
        )
        warned = ' '.join(str(c) for c in mock_logger.warning.call_args_list)
        assert 'DEADLINE EXPIRED' in warned, (
            f'the deadline must be named distinctly in the log; saw: {warned!r}'
        )
        assert 'drain failed' not in warned, 'expiry must not be mislabeled as grab failure'

    def test_healthy_capture_records_telemetry(self, live_scope):
        """The per-capture evidence a support bundle reads: present on
        every completed capture (it was pinned nowhere before this file)."""
        out = live_scope.imaging._capture_and_wait_impl(timeout_s=1.0)
        assert out is not None
        info = live_scope.imaging._last_capture_info
        for key in ('hold_ms', 'drained', 'rechecks', 'deadline_s', 'n_entry', 'active_s'):
            assert key in info, f'missing capture-evidence key {key!r}'
        assert 'deadline_expired' not in info


class TestSustainedStreamThreaded:
    def test_concurrent_led_stream_is_bounded(self, live_scope):
        """The F12 topology end to end: a real thread streaming LED
        commands through the public illumination API while the capture
        runs. The capture must return (frame or None) in bounded time
        instead of waiting out the stream."""
        stop = threading.Event()

        def slider_drag():
            on = True
            while not stop.is_set():
                if on:
                    live_scope.illumination.led_on('BF', 50)
                else:
                    live_scope.illumination.led_off('BF')
                on = not on
                stop.wait(0.02)

        t = threading.Thread(target=slider_drag, daemon=True)
        t.start()
        try:
            live_scope.imaging._capture_and_wait_impl(timeout_s=0.5)
            info = live_scope.imaging._last_capture_info
            # Either the drain outran the stream (fast sim frames) and a
            # frame came back, or the deadline fired -- both are bounded;
            # the pre-fix behavior (hold until the stream stops) is not.
            assert info['hold_ms'] / 1000.0 <= info['deadline_s'] + 5.0
        finally:
            stop.set()
            t.join(timeout=2.0)
