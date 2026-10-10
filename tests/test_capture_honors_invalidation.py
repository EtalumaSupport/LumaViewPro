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

from tests.scope_fakes import build_scope, bind_settings_like_a_session

_DARK = np.full((8, 8), 6, dtype=np.uint8)  # max 2.4% of full scale -- no signal


@pytest.fixture
def live_scope():
    """A full simulated scope: real IlluminationAPI over SimulatedLEDBoard
    and a streaming camera, on the lanes the scope builds. Every capture
    here goes through the public ``capture_and_wait``, so its body runs on
    the camera lane and an LED write injected from the camera driver
    dispatches to the IO lane, as a write from another caller would."""
    scope = build_scope(simulate=True)
    bind_settings_like_a_session(scope)
    scope._led_driver.set_timing_mode('fast')
    scope._motion_driver.set_timing_mode('fast')
    scope._camera_driver.set_timing_mode('fast')
    scope._camera_driver.load_cycle_images()
    scope.imaging.start_streaming()
    yield scope
    scope.imaging.stop_streaming()
    scope.disconnect()


def _during_the_first_grab(scope, monkeypatch, action, frame=None):
    """Land ``action`` inside the first grab window, from another caller.

    The camera driver's ``get_array`` is read only by the grab that becomes
    the capture, after the drain has settled and the LED state has been
    read, so its first call is the seam the pre-fix capture could not see.
    The action runs on a thread of its own, as a write from another caller
    does (a lane worker never dispatches onto another lane), and the grab
    waits for it to land. Returns an Event set once it has; the driver's
    frame is returned, or ``frame`` when one is given.
    """
    driver = scope._camera_driver
    real_get_array = driver.get_array
    landed = threading.Event()

    def write():
        action()
        landed.set()

    def get_array():
        if not landed.is_set():
            writer = threading.Thread(target=write, name='mid-window-write')
            writer.start()
            writer.join(5.0)
        return real_get_array() if frame is None else frame

    monkeypatch.setattr(driver, 'get_array', get_array)
    return landed


class TestMidWindowInvalidationHonored:
    def test_stale_lit_change_triggers_recheck_and_rederivation(self, live_scope, monkeypatch):
        """An LED commanded ON during the grab window re-runs the capture
        under the new state: the window is counted as re-checked, and the
        frame's record carries the lit channel, which is read before the
        grab -- a capture that returned the first grab would record the
        channel dark."""
        landed = _during_the_first_grab(
            live_scope, monkeypatch, lambda: live_scope.illumination.led_on('BF', 100)
        )

        out = live_scope.imaging.capture_and_wait(timeout_s=1.0)

        assert landed.is_set(), 'the mid-window write never landed'
        assert out is not None
        info = live_scope.imaging.last_capture_info
        assert info['rechecks'] >= 1, 'the dirtied window must be re-run, not returned'
        assert info['frame_record'].illumination_ma == {'BF': 100.0}, (
            'the re-run must read the NEW commanded state; '
            f'recorded {info["frame_record"].illumination_ma}'
        )

    def test_dirtied_window_rejection_recovers(self, live_scope, monkeypatch):
        """The other direction: the LED is commanded OFF during a lit
        capture whose frames are black. The re-run re-derives the
        expectation from the new state -- dark by design -- so the capture
        that comes back is not marked a dark frame under a lit LED, and its
        record names no lit channel."""
        live_scope.illumination.led_on('BF', 100)
        landed = _during_the_first_grab(
            live_scope, monkeypatch, lambda: live_scope.illumination.led_off('BF'), frame=_DARK
        )

        live_scope.imaging.capture_and_wait(timeout_s=0.05)

        assert landed.is_set(), 'the mid-window write never landed'
        info = live_scope.imaging.last_capture_info
        assert info['rechecks'] >= 1
        assert 'dark_saved' not in info, (
            'the expectation must re-derive lit -> dark; the capture was '
            'judged against the LED it no longer commands'
        )
        assert info['frame_record'].illumination_ma == {}

    def test_excluded_source_never_retriggers(self, live_scope, monkeypatch):
        """The live view's settled state excludes stage motion
        (``ScopeDisplay._camera_settled`` passes ``MOTION_SOURCES``, the
        one production caller of ``exclude_sources``); an excluded source
        landing mid-window never re-runs the capture."""
        landed = _during_the_first_grab(
            live_scope,
            monkeypatch,
            lambda: live_scope.imaging.frame_validity.invalidate('z_move'),
        )

        out = live_scope.imaging.capture_and_wait(timeout_s=1.0, exclude_sources=('z_move',))

        assert landed.is_set(), 'the mid-window write never landed'
        assert out is not None
        assert live_scope.imaging.last_capture_info['rechecks'] == 0


class TestDrainDeadline:
    @pytest.mark.slow
    def test_sustained_invalidation_returns_loud_none_in_bounded_time(
        self, live_scope, monkeypatch
    ):
        """The live-lock pin: invalidation arriving at least as fast as
        frames drain kept the pre-fix capture spinning for the stream's
        whole duration (25+ s recorded, bounded only by the harness
        watchdog). Post-fix the frozen budget expires and the capture
        returns None, recorded as a deadline expiry -- never as a grab
        failure. Slow: the budget's floor is 3 s of wall time, and the
        timing is the subject."""
        fv = live_scope.imaging.frame_validity
        real_get_array = live_scope._camera_driver.get_array

        def streaming_interference():
            fv.invalidate('led')  # each drained frame is answered by another change
            return real_get_array()

        monkeypatch.setattr(live_scope._camera_driver, 'get_array', streaming_interference)
        fv.invalidate('led')

        out = live_scope.imaging.capture_and_wait(timeout_s=0.5)

        assert out is None
        info = live_scope.imaging.last_capture_info
        assert info.get('deadline_expired') is True
        assert 'drain_failed' not in info, 'expiry must not be recorded as a grab failure'
        assert info['active_s'] <= info['deadline_s'] + 2.0, (
            'expiry must land near the budget, not an open-ended hold'
        )

    def test_healthy_capture_records_telemetry(self, live_scope):
        """The per-capture evidence a support bundle reads: present on
        every completed capture."""
        out = live_scope.imaging.capture_and_wait(timeout_s=1.0)
        assert out is not None
        info = live_scope.imaging.last_capture_info
        for key in ('hold_ms', 'drained', 'rechecks', 'deadline_s', 'n_entry', 'active_s'):
            assert key in info, f'missing capture-evidence key {key!r}'
        assert 'deadline_expired' not in info
