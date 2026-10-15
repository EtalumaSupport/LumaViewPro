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

from modules.exceptions import CaptureFailedError
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
    scope._camera_driver.load_specimen()
    scope.imaging.enable_streaming()
    yield scope
    scope.imaging.disable_streaming()
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
        bf_address = live_scope.layer_identity.find('BF').led_channel[0]
        assert info['frame_record'].illumination_ma == {bf_address: 100.0}, (
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
    def test_sustained_invalidation_raises_deadline_expired_in_bounded_time(
        self, live_scope, monkeypatch
    ):
        """The live-lock pin: invalidation arriving at least as fast as
        frames drain kept the pre-fix capture spinning for the stream's
        whole duration (25+ s recorded, bounded only by the harness
        watchdog). Post-fix the frozen budget expires and the capture
        raises ``deadline_expired`` -- never a grab failure's ``no_frame``.
        Slow: the budget's floor is 3 s of wall time, and the timing is the
        subject."""
        fv = live_scope.imaging.frame_validity
        real_get_array = live_scope._camera_driver.get_array

        def streaming_interference():
            fv.invalidate('led')  # each drained frame is answered by another change
            return real_get_array()

        monkeypatch.setattr(live_scope._camera_driver, 'get_array', streaming_interference)
        fv.invalidate('led')

        with pytest.raises(CaptureFailedError) as failed:
            live_scope.imaging.capture_and_wait(timeout_s=0.5)

        assert failed.value.reason == 'deadline_expired', 'expiry is not a grab failure'
        info = live_scope.imaging.last_capture_info
        assert info['active_s'] <= info['deadline_s'] + 2.0, (
            'expiry must land near the budget, not an open-ended hold'
        )

    def test_a_change_during_the_dark_wait_still_saves_the_dark_frame(
        self, live_scope, monkeypatch
    ):
        """A capture under a lit LED whose frames stay dark waits its retry
        window for a lit one, then saves the dark frame flagged. A turret
        move landing during that wait re-runs the grab; the budget holds
        the grab's own wait, so the re-run ends as the motionless capture
        does -- saved, flagged dark -- not refused as invalidation churn
        (#852). The floor is shortened on this instance so the window it
        must hold is under a second."""
        monkeypatch.setattr(live_scope.imaging, '_CAPTURE_DEADLINE_FLOOR_S', 0.2)
        live_scope.illumination.led_on('BF', 67)
        # Settled before the capture under test, as a lit scope is: frames
        # still owed to the LED would size its budget past the dark wait.
        assert live_scope.imaging.capture_and_wait(timeout_s=1.0) is not None
        landed = _during_the_first_grab(
            live_scope,
            monkeypatch,
            lambda: live_scope.imaging.frame_validity.invalidate('turret'),
            frame=_DARK,
        )

        out = live_scope.imaging.capture_and_wait(timeout_s=0.5)

        assert landed.is_set(), 'the mid-window write never landed'
        info = live_scope.imaging.last_capture_info
        assert info['n_entry'] == 0, 'precondition: nothing owed at entry'
        assert info['rechecks'] >= 1, 'the dirtied window must be re-run'
        assert out is not None, 'the grab waited out its own retry window and was refused'
        assert info.get('dark_saved') is True

    def test_healthy_capture_records_telemetry(self, live_scope):
        """The per-capture evidence a support bundle reads: present on
        every completed capture."""
        out = live_scope.imaging.capture_and_wait(timeout_s=1.0)
        assert out is not None
        info = live_scope.imaging.last_capture_info
        for key in ('hold_ms', 'drained', 'rechecks', 'deadline_s', 'n_entry', 'active_s'):
            assert key in info, f'missing capture-evidence key {key!r}'
