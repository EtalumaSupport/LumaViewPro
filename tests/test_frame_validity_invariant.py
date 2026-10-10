"""Regression: the frame-validity invariant -- any camera/illumination/
motion state change must turn the validity marker RED, and the marker
must not be suppressed by a stale software cache.

Bug shape (gain axis): a pre-scan live-mode auto-gain cycle drove
hardware gain to ~10.8 dB while the API-layer ``_camera_cache`` still
held the per-step ~0 dB. ``set_gain_db(0)`` hit a cache-equality
short-circuit that returned BEFORE ``frame_validity.invalidate('gain')``,
so the marker never went RED and a stale-gain (saturated) frame was
captured as valid. The cache-equality skip is removed: the setter
always invalidates, and redundant-SDK avoidance is left to the driver,
which compares against live hardware (cannot desync).

Each class below targets one concern of the consolidated fix. Tests
fail before the fix and pass after.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.exceptions import CameraSettingRejected
from modules.lumascope_api.imaging import ImagingAPI
from modules.notification_center import Severity
from tests.scope_fakes import bind_settings_like_a_session, build_scope

_MOTION_SOURCES = ('xy_move', 'z_move', 'turret')


@pytest.fixture
def sim_imaging():
    """A simulated scope's ImagingAPI and its SimulatedCamera, streaming.

    Bound to the template's settings as a session binds its scope. The
    stream never credits a pending source (frames are counted only by a
    grab), so a setter's invalidation is still pending at its assertion.
    The scope is disconnected by the test's teardown.
    """
    scope = build_scope(simulate=True)
    bind_settings_like_a_session(scope)
    scope.imaging.start_streaming()
    yield scope.imaging, scope._camera_driver
    scope.imaging.stop_streaming()


class TestSetGainAlwaysInvalidatesDespiteStaleCache:
    """A desynced cache must NOT suppress the hardware write or the
    invalidate -- the heart of the brightfield-saturation bug."""

    def test_desynced_cache_still_drives_hardware(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_gain_db(0.0)  # cache = 0, hw = 0
        # Hardware drifts (a live-mode auto-gain cycle drove it) without
        # the cache being updated -- the desynced precondition. A write to the
        # simulated camera's private register is the fault injection: no
        # public member moves the hardware behind the API's cache.
        with cam._lock:
            cam._gain = 10.8
        imaging.set_gain_db(0.0)  # request the per-step intended gain
        assert cam.get_gain() == pytest.approx(0.0, abs=0.001), (
            'set_gain_db must drive hardware to the requested value even '
            f'when the cache already reads it; got {cam.get_gain()}'
        )

    def test_desynced_cache_still_turns_marker_red(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_gain_db(0.0)
        imaging.frame_validity.reset()  # GREEN baseline
        assert imaging.frame_validity.is_valid
        # Fault injection, as above: hardware drifted behind the cache.
        with cam._lock:
            cam._gain = 10.8
        imaging.set_gain_db(0.0)
        assert not imaging.frame_validity.is_valid, (
            'set_gain_db must invalidate frame validity (marker RED) even '
            'when the cache already reads the requested value'
        )
        assert 'gain' in imaging.frame_validity.pending_sources


class TestSetExposureAlwaysInvalidatesDespiteStaleCache:
    """Symmetric exposure-axis sibling (the #679 axis)."""

    def test_desynced_cache_still_drives_hardware(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_exposure_ms(0.1)  # cache = 0.1, hw = 0.1
        # Fault injection: hardware drifted behind the cache (no public form).
        with cam._lock:
            cam._exposure_us = 14.0  # hw drifted to 0.014 ms
        imaging.set_exposure_ms(0.1)
        assert cam.get_exposure_t() == pytest.approx(0.1, abs=0.001), (
            'set_exposure_ms must drive hardware even when the cache '
            f'already reads the requested value; got {cam.get_exposure_t()}'
        )

    def test_desynced_cache_still_turns_marker_red(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_exposure_ms(0.1)
        imaging.frame_validity.reset()
        assert imaging.frame_validity.is_valid
        # Fault injection, as above.
        with cam._lock:
            cam._exposure_us = 14.0
        imaging.set_exposure_ms(0.1)
        assert not imaging.frame_validity.is_valid, (
            'set_exposure_ms must invalidate frame validity even when '
            'the cache already reads the requested value'
        )
        assert 'exposure' in imaging.frame_validity.pending_sources


class TestAutoGainOnceInvalidates:
    """One-shot auto-gain mutates gain AND exposure on the camera but
    historically called no invalidate at all -- a capture right after
    could grab before the converged values flushed the pipeline. The SDK
    chooses the converged values, so the manual chunk targets are dropped."""

    def test_auto_gain_once_turns_marker_red(self, sim_imaging):
        imaging, _cam = sim_imaging
        fv = imaging.frame_validity
        imaging.set_gain_db(5.0)
        imaging.set_exposure_ms(10.0)
        assert fv.target('gain') is not None and fv.target('exposure') is not None
        before = fv.invalidation_counts
        imaging.auto_gain_once(
            state=True,
            target_brightness=0.5,
            min_gain_db=0.0,
            max_gain_db=24.0,
        )
        after = fv.invalidation_counts
        assert (
            after['gain'] == before['gain'] + 1 and after['exposure'] == before['exposure'] + 1
        ), (
            'auto_gain_once must invalidate frame validity; it changes '
            f'gain and exposure on the camera. before {before}, after {after}'
        )
        assert {s: c for s, c in after.items() if s not in ('gain', 'exposure')} == {
            s: c for s, c in before.items() if s not in ('gain', 'exposure')
        }
        pending = fv.pending_sources
        assert 'gain' in pending and 'exposure' in pending
        assert fv.target('gain') is None and fv.target('exposure') is None


class TestMotionValiditySources:
    """A move turns readiness red for the source of the axis it moves --
    X and Y 'xy_move', Z 'z_move', the turret 'turret' -- and for no other
    axis's source, so the settle check gates on the axis that moved reaching
    IDLE. The old 2-way ternary mis-routed turret moves to 'xy_move', so the
    settle check cleared before the turret physically arrived.

    One homed simulated LS850T for the module (``sim_turreted_session``);
    each test places its axis first and reads the invalidation history
    across its own move only.
    """

    @staticmethod
    def _growth(fv, before):
        after = fv.invalidation_counts
        return {s: after.get(s, 0) - before.get(s, 0) for s in _MOTION_SOURCES}

    @staticmethod
    def _middle_of_travel(motion, axis):
        limits = motion.get_axis_limits(axis)
        return (limits['min'] + limits['max']) / 2

    @pytest.mark.parametrize(
        ('axis', 'source'),
        [('X', 'xy_move'), ('Y', 'xy_move'), ('Z', 'z_move')],
    )
    def test_move_absolute_invalidates_axis_source(self, sim_turreted_session, axis, source):
        motion = sim_turreted_session.scope.motion
        fv = sim_turreted_session.scope.imaging.frame_validity
        target = self._middle_of_travel(motion, axis)
        motion.move_absolute(axis, target - 500.0)
        before = fv.invalidation_counts

        motion.move_absolute(axis, target)

        growth = self._growth(fv, before)
        assert growth[source] > 0, f'move_absolute({axis!r}) must invalidate {source!r}: {growth}'
        others = {s: n for s, n in growth.items() if s != source}
        assert not any(others.values()), (
            f'move_absolute({axis!r}) invalidated another axis source: {growth}'
        )

    @pytest.mark.parametrize(
        ('axis', 'source'),
        [('X', 'xy_move'), ('Y', 'xy_move'), ('Z', 'z_move')],
    )
    def test_move_relative_invalidates_axis_source(self, sim_turreted_session, axis, source):
        motion = sim_turreted_session.scope.motion
        fv = sim_turreted_session.scope.imaging.frame_validity
        motion.move_absolute(axis, self._middle_of_travel(motion, axis))
        before = fv.invalidation_counts

        motion.move_relative(axis, 100.0)

        growth = self._growth(fv, before)
        assert growth[source] > 0, f'move_relative({axis!r}) must invalidate {source!r}: {growth}'
        others = {s: n for s, n in growth.items() if s != source}
        assert not any(others.values()), (
            f'move_relative({axis!r}) invalidated another axis source: {growth}'
        )

    def test_move_turret_invalidates_turret(self, sim_turreted_session):
        """The turret's move records 'turret', never 'xy_move'. Its Z park
        and restore are Z moves of their own and record 'z_move'."""
        motion = sim_turreted_session.scope.motion
        fv = sim_turreted_session.scope.imaging.frame_validity
        slot = 2 if motion.get_turret_slot() != 2 else 1
        before = fv.invalidation_counts

        motion.move_turret(slot)

        growth = self._growth(fv, before)
        assert motion.get_turret_slot() == slot
        assert growth['turret'] > 0, f"move_turret must invalidate 'turret': {growth}"
        assert growth['xy_move'] == 0, f"move_turret invalidated 'xy_move': {growth}"


class TestGeometryFormatInvalidates:
    """Pixel-format, frame-size, and binning changes realloc the camera
    buffer / restart the grab engine; each must turn the marker RED so a
    capture waits for the old geometry to flush."""

    def test_set_frame_size_turns_marker_red(self, sim_imaging):
        imaging, _cam = sim_imaging
        imaging.frame_validity.reset()
        imaging.set_frame_size(640, 480)
        assert 'frame_size' in imaging.frame_validity.pending_sources

    def test_set_pixel_format_turns_marker_red(self, sim_imaging):
        imaging, _cam = sim_imaging
        imaging.frame_validity.reset()
        imaging.set_pixel_format('Mono12')
        assert imaging.pixel_format_cached == 'Mono12'
        assert 'pixel_format' in imaging.frame_validity.pending_sources

    def test_set_binning_size_turns_marker_red(self, sim_imaging):
        imaging, _cam = sim_imaging
        imaging.frame_validity.reset()
        imaging.set_binning_size(2)
        assert imaging.get_binning_size() == 2
        assert 'binning' in imaging.frame_validity.pending_sources


class TestSaturationGuard:
    """The save-path saturation check must catch a near-fully-saturated
    (blown) frame and surface it, instead of only catching the all-pixels-
    exactly-max case and then accepting it silently."""

    def test_saturated_fraction_math(self):
        full8 = np.full((4, 4), 255, dtype=np.uint8)
        empty8 = np.zeros((4, 4), dtype=np.uint8)
        assert ImagingAPI.saturated_fraction(full8, 8) == pytest.approx(1.0)
        assert ImagingAPI.saturated_fraction(empty8, 8) == pytest.approx(0.0)
        # A 16-bit frame just below full scale still reads as saturated
        # (the near-max threshold, not exact-max).
        near16 = np.full((2, 2), int(65535 * 0.995), dtype=np.uint16)
        assert ImagingAPI.saturated_fraction(near16, 16) == pytest.approx(1.0)
        # Full scale follows the frame's PAYLOAD depth, not the container
        # dtype: a blown 12-bit frame (4095 in a uint16 container) reads
        # saturated at depth 12; against the container max it would
        # misread as 0% and slip past the evidence check.
        blown12 = np.full((2, 2), 4095, dtype=np.uint16)
        assert ImagingAPI.saturated_fraction(blown12, 12) == pytest.approx(1.0)
        assert ImagingAPI.saturated_fraction(blown12, 16) == pytest.approx(0.0)
        assert ImagingAPI.saturated_fraction(None, 8) == 0.0

    def test_blown_frame_logged_not_silent(self, sim_imaging, monkeypatch):
        # A blown frame that stays blown on retry must be logged as a
        # warning (visible in the post-mortem), not silently accepted. No
        # user notification -- a blown image is self-evident on screen.
        from modules.lumascope_api import imaging as imaging_mod

        imaging, cam = sim_imaging
        blown = np.full((4, 4), 255, dtype=np.uint8)
        frames = [blown, blown]  # blown on first grab AND on the retry grab
        monkeypatch.setattr(cam, 'get_array', lambda: frames.pop(0))
        warnings = []
        monkeypatch.setattr(
            imaging_mod.logger, 'warning', lambda msg, *a, **k: warnings.append(msg)
        )

        out = imaging.get_image(all_ones_check=True)

        assert not frames, 'a blown first frame must trigger exactly one retry grab'
        assert any('saturated' in w for w in warnings), (
            'a persistently blown capture must be logged as a warning, not silently accepted'
        )
        assert np.array_equal(out, blown)


class TestRejectedSettingRaisesAndKeepsCache:
    """A driver that CONFIRMS a settings-write rejection (returns False;
    drivers with no confirmation signal return None) must reach the caller
    as a raise -- which carries the words its reporter shows, so the API
    posts nothing itself -- and the requested value must NOT be recorded in
    the camera cache as if it took. IDS has no chunk data, so without this
    the camera streams at the old setting while the cache claims the
    new one -- the silent stale-settings shape."""

    def test_rejected_gain_raises_and_keeps_cache(self, sim_imaging, monkeypatch, centre_posts):
        imaging, cam = sim_imaging
        imaging.set_gain_db(2.0)  # establish a known cache value
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        # The public setter raises the rejection to its caller; what this test
        # pins is what happens on the way out -- the API posts nothing, and the
        # cache keeps the value the camera actually holds.
        with pytest.raises(CameraSettingRejected):
            imaging.set_gain_db(7.0)

        captured = [n for n in centre_posts if n.severity == Severity.ERROR]
        assert not captured, 'A confirmed gain rejection is shown by its reporter, not the API'
        assert imaging.gain_db_cached == 2.0, (
            'A rejected gain write must not be recorded in the cache'
        )

    def test_rejected_exposure_raises_and_keeps_cache(self, sim_imaging, monkeypatch, centre_posts):
        imaging, cam = sim_imaging
        imaging.set_exposure_ms(20.0)  # establish a known cache value
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        # See the gain case above: the raise is the public setter's contract,
        # the silent API and the held cache are what this test pins.
        with pytest.raises(CameraSettingRejected):
            imaging.set_exposure_ms(50.0)

        captured = [n for n in centre_posts if n.severity == Severity.ERROR]
        assert not captured, 'A confirmed exposure rejection is shown by its reporter, not the API'
        assert imaging.exposure_ms_cached == 20.0, (
            'A rejected exposure write must not be recorded in the cache'
        )
