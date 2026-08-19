"""Regression: the frame-validity invariant -- any camera/illumination/
motion state change must turn the validity marker RED, and the marker
must not be suppressed by a stale software cache.

Bug shape (gain axis): a pre-scan live-mode auto-gain cycle drove
hardware gain to ~10.8 dB while the API-layer ``_camera_cache`` still
held the per-step ~0 dB. ``set_gain(0)`` hit a cache-equality
short-circuit that returned BEFORE ``frame_validity.invalidate('gain')``,
so the marker never went RED and a stale-gain (saturated) frame was
captured as valid. The cache-equality skip is removed: the setter
always invalidates, and redundant-SDK avoidance is left to the driver,
which compares against live hardware (cannot desync).

Each class below targets one concern of the consolidated fix. Tests
fail before the fix and pass after.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from modules.lumascope_api.motion import MotionAPI


@pytest.fixture
def sim_imaging():
    """SimulatedCamera-backed ImagingAPI with a minimal scope stub
    holding the locks the setters acquire."""
    cam = SimulatedCamera()
    cam.active = True
    cam.open_and_start()
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging, cam


class TestSetGainAlwaysInvalidatesDespiteStaleCache:
    """A desynced cache must NOT suppress the hardware write or the
    invalidate -- the heart of the brightfield-saturation bug."""

    def test_desynced_cache_still_drives_hardware(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_gain(0.0)  # cache = 0, hw = 0
        # Hardware drifts (a live-mode auto-gain cycle drove it) without
        # the cache being updated -- the desynced precondition.
        with cam._lock:
            cam._gain = 10.8
        imaging.set_gain(0.0)  # request the per-step intended gain
        assert cam.get_gain() == pytest.approx(0.0, abs=0.001), (
            'set_gain must drive hardware to the requested value even '
            f'when the cache already reads it; got {cam.get_gain()}'
        )

    def test_desynced_cache_still_turns_marker_red(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_gain(0.0)
        imaging.frame_validity.reset()  # GREEN baseline
        assert imaging.frame_validity.is_valid
        with cam._lock:
            cam._gain = 10.8
        imaging.set_gain(0.0)
        assert not imaging.frame_validity.is_valid, (
            'set_gain must invalidate frame validity (marker RED) even '
            'when the cache already reads the requested value'
        )
        assert 'gain' in imaging.frame_validity.pending_sources


class TestSetExposureAlwaysInvalidatesDespiteStaleCache:
    """Symmetric exposure-axis sibling (the #679 axis)."""

    def test_desynced_cache_still_drives_hardware(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_exposure_time(0.1)  # cache = 0.1, hw = 0.1
        with cam._lock:
            cam._exposure_us = 14.0  # hw drifted to 0.014 ms
        imaging.set_exposure_time(0.1)
        assert cam.get_exposure_t() == pytest.approx(0.1, abs=0.001), (
            'set_exposure_time must drive hardware even when the cache '
            f'already reads the requested value; got {cam.get_exposure_t()}'
        )

    def test_desynced_cache_still_turns_marker_red(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_exposure_time(0.1)
        imaging.frame_validity.reset()
        assert imaging.frame_validity.is_valid
        with cam._lock:
            cam._exposure_us = 14.0
        imaging.set_exposure_time(0.1)
        assert not imaging.frame_validity.is_valid, (
            'set_exposure_time must invalidate frame validity even when '
            'the cache already reads the requested value'
        )
        assert 'exposure' in imaging.frame_validity.pending_sources


class TestAutoGainOnceInvalidates:
    """One-shot auto-gain mutates gain AND exposure on the camera but
    historically called no invalidate at all -- a capture right after
    could grab before the converged values flushed the pipeline."""

    def test_auto_gain_once_turns_marker_red(self, sim_imaging):
        imaging, _cam = sim_imaging
        imaging.set_gain(5.0)
        imaging.frame_validity.reset()
        assert imaging.frame_validity.is_valid
        imaging.auto_gain_once(
            state=True,
            target_brightness=0.5,
            min_gain_db=0.0,
            max_gain_db=24.0,
        )
        assert not imaging.frame_validity.is_valid, (
            'auto_gain_once must invalidate frame validity; it changes '
            'gain and exposure on the camera'
        )
        pending = imaging.frame_validity.pending_sources
        assert 'gain' in pending and 'exposure' in pending


class TestMotionValiditySources:
    """Turret moves must record the 'turret' source, so the settle-check
    gates on the turret reaching IDLE rather than on X/Y."""

    def test_turret_axis_maps_to_turret_source(self):
        assert MotionAPI._AXIS_VALIDITY_SOURCE.get('T', 'xy_move') == 'turret'

    def test_z_axis_maps_to_z_move(self):
        assert MotionAPI._AXIS_VALIDITY_SOURCE.get('Z', 'xy_move') == 'z_move'

    def test_xy_axes_default_to_xy_move(self):
        assert MotionAPI._AXIS_VALIDITY_SOURCE.get('X', 'xy_move') == 'xy_move'
        assert MotionAPI._AXIS_VALIDITY_SOURCE.get('Y', 'xy_move') == 'xy_move'

    @staticmethod
    def _scope_with_invalidate_recorder():
        """Simulated scope whose frame_validity.invalidate records sources."""
        scope = Lumascope(simulate=True)
        scope._motion_driver.set_timing_mode('instant')
        recorded = []
        orig_invalidate = scope.imaging.frame_validity.invalidate

        def recording_invalidate(source):
            recorded.append(source)
            return orig_invalidate(source)

        scope.imaging.frame_validity.invalidate = recording_invalidate
        return scope, recorded

    @pytest.mark.parametrize(
        ('axis', 'pos', 'source'),
        [
            ('X', 1000.0, 'xy_move'),
            ('Y', 1000.0, 'xy_move'),
            ('Z', 1000.0, 'z_move'),
            ('T', 1, 'turret'),
        ],
    )
    def test_move_absolute_invalidates_axis_source(self, axis, pos, source):
        """A move on each axis must record THAT axis's validity source --
        the old 2-way ternary mis-routed turret moves to 'xy_move', so
        the settle-check cleared before the turret physically arrived."""
        scope, recorded = self._scope_with_invalidate_recorder()
        try:
            scope.motion.move_absolute_position(axis, pos, wait_until_complete=True)
            assert source in recorded, (
                f'move_absolute_position({axis!r}) must invalidate {source!r}; recorded {recorded}'
            )
            wrong = {'xy_move', 'z_move', 'turret'} - {source}
            assert not wrong.intersection(recorded), (
                f'move_absolute_position({axis!r}) invalidated the wrong '
                f'source(s) {wrong.intersection(recorded)}; recorded {recorded}'
            )
        finally:
            scope.disconnect()

    @pytest.mark.parametrize(
        ('axis', 'source'),
        [('X', 'xy_move'), ('Y', 'xy_move'), ('Z', 'z_move')],
    )
    def test_move_relative_invalidates_axis_source(self, axis, source):
        scope, recorded = self._scope_with_invalidate_recorder()
        try:
            scope.motion.move_relative_position(axis, 100.0, wait_until_complete=True)
            assert source in recorded, (
                f'move_relative_position({axis!r}) must invalidate {source!r}; recorded {recorded}'
            )
        finally:
            scope.disconnect()


class TestGeometryFormatInvalidates:
    """Pixel-format, frame-size, and binning changes realloc the camera
    buffer / restart the grab engine; each must turn the marker RED so a
    capture waits for the old geometry to flush."""

    def test_geometry_sources_registered(self):
        from modules.frame_validity import FrameValidity

        for source in ('pixel_format', 'frame_size', 'binning'):
            assert source in FrameValidity.SKIP_FRAMES, (
                f'{source} must be a known frame-validity source'
            )

    def test_set_frame_size_turns_marker_red(self, sim_imaging):
        imaging, _cam = sim_imaging
        imaging.frame_validity.reset()
        imaging.set_frame_size(640, 480)
        assert 'frame_size' in imaging.frame_validity.pending_sources

    def test_set_pixel_format_turns_marker_red(self, sim_imaging):
        imaging, _cam = sim_imaging
        imaging.frame_validity.reset()
        assert imaging.set_pixel_format('Mono12') is True
        assert 'pixel_format' in imaging.frame_validity.pending_sources

    def test_set_binning_size_turns_marker_red(self, sim_imaging):
        imaging, _cam = sim_imaging
        imaging.frame_validity.reset()
        assert imaging.set_binning_size(2) is True
        assert 'binning' in imaging.frame_validity.pending_sources


class TestSaturationGuard:
    """The save-path saturation check must catch a near-fully-saturated
    (blown) frame and surface it, instead of only catching the all-pixels-
    exactly-max case and then accepting it silently."""

    def test_saturated_fraction_math(self):
        full8 = np.full((4, 4), 255, dtype=np.uint8)
        empty8 = np.zeros((4, 4), dtype=np.uint8)
        assert ImagingAPI._saturated_fraction(full8, 8) == pytest.approx(1.0)
        assert ImagingAPI._saturated_fraction(empty8, 8) == pytest.approx(0.0)
        # A 16-bit frame just below full scale still reads as saturated
        # (the near-max threshold, not exact-max).
        near16 = np.full((2, 2), int(65535 * 0.995), dtype=np.uint16)
        assert ImagingAPI._saturated_fraction(near16, 16) == pytest.approx(1.0)
        # Full scale follows the frame's PAYLOAD depth, not the container
        # dtype: a blown 12-bit frame (4095 in a uint16 container) reads
        # saturated at depth 12; against the container max it would
        # misread as 0% and slip past the evidence check.
        blown12 = np.full((2, 2), 4095, dtype=np.uint16)
        assert ImagingAPI._saturated_fraction(blown12, 12) == pytest.approx(1.0)
        assert ImagingAPI._saturated_fraction(blown12, 16) == pytest.approx(0.0)
        assert ImagingAPI._saturated_fraction(None, 8) == 0.0

    def test_blown_frame_logged_not_silent(self, sim_imaging, monkeypatch):
        # A blown frame that stays blown on retry must be logged as a
        # warning (visible in the post-mortem), not silently accepted. No
        # user notification -- a blown image is self-evident on screen.
        from modules.lumascope_api import imaging as imaging_mod
        from modules.lumascope_api.runtime_state import RuntimeState

        imaging, cam = sim_imaging
        # get_image's scale-bar gate reads scope.runtime_state past the
        # saturation block; the lean fixture omits it.
        imaging._scope.runtime_state = RuntimeState(imaging._scope)
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


class TestRejectedSettingNotifiesAndKeepsCache:
    """A driver that CONFIRMS a settings-write rejection (returns False;
    drivers with no confirmation signal return None) must produce a user
    notification, and the requested value must NOT be recorded in the
    camera cache as if it took. IDS has no chunk data, so without this
    the camera streams at the old setting while the cache claims the
    new one -- the silent stale-settings shape."""

    def test_rejected_gain_notifies_and_keeps_cache(self, sim_imaging, monkeypatch):
        imaging, cam = sim_imaging
        captured = []
        monkeypatch.setattr(
            'modules.lumascope_api.imaging.notifications.error',
            lambda *a, **kw: captured.append(a),
        )
        imaging.set_gain(2.0)  # establish a known cache value
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        imaging.set_gain(7.0)

        assert captured, 'A confirmed gain rejection must notify the user'
        assert imaging.camera_gain == 2.0, 'A rejected gain write must not be recorded in the cache'

    def test_rejected_exposure_notifies_and_keeps_cache(self, sim_imaging, monkeypatch):
        imaging, cam = sim_imaging
        captured = []
        monkeypatch.setattr(
            'modules.lumascope_api.imaging.notifications.error',
            lambda *a, **kw: captured.append(a),
        )
        imaging.set_exposure_time(20.0)  # establish a known cache value
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        imaging.set_exposure_time(50.0)

        assert captured, 'A confirmed exposure rejection must notify the user'
        assert imaging.camera_exposure_ms == 20.0, (
            'A rejected exposure write must not be recorded in the cache'
        )
