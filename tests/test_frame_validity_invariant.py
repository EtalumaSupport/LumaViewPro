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

import re
import threading
from pathlib import Path

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
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging, cam


def _imaging_src() -> str:
    return (
        Path(__file__).resolve().parent.parent
        / 'modules'
        / 'lumascope_api'
        / 'imaging.py'
    ).read_text(encoding='utf-8')


def _motion_src() -> str:
    return (
        Path(__file__).resolve().parent.parent
        / 'modules'
        / 'lumascope_api'
        / 'motion.py'
    ).read_text(encoding='utf-8')


def _method_body(src: str, name: str) -> str:
    m = re.search(
        rf'def {name}.*?(?=\n    def |\n    @|\nclass |\Z)',
        src,
        re.DOTALL,
    )
    assert m is not None, f'could not find {name} body in imaging.py'
    return m.group(0)


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
        imaging, cam = sim_imaging
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

    def test_auto_gain_once_body_invalidates_gain_and_exposure(self):
        body = _method_body(_imaging_src(), 'auto_gain_once')
        assert "invalidate('gain')" in body, (
            'auto_gain_once must invalidate the gain source'
        )
        assert "invalidate('exposure')" in body, (
            'auto_gain_once must invalidate the exposure source'
        )


class TestMotionValiditySources:
    """Turret moves must record the 'turret' source (so the settle-check
    gates on the turret reaching IDLE, not X/Y); xycenter must invalidate
    at all."""

    def test_turret_axis_maps_to_turret_source(self):
        assert MotionAPI._AXIS_VALIDITY_SOURCE.get('T', 'xy_move') == 'turret'

    def test_z_axis_maps_to_z_move(self):
        assert MotionAPI._AXIS_VALIDITY_SOURCE.get('Z', 'xy_move') == 'z_move'

    def test_xy_axes_default_to_xy_move(self):
        assert MotionAPI._AXIS_VALIDITY_SOURCE.get('X', 'xy_move') == 'xy_move'
        assert MotionAPI._AXIS_VALIDITY_SOURCE.get('Y', 'xy_move') == 'xy_move'

    def test_move_sites_route_through_axis_mapping(self):
        src = _motion_src()
        # The old 2-way ternary mis-routed a turret move to 'xy_move'.
        assert "'z_move' if axis == 'Z' else 'xy_move'" not in src, (
            'move sites must use the axis->source mapping, not the 2-way '
            'ternary that mis-routed turret moves to xy_move'
        )
        for name in ('move_absolute_position', 'move_relative_position'):
            assert '_AXIS_VALIDITY_SOURCE' in _method_body(src, name), (
                f'{name} must invalidate via the axis->source mapping'
            )

    def test_xycenter_invalidates_xy_move(self):
        body = _method_body(_motion_src(), 'xycenter')
        assert "invalidate('xy_move')" in body, (
            'xycenter physically moves X/Y but recorded no validity source'
        )


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

    def test_set_pixel_format_body_invalidates(self):
        body = _method_body(_imaging_src(), 'set_pixel_format')
        assert "invalidate('pixel_format')" in body

    def test_set_binning_size_body_invalidates(self):
        body = _method_body(_imaging_src(), 'set_binning_size')
        assert "invalidate('binning')" in body


class TestSaturationGuard:
    """The save-path saturation check must catch a near-fully-saturated
    (blown) frame and surface it, instead of only catching the all-pixels-
    exactly-max case and then accepting it silently."""

    def test_saturated_fraction_math(self):
        full8 = np.full((4, 4), 255, dtype=np.uint8)
        empty8 = np.zeros((4, 4), dtype=np.uint8)
        assert ImagingAPI._saturated_fraction(full8) == pytest.approx(1.0)
        assert ImagingAPI._saturated_fraction(empty8) == pytest.approx(0.0)
        # A 12-bit-in-uint16 frame just below full scale still reads as
        # saturated (the near-max threshold, not exact-max).
        near16 = np.full((2, 2), int(65535 * 0.995), dtype=np.uint16)
        assert ImagingAPI._saturated_fraction(near16) == pytest.approx(1.0)
        assert ImagingAPI._saturated_fraction(None) == 0.0

    def test_blown_frame_logged_not_silent(self):
        body = _method_body(_imaging_src(), 'get_image')
        # A blown frame must be logged as a warning (visible in the
        # post-mortem), replacing the prior silent debug-accept. No user
        # notification -- a blown image is self-evident on screen.
        assert 'logger.warning' in body and 'saturated' in body, (
            'a blown/saturated capture must be logged as a warning'
        )
        assert 'saturated frame confirmed on retry' not in body
        assert '_saturated_fraction' in body


class TestNoCacheEqualitySkipInSetters:
    """Source contract: neither setter early-returns on a cache-equality
    check before invalidating. Locks the fix against revert drift. The
    only early return permitted is the driver-inactive guard (one
    ``return`` per setter body)."""

    def test_set_gain_has_single_early_return(self):
        body = _method_body(_imaging_src(), 'set_gain')
        returns = re.findall(r'^\s+return\b', body, re.M)
        assert len(returns) == 1, (
            'set_gain must have exactly one return (the driver-inactive '
            f'guard); a cache-equality skip would add a second. Found {len(returns)}.'
        )

    def test_set_exposure_time_has_single_early_return(self):
        body = _method_body(_imaging_src(), 'set_exposure_time')
        returns = re.findall(r'^\s+return\b', body, re.M)
        assert len(returns) == 1, (
            'set_exposure_time must have exactly one return (the '
            f'driver-inactive guard). Found {len(returns)}.'
        )
