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

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI


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
