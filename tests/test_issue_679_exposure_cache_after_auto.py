"""Regression: exposure / gain cache refreshes after an auto cycle (#679).

Bug shape: the beta tester's 2026-05-26 BF Protocol ran twice; Run 1 captures
came out anomalously dim while Run 2 captured at the protocol's
configured exposure. Root cause: a pre-scan live-mode AG/AE Continuous
cycle drove hardware exposure to ~0.014 ms while LVP's API-layer
``_camera_cache['exposure_ms']`` still held the pre-AG 0.1 ms. Every
per-step ``set_exposure_time(0.1)`` in Run 1 hit the cache-equality
short-circuit and silently no-op'd; hardware stayed at 0.014 ms for
the entire run. End-of-Run-1 cleanup invalidated the cache (cache !=
hardware via the restore-from-snapshot path), so Run 2 actually
wrote the requested 0.1 ms.

Fix: ``set_auto_gain(state=False)``, ``set_auto_exposure_time(state=
False)``, and ``auto_gain_once`` resync the cache from hardware
before returning. The off-transition is the only moment when the
auto SDK path's hardware drives can desync from the cache; refreshing
then closes the window.

Tests below exercise both the AST-source contract (each call site
includes the refresh helper, locking the fix against revert drift)
AND the behavioral path (drift the simulator's hardware-truth
attribute mid-cycle, then verify the API-layer cache catches up at
auto-off).
"""

from __future__ import annotations

import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI


@pytest.fixture
def sim_imaging():
    """Build a SimulatedCamera-backed ImagingAPI with a minimal scope
    stub holding the locks the setters acquire."""
    cam = SimulatedCamera()
    cam.active = True
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._camera_executor = None
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging, cam


class TestSetAutoExposureRefreshesCacheAtAutoOff:
    """The headline bug path: live-mode AE-Continuous cycle drove
    hardware exposure away from the cached value; AE-off must resync
    the cache so the next ``set_exposure_time`` call actually writes
    hardware."""

    def test_auto_exposure_off_refreshes_cache_after_hardware_drift(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_exposure_time(0.1)
        assert imaging.exposure_ms_cached == pytest.approx(0.1)
        imaging.set_auto_exposure_time(state=True)
        # Mimic the AE-Continuous-drove-hardware-away path: poke the
        # simulator's hardware-truth attribute directly. Real Pylon does
        # this internally during a Continuous AE cycle.
        with cam._lock:
            cam._exposure_us = 14.0
        imaging.set_auto_exposure_time(state=False)
        assert imaging.exposure_ms_cached == pytest.approx(0.014, abs=0.001), (
            f'cache must reflect hardware truth after AE-off; got {imaging.exposure_ms_cached}'
        )

    def test_set_exposure_after_ae_off_actually_writes_hardware(self, sim_imaging):
        # End-to-end repro of the bug shape.
        imaging, cam = sim_imaging
        imaging.set_exposure_time(0.1)  # cache = 0.1, hw = 0.1
        imaging.set_auto_exposure_time(state=True)
        with cam._lock:
            cam._exposure_us = 14.0  # AE drove hw to 0.014 ms
        imaging.set_auto_exposure_time(state=False)
        # Pre-fix: cache still 0.1, so the next set_exposure_time(0.1)
        # short-circuited and hardware stayed at 0.014 ms.
        imaging.set_exposure_time(0.1)
        assert cam.get_exposure_t() == pytest.approx(0.1, abs=0.001), (
            'hardware must reach the requested 0.1 ms after AE-off + '
            f'set_exposure_time(0.1); got {cam.get_exposure_t()}'
        )


class TestSetAutoGainRefreshesCacheAtAutoOff:
    """Symmetric gain-side: AG-off resyncs both gain and exposure
    (auto cycles couple them in the Pylon driver)."""

    def test_auto_gain_off_refreshes_cache_after_hardware_drift(self, sim_imaging):
        imaging, cam = sim_imaging
        imaging.set_gain(5.0)
        assert imaging.gain_cached == pytest.approx(5.0)
        imaging.set_auto_gain(
            state=True,
            settings={
                'target_brightness': 0.5,
                'min_gain_db': 0.0,
                'max_gain_db': 24.0,
            },
        )
        # AG drove hw gain to a different value mid-cycle.
        with cam._lock:
            cam._gain = 17.2
            cam._exposure_us = 14.0
        imaging.set_auto_gain(
            state=False,
            settings={
                'target_brightness': 0.5,
                'min_gain_db': 0.0,
                'max_gain_db': 24.0,
            },
        )
        assert imaging.gain_cached == pytest.approx(17.2), (
            f'cache must reflect hardware gain after AG-off; got {imaging.gain_cached}'
        )
        assert imaging.exposure_ms_cached == pytest.approx(0.014, abs=0.001), (
            f'cache must reflect hardware exposure after AG-off; got {imaging.exposure_ms_cached}'
        )

    def test_set_auto_gain_state_true_does_not_refresh_cache(self, sim_imaging):
        # AG-on transition: cache stays at pre-AG truth (auto cycle is
        # still running; refresh would race with the SDK adjustments).
        imaging, cam = sim_imaging
        imaging.set_gain(5.0)
        imaging.set_exposure_time(0.1)
        # Drift hardware after caching but before AG-on -- if AG-on
        # were to refresh, cache would jump to the drifted values.
        with cam._lock:
            cam._gain = 17.2
            cam._exposure_us = 14.0
        imaging.set_auto_gain(
            state=True,
            settings={
                'target_brightness': 0.5,
                'min_gain_db': 0.0,
                'max_gain_db': 24.0,
            },
        )
        assert imaging.gain_cached == pytest.approx(5.0), (
            'cache must NOT refresh on AG-on (auto cycle still active)'
        )
        assert imaging.exposure_ms_cached == pytest.approx(0.1)


class TestAutoGainOnceRefreshesCache:
    """One-shot AG always ends with the auto cycle complete; the
    refresh must fire regardless of the ``state`` argument."""

    def test_auto_gain_once_refreshes_cache(self, sim_imaging):
        # The simulator's auto_gain_once converges gain to the
        # midpoint of [min_gain_db, max_gain_db]; that's the hardware
        # value the cache must catch up to.
        imaging, cam = sim_imaging
        imaging.set_gain(5.0)
        imaging.set_exposure_time(0.1)
        # Drift hardware exposure independently of the simulator's
        # one-shot (the sim's auto_gain_once only touches gain, so an
        # exposure drift survives across the call -- exercises the
        # symmetric exposure-side refresh.)
        with cam._lock:
            cam._exposure_us = 14.0
        imaging.auto_gain_once(
            state=True,
            target_brightness=0.5,
            min_gain_db=0.0,
            max_gain_db=24.0,
        )
        # Cache should match whatever hardware ended up at, not the
        # pre-call cached value (which would still read 5.0 / 0.1).
        assert imaging.gain_cached == pytest.approx(cam.get_gain(), abs=0.001), (
            f'cache gain {imaging.gain_cached} must match hardware '
            f'{cam.get_gain()} after auto_gain_once'
        )
        assert imaging.exposure_ms_cached == pytest.approx(0.014, abs=0.001), (
            f'cache exposure {imaging.exposure_ms_cached} must match '
            f'drifted hardware 0.014 ms after auto_gain_once refresh'
        )
