"""Invariant net for the camera-write authority consolidation.

Each camera-state setter on ImagingAPI emits a precise, ordered sequence of
frame-validity operations (invalidate + set_target) plus a cache-snapshot
update. An upcoming refactor routes every setter through one ``_camera_write``
authority so a future setter cannot forget to invalidate. That refactor must be
behavior-preserving: the emitted sequence below is the contract it may not
change.

These tests run the real setters against a SimulatedCamera (the production
driver path) and record the exact validity-op sequence via a spy on the live
FrameValidity instance. They pass on the pre-refactor code and must stay green
through every migration commit. If a migration alters any sequence, the
matching test fails -- that is the regression catch.

The two SDK-perf setters (conversion gain mode, line noise reduction) are
Pylon-only; SimulatedCamera does not implement them, so a capable subclass adds
them here to pin their success path, and the plain sim pins the
driver-lacks-method path.
"""

from __future__ import annotations

import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI


class _CamWriteCapableSim(SimulatedCamera):
    """SimulatedCamera plus the two Pylon-only SDK setters, so the success
    path of set_conversion_gain_mode / set_line_noise_reduction (driver
    implements the method and returns True) is exercisable in the sim."""

    def __init__(self):
        super().__init__()
        self._conversion_gain_mode = 'Low'
        self._line_noise_reduction = False

    def set_conversion_gain_mode(self, mode: str) -> bool:
        self._conversion_gain_mode = mode
        return True

    def set_line_noise_reduction(self, enabled: bool) -> bool:
        self._line_noise_reduction = enabled
        return True


def _build_imaging(cam):
    cam.active = True
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging


@pytest.fixture
def imaging_capable():
    """ImagingAPI on a sim that implements every camera setter."""
    return _build_imaging(_CamWriteCapableSim())


@pytest.fixture
def imaging_plain():
    """ImagingAPI on a stock SimulatedCamera (no conversion-gain / line-noise)."""
    return _build_imaging(SimulatedCamera())


def _record_validity_events(imaging):
    """Patch invalidate + set_target on imaging.frame_validity to append an
    ordered event log, then return the (still-live) log list. Each event is
    ('invalidate', source) or ('set_target', source, value); the real method
    still runs so downstream state stays correct.
    """
    events = []
    fv = imaging.frame_validity
    orig_invalidate = fv.invalidate
    orig_set_target = fv.set_target

    def recording_invalidate(source):
        events.append(('invalidate', source))
        return orig_invalidate(source)

    def recording_set_target(source, value):
        events.append(('set_target', source, value))
        return orig_set_target(source, value)

    fv.invalidate = recording_invalidate
    fv.set_target = recording_set_target
    return events


class TestValueSetterSequences:
    """Manual value setters: invalidate the source, then record the chunk
    target. Both fire on every successful write (never gated by the cache)."""

    def test_set_gain_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_gain(7.0)
        assert events == [
            ('invalidate', 'gain'),
            ('set_target', 'gain', 7.0),
        ]
        assert imaging_capable.camera_gain == pytest.approx(7.0)

    def test_set_exposure_time_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_exposure_time(0.1)
        # Target is recorded in microseconds (chunk-match unit); API takes ms.
        assert events == [
            ('invalidate', 'exposure'),
            ('set_target', 'exposure', 100.0),
        ]
        assert imaging_capable.camera_exposure_ms == pytest.approx(0.1)


class TestAutoSetterSequences:
    """Auto/mode setters flip a mode node while the value node is unchanged.
    Their invalidations are unconditional (force) -- the auto_gain settle
    window arms only because invalidate('auto_gain') fires here."""

    def test_set_auto_gain_enable_arms_settle_window(self, imaging_capable):
        cam = imaging_capable._driver
        expected = [('invalidate', 'gain'), ('set_target', 'gain', None)]
        if getattr(cam.profile, 'has_auto_gain', False):
            expected.append(('invalidate', 'auto_gain'))
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_auto_gain(
            True,
            {
                'target_brightness': 0.5,
                'min_gain_db': 0.0,
                'max_gain_db': 24.0,
                'max_exposure_ms': 100.0,
            },
        )
        assert events == expected

    def test_set_auto_gain_disable_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_auto_gain(
            False,
            {
                'target_brightness': 0.5,
                'min_gain_db': 0.0,
                'max_gain_db': 24.0,
                'max_exposure_ms': 100.0,
            },
        )
        # Disable does not arm the auto_gain window; it clears the gain target.
        assert events == [('invalidate', 'gain'), ('set_target', 'gain', None)]

    def test_set_auto_exposure_enable_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_auto_exposure_time(True)
        assert events == [('invalidate', 'exposure'), ('set_target', 'exposure', None)]

    def test_set_auto_exposure_disable_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_auto_exposure_time(False)
        assert events == [('invalidate', 'exposure'), ('set_target', 'exposure', None)]

    def test_auto_gain_once_invalidates_both_sources(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.auto_gain_once(
            state=True,
            target_brightness=0.5,
            min_gain_db=0.0,
            max_gain_db=24.0,
            ae_max_exposure_ms=100.0,
        )
        assert events == [
            ('invalidate', 'gain'),
            ('invalidate', 'exposure'),
            ('set_target', 'gain', None),
            ('set_target', 'exposure', None),
        ]


class TestGeometrySetterSequences:
    """Geometry setters invalidate one source; pixel_format and frame_size
    also snapshot the cache."""

    def test_set_frame_size_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_frame_size(640, 480)
        assert events == [('invalidate', 'frame_size')]
        assert imaging_capable.camera_frame_size == {'width': 640, 'height': 480}

    def test_set_binning_size_success_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_binning_size(2)
        assert result is True
        assert events == [('invalidate', 'binning')]

    def test_set_pixel_format_success_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_pixel_format('Mono8')
        assert result is True
        assert events == [('invalidate', 'pixel_format')]
        assert imaging_capable.camera_pixel_format == 'Mono8'


class TestSdkPerfSetterSequences:
    """Pylon-only setters: invalidate only when the driver implements the
    method AND returns truthy; a driver lacking the method returns False with
    no invalidation."""

    def test_set_conversion_gain_mode_success_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_conversion_gain_mode('High')
        assert result is True
        assert events == [('invalidate', 'conversion_gain_mode')]

    def test_set_line_noise_reduction_success_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_line_noise_reduction(True)
        assert result is True
        assert events == [('invalidate', 'line_noise_reduction')]

    def test_conversion_gain_mode_no_method_no_invalidate(self, imaging_plain):
        events = _record_validity_events(imaging_plain)
        result = imaging_plain.set_conversion_gain_mode('High')
        assert result is False
        assert events == []

    def test_line_noise_reduction_no_method_no_invalidate(self, imaging_plain):
        events = _record_validity_events(imaging_plain)
        result = imaging_plain.set_line_noise_reduction(True)
        assert result is False
        assert events == []


class TestCameraWriteAuthority:
    """The _camera_write authority in isolation: force vs applied-gated
    invalidation, target + cache maintenance, result-gating, and order."""

    def test_force_invalidate_fires_even_on_rejection(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable._camera_write(
            lambda: False,
            force_invalidate=('gain',),
            targets=(('gain', 5.0),),
        )
        # Rejection (False): force_invalidate still fires; the applied-only
        # target is suppressed.
        assert result is False
        assert events == [('invalidate', 'gain')]

    def test_applied_block_runs_when_not_rejected(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable._camera_write(
            lambda: None,
            force_invalidate=('gain',),
            targets=(('gain', 5.0),),
            cache_update={'gain_db': 5.0},
        )
        # None result counts as applied: force invalidate, then target + cache.
        assert events == [('invalidate', 'gain'), ('set_target', 'gain', 5.0)]
        assert imaging_capable.camera_gain == pytest.approx(5.0)

    def test_gate_on_result_skips_invalidate_when_falsey(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable._camera_write(
            lambda: False,
            invalidates=('binning',),
            gate_on_result=True,
        )
        assert result is False
        assert events == []

    def test_gate_on_result_invalidates_when_truthy(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable._camera_write(
            lambda: True,
            invalidates=('binning',),
            gate_on_result=True,
        )
        assert result is True
        assert events == [('invalidate', 'binning')]

    def test_force_precedes_applied_invalidate_in_order(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable._camera_write(
            lambda: None,
            force_invalidate=('gain',),
            invalidates=('auto_gain',),
        )
        assert events == [('invalidate', 'gain'), ('invalidate', 'auto_gain')]

    def test_multiple_sources_and_targets(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable._camera_write(
            lambda: None,
            force_invalidate=('gain', 'exposure'),
            targets=(('gain', None), ('exposure', None)),
        )
        assert events == [
            ('invalidate', 'gain'),
            ('invalidate', 'exposure'),
            ('set_target', 'gain', None),
            ('set_target', 'exposure', None),
        ]
