"""Regression: the frame-validity invariant -- any camera/illumination/
motion state change must record its change with frame validity, and the
record must not be suppressed by a stale software cache.

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

from _lumascope.modules.exceptions import CameraSettingRejected
from _lumascope.modules.lumascope_api.imaging import ImagingAPI
from _lumascope.modules.notification_center import Severity
from tests.scope_fakes import bind_settings_like_a_session, build_scope
from tests.motion_moves import axis_move, axis_move_by
from tests.stock_channels import LED0

_MOTION_SOURCES = ('xy_move', 'z_move', 'turret')


def _recorded_once(fv, before, source) -> bool:
    """Whether ``source`` was invalidated exactly once more since the
    ``invalidation_counts`` snapshot ``before``: a change recorded."""
    return fv.invalidation_counts.get(source, 0) == before.get(source, 0) + 1


@pytest.fixture
def sim_camera():
    """A simulated scope's CameraAPI and its SimulatedCamera, streaming.

    Bound to the template's settings as a session binds its scope. The
    stream never credits a pending source (frames are counted only by a
    grab), so a setter's invalidation is still pending at its assertion.
    The scope is disconnected by the test's teardown.
    """
    scope = build_scope(simulate=True)
    bind_settings_like_a_session(scope)
    scope.imaging.enable_streaming()
    yield scope.camera, scope._camera_driver
    scope.imaging.disable_streaming()


class TestSetGainAlwaysInvalidatesDespiteStaleCache:
    """A desynced cache must NOT suppress the hardware write or the
    invalidate -- the heart of the brightfield-saturation bug."""

    def test_desynced_cache_still_drives_hardware(self, sim_camera):
        camera, cam = sim_camera
        camera.set_gain_db(0.0)  # cache = 0, hw = 0
        # Hardware drifts (a live-mode auto-gain cycle drove it) without
        # the cache being updated -- the desynced precondition. A write to the
        # simulated camera's private register is the fault injection: no
        # public member moves the hardware behind the API's cache.
        with cam._lock:
            cam._gain = 10.8
        camera.set_gain_db(0.0)  # request the per-step intended gain
        assert cam.get_gain() == pytest.approx(0.0, abs=0.001), (
            'set_gain_db must drive hardware to the requested value even '
            f'when the cache already reads it; got {cam.get_gain()}'
        )

    def test_desynced_cache_still_records_the_change(self, sim_camera):
        camera, cam = sim_camera
        camera.set_gain_db(0.0)
        before = camera._frame_validity.invalidation_counts
        # Fault injection, as above: hardware drifted behind the cache.
        with cam._lock:
            cam._gain = 10.8
        camera.set_gain_db(0.0)
        assert _recorded_once(camera._frame_validity, before, 'gain'), (
            'set_gain_db must record its change with frame validity even '
            'when the cache already reads the requested value'
        )


class TestSetExposureAlwaysInvalidatesDespiteStaleCache:
    """Symmetric exposure-axis sibling (the #679 axis)."""

    def test_desynced_cache_still_drives_hardware(self, sim_camera):
        camera, cam = sim_camera
        camera.set_exposure_ms(0.1)  # cache = 0.1, hw = 0.1
        # Fault injection: hardware drifted behind the cache (no public form).
        with cam._lock:
            cam._exposure_us = 14.0  # hw drifted to 0.014 ms
        camera.set_exposure_ms(0.1)
        assert cam.get_exposure_t() == pytest.approx(0.1, abs=0.001), (
            'set_exposure_ms must drive hardware even when the cache '
            f'already reads the requested value; got {cam.get_exposure_t()}'
        )

    def test_desynced_cache_still_records_the_change(self, sim_camera):
        camera, cam = sim_camera
        camera.set_exposure_ms(0.1)
        before = camera._frame_validity.invalidation_counts
        # Fault injection, as above.
        with cam._lock:
            cam._exposure_us = 14.0
        camera.set_exposure_ms(0.1)
        assert _recorded_once(camera._frame_validity, before, 'exposure'), (
            'set_exposure_ms must record its change with frame validity even '
            'when the cache already reads the requested value'
        )


class TestAutoGainOnceInvalidates:
    """One-shot auto-gain mutates gain AND exposure on the camera but
    historically called no invalidate at all -- a capture right after
    could grab before the converged values flushed the pipeline. The SDK
    chooses the converged values, so the manual chunk targets are dropped."""

    def test_auto_gain_once_records_its_changes(self, sim_camera):
        camera, _cam = sim_camera
        fv = camera._frame_validity
        camera.set_gain_db(5.0)
        camera.set_exposure_ms(10.0)
        assert fv.target('gain') is not None and fv.target('exposure') is not None
        before = fv.invalidation_counts
        camera.auto_gain_once(
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
        return (limits.min_um + limits.max_um) / 2

    @pytest.mark.parametrize(
        ('axis', 'source'),
        [('X', 'xy_move'), ('Y', 'xy_move'), ('Z', 'z_move')],
    )
    def test_a_move_to_a_place_invalidates_its_axis_source(
        self, sim_turreted_session, axis, source
    ):
        motion = sim_turreted_session.scope.motion
        fv = sim_turreted_session.scope.imaging._frame_validity
        target = self._middle_of_travel(motion, axis)
        axis_move(motion, axis, target - 500.0)
        before = fv.invalidation_counts

        axis_move(motion, axis, target)

        growth = self._growth(fv, before)
        assert growth[source] > 0, (
            f'a move of {axis} to a place must invalidate {source!r}: {growth}'
        )
        others = {s: n for s, n in growth.items() if s != source}
        assert not any(others.values()), (
            f'a move of {axis} to a place invalidated another axis source: {growth}'
        )

    @pytest.mark.parametrize(
        ('axis', 'source'),
        [('X', 'xy_move'), ('Y', 'xy_move'), ('Z', 'z_move')],
    )
    def test_a_move_by_a_distance_invalidates_its_axis_source(
        self, sim_turreted_session, axis, source
    ):
        motion = sim_turreted_session.scope.motion
        fv = sim_turreted_session.scope.imaging._frame_validity
        axis_move(motion, axis, self._middle_of_travel(motion, axis))
        before = fv.invalidation_counts

        axis_move_by(motion, axis, 100.0)

        growth = self._growth(fv, before)
        assert growth[source] > 0, (
            f'a move of {axis} by a distance must invalidate {source!r}: {growth}'
        )
        others = {s: n for s, n in growth.items() if s != source}
        assert not any(others.values()), (
            f'a move of {axis} by a distance invalidated another axis source: {growth}'
        )

    def test_move_turret_invalidates_turret(self, sim_turreted_session):
        """The turret's move records 'turret', never 'xy_move'. Its Z park
        and restore are Z moves of their own and record 'z_move'."""
        motion = sim_turreted_session.scope.motion
        fv = sim_turreted_session.scope.imaging._frame_validity
        slot = 2 if motion.turret_status().slot != 2 else 1
        before = fv.invalidation_counts

        motion.move_turret(slot)

        growth = self._growth(fv, before)
        assert motion.turret_status().slot == slot
        assert growth['turret'] > 0, f"move_turret must invalidate 'turret': {growth}"
        assert growth['xy_move'] == 0, f"move_turret invalidated 'xy_move': {growth}"


class TestGeometryFormatInvalidates:
    """Pixel-format, frame-size, and binning changes realloc the camera
    buffer / restart the grab engine; each must record its change so a
    capture waits for the old geometry to flush."""

    def test_set_frame_size_records_its_change(self, sim_camera):
        camera, _cam = sim_camera
        before = camera._frame_validity.invalidation_counts
        camera.set_frame_size(640, 480)
        assert _recorded_once(camera._frame_validity, before, 'frame_size')

    def test_set_pixel_format_records_its_change(self, sim_camera):
        camera, _cam = sim_camera
        before = camera._frame_validity.invalidation_counts
        # A pixel format is set only through the image mode; 12-bit
        # scientific selects Mono12 on the simulated camera.
        camera.set_image_mode('12bit_scientific')
        assert camera.pixel_format == 'Mono12'
        assert _recorded_once(camera._frame_validity, before, 'pixel_format')

    def test_set_binning_records_its_change(self, sim_camera):
        camera, _cam = sim_camera
        before = camera._frame_validity.invalidation_counts
        camera.set_binning(2)
        assert camera.read_binning() == 2
        assert _recorded_once(camera._frame_validity, before, 'binning')


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

    def test_a_depth_no_frame_has_is_refused_not_measured(self):
        # A negative depth raised a bare ValueError from the shift, and 0
        # made full scale 0, so every pixel read as saturated.
        from _lumascope.modules.exceptions import ArgumentRefusedError

        frame = np.zeros((2, 2), dtype=np.uint8)
        refused = []
        for depth in (-1, 0, 17):
            with pytest.raises(ArgumentRefusedError) as raised:
                ImagingAPI.saturated_fraction(frame, depth)
            refused.append(raised.value.reason)
        with pytest.raises(ArgumentRefusedError) as dark:
            ImagingAPI._lit_fraction(frame, 0)

        assert refused == ['out_of_range'] * 3
        assert dark.value.reason == 'out_of_range'


class TestRejectedSettingRaisesAndKeepsCache:
    """A driver that CONFIRMS a settings-write rejection (returns False;
    drivers with no confirmation signal return None) must reach the caller
    as a raise -- which carries the words its reporter shows, so the API
    posts nothing itself -- and the requested value must NOT be recorded in
    the camera cache as if it took. IDS has no chunk data, so without this
    the camera streams at the old setting while the cache claims the
    new one -- the silent stale-settings shape."""

    def test_rejected_gain_raises_and_keeps_cache(self, sim_camera, monkeypatch, centre_posts):
        camera, cam = sim_camera
        camera.set_gain_db(2.0)  # establish a known cache value
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        # The public setter raises the rejection to its caller; what this test
        # pins is what happens on the way out -- the API posts nothing, and the
        # cache keeps the value the camera actually holds.
        with pytest.raises(CameraSettingRejected):
            camera.set_gain_db(7.0)

        captured = [n for n in centre_posts if n.severity == Severity.ERROR]
        assert not captured, 'A confirmed gain rejection is shown by its reporter, not the API'
        assert camera.gain_db == 2.0, 'A rejected gain write must not be recorded in the cache'

    def test_rejected_exposure_raises_and_keeps_cache(self, sim_camera, monkeypatch, centre_posts):
        camera, cam = sim_camera
        camera.set_exposure_ms(20.0)  # establish a known cache value
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        # See the gain case above: the raise is the public setter's contract,
        # the silent API and the held cache are what this test pins.
        with pytest.raises(CameraSettingRejected):
            camera.set_exposure_ms(50.0)

        captured = [n for n in centre_posts if n.severity == Severity.ERROR]
        assert not captured, 'A confirmed exposure rejection is shown by its reporter, not the API'
        assert camera.exposure_ms == 20.0, (
            'A rejected exposure write must not be recorded in the cache'
        )


class TestEveryLedWriteRecordsItsChange:
    """The three public LED writes on IlluminationAPI -- ``led_on``,
    ``led_off`` and ``leds_off`` -- each record an 'led' change with the
    scope's frame validity, so a capture after any of them waits past it."""

    def test_each_led_mutator_records_an_led_change(self, sim_scope):
        illum = sim_scope.illumination
        validity = sim_scope.imaging._frame_validity
        mutator_calls = {
            'led_on': lambda: illum.led_on(channel=LED0, illumination_ma=10),
            'led_off': lambda: illum.led_off(channel=LED0),
            'leds_off': lambda: illum.leds_off(),
        }
        missing = []
        for name, call in mutator_calls.items():
            before = validity.invalidation_counts
            call()
            if not _recorded_once(validity, before, 'led'):
                missing.append(name)
        assert not missing, (
            'LED mutator coverage: each IlluminationAPI LED state-mutator must '
            "record an 'led' change with frame validity so a capture waits "
            f'past the transition. Missing: {missing!r}.'
        )
