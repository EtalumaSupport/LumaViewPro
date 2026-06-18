"""Regression tests for the significant-bits / native pixel-depth owner.

Pins the contract that the summed-capture reducer and the 8-bit display
downconvert key off a frame's significant bits, not its numpy container dtype.

Covered:
  - #570: a 12-bit frame summed (Sum > 1) overflows 4095 and, treated as a
    12-bit value, indexes the 4096-entry display LUT out of range -> IndexError.
    The summed result belongs in a 16-bit container and must downconvert as
    16-bit.
  - #424 depth half: a 10-bit frame must map full-white to full 8-bit white,
    not be treated as 12-bit (which crushes it ~4x dark).
  - The "sum into a 16-bit container" rule: accumulate, saturate at 65535.

These drive the REAL ImagingAPI.get_image path against the simulated camera in
a chosen pixel format + white test pattern, so every grabbed frame is uniform
and summed values are exact and deterministic.

xfail(strict) tests encode behavior that lands with the owner; each flips to
XPASS when its fix arrives, at which point the marker is removed in that commit.
"""

import numpy as np
import pytest

from modules.lumascope_api import Lumascope


def _configure_sim(scope, pixel_format, pattern='White'):
    """Put the simulated camera into a fixed depth + uniform bright frame."""
    cam = scope._camera_driver
    cam.set_timing_mode('fast')
    cam.set_pixel_format(pixel_format)
    cam.set_test_pattern(True, pattern)
    cam.start_grabbing()
    return scope


@pytest.fixture
def make_scope():
    """Factory for simulated scopes at a chosen pixel format; auto-teardown."""
    scopes = []

    def _make(pixel_format, pattern='White'):
        scope = _configure_sim(Lumascope(simulate=True), pixel_format, pattern)
        scopes.append(scope)
        return scope

    yield _make
    for scope in scopes:
        try:
            scope._camera_driver.stop_grabbing()
            scope.disconnect()
        except Exception:
            pass


# Uniform white-pattern pixel value the simulator emits per pixel format.
WHITE = {'Mono8': 255, 'Mono10': 1023, 'Mono12': 4095}


class TestSummedCaptureDepthCeiling:
    """The summed reducer + display downconvert must respect significant bits."""

    @pytest.mark.xfail(
        strict=True,
        reason='12-bit summed frame crashes the 12-bit display LUT; flips when the downconvert keys off significant bits',
    )
    @pytest.mark.parametrize('sum_count', [2, 3, 30])
    def test_summed_12bit_display_no_crash(self, make_scope, sum_count):
        """A summed 12-bit capture forced to 8-bit returns a uint8 array.

        Today combined > 4095 indexes the 4096-entry 12-bit LUT -> IndexError.
        """
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=True, sum_count=sum_count)
        assert img is not None
        assert img.dtype == np.uint8

    def test_sum_into_16bit_container_accumulates(self, make_scope):
        """Summing accumulates into a 16-bit container (Sum=2 of 4095 -> 8190)."""
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=False, sum_count=2)
        assert img is not None
        assert img.dtype == np.uint16
        assert int(img.max()) == 2 * WHITE['Mono12']  # 8190, no overflow, no over-clip

    def test_sum_saturates_at_container_ceiling(self, make_scope):
        """A high sum count saturates at the 16-bit container max (65535)."""
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=False, sum_count=30)
        assert img is not None
        # 4095 * 30 = 122850 > 65535 -> saturates at the container ceiling.
        assert int(img.max()) == 65535

    def test_single_frame_identity(self, make_scope):
        """Sum=1 returns the native frame unchanged (no clip, no dtype change)."""
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=False, sum_count=1)
        assert img is not None
        assert img.dtype == np.uint16
        assert int(img.max()) == WHITE['Mono12']  # 4095, untouched

    def test_8bit_camera_sum_unaffected(self, make_scope):
        """An 8-bit camera path keeps working: no crash, uint8, saturates at 255."""
        scope = make_scope('Mono8')
        img = scope.imaging.get_image(force_to_8bit=True, sum_count=2)
        assert img is not None
        assert img.dtype == np.uint8
        assert int(img.max()) == 255  # 255 * 2 -> saturates at 8-bit ceiling


class TestDisplayDownconvertGenericDepth:
    """The 8-bit display mapping must scale by the frame's real significant bits."""

    @pytest.mark.xfail(
        strict=True,
        reason='10-bit frame is downconverted as 12-bit (crushed ~4x dark); flips when the downconvert keys off significant bits',
    )
    def test_10bit_white_maps_to_full_8bit_white(self, make_scope):
        """A full-white 10-bit frame must map to 8-bit 255, not ~63.

        Treating a 10-bit value (max 1023) as 12-bit divides by 4095 and crushes
        white to ~63. The display divisor must come from the significant bits.
        """
        scope = make_scope('Mono10')
        img = scope.imaging.get_image(force_to_8bit=True, sum_count=1)
        assert img is not None
        assert img.dtype == np.uint8
        assert int(img.max()) == 255

    def test_12bit_white_maps_to_full_8bit_white(self, make_scope):
        """A full-white 12-bit frame already maps to 255 (guards against regression)."""
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=True, sum_count=1)
        assert img is not None
        assert img.dtype == np.uint8
        assert int(img.max()) == 255
