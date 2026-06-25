# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IDS Peak hardware tests -- opt-in via --run-ids-hardware.

These tests require:
  1. Real ids_peak SDK installed (not the conftest MagicMock)
  2. A connected IDS camera

Skipped by default. Run with:
    pytest tests/test_ids_hardware.py --run-ids-hardware

The `ids_hardware` marker is gated by conftest.pytest_collection_modifyitems --
test bodies do NOT need their own skip dance.
"""

import time
import unittest

import numpy as np
import pytest

# When --run-ids-hardware is set, conftest skips installing the ids_peak
# mock so the real SDK can load here. When the flag is NOT set, this
# import succeeds against the conftest mock and the marker below skips
# the tests at collection time.
from drivers.idscamera import IDSCamera


@pytest.mark.ids_hardware
class TestIDS(unittest.TestCase):
    def setUp(self):
        self.camera = IDSCamera()

    def tearDown(self):
        self.camera.disconnect()
        time.sleep(1)

    def test_connect_disconnect(self):
        self.assertTrue(self.camera.disconnect())
        self.assertTrue(self.camera.connect())

    def test_grab(self):
        self.assertTrue(self.camera.is_grabbing())
        self.camera.stop_grabbing()
        self.assertFalse(self.camera.is_grabbing())

    def test_frame_size(self):
        # Valid
        self.camera.set_frame_size(1920, 1528)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1920, 'height': 1528})
        # Out of bounds
        self.camera.set_frame_size(1919, 1529)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1872, 'height': 1528})
        # Incorrect increment
        self.camera.set_frame_size(1480, 906)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1440, 'height': 904})

    def test_pixel_format(self):
        formats = self.camera.get_supported_pixel_formats()
        self.camera.set_pixel_format(formats[0])
        self.assertTrue(self.camera.get_pixel_format() in formats)

    def test_pixel_format_logical_mono8(self):
        # Logical 'Mono8' must resolve to whichever Mono8-class entry the
        # camera actually exposes (Mono8, Mono8g, etc). Skipped on cameras
        # without any Mono8-class entry (e.g. Sony IMX676 in U3-34L0XCP-M
        # exposes only Mono10g40IDS / Mono12g24IDS) -- on those, the resolver
        # correctly returns None and set_pixel_format reports unsupported,
        # tested by the pure-logic suite TestIDSPixelFormatResolver.
        supported = self.camera.get_supported_pixel_formats()
        if not any(s.startswith('Mono8') for s in supported):
            self.skipTest(f'camera has no Mono8-class entry (supported={list(supported)})')
        self.assertTrue(self.camera.set_pixel_format('Mono8'))
        active = self.camera.get_pixel_format()
        self.assertTrue(
            active.startswith('Mono8'), f'Logical Mono8 resolved to non-Mono8 entry: {active}'
        )

    def test_exposure_t(self):
        self.camera.exposure_t(15)
        self.assertAlmostEqual(self.camera.get_exposure_t(), 15.0, delta=0.01)

    def test_binning_size(self):
        self.assertTrue(self.camera.set_binning_size(1))
        self.assertEqual(self.camera.get_binning_size(), 1)
        self.assertTrue(self.camera.set_binning_size(2))
        self.assertEqual(self.camera.get_binning_size(), 2)
        self.assertFalse(self.camera.set_binning_size(3))
        self.assertEqual(self.camera.get_binning_size(), 2)
        self.assertFalse(self.camera.set_binning_size(0))
        self.assertEqual(self.camera.get_binning_size(), 2)

    def test_binned_aoi_space_max_halves(self):
        """Resolve whether the AOI maxima are reported in binned or native px.

        The oversize-then-crop framing math is planned in displayed
        (post-binning) pixel space, which is correct only if Width/Height
        maxima are in binned pixels -- i.e. they roughly halve from 1x to 2x
        binning. If the max does NOT shrink, the AOI is in native pixels and
        the framing plan must divide once at the end instead. This records the
        fact so the math can be trusted; run with -s to see the printed values.
        """
        self.camera.set_binning_size(1)
        max_1x = self.camera.get_max_frame_size()
        self.camera.set_binning_size(2)
        max_2x = self.camera.get_max_frame_size()
        self.camera.set_binning_size(1)  # restore
        print(f'[binned-aoi-space] max 1x={max_1x} 2x={max_2x}')
        self.assertLess(
            max_2x['width'],
            max_1x['width'],
            f'AOI max did not shrink with binning ({max_1x} -> {max_2x}): AOI is in '
            'native pixels, so the oversize-crop plan must divide by binning at the end',
        )
        self.assertAlmostEqual(max_2x['width'], max_1x['width'] / 2, delta=max_1x['width'] * 0.02)

    def test_grab_frame(self):
        time.sleep(1)  # Allow time for the camera to start grabbing
        result, timestamp = self.camera.grab()
        self.assertTrue(result)
        self.assertTrue(len(self.camera.array) == 1528)
        self.assertIsNotNone(timestamp)

    def test_gain(self):
        self.camera.gain(10)
        self.assertAlmostEqual(self.camera.get_gain(), 10.0, delta=0.1)

    def test_native_depth_frame_is_right_aligned_uint16(self):
        """Keystone bench check: a grabbed frame arrives at the sensor's native
        depth in a uint16 container, right-aligned.

        Right-aligned means full scale is (1 << significant_bits) - 1 and every
        pixel fits in the low significant_bits of the uint16. If the SDK
        left-aligns instead, pixel values ride in the high bits and overflow
        that range -- the assertion below catches it, settling the #1
        undocumented unpack unknown. drivers/ids_unpack.py is the cross-check
        for the expected layout.

        Needs light on the sensor: a near-dark frame can't distinguish the two
        alignments (small values stay small either way), so the test skips
        rather than pass vacuously when the frame is too dark.
        """
        time.sleep(1)  # let the grab loop store a frame
        result, timestamp = self.camera.grab()
        self.assertTrue(result)
        self.assertIsNotNone(timestamp)

        frame = self.camera.array
        self.assertEqual(frame.dtype, np.uint16, 'native-depth frame must be a uint16 container')

        sig = self.camera.last_significant_bits
        self.assertIn(
            sig, (10, 12), f'IMX676 delivers packed 10/12-bit; got significant_bits={sig}'
        )

        peak = int(frame.max())
        if peak < 256:
            self.skipTest(
                f'frame too dark (max={peak}) to judge alignment -- put light on the sensor'
            )
        full_scale = (1 << sig) - 1
        self.assertLessEqual(
            peak,
            full_scale,
            f'frame.max()={peak} exceeds {full_scale}: the SDK is NOT right-aligned at '
            f'significant_bits={sig} (values ride in the high bits) -- the consuming code '
            f'must shift. This is the alignment unknown the rebuild flagged.',
        )
        self.assertGreaterEqual(int(frame.min()), 0)

    def test_unpack_benchmark(self):
        """Head-to-head: the SDK ConvertTo unpack vs the numpy ids_unpack path,
        on real packed frames. Two questions in one run:

          1. Correctness -- ConvertTo is the oracle. A zero-mismatch result
             proves the derived packed-bit layout (and right-alignment) is
             correct, since IDS documents the layout only as a figure.
          2. Speed -- the per-frame timings show whether the numpy unpack is
             actually faster than ConvertTo on this host (the ~18 fps display
             cap is the ConvertTo throughput).

        Run with output visible:
            pytest tests/test_ids_hardware.py -k unpack_benchmark --run-ids-hardware -s

        Needs light on the sensor so a real (non-zero) image exercises the
        full-scale bits; an all-dark frame would match trivially on both paths.
        """
        time.sleep(1)  # let acquisition settle
        res = self.camera.benchmark_unpack(n_frames=200)

        print('\n[IDS unpack benchmark]')
        for k in (
            'wire_format',
            'width',
            'height',
            'available_formats',
            'packed_dtype',
            'n_compared',
            'mismatches',
            'first_mismatch',
            'convert',
            'numpy',
            'icv',
        ):
            print(f'  {k}: {res.get(k)}')

        self.assertGreater(res['n_compared'], 0, f'no frames were compared: {res.get("error")}')
        self.assertEqual(
            res['mismatches'],
            0,
            f'numpy unpack disagrees with ConvertTo on {res["mismatches"]}/'
            f'{res["n_compared"]} frames -- the derived layout is wrong: '
            f'{res.get("first_mismatch")}',
        )


class TestIDSPixelFormatResolver(unittest.TestCase):
    """Pure-logic tests for IDSCamera._resolve_logical_format_name.

    Runs without hardware -- the resolver is a @staticmethod that takes the
    supported-list as a parameter, so we don't need a connected camera or
    SDK access. Regression test for the 2026-05-04 bench bug where
    set_pixel_format('Mono8') silently fell through to formats[0] on cameras
    whose PixelFormat node uses sensor-specific names (Mono10g40IDS etc).
    """

    def test_exact_match(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(IDSCamera._resolve_logical_format_name('Mono8', ('Mono8',)), 'Mono8')

    def test_mono8_prefix_match(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._resolve_logical_format_name(
                'Mono8', ('Mono10g40IDS', 'Mono12g24IDS', 'Mono8g')
            ),
            'Mono8g',
        )

    def test_mono8_no_family_returns_none(self):
        from drivers.idscamera import IDSCamera

        self.assertIsNone(
            IDSCamera._resolve_logical_format_name('Mono8', ('Mono10g40IDS', 'Mono12g24IDS'))
        )

    def test_mono12_prefix_match(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._resolve_logical_format_name('Mono12', ('Mono8', 'Mono12g24IDS')),
            'Mono12g24IDS',
        )

    def test_mono12_falls_back_to_mono10(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._resolve_logical_format_name('Mono12', ('Mono8', 'Mono10g40IDS')),
            'Mono10g40IDS',
        )

    def test_camera_native_name_passes_through(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._resolve_logical_format_name('Mono10g40IDS', ('Mono8', 'Mono10g40IDS')),
            'Mono10g40IDS',
        )

    def test_unknown_logical_returns_none(self):
        from drivers.idscamera import IDSCamera

        self.assertIsNone(IDSCamera._resolve_logical_format_name('Mono99', ('Mono8',)))

    def test_empty_supported_returns_none(self):
        from drivers.idscamera import IDSCamera

        self.assertIsNone(IDSCamera._resolve_logical_format_name('Mono8', ()))


if __name__ == '__main__':
    unittest.main()
