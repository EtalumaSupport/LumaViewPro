"""Tests for ``modules.image_utils`` -- convert helpers + boundary wrappers.

Phase 1c.4 adds ``convert_12bit_to_8bit(out=...)`` as a sibling to the
existing PIW-5 ``convert_12bit_to_16bit(out=...)`` pattern. Saves
~120 MB/s allocator churn on the 30fps Pylon 12-bit preview path.

The ``test_audit_fixes.TestPIW5_Convert12to16OutBuffer`` test covers the
16-bit sibling. This file covers the 8-bit variant.
"""

from __future__ import annotations

import numpy as np

from modules.image_utils import convert_12bit_to_8bit


class TestConvert12to8OutBuffer:
    """``convert_12bit_to_8bit(image, out=None)`` accepts a caller-supplied
    output buffer; matched shape/dtype reuses the buffer, mismatched falls
    back to fresh allocation."""

    def test_matched_out_buffer_is_reused(self):
        src = np.array([[0, 1, 2], [4093, 4094, 4095]], dtype=np.uint16)
        buf = np.zeros((2, 3), dtype=np.uint8)
        result = convert_12bit_to_8bit(src, out=buf)
        assert result is buf, 'matched-shape out buffer should be returned'
        # LUT compresses 12-bit (0..4095) -> 8-bit (0..255).
        assert result[0, 0] == 0
        assert result[1, 2] == 255

    def test_mismatched_shape_falls_back(self):
        src = np.array([[100, 200]], dtype=np.uint16)
        bad_buf = np.zeros((3, 3), dtype=np.uint8)
        result = convert_12bit_to_8bit(src, out=bad_buf)
        assert result is not bad_buf, 'shape mismatch must fall back to fresh allocation'
        assert result.shape == src.shape

    def test_mismatched_dtype_falls_back(self):
        src = np.array([[100, 200]], dtype=np.uint16)
        bad_buf = np.zeros((1, 2), dtype=np.uint16)  # wrong dtype
        result = convert_12bit_to_8bit(src, out=bad_buf)
        assert result is not bad_buf, 'dtype mismatch must fall back to fresh allocation'
        assert result.dtype == np.uint8

    def test_no_out_param_preserves_legacy_behavior(self):
        src = np.array([[100, 200]], dtype=np.uint16)
        result = convert_12bit_to_8bit(src)
        assert result is not src
        assert result.dtype == np.uint8

    def test_uint8_input_short_circuits(self):
        src = np.array([[10, 20]], dtype=np.uint8)
        result = convert_12bit_to_8bit(src)
        assert result is src, 'uint8 input must short-circuit (no LUT, no copy)'

    def test_o1_allocations_across_sequential_calls(self):
        """100 sequential calls with a single reused out buffer should
        produce O(1) frame-sized np.empty allocations (the ``np.take``
        path with ``out=`` writes in place; no fresh frame buffer)."""
        from unittest.mock import patch

        src = np.full((64, 64), 2000, dtype=np.uint16)
        out = np.zeros_like(src, dtype=np.uint8)

        FRAME_SIZE = 64 * 64
        alloc_count = {'n': 0}
        orig_empty = np.empty

        def counting_empty(shape, *args, **kwargs):
            try:
                n = 1
                if hasattr(shape, '__iter__'):
                    for d in shape:
                        n *= int(d)
                else:
                    n = int(shape)
            except Exception:
                n = 0
            if n >= FRAME_SIZE:
                alloc_count['n'] += 1
            return orig_empty(shape, *args, **kwargs)

        with patch('numpy.empty', side_effect=counting_empty):
            for _ in range(100):
                convert_12bit_to_8bit(src, out=out)

        assert alloc_count['n'] < 10, (
            f'Expected O(1) frame-sized allocations with reused out buffer; got {alloc_count["n"]}.'
        )
