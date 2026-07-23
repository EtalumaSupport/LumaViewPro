"""Tests for ``modules.image_utils`` -- convert helpers + boundary wrappers.

Covers ``convert_12bit_to_8bit(out=...)``, whose reusable out buffer saves
~120 MB/s allocator churn on the 30fps Pylon 12-bit preview path.
"""

from __future__ import annotations

import numpy as np

import modules.image_utils as image_utils
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


class TestIsColorShape:
    """is_color_shape is the one channel-count rule; is_color_image delegates to
    it so a caller holding only a shape tuple uses the same definition."""

    def test_three_channel_shape_is_color(self):
        assert image_utils.is_color_shape((8, 8, 3)) is True

    def test_mono_shape_is_not_color(self):
        assert image_utils.is_color_shape((8, 8)) is False
        assert image_utils.is_color_shape((8, 8, 1)) is False

    def test_is_color_image_delegates_to_shape(self):
        assert image_utils.is_color_image(np.zeros((4, 4, 3), dtype=np.uint8)) is True
        assert image_utils.is_color_image(np.zeros((4, 4), dtype=np.uint8)) is False


class TestTiffDiscovery:
    """is_tiff / find_tiff_files are the single answer to "is this a TIFF" and
    "which TIFFs are in this folder". The bug they exist to prevent: a folder
    scan that unions a `*.tiff` glob with a `*.ome.tiff` glob counts every
    OME-TIFF twice (its suffix is plain `.tiff`) while missing single-`f` `.tif`
    files entirely -- so a 4-tile OME capture loaded as 8 rows, four of them
    duplicates colliding on the stitcher's lattice key."""

    def _touch(self, directory, *names):
        for name in names:
            (directory / name).write_bytes(b'')

    def test_is_tiff_accepts_every_form(self, tmp_path):
        assert image_utils.is_tiff(tmp_path / 'a.tiff') is True
        assert image_utils.is_tiff(tmp_path / 'a.tif') is True
        assert image_utils.is_tiff(tmp_path / 'a.ome.tiff') is True
        assert image_utils.is_tiff(tmp_path / 'a.ome.tif') is True
        assert image_utils.is_tiff(tmp_path / 'A.TIFF') is True

    def test_is_tiff_rejects_non_tiff(self, tmp_path):
        assert image_utils.is_tiff(tmp_path / 'a.png') is False
        assert image_utils.is_tiff(tmp_path / 'a.jpg') is False

    def test_ome_tiff_counted_once(self, tmp_path):
        # The exact failure: an OME-TIFF must yield ONE path, not two.
        self._touch(tmp_path, 'tile.ome.tiff')
        found = image_utils.find_tiff_files(tmp_path)
        assert found == [tmp_path / 'tile.ome.tiff']

    def test_single_f_tif_is_found(self, tmp_path):
        # The other half: `.tif` must not be dropped (Quick Enhance writes it).
        self._touch(tmp_path, 'enhanced.tif')
        assert image_utils.find_tiff_files(tmp_path) == [tmp_path / 'enhanced.tif']

    def test_one_row_per_file_all_forms(self, tmp_path):
        self._touch(
            tmp_path,
            'a.tiff',
            'b.ome.tiff',
            'c.tif',
            'd.ome.tif',
            'skip.png',
        )
        found = {p.name for p in image_utils.find_tiff_files(tmp_path)}
        assert found == {'a.tiff', 'b.ome.tiff', 'c.tif', 'd.ome.tif'}

    def test_no_duplicates_ever(self, tmp_path):
        self._touch(tmp_path, 'x.ome.tiff', 'y.tiff', 'z.ome.tif')
        found = image_utils.find_tiff_files(tmp_path)
        assert len(found) == len(set(found)), 'find_tiff_files must not return a path twice'

    def test_recursive_flag(self, tmp_path):
        sub = tmp_path / 'Blue'
        sub.mkdir()
        self._touch(tmp_path, 'top.tiff')
        self._touch(sub, 'nested.ome.tiff')
        flat = {p.name for p in image_utils.find_tiff_files(tmp_path, recursive=False)}
        deep = {p.name for p in image_utils.find_tiff_files(tmp_path, recursive=True)}
        assert flat == {'top.tiff'}
        assert deep == {'top.tiff', 'nested.ome.tiff'}
