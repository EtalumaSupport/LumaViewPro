# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for stitcher modules -- stitch_algorithms.py (feature-based) and stitcher.py (grid-based)."""

import pathlib

import cv2
import numpy as np
import pandas as pd
import pytest
import tifffile


# ---------------------------------------------------------------------------
# stitch_algorithms.py -- feature-based stitching, color transfer, border crop
# ---------------------------------------------------------------------------

from modules.stitch_algorithms import (
    _image_stats,
    align_tile_positions,
    color_transfer,
    _grab_contours,
    crop_to_content,
    stitch_registered_tiles,
)


class TestImageStats:
    """Test _image_stats -- computes L*a*b* channel statistics."""

    def test_uniform_image(self):
        # Uniform gray -> known LAB values
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        stats = _image_stats(lab)
        assert len(stats) == 6  # (lMean, lStd, aMean, aStd, bMean, bStd)
        l_mean, l_std, a_mean, a_std, b_mean, b_std = stats
        assert l_std == 0.0  # uniform -> zero std
        assert a_std == 0.0
        assert b_std == 0.0

    def test_random_image_has_nonzero_std(self):
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        stats = _image_stats(lab)
        l_mean, l_std, a_mean, a_std, b_mean, b_std = stats
        assert l_std > 0
        assert a_std > 0
        assert b_std > 0


class TestColorTransfer:
    """Test color_transfer -- LAB color distribution transfer (Reinhard et al.)."""

    def test_output_shape_matches_target(self):
        source = np.full((50, 50, 3), 200, dtype=np.uint8)
        target = np.full((80, 60, 3), 100, dtype=np.uint8)
        result = color_transfer(source, target)
        assert result.shape == target.shape
        assert result.dtype == np.uint8

    def test_identical_images_unchanged(self):
        # Uniform images have zero std -- division guard returns identity-like result
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = color_transfer(img.copy(), img.copy())
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        # With zero-std guard, uniform -> uniform (LAB round-trip may shift slightly)
        assert np.allclose(result, result[0, 0], atol=1)  # all pixels same

    def test_varied_identical_images(self):
        # Non-uniform identical images -> result ~= input
        rng = np.random.RandomState(7)
        img = rng.randint(80, 180, (50, 50, 3), dtype=np.uint8)
        result = color_transfer(img.copy(), img.copy())
        assert np.allclose(result, img, atol=5)

    def test_different_colors_shifts_target(self):
        # Bright source, dark target -> result should be brighter than original target
        source = np.full((50, 50, 3), 220, dtype=np.uint8)
        target = np.full((50, 50, 3), 50, dtype=np.uint8)
        result = color_transfer(source, target)
        # Result's mean brightness should be closer to source than original target
        assert result.mean() > target.mean()

    def test_handles_color_images(self):
        rng = np.random.RandomState(123)
        source = rng.randint(50, 200, (30, 30, 3), dtype=np.uint8)
        target = rng.randint(50, 200, (30, 30, 3), dtype=np.uint8)
        result = color_transfer(source, target)
        assert result.shape == target.shape
        # Values should be valid uint8
        assert result.min() >= 0
        assert result.max() <= 255


class TestGrabContours:
    """Test _grab_contours -- OpenCV 4.x contour extraction."""

    def test_two_element_tuple(self):
        # OpenCV 4.x returns (contours, hierarchy)
        contours = [np.array([[0, 0], [1, 0], [1, 1]])]
        hierarchy = np.array([[[0, 0, 0, 0]]])
        result = _grab_contours((contours, hierarchy))
        assert result is contours

    def test_three_element_tuple(self):
        # OpenCV 3.x returned (image, contours, hierarchy)
        img = np.zeros((10, 10), dtype=np.uint8)
        contours = [np.array([[0, 0], [1, 0], [1, 1]])]
        hierarchy = np.array([[[0, 0, 0, 0]]])
        result = _grab_contours((img, contours, hierarchy))
        assert result is contours

    def test_with_real_findcontours(self):
        # Create a simple image with a white rectangle
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
        raw = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = _grab_contours(raw)
        assert len(contours) == 1
        # Bounding rect should roughly match the rectangle
        x, y, w, h = cv2.boundingRect(contours[0])
        assert 15 <= x <= 25
        assert 55 <= w <= 65


class TestCropToContent:
    """Test crop_to_content -- crops stitched image to content area."""

    def test_crops_black_border(self):
        # Create an image with content in the center and black border
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[40:160, 60:240] = 128  # gray content area
        result = crop_to_content(img)
        # Result should be smaller than input (border removed)
        assert result.shape[0] < img.shape[0]
        assert result.shape[1] < img.shape[1]
        # Content should be preserved
        assert result.mean() > 0

    def test_full_content_image(self):
        # No border -> result ~= input size (only the 10px border padding matters)
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = crop_to_content(img)
        # Should be close to original dimensions (+/-20 from the added border)
        assert abs(result.shape[0] - 100) <= 22
        assert abs(result.shape[1] - 100) <= 22


# ---------------------------------------------------------------------------
# Current stitcher.py -- _simple_position_stitcher
# ---------------------------------------------------------------------------

from modules.stitcher import Stitcher
import modules.common_utils as common_utils
import modules.image_utils as image_utils


class TestSimplePositionStitcher:
    """Test Stitcher._simple_position_stitcher with synthetic tile images."""

    @pytest.fixture
    def tile_dir(self, tmp_path):
        """Create a 2x2 grid of grayscale tiles with known pixel values."""
        tiles = {
            'tile_0_0.tiff': np.full((50, 50), 50, dtype=np.uint8),
            'tile_1_0.tiff': np.full((50, 50), 100, dtype=np.uint8),
            'tile_0_1.tiff': np.full((50, 50), 150, dtype=np.uint8),
            'tile_1_1.tiff': np.full((50, 50), 200, dtype=np.uint8),
        }
        for name, img in tiles.items():
            cv2.imwrite(str(tmp_path / name), img)
        return tmp_path

    @pytest.fixture
    def tile_df(self):
        """DataFrame describing a 2x2 grid of tiles."""
        return pd.DataFrame(
            [
                {'Filepath': 'tile_0_0.tiff', 'X': 0.0, 'Y': 0.0},
                {'Filepath': 'tile_1_0.tiff', 'X': 1.0, 'Y': 0.0},
                {'Filepath': 'tile_0_1.tiff', 'X': 0.0, 'Y': 1.0},
                {'Filepath': 'tile_1_1.tiff', 'X': 1.0, 'Y': 1.0},
            ]
        )

    def test_output_dimensions(self, tile_dir, tile_df):
        result = Stitcher._simple_position_stitcher(tile_dir, tile_df)
        assert result['status'] is True
        img = result['image']
        # 2x2 grid of 50x50 tiles -> 100x100
        assert img.shape == (100, 100)

    def test_center_metadata(self, tile_dir, tile_df):
        result = Stitcher._simple_position_stitcher(tile_dir, tile_df)
        center = result['metadata']['center']
        assert center['x'] == 0.5  # mean of [0.0, 1.0]
        assert center['y'] == 0.5

    def test_all_pixels_filled(self, tile_dir, tile_df):
        result = Stitcher._simple_position_stitcher(tile_dir, tile_df)
        img = result['image']
        # No black pixels -- all tiles have nonzero values
        assert img.min() > 0

    def test_single_tile(self, tmp_path):
        tile = np.full((64, 64), 42, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'single.tiff'), tile)
        df = pd.DataFrame([{'Filepath': 'single.tiff', 'X': 0.0, 'Y': 0.0}])
        result = Stitcher._simple_position_stitcher(tmp_path, df)
        assert result['status'] is True
        assert result['image'].shape == (64, 64)
        assert np.all(result['image'] == 42)

    def test_color_tiles(self, tmp_path):
        """3-channel color tiles should produce a 3-channel stitched image."""
        t1 = np.full((40, 40, 3), [255, 0, 0], dtype=np.uint8)
        t2 = np.full((40, 40, 3), [0, 255, 0], dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'a.tiff'), t1)
        cv2.imwrite(str(tmp_path / 'b.tiff'), t2)
        df = pd.DataFrame(
            [
                {'Filepath': 'a.tiff', 'X': 0.0, 'Y': 0.0},
                {'Filepath': 'b.tiff', 'X': 1.0, 'Y': 0.0},
            ]
        )
        result = Stitcher._simple_position_stitcher(tmp_path, df)
        assert result['status'] is True
        img = result['image']
        assert img.shape == (40, 80, 3)

    def test_3x1_grid(self, tmp_path):
        """3 tiles in a row."""
        for i in range(3):
            tile = np.full((30, 30), (i + 1) * 60, dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f't{i}.tiff'), tile)
        df = pd.DataFrame([{'Filepath': f't{i}.tiff', 'X': float(i), 'Y': 0.0} for i in range(3)])
        result = Stitcher._simple_position_stitcher(tmp_path, df)
        assert result['status'] is True
        assert result['image'].shape == (30, 90)

    def test_16bit_tiles(self, tmp_path):
        """16-bit grayscale tiles."""
        t1 = np.full((32, 32), 1000, dtype=np.uint16)
        t2 = np.full((32, 32), 50000, dtype=np.uint16)
        cv2.imwrite(str(tmp_path / 'a.tiff'), t1)
        cv2.imwrite(str(tmp_path / 'b.tiff'), t2)
        df = pd.DataFrame(
            [
                {'Filepath': 'a.tiff', 'X': 0.0, 'Y': 0.0},
                {'Filepath': 'b.tiff', 'X': 0.0, 'Y': 1.0},
            ]
        )
        result = Stitcher._simple_position_stitcher(tmp_path, df)
        assert result['status'] is True
        assert result['image'].dtype == np.uint16
        assert result['image'].shape == (64, 32)


class TestPositionAwareStitcher:
    def test_preserves_overlap_from_stage_positions(self, tmp_path):
        left = np.full((50, 50), 100, dtype=np.uint8)
        right = np.full((50, 50), 200, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'left.tiff'), left)
        cv2.imwrite(str(tmp_path / 'right.tiff'), right)

        fov = common_utils.get_field_of_view(
            focal_length=18.0,
            frame_size={'width': 50, 'height': 50},
            binning_size=1,
        )
        half_fov_mm = fov['width'] / 2 / 1000
        df = pd.DataFrame(
            [
                {'Filepath': 'left.tiff', 'X': 0.0, 'Y': 0.0, 'Objective': '10x Oly'},
                {'Filepath': 'right.tiff', 'X': -half_fov_mm, 'Y': 0.0, 'Objective': '10x Oly'},
            ]
        )

        result = Stitcher(has_turret=False)._position_stitcher(tmp_path, df)

        assert result['status'] is True
        assert result['image'].shape == (50, 75)
        assert result['image'][:, :25].mean() == pytest.approx(100)
        assert result['image'][:, 25:50].mean() == pytest.approx(150)
        assert result['image'][:, 50:].mean() == pytest.approx(200)

    def test_registered_stitch_recovers_neighbor_jitter(self):
        rng = np.random.default_rng(123)
        base = rng.integers(0, 255, (80, 180, 3), dtype=np.uint8)
        tile_w = 100
        tile_h = 80
        nominal_step = 75
        left_jitter = (0, 0)
        right_jitter = (4, -2)
        pad = 8
        padded = cv2.copyMakeBorder(base, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        left = padded[
            pad + left_jitter[1] : pad + left_jitter[1] + tile_h,
            pad + left_jitter[0] : pad + left_jitter[0] + tile_w,
        ]
        right = padded[
            pad + right_jitter[1] : pad + right_jitter[1] + tile_h,
            pad + nominal_step + right_jitter[0] : pad + nominal_step + right_jitter[0] + tile_w,
        ]

        _, registered = stitch_registered_tiles(
            [
                {'tile': left, 'x_px': 0, 'y_px': 0},
                {'tile': right, 'x_px': nominal_step, 'y_px': 0},
            ],
            output_shape=(tile_h, nominal_step + tile_w),
        )

        assert registered[1]['registration_offset_x_px'] == pytest.approx(4, abs=1)
        assert registered[1]['registration_offset_y_px'] == pytest.approx(-2, abs=1)

    def test_sparse_grid_registers_around_hole(self):
        """A 3x3 group with a top-middle hole: the top-right tile is reachable
        only via up from the row below. The old right/down-only walk stranded
        it at zero offset; the 4-neighbor flood must register it -- and every
        present tile except the anchor.
        """
        rng = np.random.default_rng(7)
        base = rng.integers(0, 255, (140, 140), dtype=np.uint8)

        def crop(gx, gy):
            return base[gy * 40 : gy * 40 + 60, gx * 40 : gx * 40 + 60].copy()

        tiles = []
        for gy in range(3):
            for gx in range(3):
                if (gx, gy) == (1, 0):  # top-middle hole
                    continue
                tiles.append({'tile': crop(gx, gy), 'x_px': gx * 40, 'y_px': gy * 40})

        registered = align_tile_positions(tiles, max_correction_px=8, min_overlap_px=16)

        by_pos = {(int(t['x_px']), int(t['y_px'])): t for t in registered}
        assert 'registration_score' in by_pos[(80, 0)], (
            'top-right tile (reachable only via up) must be registered by the '
            '4-neighbor flood, not stranded at zero offset'
        )
        for (x, y), t in by_pos.items():
            if (x, y) == (0, 0):  # the anchor carries no registration score
                continue
            assert 'registration_score' in t, f'tile at ({x},{y}) was not registered'

    def test_position_stitch_save_restores_false_color_and_metadata(self, tmp_path):
        """Saving via the primary position-aware path must carry the 8-bit
        PALETTE false-color colormap and acquisition metadata -- mirroring the
        simple-grid fallback -- not a bare grayscale, metadata-less TIFF.
        """
        red = np.full((50, 50), 120, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'r0.tiff'), red)
        cv2.imwrite(str(tmp_path / 'r1.tiff'), red)
        df = pd.DataFrame(
            [
                {'Filepath': 'r0.tiff', 'X': 0.0, 'Y': 0.0, 'Objective': '10x Oly', 'Color': 'Red'},
                {'Filepath': 'r1.tiff', 'X': 1.0, 'Y': 0.0, 'Objective': '10x Oly', 'Color': 'Red'},
            ]
        )
        out = pathlib.Path('stitched_red.tiff')

        result = Stitcher(has_turret=False)._position_stitcher(tmp_path, df, output_file_loc=out)

        assert result['status'] is True
        assert result['image'] is None  # subclass-wrote signal
        written = tmp_path / out
        assert written.exists()
        with tifffile.TiffFile(str(written)) as t:
            page = t.pages[0]
            assert page.photometric == tifffile.PHOTOMETRIC.PALETTE
            assert page.colormap is not None
        assert image_utils.read_postproc_input_metadata(written) is not None


def _reference_blend_float64(registered, sample):
    """The average-blend as it was before the float32 change: float64
    accumulator + weights, separate output canvas. Ground truth that the
    float32 in-place version must reproduce byte-for-byte."""
    min_x = min(int(t['registered_x_px']) for t in registered)
    min_y = min(int(t['registered_y_px']) for t in registered)
    max_x = max(int(t['registered_x_px']) + t['tile'].shape[1] for t in registered)
    max_y = max(int(t['registered_y_px']) + t['tile'].shape[0] for t in registered)
    if sample.ndim == 2:
        acc_shape = (max_y - min_y, max_x - min_x)
        weight_shape = acc_shape
    else:
        acc_shape = (max_y - min_y, max_x - min_x, sample.shape[2])
        weight_shape = (max_y - min_y, max_x - min_x, 1)
    acc = np.zeros(acc_shape, dtype=np.float64)
    wts = np.zeros(weight_shape, dtype=np.float64)
    for t in registered:
        image = t['tile']
        x0 = int(t['registered_x_px']) - min_x
        y0 = int(t['registered_y_px']) - min_y
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1 = min(acc_shape[1], x0 + image.shape[1])
        dst_y1 = min(acc_shape[0], y0 + image.shape[0])
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            continue
        sx0, sy0 = dst_x0 - x0, dst_y0 - y0
        acc[dst_y0:dst_y1, dst_x0:dst_x1] += image[
            sy0 : sy0 + (dst_y1 - dst_y0), sx0 : sx0 + (dst_x1 - dst_x0)
        ].astype(np.float64)
        wts[dst_y0:dst_y1, dst_x0:dst_x1] += 1.0
    out = np.zeros(acc_shape, dtype=np.float64)
    np.divide(acc, wts, out=out, where=wts > 0)
    if np.issubdtype(sample.dtype, np.integer):
        info = np.iinfo(sample.dtype)
        out = np.clip(out, info.min, info.max)
    return out.astype(sample.dtype)


@pytest.mark.parametrize('dtype,high', [(np.uint8, 255), (np.uint16, 65535)])
@pytest.mark.parametrize('channels', [1, 3])
def test_float32_blend_is_byte_identical_to_float64(dtype, high, channels):
    # Two overlapping featured tiles cut from a common base so registration
    # is deterministic and the overlap column is genuinely 2-tile-averaged.
    rng = np.random.default_rng(99)
    shape = (80, 180, channels) if channels == 3 else (80, 180)
    base = rng.integers(0, high, shape, dtype=dtype)
    tile_w, tile_h, step = 100, 80, 75
    left = base[:tile_h, :tile_w].copy()
    right = base[:tile_h, step : step + tile_w].copy()

    out, registered = stitch_registered_tiles(
        [{'tile': left, 'x_px': 0, 'y_px': 0}, {'tile': right, 'x_px': step, 'y_px': 0}],
        output_shape=(tile_h, step + tile_w),
    )
    reference = _reference_blend_float64(registered, left)
    assert np.array_equal(out, reference), 'float32 blend diverged from float64'
