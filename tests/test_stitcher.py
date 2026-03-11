# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for image stitcher modules — both legacy (image_stitcher.py) and current (stitcher.py)."""

import pathlib
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Legacy image_stitcher.py — inlined functions (replaced color_transfer, imutils)
# ---------------------------------------------------------------------------

from modules.image_stitcher import _image_stats, _color_transfer, _grab_contours, zoom_frame


class TestImageStats:
    """Test _image_stats — computes L*a*b* channel statistics."""

    def test_uniform_image(self):
        # Uniform gray → known LAB values
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        stats = _image_stats(lab)
        assert len(stats) == 6  # (lMean, lStd, aMean, aStd, bMean, bStd)
        l_mean, l_std, a_mean, a_std, b_mean, b_std = stats
        assert l_std == 0.0  # uniform → zero std
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
    """Test _color_transfer — LAB color distribution transfer (Reinhard et al.)."""

    def test_output_shape_matches_target(self):
        source = np.full((50, 50, 3), 200, dtype=np.uint8)
        target = np.full((80, 60, 3), 100, dtype=np.uint8)
        result = _color_transfer(source, target)
        assert result.shape == target.shape
        assert result.dtype == np.uint8

    def test_identical_images_unchanged(self):
        # Uniform images have zero std — division guard returns identity-like result
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = _color_transfer(img.copy(), img.copy())
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        # With zero-std guard, uniform → uniform (LAB round-trip may shift slightly)
        assert np.allclose(result, result[0, 0], atol=1)  # all pixels same

    def test_varied_identical_images(self):
        # Non-uniform identical images → result ≈ input
        rng = np.random.RandomState(7)
        img = rng.randint(80, 180, (50, 50, 3), dtype=np.uint8)
        result = _color_transfer(img.copy(), img.copy())
        assert np.allclose(result, img, atol=5)

    def test_different_colors_shifts_target(self):
        # Bright source, dark target → result should be brighter than original target
        source = np.full((50, 50, 3), 220, dtype=np.uint8)
        target = np.full((50, 50, 3), 50, dtype=np.uint8)
        result = _color_transfer(source, target)
        # Result's mean brightness should be closer to source than original target
        assert result.mean() > target.mean()

    def test_handles_color_images(self):
        rng = np.random.RandomState(123)
        source = rng.randint(50, 200, (30, 30, 3), dtype=np.uint8)
        target = rng.randint(50, 200, (30, 30, 3), dtype=np.uint8)
        result = _color_transfer(source, target)
        assert result.shape == target.shape
        # Values should be valid uint8
        assert result.min() >= 0
        assert result.max() <= 255


class TestGrabContours:
    """Test _grab_contours — OpenCV 4.x contour extraction."""

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


class TestZoomFrame:
    """Test zoom_frame — crops stitched image to content area."""

    def test_crops_black_border(self):
        # Create an image with content in the center and black border
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[40:160, 60:240] = 128  # gray content area
        result = zoom_frame(img)
        # Result should be smaller than input (border removed)
        assert result.shape[0] < img.shape[0]
        assert result.shape[1] < img.shape[1]
        # Content should be preserved
        assert result.mean() > 0

    def test_full_content_image(self):
        # No border → result ≈ input size (only the 10px border padding matters)
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = zoom_frame(img)
        # Should be close to original dimensions (±20 from the added border)
        assert abs(result.shape[0] - 100) <= 22
        assert abs(result.shape[1] - 100) <= 22


# ---------------------------------------------------------------------------
# Current stitcher.py — _simple_position_stitcher
# ---------------------------------------------------------------------------

from modules.stitcher import Stitcher


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
        return pd.DataFrame([
            {'Filepath': 'tile_0_0.tiff', 'X': 0.0, 'Y': 0.0},
            {'Filepath': 'tile_1_0.tiff', 'X': 1.0, 'Y': 0.0},
            {'Filepath': 'tile_0_1.tiff', 'X': 0.0, 'Y': 1.0},
            {'Filepath': 'tile_1_1.tiff', 'X': 1.0, 'Y': 1.0},
        ])

    def test_output_dimensions(self, tile_dir, tile_df):
        result = Stitcher._simple_position_stitcher(tile_dir, tile_df)
        assert result['status'] is True
        img = result['image']
        # 2x2 grid of 50x50 tiles → 100x100
        assert img.shape == (100, 100)

    def test_center_metadata(self, tile_dir, tile_df):
        result = Stitcher._simple_position_stitcher(tile_dir, tile_df)
        center = result['metadata']['center']
        assert center['x'] == 0.5  # mean of [0.0, 1.0]
        assert center['y'] == 0.5

    def test_all_pixels_filled(self, tile_dir, tile_df):
        result = Stitcher._simple_position_stitcher(tile_dir, tile_df)
        img = result['image']
        # No black pixels — all tiles have nonzero values
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
        df = pd.DataFrame([
            {'Filepath': 'a.tiff', 'X': 0.0, 'Y': 0.0},
            {'Filepath': 'b.tiff', 'X': 1.0, 'Y': 0.0},
        ])
        result = Stitcher._simple_position_stitcher(tmp_path, df)
        assert result['status'] is True
        img = result['image']
        assert img.shape == (40, 80, 3)

    def test_3x1_grid(self, tmp_path):
        """3 tiles in a row."""
        for i in range(3):
            tile = np.full((30, 30), (i + 1) * 60, dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f't{i}.tiff'), tile)
        df = pd.DataFrame([
            {'Filepath': f't{i}.tiff', 'X': float(i), 'Y': 0.0}
            for i in range(3)
        ])
        result = Stitcher._simple_position_stitcher(tmp_path, df)
        assert result['status'] is True
        assert result['image'].shape == (30, 90)

    def test_16bit_tiles(self, tmp_path):
        """16-bit grayscale tiles."""
        t1 = np.full((32, 32), 1000, dtype=np.uint16)
        t2 = np.full((32, 32), 50000, dtype=np.uint16)
        cv2.imwrite(str(tmp_path / 'a.tiff'), t1)
        cv2.imwrite(str(tmp_path / 'b.tiff'), t2)
        df = pd.DataFrame([
            {'Filepath': 'a.tiff', 'X': 0.0, 'Y': 0.0},
            {'Filepath': 'b.tiff', 'X': 0.0, 'Y': 1.0},
        ])
        result = Stitcher._simple_position_stitcher(tmp_path, df)
        assert result['status'] is True
        assert result['image'].dtype == np.uint16
        assert result['image'].shape == (64, 32)
