# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for stitcher modules -- stitch_algorithms.py (feature-based) and stitcher.py (grid-based)."""

import ast
import logging
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
    estimate_phase_offset,
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
        _l_mean, l_std, _a_mean, a_std, _b_mean, b_std = stats
        assert l_std == 0.0  # uniform -> zero std
        assert a_std == 0.0
        assert b_std == 0.0

    def test_random_image_has_nonzero_std(self):
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        stats = _image_stats(lab)
        _l_mean, l_std, _a_mean, a_std, _b_mean, b_std = stats
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
        x, _y, w, _h = cv2.boundingRect(contours[0])
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

from modules.stitching_core import channel_aware_stitcher, infer_stage_overlap
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

    def test_first_tile_not_opened_repeatedly(self, tile_dir, tile_df, monkeypatch):
        """The canvas-sizing geometry probe reads a header only, so the first
        tile is opened once for geometry plus once for its placement -- not
        re-decoded a third/fourth time. Each other tile is opened once. The
        stitched output stays byte-identical to the unspied run."""
        from modules import image_utils

        baseline = Stitcher._simple_position_stitcher(tile_dir, tile_df)['image']

        real_tifffile = image_utils.tf.TiffFile
        opens = {}

        def counting_tifffile(arg, *args, **kwargs):
            opens[pathlib.Path(arg).name] = opens.get(pathlib.Path(arg).name, 0) + 1
            return real_tifffile(arg, *args, **kwargs)

        monkeypatch.setattr(image_utils.tf, 'TiffFile', counting_tifffile)
        result = Stitcher._simple_position_stitcher(tile_dir, tile_df)

        # df.iloc[0] is the geometry-probe + placement tile; up to one open each.
        assert opens['tile_0_0.tiff'] == 2
        assert opens['tile_1_0.tiff'] == 1
        assert opens['tile_0_1.tiff'] == 1
        assert opens['tile_1_1.tiff'] == 1
        assert np.array_equal(result['image'], baseline)

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

    def test_channel_aware_bf_output_shape_and_dtype(self, tmp_path):
        for ix, x in enumerate((0.0, 1.0)):
            for iy, y in enumerate((0.0, 1.0)):
                tile = np.full((12, 10), ix * 40 + iy * 20 + 50, dtype=np.uint8)
                tifffile.imwrite(str(tmp_path / f'bf_{ix}_{iy}.tiff'), tile)

        df = pd.DataFrame(
            [
                {
                    'Filepath': f'bf_{ix}_{iy}.tiff',
                    'X': x,
                    'Y': y,
                    'Objective': '10x Oly',
                    'Color': 'BF',
                    'Well': 'A1',
                    'Tile Group ID': 1,
                }
                for ix, x in enumerate((0.0, 1.0))
                for iy, y in enumerate((0.0, 1.0))
            ]
        )

        result = channel_aware_stitcher(tmp_path, df, pixel_size_um=None)

        assert result['status'] is True
        assert result['image'].shape == (24, 20)
        assert result['image'].dtype == np.uint8

    def test_channel_aware_fluorescence_output_shape_and_dtype(self, tmp_path):
        for ix, x in enumerate((0.0, 1.0)):
            for iy, y in enumerate((0.0, 1.0)):
                tile = np.full((8, 6), ix * 1000 + iy * 2000 + 100, dtype=np.uint16)
                tifffile.imwrite(str(tmp_path / f'green_{ix}_{iy}.tiff'), tile)

        df = pd.DataFrame(
            [
                {
                    'Filepath': f'green_{ix}_{iy}.tiff',
                    'X': x,
                    'Y': y,
                    'Objective': '10x Oly',
                    'Color': 'Green',
                    'Well': 'B2',
                    'Tile Group ID': 2,
                }
                for ix, x in enumerate((0.0, 1.0))
                for iy, y in enumerate((0.0, 1.0))
            ]
        )

        result = channel_aware_stitcher(tmp_path, df, pixel_size_um=None)

        assert result['status'] is True
        assert result['image'].shape == (16, 12)
        assert result['image'].dtype == np.uint16

    def test_load_folder_surfaces_degraded_success_to_callers(self, tmp_path, monkeypatch):
        rows = []
        for tile_idx, x in enumerate((0.0, 1.0)):
            row = {
                'Filepath': f'tile_{tile_idx}.tiff',
                'Timestamp': '2026-06-19T00:00:00',
                'Name': 'scan_BF',
                'Scan Count': 0,
                'X': x,
                'Y': 0.0,
                'Z': 0.0,
                'Z-Slice': 0,
                'Well': 'A1',
                'Color': 'BF',
                'Objective': '10x Oly',
                'Tile Group ID': 1,
                'Tile': str(tile_idx),
                'Custom Step': False,
                'Raw': True,
            }
            for post_function in common_utils.PostFunction.list_values():
                row[post_function] = False
            rows.append(row)

        class FakePostRecord:
            def file_exists_in_records(self, filepath):
                return False

            def complete(self):
                pass

        class FakeHelper:
            def load_folder(self, path, tiling_configs_file_loc):
                return {
                    'status': True,
                    'images_df': pd.DataFrame(rows),
                    'root_path': tmp_path,
                    'protocol_post_record': FakePostRecord(),
                    'protocol': None,
                }

            def generate_output_dir_name(self, record):
                return 'Stitched'

        stitcher = Stitcher(has_turret=False)
        stitcher._post_processing_helper = FakeHelper()
        monkeypatch.setattr(stitcher, '_generate_filename', lambda df, **kwargs: 'stitched.tiff')
        monkeypatch.setattr(stitcher, '_add_record', lambda **kwargs: None)

        def degraded_group_algorithm(**kwargs):
            from modules.protocol_post_processing_result import PostProcResult

            return PostProcResult.ok(
                significant_bits=16,
                record_metadata={
                    'center': {'x': 0.5, 'y': 0.0},
                    'algorithm': 'simple_position_stitcher',
                    'fallback_from': 'bf_feature_stitcher',
                    'fallback_reason': 'bf_feature_stitcher: BF feature stitching failed',
                },
            )

        monkeypatch.setattr(stitcher, '_group_algorithm', degraded_group_algorithm)

        result = stitcher.load_folder(tmp_path, tmp_path / 'tiling.json')

        assert result['status'] is True
        assert result['degraded'] is True
        assert 'degraded output' in result['message']
        assert result['degraded_outputs'] == [
            {
                'filepath': 'Stitched/stitched.tiff',
                'algorithm': 'simple_position_stitcher',
                'fallback_from': 'bf_feature_stitcher',
                'fallback_reason': 'bf_feature_stitcher: BF feature stitching failed',
            }
        ]

    def test_stitcher_callback_has_degraded_operator_surface(self):
        source = (
            pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'post_processing.py'
        ).read_text()
        tree = ast.parse(source)
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'stitcher_callback'
        )

        degraded_branch = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.If) and 'degraded' in ast.unparse(node.test)
        ]

        assert degraded_branch, 'stitcher_callback must branch on result["degraded"]'
        branch_source = ast.unparse(degraded_branch[0])
        assert 'geometry-only fallback' in branch_source
        assert 'popup.text' in branch_source

    def test_stitcher_popup_never_surfaces_raw_worker_message(self):
        source = (
            pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'post_processing.py'
        ).read_text()
        tree = ast.parse(source)
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'stitcher_callback'
        )
        callback_source = ast.unparse(callback)
        assert 'result[\'message\']' not in callback_source
        assert 'Support > Logs' in callback_source

    def test_stitch_ui_explains_modes_and_time_estimation(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        kv_source = (root / 'ui' / 'lumaviewpro.kv').read_text()
        ui_source = (root / 'ui' / 'post_processing.py').read_text()
        assert 'Fast Preview: bounded FFT geometry check' in kv_source
        assert 'Quality: bounded local registration' in kv_source
        assert 'Estimated remaining time' in (root / 'modules' / 'protocol_post_processor.py').read_text()
        assert 'stitching_mode' in ui_source

    def test_bf_quality_route_never_uses_unconstrained_feature_stitcher(
        self,
        tmp_path,
        monkeypatch,
    ):
        df = pd.DataFrame(
            [
                {
                    'Filepath': 'a.tiff',
                    'X': 0.0,
                    'Y': 0.0,
                    'Objective': '10x Oly',
                    'Color': 'BF',
                },
                {
                    'Filepath': 'b.tiff',
                    'X': 0.050,
                    'Y': 0.0,
                    'Objective': '10x Oly',
                    'Color': 'BF',
                },
            ]
        )
        calls = []

        def fail(name):
            def runner(*args, **kwargs):
                calls.append(name)
                return {
                    'status': False,
                    'error': f'{name} failed',
                    'image': None,
                    'metadata': {'center': {'x': 0.5, 'y': 0.0}},
                }

            return runner

        def simple_success(*args, **kwargs):
            calls.append('simple_position_stitcher')
            return {
                'status': True,
                'error': None,
                'image': np.zeros((4, 8), dtype=np.uint8),
                'metadata': {
                    'center': {'x': 0.5, 'y': 0.0},
                    'algorithm': 'simple_position_stitcher',
                },
            }

        monkeypatch.setattr(
            'modules.stitching_core._read_tile_with_depth',
            lambda *_: (np.ones((100, 100), dtype=np.uint8), 8),
        )
        monkeypatch.setattr(
            'modules.stitching_core.bf_feature_stitcher',
            lambda *_args, **_kwargs: pytest.fail('BF feature stitch must not be automatic'),
        )
        monkeypatch.setattr('modules.stitching_core.overlap_stitcher', fail('overlap_stitcher'))
        monkeypatch.setattr(
            'modules.stitching_core.stage_position_stitcher',
            fail('stage_position_stitcher'),
        )
        monkeypatch.setattr('modules.stitching_core.simple_position_stitcher', simple_success)

        result = channel_aware_stitcher(tmp_path, df, pixel_size_um=1.0)

        assert result['status'] is True
        assert calls == [
            'overlap_stitcher',
            'stage_position_stitcher',
            'simple_position_stitcher',
        ]
        assert result['metadata']['algorithm'] == 'simple_position_stitcher'
        assert result['metadata']['fallback_from'] == 'quality_local_ncc'

    def test_fallback_logs_operator_visible_warning(
        self,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        df = pd.DataFrame(
            [
                {
                    'Filepath': 'a.tiff',
                    'X': 0.0,
                    'Y': 0.0,
                    'Objective': '10x Oly',
                    'Color': 'Green',
                    'Well': 'A1',
                    'Tile Group ID': 3,
                },
                {
                    'Filepath': 'b.tiff',
                    'X': 0.050,
                    'Y': 0.0,
                    'Objective': '10x Oly',
                    'Color': 'Green',
                    'Well': 'A1',
                    'Tile Group ID': 3,
                },
            ]
        )

        def overlap_fail(*args, **kwargs):
            return {
                'status': False,
                'error': 'registration failed',
                'image': None,
                'metadata': {'center': {'x': 0.5, 'y': 0.0}},
            }

        def stage_success(*args, **kwargs):
            return {
                'status': True,
                'error': None,
                'image': np.zeros((4, 8), dtype=np.uint8),
                'metadata': {
                    'center': {'x': 0.5, 'y': 0.0},
                    'algorithm': 'stage_position_stitcher',
                },
            }

        monkeypatch.setattr('modules.stitching_core.overlap_stitcher', overlap_fail)
        monkeypatch.setattr('modules.stitching_core.stage_position_stitcher', stage_success)
        monkeypatch.setattr(
            'modules.stitching_core._read_tile_with_depth',
            lambda *_: (np.ones((100, 100), dtype=np.uint8), 8),
        )

        with caplog.at_level(logging.WARNING, logger='LVP.modules.stitching_core'):
            result = channel_aware_stitcher(tmp_path, df, pixel_size_um=1.0)

        assert result['status'] is True
        assert result['metadata']['fallback_from'] == 'quality_local_ncc'
        assert 'using stage_position_stitcher for well=A1 color=Green tile_group=3' in caplog.text

    def test_fluorescence_route_uses_overlap_then_stage_then_simple(
        self,
        tmp_path,
        monkeypatch,
    ):
        df = pd.DataFrame(
            [
                {
                    'Filepath': 'a.tiff',
                    'X': 0.0,
                    'Y': 0.0,
                    'Objective': '10x Oly',
                    'Color': 'Green',
                },
                {
                    'Filepath': 'b.tiff',
                    'X': 0.050,
                    'Y': 0.0,
                    'Objective': '10x Oly',
                    'Color': 'Green',
                },
            ]
        )
        calls = []

        def fail(name):
            def runner(*args, **kwargs):
                calls.append(name)
                return {
                    'status': False,
                    'error': f'{name} failed',
                    'image': None,
                    'metadata': {'center': {'x': 0.5, 'y': 0.0}},
                }

            return runner

        def simple_success(*args, **kwargs):
            calls.append('simple_position_stitcher')
            return {
                'status': True,
                'error': None,
                'image': np.zeros((4, 8), dtype=np.uint8),
                'metadata': {
                    'center': {'x': 0.5, 'y': 0.0},
                    'algorithm': 'simple_position_stitcher',
                },
            }

        def bf_should_not_run(*args, **kwargs):
            raise AssertionError('fluorescence should not use BF feature stitching')

        monkeypatch.setattr('modules.stitching_core.bf_feature_stitcher', bf_should_not_run)
        monkeypatch.setattr('modules.stitching_core.overlap_stitcher', fail('overlap_stitcher'))
        monkeypatch.setattr(
            'modules.stitching_core.stage_position_stitcher',
            fail('stage_position_stitcher'),
        )
        monkeypatch.setattr('modules.stitching_core.simple_position_stitcher', simple_success)
        monkeypatch.setattr(
            'modules.stitching_core._read_tile_with_depth',
            lambda *_: (np.ones((100, 100), dtype=np.uint8), 8),
        )

        result = channel_aware_stitcher(tmp_path, df, pixel_size_um=1.0)

        assert result['status'] is True
        assert calls == ['overlap_stitcher', 'stage_position_stitcher', 'simple_position_stitcher']
        assert result['metadata']['algorithm'] == 'simple_position_stitcher'
        assert result['metadata']['fallback_from'] == 'quality_local_ncc'

    def test_fast_preview_route_uses_fft_then_simple_only(
        self,
        tmp_path,
        monkeypatch,
    ):
        df = pd.DataFrame(
            [
                {
                    'Filepath': 'a.tiff',
                    'X': 0.0,
                    'Y': 0.0,
                    'Objective': '10x Oly',
                    'Color': 'Green',
                },
                {
                    'Filepath': 'b.tiff',
                    'X': 0.05,
                    'Y': 0.0,
                    'Objective': '10x Oly',
                    'Color': 'Green',
                },
            ]
        )
        calls = []

        def fft_fail(*args, **kwargs):
            calls.append('fft_phase_stitcher')
            return {
                'status': False,
                'error': 'fft failed',
                'image': None,
                'metadata': {'center': {'x': 0.5, 'y': 0.0}},
            }

        def simple_success(*args, **kwargs):
            calls.append('simple_position_stitcher')
            return {
                'status': True,
                'error': None,
                'image': np.zeros((4, 8), dtype=np.uint8),
                'metadata': {
                    'center': {'x': 0.5, 'y': 0.0},
                    'algorithm': 'simple_position_stitcher',
                },
            }

        def stage_fail(*args, **kwargs):
            calls.append('stage_position_stitcher')
            return {
                'status': False,
                'error': 'stage failed',
                'image': None,
                'metadata': {'center': {'x': 0.025, 'y': 0.0}},
            }

        def slow_quality_should_not_run(*args, **kwargs):
            raise AssertionError('fast preview should not use current LVP quality registration')

        monkeypatch.setattr('modules.stitching_core.fft_phase_stitcher', fft_fail)
        monkeypatch.setattr('modules.stitching_core.overlap_stitcher', slow_quality_should_not_run)
        monkeypatch.setattr('modules.stitching_core.stage_position_stitcher', stage_fail)
        monkeypatch.setattr('modules.stitching_core.simple_position_stitcher', simple_success)
        monkeypatch.setattr(
            'modules.stitching_core._read_tile_with_depth',
            lambda *_args, **_kwargs: (np.zeros((100, 100), dtype=np.uint8), 8),
        )

        result = channel_aware_stitcher(
            tmp_path,
            df,
            pixel_size_um=1.0,
            stitching_mode=Stitcher.FAST_PREVIEW_MODE,
        )

        assert result['status'] is True
        assert calls == ['fft_phase_stitcher', 'stage_position_stitcher', 'simple_position_stitcher']
        assert result['metadata']['algorithm'] == 'simple_position_stitcher'
        assert result['metadata']['fallback_from'] == 'fast_fft_phase'

    def test_fast_preview_filename_is_distinct_from_quality(self):
        stitcher = Stitcher(has_turret=False)
        df = pd.DataFrame(
            [
                {
                    'Name': 'A1_Green_T00',
                    'Tile': 'T00',
                    'Color': 'Green',
                    'Objective': '10x Oly',
                    'Well': 'A1',
                    'Label': '',
                    'Z-Slice': -1,
                    'Scan Count': 0,
                }
            ]
        )

        quality = stitcher._generate_filename(df, capture_root='')
        preview = stitcher._generate_filename(
            df,
            stitching_mode=Stitcher.FAST_PREVIEW_MODE,
            capture_root='',
        )

        assert quality != preview
        assert 'FastPreview' in preview


class TestPositionAwareStitcher:
    def test_phase_offset_recovers_neighbor_jitter(self):
        rng = np.random.default_rng(321)
        base = rng.integers(0, 255, (80, 180), dtype=np.uint8)
        tile_w = 100
        tile_h = 80
        nominal_step = 75
        jitter = (3, -2)
        pad = 8
        padded = cv2.copyMakeBorder(base, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        left = padded[pad : pad + tile_h, pad : pad + tile_w]
        right = padded[
            pad + jitter[1] : pad + jitter[1] + tile_h,
            pad + nominal_step + jitter[0] : pad + nominal_step + jitter[0] + tile_w,
        ]

        corr_x, corr_y, score = estimate_phase_offset(
            left,
            right,
            nominal_dx=nominal_step,
            nominal_dy=0,
        )

        assert corr_x == pytest.approx(jitter[0], abs=1)
        assert corr_y == pytest.approx(jitter[1], abs=1)
        assert score > 0

    def test_position_stitcher_passes_shared_output_shape(self, tmp_path, monkeypatch):
        # The stitched canvas size must come from the nominal stage grid --
        # identical for every channel / Z-slice of a tile-group -- not from each
        # layer's own content registration. Otherwise composite / z-projection
        # of the per-layer outputs get mismatched shapes and cannot combine.
        import modules.stitcher as stitcher_mod

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

        recorded = {}
        real = stitcher_mod.stitch_registered_tiles

        def _spy(tiles, **kwargs):
            recorded['output_shape'] = kwargs.get('output_shape')
            return real(tiles, **kwargs)

        monkeypatch.setattr(stitcher_mod, 'stitch_registered_tiles', _spy)
        result = Stitcher(has_turret=False)._position_stitcher(tmp_path, df)

        assert result['status'] is True
        assert recorded['output_shape'] is not None, (
            'position stitcher must pass a shared nominal output_shape so every '
            'layer of a tile-group stitches to one canvas'
        )
        assert tuple(result['image'].shape[:2]) == recorded['output_shape']

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


class TestStageConstrainedStitchModes:
    """Production routing must never use unconstrained panorama matching."""

    def test_infers_zero_overlap_from_stage_spacing(self):
        frame = pd.DataFrame(
            [
                {'X': 0.0, 'Y': 0.0},
                {'X': 0.100, 'Y': 0.0},
                {'X': 0.0, 'Y': 0.100},
                {'X': 0.100, 'Y': 0.100},
            ]
        )
        overlap = infer_stage_overlap(frame, pixel_size_um=1.0, tile_shape=(100, 100))
        assert overlap['has_overlap'] is False
        assert overlap['x_percent'] == pytest.approx(0.0)
        assert overlap['y_percent'] == pytest.approx(0.0)

    def test_quality_at_zero_overlap_uses_geometry_without_feature_matching(
        self, tmp_path, monkeypatch
    ):
        frame = pd.DataFrame(
            [
                {'Filepath': 'a.tiff', 'X': 0.0, 'Y': 0.0, 'Color': 'BF'},
                {'Filepath': 'b.tiff', 'X': 0.100, 'Y': 0.0, 'Color': 'BF'},
            ]
        )
        calls = []

        def should_not_run(*args, **kwargs):
            raise AssertionError('unconstrained registration must not run at 0% overlap')

        def stage_success(*args, **kwargs):
            calls.append('stage')
            return {
                'status': True,
                'error': None,
                'image': np.zeros((10, 20), dtype=np.uint8),
                'significant_bits': 8,
                'metadata': {'center': {'x': 0.05, 'y': 0.0}},
            }

        monkeypatch.setattr('modules.stitching_core._read_tile_with_depth', lambda *_: (np.ones((100, 100), dtype=np.uint8), 8))
        monkeypatch.setattr('modules.stitching_core.bf_feature_stitcher', should_not_run)
        monkeypatch.setattr('modules.stitching_core.overlap_stitcher', should_not_run)
        monkeypatch.setattr('modules.stitching_core.stage_position_stitcher', stage_success)

        result = channel_aware_stitcher(
            tmp_path, frame, pixel_size_um=1.0, stitching_mode='quality'
        )

        assert result['status'] is True
        assert calls == ['stage']
        assert result['metadata']['algorithm'] == 'stage_position_stitcher'
        assert result['metadata']['overlap']['has_overlap'] is False

    def test_source_preserving_mode_never_averages_overlap_pixels(self):
        first = np.full((4, 4), 10, dtype=np.uint8)
        second = np.full((4, 4), 200, dtype=np.uint8)
        output, _ = stitch_registered_tiles(
            [
                {'tile': first, 'x_px': 0, 'y_px': 0},
                {'tile': second, 'x_px': 2, 'y_px': 0},
            ],
            output_shape=(4, 6),
            blend_mode='source_preserving',
        )
        assert np.all(output[:, :4] == 10)
        assert np.all(output[:, 4:] == 200)

    def test_quality_overlap_route_uses_bounded_local_registration(self, tmp_path):
        rng = np.random.default_rng(123)
        base = rng.integers(0, 255, (100, 150), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'left.tiff'), base[:, :100])
        cv2.imwrite(str(tmp_path / 'right.tiff'), base[:, 50:150])
        frame = pd.DataFrame(
            [
                {'Filepath': 'left.tiff', 'X': 0.050, 'Y': 0.0, 'Color': 'BF'},
                {'Filepath': 'right.tiff', 'X': 0.0, 'Y': 0.0, 'Color': 'BF'},
            ]
        )

        result = channel_aware_stitcher(tmp_path, frame, pixel_size_um=1.0)

        assert result['status'] is True
        assert result['metadata']['algorithm'] == 'quality_local_ncc'
        assert result['metadata']['pixel_policy'] == 'source_preserving'
        assert result['metadata']['overlap']['has_overlap'] is True

    def test_fast_preview_overlap_route_uses_fft_not_quality_ncc(self, tmp_path, monkeypatch):
        frame = pd.DataFrame(
            [
                {'Filepath': 'a.tiff', 'X': 0.050, 'Y': 0.0, 'Color': 'BF'},
                {'Filepath': 'b.tiff', 'X': 0.0, 'Y': 0.0, 'Color': 'BF'},
            ]
        )
        calls = []

        def fft_success(*args, **kwargs):
            calls.append('fft')
            return {
                'status': True,
                'error': None,
                'image': np.zeros((100, 150), dtype=np.uint8),
                'significant_bits': 8,
                'metadata': {'center': {'x': 0.025, 'y': 0.0}},
            }

        monkeypatch.setattr(
            'modules.stitching_core._read_tile_with_depth',
            lambda *_: (np.ones((100, 100), dtype=np.uint8), 8),
        )
        monkeypatch.setattr('modules.stitching_core.fft_phase_stitcher', fft_success)
        monkeypatch.setattr(
            'modules.stitching_core.overlap_stitcher',
            lambda *_args, **_kwargs: pytest.fail('Fast Preview must not use Quality NCC'),
        )

        result = channel_aware_stitcher(
            tmp_path, frame, pixel_size_um=1.0, stitching_mode='fast_preview'
        )

        assert result['status'] is True
        assert calls == ['fft']
        assert result['metadata']['algorithm'] == 'fast_fft_phase'
        assert result['metadata']['mode'] == 'fast_preview'

    def test_stitcher_ignores_prebuilt_composites(self):
        stitcher = Stitcher(has_turret=False)
        frame = pd.DataFrame(
            [
                {
                    'Filepath': 'raw_bf.tiff',
                    'Composite': False,
                    'Stitched': False,
                    'ZProject': False,
                    'Video': False,
                    'Hyperstack': False,
                },
                {
                    'Filepath': 'derived_composite.tiff',
                    'Composite': True,
                    'Stitched': False,
                    'ZProject': False,
                    'Video': False,
                    'Hyperstack': False,
                },
            ]
        )

        filtered = stitcher._filter_ignored_types(frame)

        assert filtered['Filepath'].tolist() == ['raw_bf.tiff']
