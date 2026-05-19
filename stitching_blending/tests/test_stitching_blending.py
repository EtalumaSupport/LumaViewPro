from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd

from stitching_blending.scripts.alignment import estimate_shift
from stitching_blending.scripts.blending import blend_tiles
from stitching_blending.scripts.metrics import compute_metrics, write_metrics
from stitching_blending.scripts.optimization import optimize_tile_positions
from stitching_blending.scripts.pipeline import run_stitching_pipeline
from stitching_blending.scripts.synthetic import SyntheticConfig, generate_synthetic_dataset


def test_no_overlap_montage_layout():
    tiles = {
        'a': np.full((4, 4), 10, dtype=np.uint8),
        'b': np.full((4, 4), 20, dtype=np.uint8),
        'c': np.full((4, 4), 30, dtype=np.uint8),
        'd': np.full((4, 4), 40, dtype=np.uint8),
    }
    placements = pd.DataFrame(
        [
            {'tile_id': 'a', 'metadata_x_px': 0, 'metadata_y_px': 0, 'x_px': 0, 'y_px': 0},
            {'tile_id': 'b', 'metadata_x_px': 4, 'metadata_y_px': 0, 'x_px': 4, 'y_px': 0},
            {'tile_id': 'c', 'metadata_x_px': 0, 'metadata_y_px': 4, 'x_px': 0, 'y_px': 4},
            {'tile_id': 'd', 'metadata_x_px': 4, 'metadata_y_px': 4, 'x_px': 4, 'y_px': 4},
        ]
    )
    stitched, _ = blend_tiles(tiles, placements, mode='average')
    assert stitched.shape == (8, 8)
    assert np.all(stitched[:4, :4] == 10)
    assert np.all(stitched[:4, 4:] == 20)
    assert np.all(stitched[4:, :4] == 30)
    assert np.all(stitched[4:, 4:] == 40)


def test_overlap_output_size():
    tiles = {
        'a': np.ones((10, 10), dtype=np.uint8),
        'b': np.ones((10, 10), dtype=np.uint8),
        'c': np.ones((10, 10), dtype=np.uint8),
        'd': np.ones((10, 10), dtype=np.uint8),
    }
    placements = pd.DataFrame(
        [
            {'tile_id': 'a', 'metadata_x_px': 0, 'metadata_y_px': 0, 'x_px': 0, 'y_px': 0},
            {'tile_id': 'b', 'metadata_x_px': 8, 'metadata_y_px': 0, 'x_px': 8, 'y_px': 0},
            {'tile_id': 'c', 'metadata_x_px': 0, 'metadata_y_px': 8, 'x_px': 0, 'y_px': 8},
            {'tile_id': 'd', 'metadata_x_px': 8, 'metadata_y_px': 8, 'x_px': 8, 'y_px': 8},
        ]
    )
    stitched, _ = blend_tiles(tiles, placements, mode='average')
    assert stitched.shape == (18, 18)


def test_average_blending_does_not_overwrite():
    tiles = {
        'left': np.full((4, 4), 10, dtype=np.uint8),
        'right': np.full((4, 4), 30, dtype=np.uint8),
    }
    placements = pd.DataFrame(
        [
            {'tile_id': 'left', 'metadata_x_px': 0, 'metadata_y_px': 0, 'x_px': 0, 'y_px': 0},
            {'tile_id': 'right', 'metadata_x_px': 2, 'metadata_y_px': 0, 'x_px': 2, 'y_px': 0},
        ]
    )
    stitched, _ = blend_tiles(tiles, placements, mode='average')
    assert np.all(stitched[:, 2:4] == 20)


def test_phase_correlation_recovers_known_shift():
    image = np.zeros((64, 64), dtype=np.float32)
    image[18:38, 20:45] = 1.0
    image = cv2.GaussianBlur(image, (5, 5), 0)
    matrix = np.float32([[1, 0, 4], [0, 1, -3]])
    shifted = cv2.warpAffine(image, matrix, (64, 64))
    estimate = estimate_shift(image, shifted, max_shift_px=8, phase_confidence_threshold=0.01)
    assert estimate['method'] == 'phase_correlation'
    assert abs(float(estimate['dx_px']) - 4) < 0.5
    assert abs(float(estimate['dy_px']) + 3) < 0.5


def test_ncc_fallback_works_on_controlled_example():
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:20, 10:18] = 1.0
    shifted = np.zeros_like(image)
    shifted[10:22, 7:15] = 1.0
    estimate = estimate_shift(image, shifted, max_shift_px=5, phase_confidence_threshold=2.0)
    assert estimate['method'] == 'ncc'
    assert estimate['accepted'] is True
    assert estimate['dx_px'] == -3
    assert estimate['dy_px'] == 2


def test_global_optimization_preserves_or_improves_consistency():
    metadata = pd.DataFrame(
        [
            {'tile_id': 'a', 'metadata_x_px': 0.0, 'metadata_y_px': 0.0},
            {'tile_id': 'b', 'metadata_x_px': 11.0, 'metadata_y_px': 0.0},
            {'tile_id': 'c', 'metadata_x_px': 22.0, 'metadata_y_px': 0.0},
        ]
    )
    alignments = pd.DataFrame(
        [
            {
                'source_tile_id': 'a',
                'target_tile_id': 'b',
                'dx_px': 10.0,
                'dy_px': 0.0,
                'confidence': 1.0,
                'accepted': True,
            },
            {
                'source_tile_id': 'b',
                'target_tile_id': 'c',
                'dx_px': 10.0,
                'dy_px': 0.0,
                'confidence': 1.0,
                'accepted': True,
            },
        ]
    )
    optimized = optimize_tile_positions(metadata, alignments)
    metadata_error = abs((metadata.iloc[1].metadata_x_px - metadata.iloc[0].metadata_x_px) - 10)
    optimized_error = abs((optimized.iloc[1].x_px - optimized.iloc[0].x_px) - 10)
    assert optimized_error <= metadata_error
    assert optimized['placement_mode'].iloc[0] == 'optimized'


def test_dtype_preservation_uint8_and_uint16():
    placements = pd.DataFrame([{'tile_id': 'a', 'metadata_x_px': 0, 'metadata_y_px': 0, 'x_px': 0, 'y_px': 0}])
    for dtype in (np.uint8, np.uint16):
        stitched, _ = blend_tiles({'a': np.full((4, 4), 7, dtype=dtype)}, placements, mode='average')
        assert stitched.dtype == dtype


def test_metrics_file_written(tmp_path):
    image = np.zeros((8, 8), dtype=np.uint8)
    metrics = compute_metrics(image, reference=image)
    metrics_path = tmp_path / 'metrics.json'
    write_metrics(metrics_path, metrics)
    loaded = json.loads(metrics_path.read_text())
    assert {'mse', 'psnr_db', 'ssim', 'seam_energy', 'approx_memory_bytes'} <= set(loaded)


def test_manifest_written(tmp_path):
    dataset_dir = tmp_path / 'dataset'
    output_dir = tmp_path / 'output'
    synthetic = generate_synthetic_dataset(
        dataset_dir,
        SyntheticConfig(
            grid_shape=(2, 2),
            tile_shape=(48, 48),
            overlap_px=12,
            noise_sigma=0,
            brightness_variation=0,
            max_translation_error_px=0,
            dtype='uint8',
        ),
    )
    result = run_stitching_pipeline(
        synthetic['metadata_path'],
        output_dir=output_dir,
        run_name='pytest_demo',
        blend_mode='average',
        ground_truth_path=synthetic['ground_truth_path'],
    )
    manifest = json.loads(result['manifest_path'].read_text())
    assert manifest['outputs']['stitched_image'] == 'stitched.tif'
    assert (output_dir / manifest['outputs']['metrics_json']).exists()
    assert result['metrics_path'].exists()
