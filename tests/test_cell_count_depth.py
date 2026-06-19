# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Cell-count downconvert must use each file's true significant-bit depth.

A right-aligned 12-bit scientific capture (0..4095 in a uint16 container) must be
scaled to 8-bit by 4095, not the 16-bit container width; otherwise full-white
cells map to ~15/255 and detection collapses on a scientific user's primary
output. The folder walker reads the depth per file and threads it into the
converter; a legacy left-justified file (no narrow tag) correctly reads as 16.
"""

from __future__ import annotations

import numpy as np

from modules import image_utils
from modules.post_processing import PostProcessing


def _meta(significant_bits):
    return {
        'pixel_size_um': 0.5,
        'channel': 'Green',
        'objective': '10x',
        'exposure_time_ms': 50.0,
        'gain_db': 0.0,
        'illumination_ma': 100.0,
        'z_pos_um': 1000.0,
        'plate_pos_mm': {'x': 10.0, 'y': 20.0},
        'datetime': '2026:06:18 12:00:00',
        'camera_make': 'Test',
        'microscope': 'TestScope',
        'well_label': 'A1',
        'significant_bits': significant_bits,
    }


def _write_plain_tiff(path, value, significant_bits):
    arr = np.full((8, 8), value, dtype=np.uint16)
    image_utils.write_tiff(
        data=arr,
        file_loc=path,
        metadata=_meta(significant_bits),
        ome=False,
        color='Green',
        significant_bits=significant_bits,
    )


def _capture_depth(post):
    """Replace the leaf process_image with a recorder; return the capture dict."""
    captured = {}

    def fake_process_image(image, settings, include_images=None, significant_bits=16):
        captured['significant_bits'] = significant_bits
        return (
            {'filtered_contours': np.zeros((8, 8, 3), dtype=np.uint8)},
            {'summary': {'num_regions': 0, 'total_object_area': 0, 'total_object_intensity': 0.0}},
        )

    post._cell_count.process_image = fake_process_image
    return captured


def test_folder_walk_threads_12bit_depth(tmp_path):
    """A right-aligned 12-bit TIFF is counted at its true depth, not container width."""
    _write_plain_tiff(tmp_path / 'cells.tif', 4095, significant_bits=12)
    post = PostProcessing()
    captured = _capture_depth(post)

    list(post.apply_cell_count_to_folder(path=str(tmp_path), settings={}))

    assert captured['significant_bits'] == 12


def test_folder_walk_legacy_file_defaults_to_container_width(tmp_path):
    """A legacy left-justified file (no narrow tag) reads as 16, the container width."""
    _write_plain_tiff(tmp_path / 'legacy.tif', 4095 * 16, significant_bits=16)
    post = PostProcessing()
    captured = _capture_depth(post)

    list(post.apply_cell_count_to_folder(path=str(tmp_path), settings={}))

    assert captured['significant_bits'] == 16
