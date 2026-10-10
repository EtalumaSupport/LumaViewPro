# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved TIFF that cannot be decoded is a ValueError naming the file.

The readers document ValueError for an undecodable file, and their callers
(cell count's skip, Quick Enhance's skip, the hyperstack pre-scan) catch that.
A file read while it is still being written -- a run folder read before its
files drain -- or a damaged one fails inside tifffile with whatever its
parser hit first. Each case below is a byte state a real write passes
through, found by racing reads against write_tiff and by truncating its
output.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from modules import image_utils

_METADATA = {
    'pixel_size_um': 0.5,
    'channel': 'BF',
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
}


def _saved_12bit_tiff(path):
    """A plain 12-bit frame as a protocol saves it (deflate-compressed)."""
    pixels = np.random.default_rng(0).integers(0, 4096, (256, 256)).astype(np.uint16)
    image_utils.write_tiff(
        pixels,
        path,
        {**_METADATA, 'significant_bits': 12},
        ome=False,
        color='BF',
        significant_bits=12,
        save_encoding='right_aligned',
    )
    return path.read_bytes()


def _created_but_empty(path, saved):
    path.write_bytes(b'')


def _header_cut_short(path, saved):
    path.write_bytes(saved[:2])


def _header_without_its_page(path, saved):
    path.write_bytes(saved[:8])


def _page_header_not_yet_written(path, saved):
    # tifffile reserves the first page's header as zeros and fills it in after
    # the pixels, so a reader mid-write sees a page with no tags.
    path.write_bytes(saved[:8] + bytes(4096))


def _compressed_pixels_cut_short(path, saved):
    path.write_bytes(saved[: len(saved) // 2])


_STATES = [
    _created_but_empty,
    _header_cut_short,
    _header_without_its_page,
    _page_header_not_yet_written,
    _compressed_pixels_cut_short,
]

_READERS = [
    image_utils.load_pixels,
    image_utils.load_pixels_with_timestamp,
    image_utils.read_tiff_significant_bits,
    image_utils.read_tiff_depth_and_timestamp,
    image_utils.read_image_geometry,
]


@pytest.mark.parametrize('state', _STATES, ids=lambda f: f.__name__.lstrip('_'))
@pytest.mark.parametrize('reader', _READERS, ids=lambda f: f.__name__)
def test_an_undecodable_tiff_is_a_value_error_naming_the_file(tmp_path, reader, state):
    saved = _saved_12bit_tiff(tmp_path / 'whole.tif')
    broken = tmp_path / 'A1_BF_0000.tif'
    state(broken, saved)

    header_only = reader in (
        image_utils.read_tiff_significant_bits,
        image_utils.read_tiff_depth_and_timestamp,
        image_utils.read_image_geometry,
    )
    if header_only and state is _compressed_pixels_cut_short:
        # These read the header alone, which is whole; the pixels are not theirs.
        reader(broken)
        return

    with pytest.raises(ValueError, match=re.escape(broken.name)):
        reader(broken)


def test_a_whole_file_still_loads(tmp_path):
    path = tmp_path / 'A1_BF_0000.tif'
    _saved_12bit_tiff(path)
    image, significant_bits = image_utils.load_pixels(path)
    assert image.shape == (256, 256)
    assert significant_bits == 12
