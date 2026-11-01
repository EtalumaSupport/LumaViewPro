# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every TIFF LVP writes is compressed one way: deflate, no predictor.

The codec belongs to the two writers, ``image_utils.write_tiff`` and
``image_utils.write_hyperstack_tiff``; no caller chooses it or can leave it
out. A file of the same pixels is then the same size, within its header,
whichever container the user picked. Each case reads its file back through
tifffile and asserts the codec on every page and the pixels unchanged.
"""

import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from modules import image_utils
from modules.layer_record import release_color, release_colors
from modules.stack_builder import StackBuilder
from tests.stock_channels import LED1


def _metadata():
    return {
        'pixel_size_um': 0.5,
        'channel': LED1,
        'significant_bits': 12,
        'objective': {'id': '10x'},
        'exposure_time_ms': 50.0,
        'gain_db': 0.0,
        'illumination_ma': 100.0,
        'z_pos_um': 1000.0,
        'plate_pos_mm': {'x': 10.0, 'y': 20.0},
        'datetime': datetime.datetime(2026, 10, 31, 12, 0, 0),
        'well_site': '1',
    }


def _assert_deflate_everywhere(path: pathlib.Path) -> np.ndarray:
    with tf.TiffFile(str(path)) as tif:
        codecs = {(page.compression, page.predictor) for page in tif.pages}
        pixels = tif.asarray()
    assert codecs == {(tf.COMPRESSION.ADOBE_DEFLATE, tf.PREDICTOR.NONE)}, (
        f'{path.name}: every page must be deflate with no predictor; got {codecs}'
    )
    return pixels


_RNG = np.random.default_rng(0)

# parametrize-justified: each case is a different container branch of the
# writer's options (OME, ImageJ, plain mono, plain RGB), each its own file.
_CASES = {
    'ome_12bit_mono': {
        'data': _RNG.integers(0, 4096, (64, 64)).astype(np.uint16),
        'ome': True,
        'significant_bits': 12,
        'save_encoding': 'right_aligned',
    },
    'imagej_12bit_mono': {
        'data': _RNG.integers(0, 4096, (64, 64)).astype(np.uint16),
        'ome': False,
        'significant_bits': 12,
        'save_encoding': 'right_aligned',
    },
    'plain_8bit_palette': {
        'data': _RNG.integers(0, 256, (64, 64)).astype(np.uint8),
        'ome': False,
        'significant_bits': 8,
        'save_encoding': '8bit',
    },
    'plain_16bit_rgb': {
        'data': _RNG.integers(0, 65536, (64, 64, 3)).astype(np.uint16),
        'ome': False,
        'significant_bits': 16,
        'save_encoding': 'right_aligned',
    },
}


@pytest.mark.parametrize('case', sorted(_CASES))
def test_a_frame_is_written_deflate_whatever_its_container(case, tmp_path):
    spec = _CASES[case]
    path = tmp_path / f'{case}.tiff'
    image_utils.write_tiff(
        data=spec['data'],
        file_loc=path,
        metadata=_metadata(),
        ome=spec['ome'],
        color=release_color(LED1),
        significant_bits=spec['significant_bits'],
        save_encoding=spec['save_encoding'],
        white_bits=None,
    )
    pixels = _assert_deflate_everywhere(path)
    assert np.array_equal(pixels, spec['data'])


def test_a_hyperstack_is_written_deflate(tmp_path):
    frames = []
    rows = []
    for z_idx in range(2):
        fname = f'frame_z{z_idx}.tiff'
        frame = _RNG.integers(0, 4096, (16, 16)).astype(np.uint16)
        frames.append(frame)
        image_utils.write_tiff(
            data=frame,
            file_loc=tmp_path / fname,
            metadata={**_metadata(), 'z_pos_um': float(z_idx) * 10},
            ome=False,
            color=release_color(LED1),
            significant_bits=12,
            save_encoding='right_aligned',
            white_bits=None,
        )
        rows.append(
            {
                'Filepath': fname,
                'Color': LED1,
                'Scan Count': 0,
                'Z-Slice': z_idx,
                'X': 10.0,
                'Y': 20.0,
                'Z': float(z_idx) * 10,
            }
        )
    result = StackBuilder._create_stack(
        path=tmp_path,
        df=pd.DataFrame(rows),
        output_file_loc=pathlib.Path('out.ome.tiff'),
        save_encoding='right_aligned',
        channel_colors=release_colors(),
    )
    assert result['status'], result.get('error')

    pixels = _assert_deflate_everywhere(tmp_path / 'out.ome.tiff')
    assert np.array_equal(pixels.reshape(2, 16, 16), np.stack(frames))


def test_a_caller_cannot_choose_the_hyperstack_codec(tmp_path):
    plane = np.zeros((4, 4), dtype=np.uint8)
    reference = tmp_path / 'ref.tiff'
    tf.imwrite(reference, plane)
    metadata = image_utils.build_hyperstack_output_metadata(
        input_records=[image_utils.read_file_record(reference)],
        channel_names=[LED1],
        channel_colors=[release_color(LED1)],
        plane_positions={
            'PositionX': [0.0],
            'PositionY': [0.0],
            'PositionZ': [0.0],
            'DeltaT': None,
        },
        significant_bits=8,
        pixel_size_um=None,
    )
    with pytest.raises(TypeError, match='compression'):
        image_utils.write_hyperstack_tiff(
            planes=iter([plane]),
            file_loc=tmp_path / 'chosen.ome.tiff',
            shape=(1, 1, 1, 4, 4),
            dtype=np.uint8,
            hyperstack_metadata=metadata,
            hyperstack_options={'photometric': 'minisblack', 'compression': 'lzw'},
        )
    assert not (tmp_path / 'chosen.ome.tiff').exists()
