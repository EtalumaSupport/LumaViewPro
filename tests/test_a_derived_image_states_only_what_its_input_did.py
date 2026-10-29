# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stitch, a projection, a composite or an enhanced image states only what its input stated.

Unknown stays unknown (Eric, 2026-10-03). The derived-output path used to fill
what it could not read with numbers: a pixel size of 1.0 um, a position of
0, 0, 0 and an exposure, gain and illumination of 0. A number in a TIFF tag
is read downstream as a measurement, so each was a plausible wrong result.
And a capture saved with no scale -- which the writer marks by omitting
PhysicalSizeX -- was read back as having no metadata at all, losing the
position, exposure and channel it did state (#810).

Every case writes through the real writer and reads back from disk.
"""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import pandas as pd
import tifffile as tf
from PIL import Image

from modules import image_utils
from modules.layer_record import release_color, release_colors
from modules.stitcher import Stitcher
from modules.zproject_method import ZProjectMethod
from modules.zprojector import ZProjector
from tests.stock_channels import LED1, LED2

_ACQUISITION_FIELDS = ('plate_pos_mm', 'z_pos_um', 'exposure_time_ms', 'gain_db', 'illumination_ma')


def _write(path, metadata, *, ome=False):
    image_utils.write_tiff(
        data=np.zeros((16, 16), dtype=np.uint16),
        file_loc=path,
        metadata=metadata,
        ome=ome,
        color=release_color(LED1),
        significant_bits=12,
        save_encoding='right_aligned',
        white_bits=None,
    )
    return path


def _resolution_unit(path):
    with tf.TiffFile(str(path)) as tif:
        return tif.pages[0].tags['ResolutionUnit'].value


def test_an_input_with_no_metadata_gives_an_output_that_claims_nothing(tmp_path):
    png = tmp_path / 'external.png'
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(png)

    metadata = image_utils.build_postproc_output_metadata(
        [image_utils.read_file_record(png)], LED1, significant_bits=8, adds_frames=False
    )

    assert metadata['pixel_size_um'] is None
    for field in _ACQUISITION_FIELDS:
        assert field not in metadata, f'{field} was invented for an input that states none'

    out = _write(tmp_path / 'enhanced.tiff', metadata)
    assert image_utils.read_pixel_size_um(out) is None
    assert _resolution_unit(out) == tf.RESUNIT.NONE
    back = image_utils.read_file_record(out)
    for field in _ACQUISITION_FIELDS:
        assert field not in back


def test_a_capture_with_no_scale_keeps_everything_else_it_states(tmp_path):
    capture = _write(
        tmp_path / 'capture.tiff',
        {
            'objective': {},
            'pixel_size_um': None,
            'channel': LED1,
            'significant_bits': 12,
            'datetime': datetime.datetime(2026, 10, 4, 0, 0, 0),
            'plate_pos_mm': {'x': 12.5, 'y': 30.25},
            'z_pos_um': 4321.0,
            'exposure_time_ms': 50.0,
            'gain_db': 3.0,
            'illumination_ma': 120.0,
        },
    )

    back = image_utils.read_file_record(capture)

    assert back is not None, 'a capture with no scale was read as having no metadata (#810)'
    assert back['pixel_size_um'] is None
    assert back['plate_pos_mm'] == {'x': 12.5, 'y': 30.25}
    assert back['z_pos_um'] == 4321.0
    assert back['exposure_time_ms'] == 50.0
    assert back['gain_db'] == 3.0
    assert back['illumination_ma'] == 120.0

    derived = image_utils.build_postproc_output_metadata(
        [image_utils.read_file_record(capture)], LED1, significant_bits=12, adds_frames=False
    )
    assert derived['pixel_size_um'] is None
    assert derived['plate_pos_mm'] == {'x': 12.5, 'y': 30.25}
    assert derived['exposure_time_ms'] == 50.0


def test_an_ome_input_states_only_what_its_xml_carries():
    ome = (
        '<OME><Image><Pixels SizeX="16" SizeY="16">'
        f'<Channel Name="{LED1}"/><Plane/>'
        '</Pixels></Image></OME>'
    )

    back = image_utils._read_ome_input_metadata(ome)

    assert back['pixel_size_um'] is None
    for field in _ACQUISITION_FIELDS:
        assert field not in back, f'{field} was invented for an OME input that states none'


def test_an_ome_input_keeps_the_scale_and_exposure_it_states():
    ome = (
        '<OME><Image><Pixels SizeX="16" SizeY="16" PhysicalSizeX="0.65">'
        f'<Channel Name="{LED1}"/><Plane ExposureTime="50.0"/>'
        '</Pixels></Image></OME>'
    )

    back = image_utils._read_ome_input_metadata(ome)

    assert back['pixel_size_um'] == 0.65
    assert back['exposure_time_ms'] == 50.0
    assert 'gain_db' not in back
    assert 'illumination_ma' not in back


def test_a_composite_states_no_per_channel_acquisition_setting(tmp_path):
    reference = _write(
        tmp_path / 'red.tiff',
        {
            'objective': {},
            'pixel_size_um': 0.65,
            'channel': LED2,
            'significant_bits': 12,
            'datetime': datetime.datetime(2026, 10, 4, 0, 0, 0),
            'exposure_time_ms': 50.0,
            'gain_db': 3.0,
            'black_level': 4.0,
            'illumination_ma': 120.0,
            # The sum count is a layer's setting too, so the channels differ in it.
            'frames_summed': 4,
        },
    )

    metadata = image_utils.build_postproc_output_metadata(
        [image_utils.read_file_record(reference)],
        'Composite',
        significant_bits=12,
        adds_frames=False,
    )

    for field in ('exposure_time_ms', 'gain_db', 'black_level', 'illumination_ma', 'frames_summed'):
        assert field not in metadata, f'a merged image states one {field}'
    assert metadata['pixel_size_um'] == 0.65


def test_an_ome_file_stating_a_zero_scale_reads_as_no_scale(tmp_path):
    # A non-positive scale is not a measurement; every reader takes the
    # file's scale from the one place that says so.
    path = tmp_path / 'zero_scale.ome.tiff'
    tf.imwrite(
        str(path),
        np.zeros((16, 16), dtype=np.uint16),
        ome=True,
        metadata={
            'PhysicalSizeX': 0.0,
            'Channel': {'Name': [LED1]},
            'Plane': {'ExposureTime': 5.0},
        },
    )

    back = image_utils.read_file_record(path)

    assert back['pixel_size_um'] is None
    assert back['exposure_time_ms'] == 5.0
    assert image_utils.read_pixel_size_um(path) is None


def _tile(path, *, x, gain_db=3.0, z_pos_um=100.0, pixel_size_um=0.65):
    """An 8-bit BF tile at ``x`` mm stating its gain, Z and scale."""
    image_utils.write_tiff(
        data=np.full((16, 16), 100, dtype=np.uint8),
        file_loc=path,
        metadata={
            'pixel_size_um': pixel_size_um,
            'channel': 'BF',
            'datetime': datetime.datetime(2026, 10, 28, 12, 0, 0),
            'plate_pos_mm': {'x': x, 'y': 0.0},
            'z_pos_um': z_pos_um,
            'gain_db': gain_db,
            'exposure_time_ms': 10.0,
        },
        ome=False,
        color=release_color('BF'),
        significant_bits=8,
        save_encoding='8bit',
        white_bits=None,
    )


def _tiles(folder, **per_tile):
    """Two tiles side by side, the i-th taking each keyword's i-th value."""
    rows = []
    for i in range(2):
        name = f'tile_{i}.tiff'
        _tile(folder / name, x=i * 0.01, **{key: values[i] for key, values in per_tile.items()})
        rows.append(
            {
                'Filepath': name,
                'Color': 'BF',
                'X': i * 0.01,
                'Y': 0.0,
                'Objective': '20x',
                'Channel Group ID': 0,
            }
        )
    return pd.DataFrame(rows)


def test_a_derived_file_states_a_setting_only_where_its_inputs_agree(tmp_path):
    """A stitch of tiles at 3 and 9 dB states no gain, and of tiles at one gain
    that gain; a projection of slices at different Z states no Z."""
    stated = {}
    for name, gains in (('differ', (3.0, 9.0)), ('agree', (3.0, 3.0))):
        folder = tmp_path / name
        folder.mkdir()
        Stitcher._simple_position_stitcher(
            folder,
            _tiles(folder, gain_db=gains),
            output_file_loc=pathlib.Path('stitched.tiff'),
            channel_colors=release_colors(),
        )
        stated[name] = image_utils.read_file_record(folder / 'stitched.tiff').get('gain_db')
    slices = tmp_path / 'slices'
    slices.mkdir()
    ZProjector(has_turret=False)._zproject(
        slices,
        _tiles(slices, z_pos_um=(100.0, 110.0)),
        ZProjectMethod.Max,
        pathlib.Path('projected.tiff'),
        release_colors(),
    )

    assert stated == {'differ': None, 'agree': 3.0}
    assert 'z_pos_um' not in image_utils.read_file_record(slices / 'projected.tiff')


def test_a_stitch_whose_tiles_state_no_one_scale_is_unregistered_and_states_none(tmp_path):
    """Tiles laid out at the first tile's scale are misregistered when another
    tile is at its own: tiles that disagree on scale, or of which one states
    none, have no scale, and the stitch is the unregistered montage."""
    for name, scales in (('differ', (0.65, 1.3)), ('one_none', (0.65, None))):
        folder = tmp_path / name
        folder.mkdir()
        Stitcher.__new__(Stitcher)._group_algorithm(
            path=folder,
            df=_tiles(folder, pixel_size_um=scales),
            output_file_loc=pathlib.Path('stitched.tiff'),
            channel_colors=release_colors(),
        )
        record = image_utils.read_file_record(folder / 'stitched.tiff')
        assert (name, record['pixel_size_um'], record['algorithm']) == (
            name,
            None,
            'simple_position_stitcher',
        )
