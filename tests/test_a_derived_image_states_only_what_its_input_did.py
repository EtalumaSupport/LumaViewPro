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

import numpy as np
import tifffile as tf
from PIL import Image

from modules import image_utils

_ACQUISITION_FIELDS = ('plate_pos_mm', 'z_pos_um', 'exposure_time_ms', 'gain_db', 'illumination_ma')


def _write(path, metadata, *, ome=False):
    image_utils.write_tiff(
        data=np.zeros((16, 16), dtype=np.uint16),
        file_loc=path,
        metadata=metadata,
        ome=ome,
        color='Green',
        significant_bits=12,
        save_encoding='right_aligned',
    )
    return path


def _resolution_unit(path):
    with tf.TiffFile(str(path)) as tif:
        return tif.pages[0].tags['ResolutionUnit'].value


def test_an_input_with_no_metadata_gives_an_output_that_claims_nothing(tmp_path):
    png = tmp_path / 'external.png'
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(png)

    metadata = image_utils.build_postproc_output_metadata(png, 'Green', significant_bits=8)

    assert metadata['pixel_size_um'] is None
    for field in _ACQUISITION_FIELDS:
        assert field not in metadata, f'{field} was invented for an input that states none'

    out = _write(tmp_path / 'enhanced.tiff', metadata)
    assert image_utils.read_pixel_size_um(out) is None
    assert _resolution_unit(out) == tf.RESUNIT.NONE
    back = image_utils.read_postproc_input_metadata(out)
    for field in _ACQUISITION_FIELDS:
        assert field not in back


def test_a_capture_with_no_scale_keeps_everything_else_it_states(tmp_path):
    capture = _write(
        tmp_path / 'capture.tiff',
        {
            'objective': {},
            'pixel_size_um': None,
            'channel': 'Green',
            'significant_bits': 12,
            'datetime': '2026-10-04T00:00:00',
            'plate_pos_mm': {'x': 12.5, 'y': 30.25},
            'z_pos_um': 4321.0,
            'exposure_time_ms': 50.0,
            'gain_db': 3.0,
            'illumination_ma': 120.0,
        },
    )

    back = image_utils.read_postproc_input_metadata(capture)

    assert back is not None, 'a capture with no scale was read as having no metadata (#810)'
    assert back['pixel_size_um'] is None
    assert back['plate_pos_mm'] == {'x': 12.5, 'y': 30.25}
    assert back['z_pos_um'] == 4321.0
    assert back['exposure_time_ms'] == 50.0
    assert back['gain_db'] == 3.0
    assert back['illumination_ma'] == 120.0

    derived = image_utils.build_postproc_output_metadata(capture, 'Green', significant_bits=12)
    assert derived['pixel_size_um'] is None
    assert derived['plate_pos_mm'] == {'x': 12.5, 'y': 30.25}
    assert derived['exposure_time_ms'] == 50.0


def test_an_ome_input_states_only_what_its_xml_carries():
    ome = (
        '<OME><Image><Pixels SizeX="16" SizeY="16">'
        '<Channel Name="Green"/><Plane/>'
        '</Pixels></Image></OME>'
    )

    back = image_utils._read_ome_input_metadata(ome, None)

    assert back['pixel_size_um'] is None
    for field in _ACQUISITION_FIELDS:
        assert field not in back, f'{field} was invented for an OME input that states none'


def test_an_ome_input_keeps_the_scale_and_exposure_it_states():
    ome = (
        '<OME><Image><Pixels SizeX="16" SizeY="16" PhysicalSizeX="0.65">'
        '<Channel Name="Green"/><Plane ExposureTime="50.0"/>'
        '</Pixels></Image></OME>'
    )

    back = image_utils._read_ome_input_metadata(ome, None)

    assert back['pixel_size_um'] == 0.65
    assert back['exposure_time_ms'] == 50.0
    assert 'gain_db' not in back
    assert 'illumination_ma' not in back


def test_a_composite_states_no_exposure_gain_or_illumination(tmp_path):
    reference = _write(
        tmp_path / 'red.tiff',
        {
            'objective': {},
            'pixel_size_um': 0.65,
            'channel': 'Red',
            'significant_bits': 12,
            'datetime': '2026-10-04T00:00:00',
            'exposure_time_ms': 50.0,
            'gain_db': 3.0,
            'illumination_ma': 120.0,
        },
    )

    metadata = image_utils.build_composite_output_metadata(reference, significant_bits=12)

    for field in ('exposure_time_ms', 'gain_db', 'illumination_ma'):
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
            'Channel': {'Name': ['Green']},
            'Plane': {'ExposureTime': 5.0},
        },
    )

    back = image_utils.read_postproc_input_metadata(path)

    assert back['pixel_size_um'] is None
    assert back['exposure_time_ms'] == 5.0
    assert image_utils.read_pixel_size_um(path) is None
