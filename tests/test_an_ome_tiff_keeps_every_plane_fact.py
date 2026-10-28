# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An OME-TIFF keeps every Plane fact LVP writes, and LVP reads them back.

The OME schema's Plane has attributes for the exposure, the position and the
time offset only, and tifffile's OME writer drops every other key without a
word. An OME-TIFF of a 4-frame sum then stated its per-frame exposure and no
count, so it read as a quarter of its integration; a dark frame, a frame
taken inside a change's settle and one after a raw camera write read as good
ones. The facts the schema has no attribute for ride in the image's
``etaluma.com/lvp`` MapAnnotation, OME's own key/value slot, which
Bio-Formats shows; an ImageJ file of the same frame keeps them in its Plane.

Every case writes through the real writer and reads the file back from disk.
"""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET

import numpy as np
import tifffile as tf

from modules import image_utils
from modules.layer_record import release_color
from tests.stock_channels import LED1

_OME_NS = '{http://www.openmicroscopy.org/Schemas/OME/2016-06}'
_SCHEMA_NUMBERS = ('ExposureTime', 'PositionX', 'PositionY', 'PositionZ')
# The micron spelling is the container's: OME's schema form, ImageJ's ASCII.
_UNITS = ('ExposureTimeUnit', 'PositionXUnit', 'PositionYUnit', 'PositionZUnit')


def _every_plane_field() -> dict:
    """A capture's metadata with every optional Plane field set."""
    return {
        'objective': {'model': 'PlanFluor20x', 'magnification': 20, 'aperture': 0.45},
        'pixel_size_um': 0.65,
        'channel': LED1,
        'excitation_nm': 470.0,
        'datetime': '2026-10-19T12:00:00',
        'plate_pos_mm': {'x': 12.5, 'y': 30.25},
        'z_pos_um': 4321.0,
        'illumination_ma': 120.0,
        'exposure_time_ms': 50.0,
        'gain_db': 3.0,
        'black_level': 4,
        'frames_summed': 4,
        'binning_size': 2,
        'timestamp_iso': '2026-10-19T12:00:00.123',
        'timestamp_source': 'camera_rebased',
        'illumination_timing': 'on_arrival',
        'recording_frame': 3,
        'stage_moving': True,
        'timestamp_camera_ticks': 123456789,
        'timestamp_camera_tick_hz': 1_000_000_000,
        'frame_id': 7,
        'frame_validity': [
            {
                'source': 'led',
                'counts': True,
                'frames_after': 2,
                'after_write_ms': 40.0,
                'skip': 1,
                'counted_by': ['3:camera'],
            }
        ],
        'saturated_fraction': 0.0,
        'dark': False,
        'camera_held_a_channel': True,
        'camera_held_channel': LED1,
        'fallback_focus': 'Autofocus found no focus: the focus curve is flat.',
    }


def _write(path, metadata, *, ome):
    image_utils.write_tiff(
        data=np.full((16, 16), 4000, dtype=np.uint16),
        file_loc=path,
        metadata=metadata,
        ome=ome,
        color=release_color(LED1),
        significant_bits=14,
        save_encoding='right_aligned',
        white_bits=12,
    )
    return path


def _ome_image(path) -> ET.Element:
    with tf.TiffFile(str(path)) as tif:
        return ET.fromstring(tif.ome_metadata)


def _lvp_annotation(root: ET.Element) -> ET.Element:
    """The image's MapAnnotation in LVP's namespace, found through the image's own reference."""
    image = root.find(f'{_OME_NS}Image')
    refs = {ref.get('ID') for ref in image.findall(f'{_OME_NS}AnnotationRef')}
    (annotation,) = [
        a
        for a in root.iter(f'{_OME_NS}MapAnnotation')
        if a.get('ID') in refs and a.get('Namespace') == 'etaluma.com/lvp'
    ]
    return annotation


def test_the_ome_file_keeps_the_plane_an_imagej_file_keeps(tmp_path):
    metadata = _every_plane_field()
    ome_root = _ome_image(_write(tmp_path / 'frame.ome.tiff', metadata, ome=True))
    with tf.TiffFile(str(_write(tmp_path / 'frame.tiff', metadata, ome=False))) as tif:
        imagej_plane = ast.literal_eval(tif.imagej_metadata['plane'])

    schema = ome_root.find(f'{_OME_NS}Image/{_OME_NS}Pixels/{_OME_NS}Plane').attrib
    annotated = {
        m.get('K'): json.loads(m.text) for m in _lvp_annotation(ome_root).iter(f'{_OME_NS}M')
    }

    ome_plane = {key: float(schema[key]) for key in _SCHEMA_NUMBERS if key in schema}
    ome_plane |= {key: schema[key] for key in _UNITS if key in schema}
    assert set(ome_plane) | set(annotated) == set(imagej_plane)
    assert not set(ome_plane) & set(annotated), 'a fact has two homes in one file'
    for key, value in annotated.items():
        assert value == imagej_plane[key], key
    for key in _SCHEMA_NUMBERS:
        assert ome_plane[key] == imagej_plane[key], key
    assert annotated['FramesSummed'] == 4


def test_lvp_reads_an_ome_file_as_it_reads_an_imagej_file(tmp_path):
    metadata = _every_plane_field()
    ome = image_utils.read_postproc_input_metadata(
        _write(tmp_path / 'frame.ome.tiff', metadata, ome=True)
    )
    imagej = image_utils.read_postproc_input_metadata(
        _write(tmp_path / 'frame.tiff', metadata, ome=False)
    )

    assert ome == imagej
    assert ome['frames_summed'] == 4
    assert ome['exposure_time_ms'] == 50.0


def test_a_numpy_value_is_kept_as_its_number(tmp_path):
    # A producer's value that stayed a numpy scalar saves as the number it is,
    # not as a failed save.
    metadata = _every_plane_field() | {
        'frames_summed': np.int64(4),
        'dark': np.bool_(False),
        'saturated_fraction': np.float32(0.5),
    }
    back = image_utils.read_postproc_input_metadata(
        _write(tmp_path / 'frame.ome.tiff', metadata, ome=True)
    )

    assert (back['frames_summed'], back['dark'], back['saturated_fraction']) == (4, False, 0.5)


def test_an_ome_file_whose_annotation_will_not_parse_states_nothing(tmp_path):
    path = tmp_path / 'broken.ome.tiff'
    tf.imwrite(
        str(path),
        np.zeros((8, 8), dtype=np.uint16),
        ome=True,
        metadata={
            'axes': 'YX',
            'Plane': {'ExposureTime': 50.0},
            'MapAnnotation': {'Namespace': 'etaluma.com/lvp', 'Value': {'FramesSummed': '{4'}},
        },
    )

    assert image_utils.read_postproc_input_metadata(path) is None
