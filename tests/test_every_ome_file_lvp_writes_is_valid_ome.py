# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every OME file LVP writes is valid against the OME 2016-06 schema.

tifffile's OME writer drops the Instrument, so ``image_utils`` splices it
into the description after the write; the schema orders the Instrument
before every Image and an Image's InstrumentRef and ObjectiveSettings
before its Pixels, and its keyrefs refuse a reference to an absent part.
The gate is the XSD itself (tests/ome_schema.py), which ``ome_types``
alone is not.
"""

from __future__ import annotations

import datetime
import xml.etree.ElementTree as ET

import numpy as np
import tifffile as tf

from modules import image_utils
from modules.layer_record import GREY, release_color
from tests.ome_schema import q, schema_errors, valid_description

_RECORD = {
    'pixel_size_um': 0.65,
    'channel': 'BF',
    'datetime': datetime.datetime(2026, 10, 29, 12, 0, 0),
    'objective': {'id': '20x Oly', 'magnification': 20, 'numerical_aperture': 0.45},
    'plate_pos_mm': {'x': 12.5, 'y': 30.25},
    'z_pos_um': 4321.0,
    'exposure_time_ms': 50.0,
    'instrument': {'manufacturer': 'Etaluma', 'model': 'LS850', 'camera_model': 'cam'},
    'plate': {'name': '96 well microplate', 'rows': 8, 'columns': 12, 'well_label': 'A1'},
}


def test_every_shape_of_ome_still_is_valid_ome(tmp_path):
    """Mono 8 and 16 bit, an 8-bit palette, a 16-bit false colour and an RGB
    composite on an OME run each write a schema-valid description carrying
    the Instrument."""
    shapes = {
        'mono 8': (np.uint8, 8, '8bit', GREY, {}),
        'mono 16': (np.uint16, 12, 'right_aligned', GREY, {}),
        'palette': (np.uint8, 8, '8bit', release_color('Ex488'), {'channel': 'Ex488'}),
        'false colour 16': (np.uint16, 12, 'rgb', release_color('Ex488'), {'channel': 'Ex488'}),
        'RGB composite': (np.uint8, 8, '8bit', GREY, {'channel': 'Composite', 'rgb': True}),
    }
    offenders = {}
    for name, (dtype, bits, encoding, color, overrides) in shapes.items():
        shape = (8, 8, 3) if overrides.pop('rgb', False) else (8, 8)
        path = tmp_path / f'{name}.ome.tiff'
        image_utils.write_tiff(
            data=np.zeros(shape, dtype=dtype),
            file_loc=path,
            metadata={**_RECORD, **overrides},
            ome=True,
            color=color,
            significant_bits=bits,
            save_encoding=encoding,
            white_bits=None,
        )
        description = tf.tiffcomment(path)
        errors = schema_errors(description)
        if errors or ET.fromstring(description).find(q('Instrument')) is None:
            offenders[name] = errors or 'no Instrument'
    assert offenders == {}


def test_the_gate_refuses_a_misordered_instrument_and_the_splice_orders_it(tmp_path):
    """The schema orders the Instrument before the Image: the gate refuses a
    description with it after, and the splice writes it before the first
    Image with the Image's references immediately before its Pixels."""
    path = tmp_path / 'still.ome.tiff'
    image_utils.write_tiff(
        data=np.zeros((8, 8), dtype=np.uint16),
        file_loc=path,
        metadata=_RECORD,
        ome=True,
        color=GREY,
        significant_bits=12,
        save_encoding='right_aligned',
        white_bits=None,
    )

    root = valid_description(path)
    assert [child.tag for child in root][:2] == [q('Instrument'), q('Image')]
    image = [child.tag for child in root.find(q('Image'))]
    pixels = image.index(q('Pixels'))
    assert image[pixels - 2 : pixels] == [q('InstrumentRef'), q('ObjectiveSettings')]

    instrument = root.find(q('Instrument'))
    root.remove(instrument)
    root.append(instrument)
    misordered = ET.tostring(root, encoding='unicode')
    assert schema_errors(misordered) != []
