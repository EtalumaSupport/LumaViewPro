# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#491: a saved file names its instrument and plate -- the ImageJ
Instrument and Plate blocks, and the OME schema's Instrument.

Bug
---
Customer asked for richer TIFF metadata so downstream tooling can
extract instrument identity + plate identity per image. The existing
ImageJ Document block carried only Manufacturer/Device/WellLabel/
WellSite; the OME path had no Plate or Instrument blocks at all.

Fix
---
Per the Option-A scope agreed with Eric:
- An ImageJ file names its scope in an Instrument block and its plate and
  well in a Plate block, each once; the Document block that repeated them
  is gone.
- An OME file carries the schema's Instrument (Microscope, Detector,
  Objective), spliced into the description tifffile writes, which drops
  it; its plate and well ride in LVP's annotation, since a schema Plate
  in each file would import into OMERO as a plate per file.

Not shipped in this commit (documented as gap): OME LightSource
(LED wavelength + power; not tracked per-color), OME Detector gain/
zoom (partial -- model only), OME FilterSet (filter wheel + dichroic
not tracked).

Test approach
-------------
The OME tests write the file and read its description, checked against
the schema; the ImageJ tests read generate_tiff_data's structured block,
which tifffile serializes as given.
"""

from __future__ import annotations

import datetime
import numpy as np

from modules import image_utils
from modules.layer_record import release_color
from tests.ome_schema import OME, q, valid_description


def _build_metadata():
    return {
        'channel': 'BF',
        'significant_bits': 8,
        'datetime': datetime.datetime(2026, 5, 26, 23, 0, 0),
        'objective': {
            'id': '4x Oly',
            'magnification': 4,
            'numerical_aperture': 0.16,
            'working_distance_mm': 13.0,
            'focal_length_mm': 45.0,
        },
        'plate_pos_mm': {'x': 12.5, 'y': 30.0},
        'z_pos_um': 1500.0,
        'exposure_time_ms': 10.0,
        'gain_db': 1.0,
        'illumination_ma': 50.0,
        'pixel_size_um': 2.5,
        'well_site': 0,
        'instrument': {
            'manufacturer': 'Etaluma',
            'model': 'LS850',
            'serial_number': 'EL0940-05',
            'camera_model': 'Basler a2A3536-31umBAS',
        },
        'plate': {
            'name': '96 well microplate',
            'rows': 8,
            'columns': 12,
            'standard': 'ANSI SLAS 2004',
            'well_label': 'A1',
        },
    }


def _ome_still(tmp_path, md: dict, name: str = 'still.ome.tiff'):
    """The schema-valid description of an OME still written from ``md``."""
    path = tmp_path / name
    image_utils.write_tiff(
        data=np.zeros((16, 16), dtype=np.uint16),
        file_loc=path,
        metadata=md,
        ome=True,
        color=release_color('BF'),
        significant_bits=12,
        save_encoding='right_aligned',
        white_bits=None,
    )
    return valid_description(path)


def _attributes(root, path: str) -> dict | None:
    element = root.find(path)
    return None if element is None else dict(element.attrib)


def test_an_ome_still_states_its_instrument_in_the_schema(tmp_path):
    """An OME still names its microscope, camera and objective in the schema's
    Instrument, referenced from its Image, so Bio-Formats and OMERO show them;
    it writes no schema Plate (each file is its own fileset, and OMERO would
    import a plate per file), and the file is valid OME."""
    root = _ome_still(tmp_path, _build_metadata())

    instrument = q('Instrument')
    assert _attributes(root, f'{instrument}/{q("Microscope")}') == {
        'Manufacturer': 'Etaluma',
        'Model': 'LS850',
        'SerialNumber': 'EL0940-05',
    }
    assert _attributes(root, f'{instrument}/{q("Detector")}') == {
        'ID': 'Detector:0:0',
        'Model': 'Basler a2A3536-31umBAS',
    }
    assert _attributes(root, f'{instrument}/{q("Objective")}') == {
        'ID': 'Objective:0:0',
        'NominalMagnification': '4',
        'LensNA': '0.16',
        'WorkingDistance': '13.0',
        'WorkingDistanceUnit': 'mm',
    }
    image = root.find(q('Image'))
    assert _attributes(image, q('InstrumentRef')) == {'ID': 'Instrument:0'}
    assert _attributes(image, q('ObjectiveSettings')) == {'ID': 'Objective:0:0'}
    assert root.find(q('Plate')) is None


def test_an_ome_still_states_only_the_parts_its_record_gives(tmp_path):
    """Each part of the Instrument appears only when it states something, and
    each reference only with what it names: a scope with no serial gives a
    Microscope with none; a frame with no objective gives no Objective and
    no ObjectiveSettings; a record with neither instrument nor objective
    gives no Instrument at all. Each file is valid OME."""
    no_serial = _build_metadata()
    no_serial['instrument'] = {
        key: value for key, value in no_serial['instrument'].items() if key != 'serial_number'
    }
    root = _ome_still(tmp_path, no_serial, 'no_serial.ome.tiff')
    assert _attributes(root, f'{q("Instrument")}/{q("Microscope")}') == {
        'Manufacturer': 'Etaluma',
        'Model': 'LS850',
    }
    assert root.find(f'{q("Image")}/{q("InstrumentRef")}') is not None

    no_objective = _build_metadata()
    del no_objective['objective']
    root = _ome_still(tmp_path, no_objective, 'no_objective.ome.tiff')
    assert [part.tag for part in root.find(q('Instrument'))] == [q('Microscope'), q('Detector')]
    assert root.find(f'{q("Image")}/{q("ObjectiveSettings")}') is None

    objective_only = _build_metadata()
    del objective_only['instrument']
    root = _ome_still(tmp_path, objective_only, 'objective_only.ome.tiff')
    assert [part.tag for part in root.find(q('Instrument'))] == [q('Objective')]

    neither = _build_metadata()
    del neither['instrument'], neither['objective']
    root = _ome_still(tmp_path, neither, 'neither.ome.tiff')
    assert root.find(q('Instrument')) is None
    assert root.find(f'{q("Image")}/{q("InstrumentRef")}') is None


def _unreported(md: dict) -> dict:
    """``md`` from a scope that reported no serial, and a frame in no well:
    the record states neither (``image_save`` omits an empty value)."""
    del md['instrument']['serial_number']
    del md['plate']['well_label']
    return md


def test_no_writer_states_an_instrument_fact_the_scope_did_not_report(tmp_path):
    """A fact the catalogue or the instrument did not give is absent from the
    file: no ``Air`` immersion, ``CMOS`` detector, empty model, manufacturer,
    serial or well label, in a still's OME or ImageJ container or a
    hyperstack built from it."""
    data = np.zeros((128, 128), dtype=np.uint16)
    imagej = image_utils.generate_tiff_data(
        data=data,
        metadata=_unreported(_build_metadata()),
        image_type='imagej',
        color=release_color('BF'),
        white_bits=None,
    )['metadata']
    still = tmp_path / 'still.ome.tiff'
    image_utils.write_tiff(
        data=data,
        file_loc=still,
        metadata=_unreported(_build_metadata()),
        ome=True,
        color=release_color('BF'),
        significant_bits=12,
        save_encoding='right_aligned',
        white_bits=None,
    )
    hyperstack = tmp_path / 'stack.ome.tiff'
    image_utils.write_hyperstack_tiff(
        iter([data]),
        hyperstack,
        (1, 1, 1, *data.shape),
        data.dtype,
        image_utils.build_hyperstack_output_metadata(
            [image_utils.read_file_record(still)],
            channel_names=['BF'],
            channel_colors=[release_color('BF')],
            plane_positions={
                'PositionX': [12.5],
                'PositionY': [30.0],
                'PositionZ': [1500.0],
                'DeltaT': [0.0],
            },
            significant_bits=12,
            pixel_size_um=None,
        ),
        {'photometric': 'minisblack'},
    )

    def _ome_parts(path) -> dict:
        instrument = valid_description(path).find(q('Instrument'))
        return {
            part.tag.removeprefix(f'{{{OME}}}').lower(): sorted(set(part.attrib) - {'ID'})
            for part in instrument
        }

    stated = {
        'imagej': {part.lower(): sorted(block) for part, block in imagej['Instrument'].items()},
        'ome': _ome_parts(still),
        'hyperstack': _ome_parts(hyperstack),
    }
    expected = {
        'microscope': ['Manufacturer', 'Model'],
        'objective': ['LensNA', 'NominalMagnification', 'WorkingDistance', 'WorkingDistanceUnit'],
        'detector': ['Model'],
    }
    assert stated == dict.fromkeys(stated, expected)
    assert sorted(imagej['Plate']) == ['Columns', 'Name', 'Rows', 'Standard']
