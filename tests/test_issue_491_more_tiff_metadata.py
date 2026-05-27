# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#491 regression: more TIFF metadata -- ImageJ Document extension +
OME Plate + Instrument blocks.

Bug
---
Customer asked for richer TIFF metadata so downstream tooling can
extract instrument identity + plate identity per image. The existing
ImageJ Document block carried only Manufacturer/Device/WellLabel/
WellSite; the OME path had no Plate or Instrument blocks at all.

Fix
---
Per the Option-A scope agreed with Eric:
- ImageJ Document block gets Model, SerialNumber, FirmwareVersion,
  CameraModel, PlateName, PlateRows, PlateColumns.
- OME tiff_metadata gets a top-level Instrument dict (Microscope +
  Objective + Detector sub-blocks) and a top-level Plate dict
  (Name + Rows + Columns + WellLabel).

Not shipped in this commit (documented as gap): OME LightSource
(LED wavelength + power; not tracked per-color), OME Detector gain/
zoom (partial -- model only), OME FilterSet (filter wheel + dichroic
not tracked).

Test approach
-------------
Direct functional test on generate_tiff_data: pass a representative
metadata dict (with 'instrument' + 'plate' sub-dicts) and assert the
output tiff_metadata contains Instrument.Microscope.{Manufacturer,
Model, SerialNumber}, Instrument.Objective.{Magnification, LensNA},
Plate.{Name, Rows, Columns}, and -- for the ImageJ path -- the
extended Document block with Model + SerialNumber + CameraModel +
PlateName.
"""

from __future__ import annotations

import numpy as np

from modules import image_utils


def _build_metadata():
    return {
        'camera_make': 'Etaluma',
        'microscope': 'LS850',
        'microscope_model': 'LS850',
        'channel': 'BF',
        'datetime': '2026:05:26 23:00:00',
        'objective': {
            'model': '4x Plan Apochromat',
            'manufacturer': 'Olympus',
            'magnification': 4,
            'aperture': 0.16,
            'working_distance': 13.0,
            'immersion': 'Air',
            'focal_length': 45.0,
        },
        'plate_pos_mm': {'x': 12.5, 'y': 30.0},
        'z_pos_um': 1500.0,
        'exposure_time_ms': 10.0,
        'gain_db': 1.0,
        'illumination_ma': 50.0,
        'pixel_size_um': 2.5,
        'well_label': 'A1',
        'well_site': 0,
        'instrument': {
            'manufacturer': 'Etaluma',
            'model': 'LS850',
            'serial_number': 'EL0940-05',
            'firmware_version': '3.0.7',
            'camera_model': 'Basler a2A3536-31umBAS',
        },
        'plate': {
            'name': '96 well microplate',
            'rows': 8,
            'columns': 12,
            'standard': 'ANSI SLAS 2004',
        },
    }


def _gen(image_type: str, dtype=np.uint8):
    data = np.zeros((128, 128), dtype=dtype)
    return image_utils.generate_tiff_data(
        data=data,
        metadata=_build_metadata(),
        image_type=image_type,
        color='BF',
    )


def test_ome_path_has_instrument_microscope_block():
    result = _gen(image_type='ome', dtype=np.uint8)
    tm = result['metadata']
    assert 'Instrument' in tm, 'OME tiff_metadata must contain Instrument block (#491)'
    micro = tm['Instrument']['Microscope']
    assert micro['Manufacturer'] == 'Etaluma'
    assert micro['Model'] == 'LS850'
    assert micro['SerialNumber'] == 'EL0940-05'
    assert micro['FirmwareVersion'] == '3.0.7'


def test_ome_path_has_instrument_objective_block():
    result = _gen(image_type='ome', dtype=np.uint8)
    obj = result['metadata']['Instrument']['Objective']
    assert obj['Manufacturer'] == 'Olympus'
    assert obj['Magnification'] == 4
    assert obj['LensNA'] == 0.16
    assert obj['WorkingDistance'] == 13.0
    assert obj['Immersion'] == 'Air'


def test_ome_path_has_instrument_detector_block():
    result = _gen(image_type='ome', dtype=np.uint8)
    det = result['metadata']['Instrument']['Detector']
    assert det['Model'] == 'Basler a2A3536-31umBAS'
    assert det['Type'] == 'CMOS'


def test_ome_path_has_plate_block():
    result = _gen(image_type='ome', dtype=np.uint8)
    tm = result['metadata']
    assert 'Plate' in tm, 'OME tiff_metadata must contain Plate block (#491)'
    plate = tm['Plate']
    assert plate['Name'] == '96 well microplate'
    assert plate['Rows'] == 8
    assert plate['Columns'] == 12
    assert plate['WellLabel'] == 'A1'
    assert plate['Standard'] == 'ANSI SLAS 2004'


def test_ome_path_omits_plate_when_dimensions_missing():
    """Plate block requires rows + columns; degraded labware (slide,
    blank) has neither -- the block should not appear."""
    md = _build_metadata()
    md['plate'] = {'name': 'Slide', 'rows': None, 'columns': None}
    data = np.zeros((128, 128), dtype=np.uint8)
    result = image_utils.generate_tiff_data(
        data=data, metadata=md, image_type='ome', color='BF'
    )
    assert 'Plate' not in result['metadata'], (
        'Plate block should be omitted when rows/columns are unset; '
        'degraded labware (slide/blank) has no plate dimensions to report.'
    )


def test_imagej_document_block_extended():
    result = _gen(image_type='imagej', dtype=np.uint8)
    doc = result['metadata']['Document']
    # Pre-existing keys must still be present.
    assert doc['Manufacturer'] == 'Etaluma'
    assert doc['Device'] == 'LS850'
    assert doc['WellLabel'] == 'A1'
    # New keys per #491.
    assert doc['Model'] == 'LS850'
    assert doc['SerialNumber'] == 'EL0940-05'
    assert doc['FirmwareVersion'] == '3.0.7'
    assert doc['CameraModel'] == 'Basler a2A3536-31umBAS'
    assert doc['PlateName'] == '96 well microplate'
    assert doc['PlateRows'] == 8
    assert doc['PlateColumns'] == 12


def test_ome_path_without_instrument_dict_does_not_crash():
    """If a downstream caller omits the instrument sub-dict (e.g. a
    post-processor that reconstructs metadata), the OME path must
    still produce valid output -- the Instrument block is optional."""
    md = _build_metadata()
    del md['instrument']
    del md['plate']
    data = np.zeros((128, 128), dtype=np.uint8)
    result = image_utils.generate_tiff_data(
        data=data, metadata=md, image_type='ome', color='BF'
    )
    assert 'Instrument' not in result['metadata']
    assert 'Plate' not in result['metadata']
    # Plane data must still be intact.
    assert result['metadata']['Plane']['PositionX'] == 12.5
