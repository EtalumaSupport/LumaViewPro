# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved frame's plate metadata names the plate it was taken on.

The plate block of a frame's metadata (the Plate a reader sees) carries the
plate's rows, columns and standard, the well the frame was taken in, and its
name: the catalogue name of the plate handed to the save, never the
microscope's model.
"""

from modules import image_save
from tests.frame_records import frame_record, plate


def test_the_plate_block_names_the_plate(sim_scope):
    metadata = image_save.generate_image_metadata(
        sim_scope,
        channel='BF',
        plate_x_mm=12.0,
        plate_y_mm=8.0,
        stage_z_um=0.0,
        objective_id='10x Oly',
        frame_record=frame_record(),
        labware=plate('384 well microplate'),
        well_label='A1',
        fallback_focus=None,
    )

    assert metadata['plate'] == {
        'name': '384 well microplate',
        'rows': 16,
        'columns': 24,
        'standard': 'ANSI SLAS 2004',
        'well_label': 'A1',
    }
