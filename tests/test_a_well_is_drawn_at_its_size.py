# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A plate's wells are drawn at their size.

The shipped plates state their wells' diameters; a plate whose file says
-1 (a custom plate, the Center Plate, the Four-Slide Holder) states none.
"""

from _lumascope.modules.labware_loader import WellPlateLoader


def test_a_stated_diameter_is_the_wells_size_on_both_axes():
    assert WellPlateLoader().get_plate('96 well microplate').well_size_mm == (7, 7)


def test_a_well_with_no_stated_diameter_fills_its_spacing_on_each_axis():
    """The Four-Slide Holder's spacing is 30.2 by 76.2: each slide's well fills it, unequal kept unequal."""
    plate = WellPlateLoader().get_plate('Four-Slide Holder')
    assert plate.well_diameter_mm is None
    assert plate.well_size_mm == (30.2, 76.2)


def test_no_shipped_plate_states_a_well_wider_than_its_spacing():
    """A stated diameter fits its spacing on both axes, so no two drawn wells overlap."""
    loader = WellPlateLoader()
    too_wide = [
        (name, plate.well_diameter_mm, plate.spacing_x_mm, plate.spacing_y_mm)
        for name in loader.get_plate_list()
        if (plate := loader.get_plate(name)).well_diameter_mm is not None
        and plate.well_diameter_mm > min(plate.spacing_x_mm, plate.spacing_y_mm)
    ]
    assert too_wide == []
