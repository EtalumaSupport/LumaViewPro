# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each shipped objective's focal length times its magnification is its maker's tube length.

An infinity-corrected objective's magnification is the tube lens's focal
length over its own, and each maker designs to one tube length: Olympus to
180 mm, Meiji to 200 mm. An entry whose product is another maker's tube
length is filed under the wrong maker, and one that matches no maker has a
mistyped focal length or magnification; either way the catalogue states
optics the glass does not have. A maker the table does not name fails too,
so a new maker's glass arrives with its tube length.
"""

import pytest

from _lumascope.modules.objectives_loader import ObjectiveLoader

TUBE_LENGTH_MM = {
    'Olympus': 180.0,
    'Meiji': 200.0,
}


def test_each_objective_is_its_makers_tube_length():
    offenders = []
    for objective in ObjectiveLoader().objectives.values():
        product = objective.focal_length_mm * objective.magnification
        tube = TUBE_LENGTH_MM.get(objective.manufacturer)
        if tube is None or product != pytest.approx(tube, abs=0.01):
            offenders.append(
                f'{objective.id} ({objective.manufacturer}): '
                f'{objective.focal_length_mm} mm x {objective.magnification} = {product:g} mm, '
                f'the maker designs to {tube} mm'
            )
    assert not offenders, '\n'.join(offenders)
