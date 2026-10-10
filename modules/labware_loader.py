# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import copy
import logging
import pathlib
import typing

import modules.labware as labware
from modules.exceptions import CatalogueNameRefusedError, ConfigError
from modules.path_utils import read_installation_file, resolve_data_file
from modules.api_surface import api

if typing.TYPE_CHECKING:
    from modules.settings_init import SettingValue

logger = logging.getLogger('LVP.modules.labware_loader')

_REQUIRED_WELLPLATE_FIELDS = {
    'columns': int,
    'rows': int,
    'dimensions': dict,
    'spacing': dict,
    'offset': dict,
}

_REQUIRED_DIMENSION_FIELDS = {'x': (int, float), 'y': (int, float)}

# The plate of a scope with no XY stage: one field, at the centre, with no
# wells to move between.
CENTER_PLATE = 'Center Plate'

# Plate names the catalogue has retired, and the key each now lives under.
# A protocol or settings file written before a rename still carries the old
# spelling; it is translated wherever a name enters the program, once, so
# nothing past that edge ever compares an old spelling to a key.
_LABWARE_ALIASES = {
    '384 well Corning Spheroid Microplate': '384 well microplate',
    # The catalogue's own rename: 'Center Dish' shipped, then became
    # 'Center Plate'; files saved before that carry the old key.
    'Center Dish': CENTER_PLATE,
}


def canonical_plate_name(plate_key: object) -> object:
    """The catalogue spelling of ``plate_key``; a name the table does not
    know is returned as given.

    This translates and does not judge: it needs no catalogue, so settings
    preparation can call it before any helper exists. Whether the plate
    exists is ``WellPlateLoader.resolve_plate_key``'s question.
    """
    if not isinstance(plate_key, str):
        return plate_key
    return _LABWARE_ALIASES.get(plate_key, plate_key)


def _validate_labware(labware: dict, filepath: pathlib.Path) -> None:
    """Validate labware.json: check structure and required fields per entry.

    Only 'Wellplate' entries require the full columns/rows/dimensions/spacing/
    offset schema. Slide and Petri dish have simpler structures and are not
    validated beyond type.
    """
    for category, items in labware.items():
        if not isinstance(items, dict):
            logger.warning(f"[Labware   ] category '{category}' should be dict in {filepath}")
            continue
        # Only Wellplate category has the full schema -- others have different shapes
        if category != 'Wellplate':
            continue
        for name, entry in items.items():
            if not isinstance(entry, dict):
                logger.warning(f"[Labware   ] '{category}/{name}' should be dict in {filepath}")
                continue
            for field, expected_type in _REQUIRED_WELLPLATE_FIELDS.items():
                if field not in entry:
                    logger.warning(
                        f"[Labware   ] '{category}/{name}' missing '{field}' in {filepath}"
                    )
                elif not isinstance(entry[field], expected_type):
                    logger.warning(
                        f"[Labware   ] '{category}/{name}'.'{field}' should be "
                        f'{expected_type.__name__}, got {type(entry[field]).__name__} in {filepath}'
                    )
            # Check nested dimension/spacing/offset dicts
            for subfield in ('dimensions', 'spacing', 'offset'):
                sub = entry.get(subfield)
                if isinstance(sub, dict):
                    for coord, _coord_type in _REQUIRED_DIMENSION_FIELDS.items():
                        if coord not in sub:
                            logger.warning(
                                f"[Labware   ] '{category}/{name}'.'{subfield}' "
                                f"missing '{coord}' in {filepath}"
                            )


class LabwareLoader:
    """A class that stores and computes actions for objective labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        self.x = 75
        self.y = 25
        self.z = 1

        # Load all Possible Labware from JSON
        filepath = resolve_data_file('labware.json', source_path=source_path)
        self.labware = read_installation_file(filepath)

        _validate_labware(self.labware, filepath)


class SlideLoader(LabwareLoader):
    """A class that stores and computes actions for slides labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super().__init__(*arg, source_path=source_path)
        self.covered = True


class WellPlateLoader(LabwareLoader):
    """A class that stores and computes actions for wellplate labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super().__init__(*arg, source_path=source_path)

    @api
    def get_plate_list(self) -> list[str]:
        return list(self.labware['Wellplate'].keys())

    @api
    def resolve_plate_key(self, plate_key: str) -> str:
        """The catalogue key for ``plate_key``, whatever spelling it arrived in.

        Raises:
            CatalogueNameRefusedError: ``'labware_unknown'``, ``plate_key``
                names a plate this catalogue does not have; ``offered``
                carries the plates it has.
        """
        resolved_key = canonical_plate_name(plate_key)
        if resolved_key not in self.labware['Wellplate']:
            raise CatalogueNameRefusedError(
                'labware_unknown',
                argument='plate_key',
                value=plate_key,
                offered=tuple(self.get_plate_list()),
            )
        return resolved_key

    @api
    def is_known_plate(self, plate_key: 'SettingValue') -> bool:
        """Whether ``plate_key`` resolves to a plate, directly or under a retired spelling.

        Use this for validation so callers accept exactly what get_plate() accepts.
        get_plate_list() returns only canonical keys and would reject legacy/alias
        names that get_plate() would resolve correctly at runtime. Takes
        whatever a stored setting can hold, since bring-up asks it of the
        stored plate: a value that is not a name is no plate.
        """
        if not isinstance(plate_key, str):
            return False
        try:
            self.resolve_plate_key(plate_key)
        except ConfigError:
            return False
        return True

    @api
    def get_plate(self, plate_key: str) -> labware.WellPlate:
        # The caller's own copy of the row: a changed plate would otherwise
        # change the catalogue every later reader gets.
        return labware.WellPlate(
            config=copy.deepcopy(self.labware['Wellplate'][self.resolve_plate_key(plate_key)])
        )


class PitriDishLoader(LabwareLoader):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super().__init__(*arg, source_path=source_path)
        self.diameter = 100
        self.z = 20
