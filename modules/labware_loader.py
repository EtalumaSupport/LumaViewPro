# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import pathlib
import types
import typing
from collections.abc import Mapping

import modules.labware as labware
from modules.exceptions import CatalogueNameRefusedError, ConfigError, InstallationFileError
from modules.path_utils import read_installation_file, resolve_data_file
from modules.api_surface import api

if typing.TYPE_CHECKING:
    from modules.settings_init import SettingValue

logger = logging.getLogger('LVP.modules.labware_loader')

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


def _number(entry: dict, path: tuple[str, ...], name: str, filepath: pathlib.Path) -> float:
    value = entry
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise InstallationFileError(filepath, f'gives plate {name!r} no {".".join(path)!r}')
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InstallationFileError(
            filepath, f'gives plate {name!r} a {".".join(path)!r} that is not a number'
        )
    return value


def _read_plate(name: str, entry: object, filepath: pathlib.Path) -> labware.WellPlate:
    """The record for one plate of the catalogue, or a refusal naming the file, the plate and the key."""
    if not isinstance(entry, dict):
        raise InstallationFileError(filepath, f'has a plate {name!r} that is not a plate')
    counts = {}
    for key in ('rows', 'columns'):
        if isinstance(entry.get(key), bool) or not isinstance(entry.get(key), int):
            raise InstallationFileError(filepath, f'gives plate {name!r} no whole-number {key!r}')
        counts[key] = entry[key]
    standard = entry.get('standard')
    if not isinstance(standard, str):
        raise InstallationFileError(filepath, f"gives plate {name!r} no 'standard'")
    diameter = _number(entry, ('diameter',), name, filepath)
    return labware.WellPlate(
        name=name,
        **counts,
        spacing_x_mm=_number(entry, ('spacing', 'x'), name, filepath),
        spacing_y_mm=_number(entry, ('spacing', 'y'), name, filepath),
        offset_x_mm=_number(entry, ('offset', 'x'), name, filepath),
        offset_y_mm=_number(entry, ('offset', 'y'), name, filepath),
        width_mm=_number(entry, ('dimensions', 'x'), name, filepath),
        height_mm=_number(entry, ('dimensions', 'y'), name, filepath),
        # The file's -1 is its word for a diameter it does not state.
        well_diameter_mm=None if diameter == -1 else diameter,
        standard=standard,
    )


class WellPlateLoader:
    """The installation's plates (labware.json's ``Wellplate``), read once.

    Slides and Petri dishes are in the file but are not labware the program
    offers yet, so they are not read.
    """

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        filepath = resolve_data_file('labware.json', source_path=source_path)
        catalogue = read_installation_file(filepath)
        if not isinstance(catalogue, dict) or not isinstance(catalogue.get('Wellplate'), dict):
            raise InstallationFileError(filepath, "has no 'Wellplate' table of plates")
        self._plates = types.MappingProxyType(
            {
                name: _read_plate(name, entry, filepath)
                for name, entry in catalogue['Wellplate'].items()
            }
        )

    @property
    def plates(self) -> Mapping[str, labware.WellPlate]:
        """The catalogue, by name; read-only, as each record is."""
        return self._plates

    @api(reads=True)
    def get_plate_list(self) -> list[str]:
        return list(self._plates)

    @api(reads=True)
    def resolve_plate_key(self, plate_key: str) -> str:
        """The catalogue key for ``plate_key``, whatever spelling it arrived in.

        Raises:
            CatalogueNameRefusedError: ``'labware_unknown'``, ``plate_key``
                names a plate this catalogue does not have; ``offered``
                carries the plates it has.
        """
        resolved_key = canonical_plate_name(plate_key)
        if resolved_key not in self._plates:
            raise CatalogueNameRefusedError(
                'labware_unknown',
                argument='plate_key',
                value=plate_key,
                offered=tuple(self.get_plate_list()),
            )
        return resolved_key

    @api(reads=True)
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

    @api(reads=True)
    def get_plate(self, plate_key: str) -> labware.WellPlate:
        """The plate ``plate_key`` names, under any spelling ``resolve_plate_key`` takes.

        Raises:
            CatalogueNameRefusedError: ``'labware_unknown'``, as
                ``resolve_plate_key``.
        """
        return self._plates[self.resolve_plate_key(plate_key)]
