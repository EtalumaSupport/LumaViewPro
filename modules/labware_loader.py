# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import json
import logging
import pathlib

import modules.labware as labware
from modules.path_utils import resolve_data_file

logger = logging.getLogger('LVP.modules.labware_loader')

_REQUIRED_WELLPLATE_FIELDS = {
    'columns': int,
    'rows': int,
    'dimensions': dict,
    'spacing': dict,
    'offset': dict,
}

_REQUIRED_DIMENSION_FIELDS = {'x': (int, float), 'y': (int, float)}


def _validate_labware(labware: dict, filepath: str) -> None:
    """Validate labware.json: check structure and required fields per entry.

    Only 'Wellplate' entries require the full columns/rows/dimensions/spacing/
    offset schema. Slide and Petri dish have simpler structures and are not
    validated beyond type.
    """
    if not isinstance(labware, dict):
        raise ValueError(f'labware.json at {filepath}: expected dict, got {type(labware).__name__}')
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
        try:
            with open(filepath) as read_file:
                self.labware = json.load(read_file)
        except FileNotFoundError as e:
            logger.error(f'[Labware   ] labware.json not found at {filepath}')
            raise RuntimeError(
                f'Required file labware.json not found at {filepath}. '
                'Please reinstall or restore from backup.'
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f'[Labware   ] labware.json is corrupt: {e}')
            raise RuntimeError(
                f'labware.json is corrupt ({e}). Please restore from backup or reinstall.'
            ) from e

        _validate_labware(self.labware, filepath)


class SlideLoader(LabwareLoader):
    """A class that stores and computes actions for slides labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super().__init__(*arg, source_path=source_path)
        self.covered = True


class WellPlateLoader(LabwareLoader):
    """A class that stores and computes actions for wellplate labware"""

    # Compatibility aliases for labware names that were renamed.
    # Protocols saved with the old name still load correctly.
    _LABWARE_ALIASES = {
        '384 well Corning Spheroid Microplate': '384 well microplate',
    }

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super().__init__(*arg, source_path=source_path)

    def get_plate_list(self):
        return list(self.labware['Wellplate'].keys())

    def is_known_plate(self, plate_key) -> bool:
        """Return True if plate_key resolves to a known plate (directly or via alias).

        Use this for validation so callers accept exactly what get_plate() accepts.
        get_plate_list() returns only canonical keys and would reject legacy/alias
        names that get_plate() would resolve correctly at runtime.
        """
        resolved_key = self._LABWARE_ALIASES.get(plate_key, plate_key)
        return resolved_key in self.labware['Wellplate']

    def get_plate(self, plate_key):
        # Apply alias mapping for backwards compatibility
        resolved_key = self._LABWARE_ALIASES.get(plate_key, plate_key)
        if resolved_key != plate_key:
            logger.info(f"[Labware   ] Aliased labware '{plate_key}' -> '{resolved_key}'")
        return labware.WellPlate(config=self.labware['Wellplate'][resolved_key])


class PitriDishLoader(LabwareLoader):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super().__init__(*arg, source_path=source_path)
        self.diameter = 100
        self.z = 20
