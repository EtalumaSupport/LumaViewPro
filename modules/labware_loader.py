# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import json
import logging
import pathlib

import modules.labware as labware
from modules.path_utils import resolve_data_file

logger = logging.getLogger('LVP.modules.labware_loader')

class LabwareLoader(object):
    """A class that stores and computes actions for objective labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        self.x = 75
        self.y = 25
        self.z = 1

        # Load all Possible Labware from JSON
        filepath = resolve_data_file("labware.json", source_path=source_path)
        try:
            with open(filepath, "r") as read_file:
                self.labware = json.load(read_file)
        except FileNotFoundError:
            logger.error(f'[Labware   ] labware.json not found at {filepath}')
            raise RuntimeError(
                f"Required file labware.json not found at {filepath}. "
                "Please reinstall or restore from backup."
            )
        except json.JSONDecodeError as e:
            logger.error(f'[Labware   ] labware.json is corrupt: {e}')
            raise RuntimeError(
                f"labware.json is corrupt ({e}). "
                "Please restore from backup or reinstall."
            )
        

class SlideLoader(LabwareLoader):
    """A class that stores and computes actions for slides labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super(SlideLoader, self).__init__(*arg, source_path=source_path)
        self.covered = True


class WellPlateLoader(LabwareLoader):
    """A class that stores and computes actions for wellplate labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super(WellPlateLoader, self).__init__(*arg, source_path=source_path)
  

    def get_plate_list(self):
        return list(self.labware['Wellplate'].keys())
    

    def get_plate(self, plate_key):
        return labware.WellPlate(config=self.labware['Wellplate'][plate_key])


class PitriDishLoader(LabwareLoader):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        super(PitriDishLoader, self).__init__(*arg, source_path=source_path)
        self.diameter = 100
        self.z = 20
