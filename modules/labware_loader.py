# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import json

import labware

class LabwareLoader(object):
    """A class that stores and computes actions for objective labware"""

    def __init__(self, *arg):
        self.x = 75
        self.y = 25
        self.z = 1

        # Load all Possible Labware from JSON
        with open('./data/labware.json', "r") as read_file:
            self.labware = json.load(read_file)
        

class SlideLoader(LabwareLoader):
    """A class that stores and computes actions for slides labware"""

    def __init__(self, *arg):
        super(SlideLoader, self).__init__()
        self.covered = True


class WellPlateLoader(LabwareLoader):
    """A class that stores and computes actions for wellplate labware"""

    def __init__(self, *arg):
        super(WellPlateLoader, self).__init__()
  

    def get_plate_list(self):
        return list(self.labware['Wellplate'].keys())
    

    def get_plate(self, plate_key):
        return labware.WellPlate(config=self.labware['Wellplate'][plate_key])


class PitriDishLoader(LabwareLoader):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, *arg):
        super(PitriDishLoader, self).__init__()
        self.diameter = 100
        self.z = 20
