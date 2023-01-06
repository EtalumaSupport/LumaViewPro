import json
import numpy as np

class LabWare(object):
    """A class that stores and computes actions for objective labware"""

    def __init__(self, *arg):
        self.x = 75
        self.y = 25
        self.z = 1
        self.load_all()

    def load_all(self):
        # Load all Possible Labware from JSON
        with open('./data/labware.json', "r") as read_file:
            self.labware = json.load(read_file)

class Slide(LabWare):
    """A class that stores and computes actions for slides labware"""

    def __init__(self, *arg):
        super(Slide, self).__init__()
        self.covered = True


class WellPlate(LabWare):
    """A class that stores and computes actions for wellplate labware"""

    def __init__(self, *arg):
        super(WellPlate, self).__init__()
 
        # PLATE SPECIFIC
        self.stage_x = 4.5-11.005 # offset from stage to bottom right corner of well plate
        self.stage_y = 1.5-7.865  # offset from stage to bottom right corner of well plate

        self.plate = []    # All plate information from JSON file
        self.ind_list = [] # ordered list of all well indices 
        self.pos_list = [] # ordered list of all well positions
        self.move = "S"    # scan pattern (S for snake)

    def load_plate(self, plate_key):
        self.plate = self.labware['Wellplate'][plate_key]

    # set indices based on plate and motion
    def set_indices(self):

        self.ind_list = []

        for j in range(self.plate['rows']):
            for i in range(self.plate['columns']):
                if j % 2 == 1:
                    i = self.plate['columns'] - i - 1

                self.ind_list.append([i, j])

    # set positions based on indices
    def set_positions(self): 

        self.set_indices()
        self.pos_list = []

        for i in self.ind_list:
            x, y = self.get_well_position(i[0], i[1])
            self.pos_list.append([x, y])
           
    # Figure out index of well based on position of xy
    def get_well_index(self, x, y):

        sx = self.stage_x
        px = self.plate['dimensions']['x']
        ox = self.plate['offset']['x']
        dx = self.plate['spacing']['x']
        i = -(x-sx-px+ox)/dx

        sy = self.stage_y
        py = self.plate['dimensions']['y']
        oy = self.plate['offset']['y']
        dy = self.plate['spacing']['y']
        j = -(y-sy-py+oy)/dy

        # i = (x - self.plate['offset']['x']) / self.plate['spacing']['x']
        # j = (y - self.plate['offset']['y']) / self.plate['spacing']['y']

        i = round(i)
        j = round(j)
        i = np.clip(i, 0, self.plate['columns']-1)
        j = np.clip(j, 0, self.plate['rows']-1)
        return i, j

    # Get real well position in mm given its index
    def get_well_position(self, i, j):
        x = self.stage_x + self.plate['dimensions']['x'] - \
            (self.plate['offset']['x'] + i*self.plate['spacing']['x'])
        y = self.stage_y + self.plate['dimensions']['y'] - \
            (self.plate['offset']['y'] + j*self.plate['spacing']['y'])
        return x, y

    # Figure out index of well based on position of xy
    def get_screen_position(self, x, y):
        sx = self.stage_x
        px = self.plate['dimensions']['x']
        ox = self.plate['offset']['x']
        dx = self.plate['spacing']['x']
        i = -(x-sx-px+ox)/dx

        sy = self.stage_y
        py = self.plate['dimensions']['y']
        oy = self.plate['offset']['y']
        dy = self.plate['spacing']['y']
        j = -(y-sy-py+oy)/dy
        
        # i = (x - self.plate['offset']['x']) / self.plate['spacing']['x']
        # j = (y - self.plate['offset']['y']) / self.plate['spacing']['y']
        return i, j


class PitriDish(LabWare):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, *arg):
        super(PitriDish, self).__init__()
        self.diameter = 100
        self.z = 20
