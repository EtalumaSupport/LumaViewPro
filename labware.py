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
 
        self.plate = []    # All plate information from JSON file
        self.ind_list = [] # ordered list of all well indices 
        self.pos_list = [] # ordered list of all well positions

    def load_plate(self, plate_key):
        self.plate = self.labware['Wellplate'][plate_key]

    # set indices based on plate and motion
    def set_indices(self):

        self.ind_list = []
        # 'i' represents column index (x-direction)
        # 'j' represents row index (y-direction)
        # (0, 0) on the bottom left looking from above

        # start at the top-left when looking from above
        for j in range(self.plate['rows']):            
            for i in range(self.plate['columns']):
                if j % 2 == 1:
                    i = self.plate['columns'] - i - 1
                self.ind_list.append([i,j])

    # set positions based on indices
    def set_positions(self): 

        self.set_indices()
        self.pos_list = []

        for i in self.ind_list:
            x, y = self.get_well_position(i[0], i[1])
            self.pos_list.append([x, y])

    # Get center position of well on plate in mm given its index (i, j)
    def get_well_position(self, i, j):

        dx = self.plate['spacing']['x'] # distance b/w wells x-dir
        ox = self.plate['offset']['x']  # offset to first well x-dir
        x = ox+i*dx                     # x position in mm of well
        
        dy = self.plate['spacing']['y']    # distance b/w wells y-dir
        oy = self.plate['offset']['y']     # offset to top well y-dir
        y = oy+j*dy
            
        return x, y
    
    # Get index of closest well based on plate position (x, y) in mm
    def get_well_index(self, x, y):

        ox = self.plate['offset']['x']     # offset to first well x-dir
        dx = self.plate['spacing']['x']    # distance b/w wells x-dir
        i = (x-ox)/dx

        dy = self.plate['spacing']['y']    # distance b/w wells y-dir
        oy = self.plate['offset']['y']     # offset to top well y-dir
        j = (y-oy)/dy
        
        i = round(i)
        j = round(j)
        i = np.clip(i, 0, self.plate['columns']-1)
        j = np.clip(j, 0, self.plate['rows']-1)
        return i, j


class PitriDish(LabWare):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, *arg):
        super(PitriDish, self).__init__()
        self.diameter = 100
        self.z = 20
