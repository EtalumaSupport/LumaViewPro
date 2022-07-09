from imageio import RETURN_BYTES


class LabWare(object):
    """A class that stores and computes actions for objective labware"""

    def __init__(self, arg):
        self.x = 75
        self.y = 25
        self.z = 1

class Slide(LabWare):
    """A class that stores and computes actions for slides labware"""

    def __init__(self, arg):
        super(, self).__init__()
        self.covered = true

class WellPlate(LabWare):
    """A class that stores and computes actions for wellplate labware"""

    def __init__(self, arg):
        super(, self).__init__()
        self.columns = 1
        self.rows = 1
        self.move = "S"
        self.wells = []
        self.pos_list = []

    def get_positions(self):
        # Generate a list of well positions
        self.pos_list = []
        for j in range(self.rows):
            for i in range(self.columns):
                if j % 2 == 1:
                    i = self.columns - i - 1

                self.pos_list.append([i, j])

    # Get real well position in mm given its index
    def get_well_position(self, i, j):
        x = self.offset['x'] + i*self.spacing['x']
        y = self.offset['y'] + j*self.spacing['y']
        return x, y

    # Figure out index of well based on position of xy
    def get_well_numbers(self, x, y):
        i = (x - self.offset['x']) / self.spacing['x']
        j = (y - self.offset['y']) / self.spacing['y']
        i = round(i)
        j = round(j)
        i = np.clip(i, 0, self.columns-1)
        j = np.clip(j, 0, self.rows-1)
        return i, j
















        
class PitriDish(LabWare):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, arg):
        super(, self).__init__()
        self.diameter = 100
        self.z = 20
