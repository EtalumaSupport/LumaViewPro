class LabWare(object):
    """A class that store and computes action for objective labware"""

    def __init__(self, arg):
        self.x = 75
        self.y = 25
        self.z = 1

class Slide(LabWare):
    """A class that store and computes action for sldes labware"""

    def __init__(self, arg):
        super(, self).__init__()
        self.covered = true

class WellPlate(LabWare):
    """A class that store and computes action for wellplate labware"""

    def __init__(self, arg):
        super(, self).__init__()
        self.columns = 1
        self.rows = 1
        self.move = "S"
        self.wells = []

class PitriDish(LabWare):
    """A class that store and computes action for peitri dish labware"""

    def __init__(self, arg):
        super(, self).__init__()
        self.diameter = 100
        self.z = 20
