# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from modules.finite_number import refuse_unless_finite_number
from modules.api_surface import api, api_fields


class LabWare:
    """A class that stores and computes actions for objective labware"""

    def __init__(self, *arg):
        self.x = 75
        self.y = 25
        self.z = 1


class Slide(LabWare):
    """A class that stores and computes actions for slides labware"""

    def __init__(self, *arg):
        super().__init__()
        self.covered = True


@api_fields('config', record=True)
class WellPlate(LabWare):
    """A class that stores and computes actions for wellplate labware"""

    config: dict

    def __init__(self, config: dict, *arg):
        super().__init__()

        self.config = config
        self.ind_list = []  # ordered list of all well indices
        self.pos_list = []  # ordered list of all well positions

    # set indices based on plate and motion
    def set_indices(self, stitch=1):

        self.ind_list = []
        self.stitch_list = []
        # 'i' represents column index (x-direction)
        # 'j' represents row index (y-direction)
        # (0, 0) on the bottom left looking from above

        # start at the top-left when looking from above
        for j in range(self.config['rows']):
            for i in range(self.config['columns']):
                if j % 2 == 1:
                    i = self.config['columns'] - i - 1
                self.ind_list.append([i, j])

        if stitch > 1:
            for i in range(stitch):
                for j in range(stitch):
                    si = stitch - i - 1 if j % 2 == 1 else i
                    self.stitch_list.append([si, j])

    # set positions based on indices
    def set_positions(self):

        self.set_indices(stitch=1)
        self.pos_list = []

        for well in self.ind_list:
            x, y = self.get_well_position(well[0], well[1])
            self.pos_list.append([x, y])

    def get_positions_with_labels(self) -> tuple[float, float, str]:
        self.set_positions()
        tmp = []
        for pos in self.pos_list:
            x, y = pos
            label = self.get_well_label(x=x, y=y)
            tmp.append((x, y, label))

        return tmp

    # Get center position of well on plate in mm given its index (i, j)
    @api(in_process=True)
    def get_well_position(self, i: int, j: int) -> tuple[float, float]:

        dx = self.config['spacing']['x']  # distance b/w wells x-dir
        ox = self.config['offset']['x']  # offset to first well x-dir
        x = ox + i * dx  # x position in mm of well

        dy = self.config['spacing']['y']  # distance b/w wells y-dir
        oy = self.config['offset']['y']  # offset to top well y-dir
        y = oy + j * dy

        return x, y

    @api(in_process=True)
    def has_wells(self) -> bool:
        """True when the plate defines at least one well. A zero-well plate
        (the Blank labware) has no well grid: no well index exists, labels
        are empty, and well UI decorations do not apply."""
        return self.config['rows'] * self.config['columns'] > 0

    @api(in_process=True)
    def get_well_index(self, x: float, y: float) -> tuple[int, int] | None:
        """The (column, row) of the well centre nearest plate position (x, y) in mm.

        None when that centre is outside the grid: the position is more
        than half a pitch beyond the outer well centres, off the plate's
        wells, or the plate has no wells at all (the Blank labware). No
        shipped plate states a well size, so within the grid a position
        names its nearest well whether or not it is inside it.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, ``x`` or ``y`` is not a
                finite number, which names no well.
        """
        for name, value in (('x', x), ('y', y)):
            refuse_unless_finite_number(value, name)
        if not self.has_wells():
            return None

        ox = self.config['offset']['x']  # offset to first well x-dir
        dx = self.config['spacing']['x']  # distance b/w wells x-dir
        i = (x - ox) / dx

        dy = self.config['spacing']['y']  # distance b/w wells y-dir
        oy = self.config['offset']['y']  # offset to top well y-dir
        j = (y - oy) / dy

        i = round(i)
        j = round(j)
        if not (0 <= i < self.config['columns'] and 0 <= j < self.config['rows']):
            return None
        return i, j

    def get_well_label(self, x: float, y: float) -> str:
        index = self.get_well_index(x=x, y=y)
        if index is None:
            # Off the grid or a plate with none: empty, not a fabricated
            # token, so filename builders and metadata writers omit the well
            # rather than stamping a fake one.
            return ''
        well_x, well_y = index

        # Handling for labware with more than 26 rows
        letter = ''
        if well_y >= 26:
            letter += 'A'
            well_y -= 26

        letter += chr(ord('A') + well_y)
        return f'{letter}{well_x + 1}'

    @api(in_process=True)
    def get_dimensions(self) -> dict:
        return self.config['dimensions']


class PitriDish(LabWare):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, *arg):
        super().__init__()
        self.diameter = 100
        self.z = 20
