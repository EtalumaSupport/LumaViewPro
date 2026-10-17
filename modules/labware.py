# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from dataclasses import dataclass

from modules.finite_number import refuse_unless_finite_number
from modules.api_surface import api, api_fields


@api_fields(
    'name',
    'rows',
    'columns',
    'spacing_x_mm',
    'spacing_y_mm',
    'offset_x_mm',
    'offset_y_mm',
    'width_mm',
    'height_mm',
    'well_diameter_mm',
    'standard',
)
@dataclass(frozen=True)
class WellPlate:
    """One multi-well plate of the installation's catalogue, read once at bring-up.

    Plate positions are in mm. ``offset_*`` is the first well's centre and
    ``spacing_*`` the pitch between well centres; ``width_mm`` and
    ``height_mm`` the plate's outline. ``well_diameter_mm`` is None where
    the catalogue states none. A plate with no wells (the Blank labware)
    has ``rows`` and ``columns`` 0.
    """

    name: str
    rows: int
    columns: int
    spacing_x_mm: float
    spacing_y_mm: float
    offset_x_mm: float
    offset_y_mm: float
    width_mm: float
    height_mm: float
    well_diameter_mm: float | None
    standard: str

    def get_positions_with_labels(self) -> list[tuple[float, float, str]]:
        """Every well's centre and label, row by row from the top, each row
        the reverse of the one before so the stage snakes across the plate."""
        positions = []
        for j in range(self.rows):
            for i in range(self.columns):
                if j % 2 == 1:
                    i = self.columns - i - 1
                x, y = self.get_well_position(i, j)
                positions.append((x, y, self.get_well_label(x=x, y=y)))
        return positions

    # Get center position of well on plate in mm given its index (i, j)
    @api(in_process=True, reads=True)
    def get_well_position(self, i: int, j: int) -> tuple[float, float]:

        x = self.offset_x_mm + i * self.spacing_x_mm
        y = self.offset_y_mm + j * self.spacing_y_mm
        return x, y

    @api(in_process=True, reads=True)
    def has_wells(self) -> bool:
        """True when the plate defines at least one well. A zero-well plate
        (the Blank labware) has no well grid: no well index exists, labels
        are empty, and well UI decorations do not apply."""
        return self.rows * self.columns > 0

    @api(in_process=True, reads=True)
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

        i = (x - self.offset_x_mm) / self.spacing_x_mm
        j = (y - self.offset_y_mm) / self.spacing_y_mm

        i = round(i)
        j = round(j)
        if not (0 <= i < self.columns and 0 <= j < self.rows):
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

    @api(in_process=True, reads=True)
    def get_dimensions(self) -> dict:
        """The plate's outline in mm, as ``{'x': width, 'y': height}``."""
        return {'x': self.width_mm, 'y': self.height_mm}
