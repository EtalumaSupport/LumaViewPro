# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Coordinate transformations between stage and plate spaces.

Stage coordinates: um from bottom-right origin (motor units).
Plate coordinates: mm from top-left origin (labware/well positions).

Both transforms use the labware dimensions and the stage offset (um) for
the current plate mounting position.
"""

import dataclasses

from modules.api_surface import api_fields
from modules.finite_number import refuse_unless_finite_number
import modules.labware as lw


@api_fields('min_um', 'max_um')
@dataclasses.dataclass(frozen=True)
class AxisLimits:
    """An axis's travel, in um: a move outside it is refused."""

    min_um: float
    max_um: float


@api_fields('x_um', 'y_um')
@dataclasses.dataclass(frozen=True)
class StagePosition:
    """A stage X/Y position, in um."""

    x_um: float
    y_um: float


@api_fields('x_mm', 'y_mm')
@dataclasses.dataclass(frozen=True)
class PlatePosition:
    """A position on a plate, in mm from the plate's top-left corner."""

    x_mm: float
    y_mm: float


@api_fields('x_mm', 'y_mm', 'z_um')
@dataclasses.dataclass(frozen=True)
class PlateLocation:
    """Where the scope is on a plate, focus included: X and Y in mm on the plate, Z in um.

    None on an axis is no position there: a move to it leaves that axis where it is.
    """

    x_mm: float | None
    y_mm: float | None
    z_um: float | None


@api_fields('x_um', 'y_um')
@dataclasses.dataclass(frozen=True)
class StageOffset:
    """Where the plate is mounted on the stage, in um: the offset the two conversions apply."""

    x_um: float
    y_um: float


def offset_axes(stage_offset: StageOffset) -> tuple[float, float]:
    """The offset's X and Y, each refused unless a finite number: a NaN would convert to a NaN position."""
    refuse_unless_finite_number(stage_offset.x_um, 'stage_offset.x_um')
    refuse_unless_finite_number(stage_offset.y_um, 'stage_offset.y_um')
    return stage_offset.x_um, stage_offset.y_um


class CoordinateTransformer:
    def stage_to_plate(
        self,
        labware: lw.Labware,
        stage_offset: StageOffset,
        x_um: float,
        y_um: float,
    ) -> PlatePosition:
        """Convert a stage position (um) to a plate position (mm).

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, ``x_um``, ``y_um`` or
                an offset axis is not a finite number.
        """
        for name, value in (('x_um', x_um), ('y_um', y_um)):
            refuse_unless_finite_number(value, name)
        offset_x, offset_y = offset_axes(stage_offset)
        dim_max = labware.get_dimensions()

        return PlatePosition(
            x_mm=dim_max['x'] - (offset_x + x_um) / 1000,
            y_mm=dim_max['y'] - (offset_y + y_um) / 1000,
        )

    def plate_to_stage(
        self,
        labware: lw.Labware,
        stage_offset: StageOffset,
        x_mm: float,
        y_mm: float,
    ) -> StagePosition:
        """Convert a plate position (mm) to a stage position (um).

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, ``x_mm``, ``y_mm`` or
                an offset axis is not a finite number.

        This is a pure transform and does NOT bound its input: callers
        that ENUMERATE candidate positions -- protocol validation, tile
        generation -- convert in order to test the result and report
        every out-of-range entry at once, which a raise here would cut
        short at the first one. Callers that COMMAND a move pass the
        plate coordinate to the motion API instead, which owns the bound
        and refuses in this frame. A warning logged here served neither:
        the enumerators do their own checking, and no user reads it.
        """
        for name, value in (('x_mm', x_mm), ('y_mm', y_mm)):
            refuse_unless_finite_number(value, name)
        offset_x, offset_y = offset_axes(stage_offset)

        dim_max = labware.get_dimensions()

        return StagePosition(
            x_um=(dim_max['x'] - offset_x / 1000 - x_mm) * 1000,  # mm -> um
            y_um=(dim_max['y'] - offset_y / 1000 - y_mm) * 1000,
        )
