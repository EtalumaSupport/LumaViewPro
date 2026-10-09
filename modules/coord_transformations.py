# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Coordinate transformations between stage, plate, and pixel spaces.

Stage coordinates: um from bottom-right origin (motor units).
Plate coordinates: mm from top-left origin (labware/well positions).
Pixel coordinates: px from bottom-left origin (screen display).

All transforms use the labware dimensions and stage_offset (um) for
the current plate mounting position.
"""

from modules.finite_number import refuse_unless_finite_number
import modules.labware as lw


class CoordinateTransformer:
    def stage_to_plate(
        self,
        labware: lw.LabWare,
        stage_offset: dict[str, float],
        sx: float,
        sy: float,
    ) -> tuple[float, float]:
        """Convert stage coordinates (um) to plate coordinates (mm).

        Args:
            labware: Labware object with get_dimensions().
            stage_offset: {'x': float, 'y': float} in um.
            sx, sy: Stage position in um.

        Returns:
            (px, py): Plate position in mm.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, ``sx`` or ``sy`` is not
                a finite number.
        """
        for name, value in (('sx', sx), ('sy', sy)):
            refuse_unless_finite_number(value, name)
        dim_max = labware.get_dimensions()

        px = dim_max['x'] - (stage_offset['x'] + sx) / 1000
        py = dim_max['y'] - (stage_offset['y'] + sy) / 1000

        return px, py

    def plate_to_stage(
        self,
        labware: lw.LabWare,
        stage_offset: dict[str, float],
        px: float,
        py: float,
    ) -> tuple[float, float]:
        """Convert plate coordinates (mm) to stage coordinates (um).

        Args:
            labware: Labware object with get_dimensions().
            stage_offset: {'x': float, 'y': float} in um.
            px, py: Plate position in mm.

        Returns:
            (sx, sy): Stage position in um.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, ``px`` or ``py`` is not
                a finite number.

        This is a pure transform and does NOT bound its input: callers
        that ENUMERATE candidate positions -- protocol validation, tile
        generation -- convert in order to test the result and report
        every out-of-range entry at once, which a raise here would cut
        short at the first one. Callers that COMMAND a move pass the
        plate coordinate to the motion API instead, which owns the bound
        and refuses in this frame. A warning logged here served neither:
        the enumerators do their own checking, and no user reads it.
        """
        for name, value in (('px', px), ('py', py)):
            refuse_unless_finite_number(value, name)

        dim_max = labware.get_dimensions()

        sx = (dim_max['x'] - stage_offset['x'] / 1000 - px) * 1000  # mm -> um
        sy = (dim_max['y'] - stage_offset['y'] / 1000 - py) * 1000

        return sx, sy

    def plate_to_pixel(
        self,
        labware: lw.LabWare,
        px: float,
        py: float,
        scale_x: float,
        scale_y: float,
    ) -> tuple[float, float]:
        """Convert plate coordinates (mm) to pixel coordinates (px).

        Args:
            labware: Labware object with get_dimensions().
            px, py: Plate position in mm.
            scale_x, scale_y: Pixels per mm.

        Returns:
            (pixel_x, pixel_y): Screen position in pixels.
        """
        dim_max = labware.get_dimensions()

        pixel_x = px * scale_x
        pixel_y = (dim_max['y'] - py) * scale_y

        return pixel_x, pixel_y

    def stage_to_pixel(
        self,
        labware: lw.LabWare,
        stage_offset: dict[str, float],
        sx: float,
        sy: float,
        scale_x: float,
        scale_y: float,
    ) -> tuple[float, float]:
        """Convert stage coordinates (um) to pixel coordinates (px)."""
        px, py = self.stage_to_plate(
            labware=labware,
            stage_offset=stage_offset,
            sx=sx,
            sy=sy,
        )
        pixel_x, pixel_y = self.plate_to_pixel(
            labware=labware,
            px=px,
            py=py,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        return pixel_x, pixel_y
