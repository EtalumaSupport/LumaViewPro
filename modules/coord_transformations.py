# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Coordinate transformations between stage, plate, and pixel spaces.

Stage coordinates: um from bottom-right origin (motor units).
Plate coordinates: mm from top-left origin (labware/well positions).
Pixel coordinates: px from bottom-left origin (screen display).

All transforms use the labware dimensions and stage_offset (um) for
the current plate mounting position.
"""

import logging

import modules.labware as lw

logger = logging.getLogger('LVP.coord_transformations')


class NoLabwareSelectedError(ValueError):
    """Raised when a coord transform is called with no labware selected.

    Per Rule 8 (`docs/CLAUDE.md`): the API boundary must reject invalid
    inputs at the boundary, not crash deep inside `labware.get_dimensions()`.
    Issue #634 fix made `get_selected_labware()` correctly return
    `(None, None)` when no labware is selected; ~12 caller sites then
    crashed with `AttributeError: 'NoneType' has no attribute
    'get_dimensions'` because they never None-checked. This exception
    is the structural answer: the boundary fails fast and informatively;
    the executor's `_safe_callback` translates it into one user-facing
    Rule 14 notification per failure class.
    """


def _require_labware(labware: lw.LabWare | None) -> lw.LabWare:
    """Boundary check shared by every public transform method."""
    if labware is None:
        raise NoLabwareSelectedError(
            "no labware selected — coordinate transforms require a wellplate; "
            "select a labware in Protocol settings"
        )
    return labware


class CoordinateTransformer:

    def stage_to_plate(
        self,
        labware: lw.LabWare,
        stage_offset: dict[str, float],
        sx: float,
        sy: float,
    ):
        """Convert stage coordinates (um) to plate coordinates (mm).

        Args:
            labware: Labware object with get_dimensions().
            stage_offset: {'x': float, 'y': float} in um.
            sx, sy: Stage position in um.

        Returns:
            (px, py): Plate position in mm.

        Raises:
            NoLabwareSelectedError: If labware is None.
        """
        labware = _require_labware(labware)
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
    ):
        """Convert plate coordinates (mm) to stage coordinates (um).

        Args:
            labware: Labware object with get_dimensions().
            stage_offset: {'x': float, 'y': float} in um.
            px, py: Plate position in mm.

        Returns:
            (sx, sy): Stage position in um.

        Raises:
            ValueError: If plate coordinates are out of labware bounds.
            NoLabwareSelectedError: If labware is None.
        """
        labware = _require_labware(labware)
        if not isinstance(px, (int, float)) or not isinstance(py, (int, float)):
            raise ValueError(f"Plate coordinates must be numeric, got ({type(px).__name__}, {type(py).__name__})")

        dim_max = labware.get_dimensions()

        if px < 0 or py < 0:
            logger.warning(f"Plate coordinates negative: ({px:.2f}, {py:.2f})mm — may be out of bounds")
        if px > dim_max['x'] or py > dim_max['y']:
            logger.warning(
                f"Plate coordinates ({px:.2f}, {py:.2f})mm exceed labware dimensions "
                f"({dim_max['x']:.1f}, {dim_max['y']:.1f})mm"
            )

        sx = (dim_max['x'] - stage_offset['x'] / 1000 - px) * 1000  # mm → um
        sy = (dim_max['y'] - stage_offset['y'] / 1000 - py) * 1000

        return sx, sy

    def plate_to_pixel(
        self,
        labware: lw.LabWare,
        px: float,
        py: float,
        scale_x: float,
        scale_y: float,
    ):
        """Convert plate coordinates (mm) to pixel coordinates (px).

        Args:
            labware: Labware object with get_dimensions().
            px, py: Plate position in mm.
            scale_x, scale_y: Pixels per mm.

        Returns:
            (pixel_x, pixel_y): Screen position in pixels.

        Raises:
            NoLabwareSelectedError: If labware is None.
        """
        labware = _require_labware(labware)
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
    ):
        """Convert stage coordinates (um) to pixel coordinates (px).

        Raises:
            NoLabwareSelectedError: If labware is None.
        """
        # _require_labware fires inside stage_to_plate and plate_to_pixel;
        # explicit check here gives a single boundary point if the inner
        # call signature ever changes.
        labware = _require_labware(labware)
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
