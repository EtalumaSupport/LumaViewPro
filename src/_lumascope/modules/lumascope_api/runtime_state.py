# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""RuntimeState -- the configuration the scope acts on, split from ScopeCapabilities.

The scope answers for itself on TWO surfaces:

- `scope.capabilities`: IMMUTABLE per Lumascope instance. Reconnect =
  new Lumascope = new capabilities.
- `scope.runtime_state`: answers for the configuration the scope acts
  on, which changes while the scope is up: labware, objective, turret
  config, stage offset, and the coordinate transformer that operates on
  those values. It holds none of them: the session's settings are their
  one store, read through the scope's ``read_setting`` on every answer. The objective catalogue is
  the scope's own (`scope.objective_helper`), read at each use.

The answers for labware / objective / turret / stage live here because
they are user configuration that callers adjust during a session;
capabilities is hardware-identity state that doesn't change without a
reconnect.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.coord_transformations as coord_transformations
from _lumascope.modules.exceptions import (
    AxisStateUnknownError,
    HardwareCommandRefusedError,
    ObjectiveUnknownError,
    PixelSizeUnknownError,
)
from _lumascope.modules.api_surface import api, api_fields
from _lumascope.modules.finite_number import refuse_outside
from _lumascope.modules.objectives_loader import OBJECTIVE_CONFIRMED_ON_A_TURRET
from _lumascope.modules.scope_capabilities import FieldOfView, StageDistance

if TYPE_CHECKING:
    from _lumascope.modules.labware import Labware
    from _lumascope.modules.lumascope_api._lumascope import Lumascope
    from _lumascope.modules.objectives_loader import Objective


def refuse_unless_frame_point(x_fraction: object, y_fraction: object) -> None:
    """Refuse a point of the frame unless each fraction is a number from 0 to 1.

    Raises:
        ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``,
            naming the fraction.
    """
    refuse_outside(x_fraction, 'x_fraction', (0.0, 1.0))
    refuse_outside(y_fraction, 'y_fraction', (0.0, 1.0))


def _distance_to(extent: FieldOfView, x_fraction: float, y_fraction: float) -> StageDistance:
    # From the frame's centre to the point: half the frame each way. Row 0 is
    # the top of the picture and up the picture is stage +Y, so a point above
    # the centre, a fraction under one half, is a positive distance.
    return StageDistance(
        dx_um=(x_fraction - 0.5) * extent.width_um,
        dy_um=(0.5 - y_fraction) * extent.height_um,
    )


@api_fields('objective', 'field_of_view')
@dataclasses.dataclass(frozen=True)
class OpticsStatus:
    """The objective in the light path and the area the camera's frame covers, each its value or why it is unknown.

    The state a display shows, read with no catch: an unknown is answered as
    the refusal that says why, its reason and sentence, never raised and
    never a bare None. One field per value, so a value and a refusal cannot
    both be held.

    Attributes:
        objective: The active objective's id, or ``ObjectiveUnknownError``:
            ``'slot_unknown'``, ``'slot_unassigned'``, ``'not_in_catalogue'``
            or ``'none_selected'``.
        field_of_view: The frame's extent on the sample, or the first reason
            it is unknown, in this order: the objective's
            ``ObjectiveUnknownError``; ``HardwareCommandRefusedError``,
            ``'not_connected'`` naming the camera, or ``'scope_disconnected'``
            after ``disconnect()``; ``PixelSizeUnknownError``,
            ``'optics_undeclared'``, ``'binning_unread'`` or
            ``'frame_size_unread'``.
    """

    objective: str | ObjectiveUnknownError
    field_of_view: (
        FieldOfView | ObjectiveUnknownError | HardwareCommandRefusedError | PixelSizeUnknownError
    )


def slots_holding(objective_id: object, turret_config: dict) -> tuple[int, ...]:
    """The turret slots ``turret_config`` assigns ``objective_id``, ascending."""
    return tuple(sorted(slot for slot, held in turret_config.items() if held == objective_id))


class RuntimeState:
    """The answers for the user configuration a Lumascope acts on
    (labware / objective / turret / stage), read from the session's settings.
    """

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope
        # The objective whose optics were last recorded, so the record is
        # written once per change of the active objective, not per read.
        self._logged_objective_id: str | None = None
        self._optics_lock = threading.Lock()

        self._coordinate_transformer = coord_transformations.CoordinateTransformer()

    def get_labware(self) -> Labware:
        """The selected labware (well plate).

        Built from the catalogue for the plate the settings name, on every
        call, so it is always the plate selected now
        (``ScopeSession.select_labware`` changes it).

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
        """
        return self._scope.wellplate_loader.get_plate(
            labware_id=self._scope.read_setting('protocol.labware')
        )

    @api(reads=True)
    def resolve_current_objective(self) -> tuple[str, Objective]:
        """The active objective's id and record, or why it is unknown.

        On a turreted scope (``capabilities.has_turret``), the objective
        assigned to the slot in the light path (``motion.turret_status``),
        read fresh every time; with no turret, the selected one, once a
        person has confirmed it (``objective_confirmed``).

        Returns:
            tuple[str, Objective]: The objective id and its catalogue record.

        Raises:
            ObjectiveUnknownError: On a turreted scope, the slot is unknown
                (``'slot_unknown'``), it has no assignment
                (``'slot_unassigned'``), or its assignment is not in the
                catalogue (``'not_in_catalogue'``); with no turret, no person
                has confirmed the selected objective (``'none_selected'``).
        """
        objective_id, objective, unknown = self._derive_current_objective()
        if unknown is not None:
            raise unknown
        return objective_id, objective

    def _derive_current_objective(
        self,
    ) -> tuple[str | None, Objective | None, ObjectiveUnknownError | None]:
        """The one derivation: (id, objective, None), or (None, None, why).

        Records the optics whenever the answer is a different objective
        from the last one recorded -- after a turret move, an assignment
        or a selection -- so the scale every later capture used is in the
        log however the objective changed.
        """
        objective_id, objective, unknown = self._derive()
        if unknown is None:
            self._record_optics_on_change(objective_id, objective)
        return objective_id, objective, unknown

    def _record_optics_on_change(self, objective_id: str, objective: Objective) -> None:
        # The record explains the scale of the images this scope takes; with
        # no camera it takes none, and the camera's absence is reported at
        # bring-up, so there is no scale to explain.
        if not self._scope.camera.active:
            return
        with self._optics_lock:
            if objective_id == self._logged_objective_id:
                return
            self._logged_objective_id = objective_id
        import _lumascope.modules.config_helpers as config_helpers

        # The binning the camera last reported, not a fresh read: an
        # objective read asks the camera nothing, and an unknown binning is
        # recorded as the scale it leaves unknown.
        config_helpers.log_resolved_optics(
            objective_id,
            objective.focal_length_mm,
            self._scope.camera.binning,
            capabilities=self._scope.capabilities,
        )

    def _derive(self) -> tuple[str | None, Objective | None, ObjectiveUnknownError | None]:
        if not self._scope.capabilities.has_turret:
            # The selected objective, from the settings, once a person has
            # confirmed it: the template's id, or one put in place of a
            # stored id the catalogue does not hold, is no person's word for
            # what is mounted. Bring-up replaced such an id, and the one
            # writer (ScopeSession.select_objective) refuses one too.
            if not self.objective_confirmed():
                return None, None, ObjectiveUnknownError('none_selected')
            objective_id = self._scope.read_setting('objective_id')
            objective = self._scope.objective_helper.get_objective_info(objective_id=objective_id)
            return objective_id, objective, None
        slot = self._scope.motion.turret_status().slot
        if slot is None:
            return None, None, ObjectiveUnknownError('slot_unknown')
        objective_id = self.get_turret_config().get(slot)
        if objective_id is None:
            return None, None, ObjectiveUnknownError('slot_unassigned', slot)
        if objective_id not in self._scope.objectives:
            return None, None, ObjectiveUnknownError('not_in_catalogue', slot)
        return objective_id, self._scope.objectives[objective_id], None

    def objective_confirmed(self) -> bool:
        """Whether a person has confirmed the objective for this light path.

        With a turret, a confirmation given on a turret scope: each slot's
        assignment names its own glass. With none, a confirmation naming
        the stored ``objective_id``, so a stored id nobody picked (the
        template's, or one put in place of a stored id the catalogue does
        not hold) is not confirmed. The two settings are read one after the
        other, so a read racing ``ScopeSession.select_objective`` can answer
        False for that instant, never True for an id nobody confirmed.
        """
        confirmed = self._scope.read_setting('objective_confirmed')
        if self._scope.capabilities.has_turret:
            return confirmed == OBJECTIVE_CONFIRMED_ON_A_TURRET
        return (
            isinstance(confirmed, str)
            and confirmed != OBJECTIVE_CONFIRMED_ON_A_TURRET
            and confirmed == self._scope.read_setting('objective_id')
        )

    def get_current_objective_id(self) -> str | None:
        """Get the ID of the currently active objective.

        Returns:
            str | None: e.g. '20x Oly', or None when it is not known
                (``resolve_current_objective`` says why).
        """
        return self._derive_current_objective()[0]

    def get_current_objective(self) -> Objective | None:
        """The active objective's record.

        Returns:
            Objective | None: The active objective, or None when it is not
                known (``resolve_current_objective`` says why).
        """
        return self._derive_current_objective()[1]

    @api(reads=True)
    def get_pixel_size_um(self) -> float | None:
        """The effective pixel size: micrometres per pixel at the current
        objective and the camera's binning, the scale a saved image carries.

        Returns:
            float | None: None when it cannot be known: no objective is
                known (``resolve_current_objective`` says why), there is no
                camera, the camera has not reported its binning, or the
                scope cannot report its optics.

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
        """
        objective = self.get_current_objective()
        if objective is None:
            return None
        return common_utils.get_pixel_size(
            objective.focal_length_mm,
            self._scope.camera.binning,
            capabilities=self._scope.capabilities,
        )

    @api(reads=True)
    def get_field_of_view(self) -> FieldOfView | None:
        """The area the camera's frame covers on the sample, at the current
        objective and the camera's binning.

        Returns:
            FieldOfView | None: Width and height in micrometres; None when
                the pixel size cannot be known (``get_pixel_size_um``) or the
                camera has not reported its frame.

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
        """
        extent = self._field_of_view('get_field_of_view', self._derive_current_objective())
        return extent if isinstance(extent, FieldOfView) else None

    @api(reads=True)
    def optics_status(self) -> OpticsStatus:
        """The objective in the light path and the frame's extent on the sample, each its value or why it is unknown.

        The read for a display: it answers an unknown as the refusal that
        says why, where ``resolve_current_objective`` and the reads that act
        on the extent refuse. Both fields come from one reading of the
        objective. No serial I/O: the turret slot and the camera's binning
        and frame are the ones last reported.

        Returns:
            OpticsStatus: Each value, or its refusal.

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
        """
        derived = self._derive_current_objective()
        objective_id, _, unknown = derived
        return OpticsStatus(
            objective=objective_id if unknown is None else unknown,
            field_of_view=self._field_of_view('optics_status', derived),
        )

    def resolve_field_of_view(self, member: str) -> FieldOfView:
        """The area the camera's frame covers on the sample, or why it is unknown.

        ``get_field_of_view``'s answer, raised where that read answers
        None, so a member that acts on the extent is refused with the cause.
        A seam for the API's own members, not part of the L2 API surface, as
        ``resolve_current_objective`` is not.

        Args:
            member: The member asking, named by a refusal for want of a camera.

        Raises:
            ObjectiveUnknownError: As ``resolve_current_objective``.
            HardwareCommandRefusedError: ``'not_connected'``, no camera is
                connected; ``'scope_disconnected'``, after ``disconnect()``.
            PixelSizeUnknownError: ``'optics_undeclared'``,
                ``'binning_unread'`` or ``'frame_size_unread'``.
        """
        extent = self._field_of_view(member, self._derive_current_objective())
        if not isinstance(extent, FieldOfView):
            raise extent
        return extent

    def _field_of_view(
        self,
        member: str,
        derived: tuple[str | None, Objective | None, ObjectiveUnknownError | None],
    ) -> FieldOfView | ObjectiveUnknownError | HardwareCommandRefusedError | PixelSizeUnknownError:
        """The one derivation of the frame's extent: the extent, or the first reason it is unknown.

        Asked in this order: the objective (``derived``, the objective
        derivation's answer), the camera (its refusal names ``member``), the
        scope's optics, the camera's binning, the camera's frame.
        """
        _, objective, unknown = derived
        if unknown is not None:
            return unknown
        no_camera = self._scope.camera.camera_not_connected_refusal(member)
        if no_camera is not None:
            return no_camera
        capabilities = self._scope.capabilities
        if capabilities.lens_focal_length_mm is None or capabilities.pixel_size_um is None:
            return PixelSizeUnknownError('optics_undeclared')
        binning = self._scope.camera.binning
        if binning is None:
            return PixelSizeUnknownError('binning_unread')
        frame = self._scope.camera.frame_size
        if frame is None:
            return PixelSizeUnknownError('frame_size_unread')
        extent = common_utils.get_field_of_view(
            focal_length=objective.focal_length_mm,
            frame_size={'width': frame.width, 'height': frame.height},
            binning_size=binning,
            capabilities=capabilities,
        )
        return FieldOfView(width_um=extent['width'], height_um=extent['height'])

    @api(reads=True)
    def frame_point_distance(self, x_fraction: float, y_fraction: float) -> StageDistance | None:
        """The stage distance from the centre of the camera's frame to a point of it.

        The point is ``x_fraction`` of the frame's width from its column 0
        and ``y_fraction`` of its height from its row 0, in the frame as the
        API returns it (``capture_still().image``): row 0 is the top of the
        picture, as in the live view and a saved file. A fraction is the same
        place on the sample at every binning and frame size. The distance is
        at the current objective and binning; moving the stage by it brings
        the point to the centre (``MotionAPI.start_center_on``).

        Args:
            x_fraction: How far across the frame's width, from 0 to 1.
            y_fraction: How far down the frame's height, from 0 at the top
                to 1.

        Returns:
            StageDistance | None: Positive ``dx_um`` for a point right of
                centre, positive ``dy_um`` for one above it; (0.5, 0.5) is
                the centre. None when the field of view cannot be known
                (``get_field_of_view``).

        Raises:
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``,
                a fraction that is not a number from 0 to 1.
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
        """
        refuse_unless_frame_point(x_fraction, y_fraction)
        extent = self.get_field_of_view()
        return None if extent is None else _distance_to(extent, x_fraction, y_fraction)

    def resolve_frame_point_distance(
        self, member: str, x_fraction: float, y_fraction: float
    ) -> StageDistance:
        """``frame_point_distance`` for fractions already refused, or why it is unknown.

        A seam for the API's own members, not part of the L2 API surface.

        Args:
            member: The member asking, named by a refusal for want of a camera.
            x_fraction: As ``frame_point_distance``, already checked
                (``refuse_unless_frame_point``).
            y_fraction: As ``frame_point_distance``, already checked.

        Raises:
            ObjectiveUnknownError: As ``resolve_field_of_view``.
            HardwareCommandRefusedError: As ``resolve_field_of_view``.
            PixelSizeUnknownError: As ``resolve_field_of_view``.
        """
        return _distance_to(self.resolve_field_of_view(member), x_fraction, y_fraction)

    @api(reads=True)
    def frame_point_plate(
        self, x_fraction: float, y_fraction: float
    ) -> (
        coord_transformations.PlatePosition
        | HardwareCommandRefusedError
        | AxisStateUnknownError
        | ObjectiveUnknownError
        | PixelSizeUnknownError
    ):
        """The plate position of a point of the camera's frame, or why it is unknown.

        Where the point is on the selected plate now: the stage's X and Y
        plus the point's distance from the frame's centre
        (``frame_point_distance``), as ``stage_to_plate`` states it. The read
        for a display, such as the plate position under a cursor: an
        unknown is answered as the refusal that says why, never raised.
        Judged on X and Y alone; Z's position does not move the point.

        Args:
            x_fraction: How far across the frame's width, from 0 to 1.
            y_fraction: How far down the frame's height, from 0 at the top
                to 1.

        Returns:
            The ``PlatePosition``, in mm; or the first reason it is unknown,
            in this order: ``HardwareCommandRefusedError``,
            ``'scope_disconnected'``, ``'not_connected'`` or
            ``'axis_absent'``, the scope has no motor controller or no X or
            Y; ``AxisStateUnknownError``, X or Y does not know its position,
            homing or lost; then the field of view's refusal
            (``optics_status``).

        Raises:
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``,
                a fraction that is not a number from 0 to 1.
            CatalogueNameRefusedError: ``'labware_unknown'``, the selected
                labware is not in the catalogue.
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
        """
        refuse_unless_frame_point(x_fraction, y_fraction)
        motion = self._scope.motion
        for axis in ('X', 'Y'):
            absent = motion.absent_refusal('frame_point_plate', axis)
            if absent is not None:
                return absent
        stage = motion.axis_positions()
        unplaced = {
            axis: stage[axis].state for axis in ('X', 'Y') if stage[axis].position_um is None
        }
        if unplaced:
            return AxisStateUnknownError(unplaced, then='read the plate position of the point')
        extent = self._field_of_view('frame_point_plate', self._derive_current_objective())
        if not isinstance(extent, FieldOfView):
            return extent
        distance = _distance_to(extent, x_fraction, y_fraction)
        return self.stage_to_plate(
            stage['X'].position_um + distance.dx_um, stage['Y'].position_um + distance.dy_um
        )

    @api(reads=True)
    def get_turret_config(self) -> dict[int, str | None]:
        """The turret's objective assignments, from the settings.

        The map ``ScopeSession.turret_objective`` reads one slot of: on a
        scope with no turret (``capabilities.has_turret``) it is empty, as
        that read refuses every slot there.

        Returns:
            Turret position (1-4) to objective ID, None for an
                unassigned slot; empty on a scope with no turret. A copy:
                changing it changes no assignment
                (``ScopeSession.assign_turret_objective`` does).

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
        """
        # Read before the scope fact is asked, so a scope no session has
        # bound refuses this read as it does every other.
        assignments = self._scope.read_setting('turret_objectives')
        if not self._scope.capabilities.has_turret:
            # The stored slots are a turret's assignments only on a scope
            # that has one; elsewhere they are kept for a change of model.
            return {}
        return assignments

    @api(reads=True)
    def get_stage_offset(self) -> coord_transformations.StageOffset:
        """The stage offset for coordinate transformations, from the settings.

        Returns:
            Where the plate is mounted on the stage, in um.

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
        """
        stored = self._scope.read_setting('stage_offset')
        return coord_transformations.StageOffset(x_um=float(stored['x']), y_um=float(stored['y']))

    def _frame(
        self,
        labware_id: str | None,
        stage_offset: coord_transformations.StageOffset | None,
    ) -> tuple[Labware, coord_transformations.StageOffset]:
        """The plate and offset a conversion is stated in: the ones given, else the selected plate and the live offset."""
        labware = (
            self.get_labware()
            if labware_id is None
            else self._scope.wellplate_loader.get_plate(labware_id=labware_id)
        )
        return labware, self.get_stage_offset() if stage_offset is None else stage_offset

    @api(reads=True)
    def stage_to_plate(
        self,
        x_um: float,
        y_um: float,
        *,
        labware_id: str | None = None,
        stage_offset: coord_transformations.StageOffset | None = None,
    ) -> coord_transformations.PlatePosition:
        """Convert a stage position (um) to a plate position (mm).

        ``plate_to_stage`` inverts it.

        Args:
            x_um: The stage X, in um.
            y_um: The stage Y, in um.
            labware_id: The plate the position is stated on; None for the
                selected plate. A protocol's positions are on the plate it
                stores.
            stage_offset: Where the plate is mounted on the stage, in um;
                None for the live offset. A run keeps the offset it started
                with.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, a coordinate or an
                offset axis is not a finite number.
            CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id``
                is not a plate the catalogue has.
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope, and ``labware_id`` or ``stage_offset`` was left to the
                settings.
        """
        labware, offset = self._frame(labware_id, stage_offset)
        return self._coordinate_transformer.stage_to_plate(
            labware=labware, stage_offset=offset, x_um=x_um, y_um=y_um
        )

    @api(reads=True)
    def plate_to_stage(
        self,
        x_mm: float,
        y_mm: float,
        *,
        labware_id: str | None = None,
        stage_offset: coord_transformations.StageOffset | None = None,
    ) -> coord_transformations.StagePosition:
        """Convert a plate position (mm) to a stage position (um), as ``stage_to_plate`` inverted.

        Not bounded: a position off the stage's travel converts, and the
        move to it is refused.

        Args:
            x_mm: The plate X, in mm.
            y_mm: The plate Y, in mm.
            labware_id: The plate the position is stated on; None for the
                selected plate. A protocol's positions are on the plate it
                stores.
            stage_offset: Where the plate is mounted on the stage, in um;
                None for the live offset. A run keeps the offset it started
                with.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, a coordinate or an
                offset axis is not a finite number.
            CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id``
                is not a plate the catalogue has.
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope, and ``labware_id`` or ``stage_offset`` was left to the
                settings.
        """
        labware, offset = self._frame(labware_id, stage_offset)
        return self._coordinate_transformer.plate_to_stage(
            labware=labware, stage_offset=offset, x_mm=x_mm, y_mm=y_mm
        )

    @api(in_process=True, reads=True)
    def plate_transform(
        self,
        *,
        labware_id: str | None = None,
        stage_offset: coord_transformations.StageOffset | None = None,
    ) -> Callable[[float, float], coord_transformations.PlatePosition]:
        """``stage_to_plate`` bound to one plate and offset, taken when this is called.

        ``stage_to_plate`` reads the selected plate and live offset on every
        call, so a caller converting many positions over time -- a recording
        writing one per frame, a run, a redraw -- would state its later ones
        against a plate or offset changed mid-way. The function returned is
        bound to the plate and offset of this call, so every position it
        converts is stated in one frame of reference and a later change moves
        none of them.

        Args:
            labware_id: The plate the positions are stated on; None for the
                plate selected when this is called.
            stage_offset: Where the plate is mounted on the stage, in um;
                None for the offset live when this is called.

        Returns:
            A function of ``(x_um, y_um)`` answering their ``PlatePosition``;
            it raises only as the transform refuses a non-finite coordinate.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, an offset axis is not a
                finite number; asked here, so the function does not.
            CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id``
                is not a plate the catalogue has.
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope, and ``labware_id`` or ``stage_offset`` was left to the
                settings.
        """
        labware, offset = self._frame(labware_id, stage_offset)
        transformer = self._coordinate_transformer
        coord_transformations.offset_axes(offset)

        def to_plate(x_um: float, y_um: float) -> coord_transformations.PlatePosition:
            return transformer.stage_to_plate(
                labware=labware, stage_offset=offset, x_um=x_um, y_um=y_um
            )

        return to_plate

    @api(reads=True)
    def get_well_label(self) -> str | None:
        """Get the well label for the current stage XY position.

        Maps the current target X/Y stage position to a plate-frame
        coordinate using the selected labware and stage offset, then
        looks up the matching well label.

        Returns:
            str | None: Well label (e.g. ``"A1"``); ``''`` when the selected
            labware has no wells (the Blank plate) or the position is more
            than half a pitch beyond its outer well centres; None when X or Y does not
            know its position, or the scope has no X or Y at all. Consumers
            omit the well from filenames and metadata for both rather than
            stamping a fabricated one -- the last number an axis had before it
            lost its reference names a real well the scope may no longer be
            over. The two stay distinct so a caller can say which it was.

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
            CatalogueNameRefusedError: ``'labware_unknown'``, the selected
                labware is not in the catalogue.
        """
        # One read: it leaves out an axis the scope lacks and answers None for
        # one whose position is not known, from the same snapshot as the
        # numbers, so a home starting between two reads cannot pass.
        targets = self._scope.motion.get_target_positions()
        x_um, y_um = targets.get('X'), targets.get('Y')
        if x_um is None or y_um is None:
            return None

        labware = self.get_labware()
        target = self.stage_to_plate(x_um, y_um)

        return labware.get_well_label(x=target.x_mm, y=target.y_mm)
