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

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations
from modules.exceptions import ObjectiveUnknownError
from modules.api_surface import api
from modules.scope_capabilities import FieldOfView

if TYPE_CHECKING:
    from modules.labware import Labware
    from modules.lumascope_api._lumascope import Lumascope
    from modules.objectives_loader import Objective


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
        read fresh every time; with no turret, the selected one.

        Returns:
            tuple[str, Objective]: The objective id and its catalogue record.

        Raises:
            ObjectiveUnknownError: On a turreted scope, the slot is unknown
                (``'slot_unknown'``), it has no assignment
                (``'slot_unassigned'``), or its assignment is not in the
                catalogue (``'not_in_catalogue'``); with no turret, nothing was
                selected (``'none_selected'``).
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
        with self._optics_lock:
            if objective_id == self._logged_objective_id:
                return
            self._logged_objective_id = objective_id
        import modules.config_helpers as config_helpers

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
            # The selected objective, from the settings; bring-up refused a
            # stored id the catalogue does not hold, and the one writer
            # (ScopeSession.select_objective) refuses one too.
            objective_id = self._scope.read_setting('objective_id')
            if objective_id is None:
                return None, None, ObjectiveUnknownError('none_selected')
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
        objective = self.get_current_objective()
        frame = self._scope.camera.frame_size
        if objective is None or frame is None:
            return None
        extent = common_utils.get_field_of_view(
            focal_length=objective.focal_length_mm,
            frame_size={'width': frame.width, 'height': frame.height},
            binning_size=self._scope.camera.binning,
            capabilities=self._scope.capabilities,
        )
        if extent is None:
            return None
        return FieldOfView(width_um=extent['width'], height_um=extent['height'])

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
            stamping a fabricated one -- an axis that lost its reference keeps
            answering the last target it had, which names a real well the
            scope may no longer be over. The two stay distinct so a caller can
            say which it was.

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope.
            CatalogueNameRefusedError: ``'labware_unknown'``, the selected
                labware is not in the catalogue.
        """
        unknown = self._scope.motion.axes_without_position()
        if 'X' in unknown or 'Y' in unknown:
            return None

        labware = self.get_labware()

        # The all-axes read carries only the axes the scope has.
        targets = self._scope.motion.get_target_positions()
        if 'X' not in targets or 'Y' not in targets:
            return None

        target = self.stage_to_plate(targets['X'], targets['Y'])

        return labware.get_well_label(x=target.x_mm, y=target.y_mm)
