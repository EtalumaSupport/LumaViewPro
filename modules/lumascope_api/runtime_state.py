# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""RuntimeState -- runtime-mutable scope state, split from ScopeCapabilities.

Design doc sec 2.5 splits the capabilities surface into TWO:

- `scope.capabilities`: IMMUTABLE per Lumascope instance. Reconnect =
  new Lumascope = new capabilities.
- `scope.runtime_state`: MUTABLE, refreshed on driver events (reflash,
  reconnect, etc.). Also answers for the configuration the scope acts
  on: labware, objective, turret config, stage offset, and the
  coordinate transformer that operates on those values. It holds none of
  them: the session's settings are their one store, read through the
  scope's ``read_setting`` on every answer. The objective catalogue is
  the scope's own (`scope.objective_helper`), read at each use.

The split exists because firmware version legitimately mutates mid-
session when boards are reflashed; a single frozen surface would lie
post-flash. Sub-APIs read from BOTH surfaces as needed -- capability-
probe gates use the immutable surface, recovery / version gates use
the runtime surface.

The answers for labware / objective / turret / stage live here because
they are user configuration that callers adjust during a session;
capabilities is hardware-identity state that doesn't change without a
reconnect.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.5 and sec 10.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import modules.coord_transformations as coord_transformations
from lvp_logger import logger
from modules.exceptions import (
    ConfigurationDefectError,
    ObjectiveUnknownError,
)
from modules.api_surface import api

if TYPE_CHECKING:
    from modules.labware import WellPlate
    from modules.lumascope_api._lumascope import Lumascope
    from modules.objectives_loader import Objective


def slots_holding(objective_id: object, turret_config: dict) -> tuple[int, ...]:
    """The turret slots ``turret_config`` assigns ``objective_id``, ascending."""
    return tuple(sorted(slot for slot, held in turret_config.items() if held == objective_id))


class RuntimeState:
    """Mutable runtime state on a Lumascope, and the answers for the user
    configuration the scope acts on (labware / objective / turret / stage),
    read from the session's settings.

    Fields land per design doc sec 2.5 as the underlying firmware +
    driver hooks ship.
    """

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope
        # Set once at bring-up from the session's one has-a-turret answer.
        # None until then: a scope nobody has configured does not know where
        # its objective comes from, and a False here would let a turret scope
        # answer with a stored objective.
        self._turreted: bool | None = None
        # The objective whose optics were last recorded, so the record is
        # written once per change of the active objective, not per read.
        self._logged_objective_id: str | None = None
        self._optics_lock = threading.Lock()

        self._coordinate_transformer = coord_transformations.CoordinateTransformer()

    @api(reads=True)
    def get_labware(self) -> WellPlate:
        """The selected labware (well plate).

        Built from the catalogue for the plate the settings name, on every
        call, so it is always the plate selected now
        (``ScopeSession.select_labware`` changes it).

        Raises:
            ConfigError: No session has bound this scope.
        """
        return self._scope.wellplate_loader.get_plate(
            plate_key=self._scope.read_setting('protocol.labware')
        )

    def set_turreted(self, turreted: bool) -> None:
        """Record whether this scope has a turret, once, at bring-up.

        This method is not part of the L2 API surface: bring-up sets it from
        the session's one has-a-turret answer (the board when it is talking,
        the declared model when it is not), so a turreted scope whose board
        is dead still derives its objective -- and, with no known slot,
        answers unknown -- rather than falling back to a stored one.
        """
        self._turreted = turreted

    @api(reads=True)
    def is_turreted(self) -> bool:
        """Whether this scope derives its objective from the turret slot.

        The answer bring-up recorded; every objective question asks this
        one, so selection and derivation cannot disagree about the turret.

        Raises:
            ConfigError: Bring-up has not recorded the answer
                (``Lumascope.initialize`` has not run).
        """
        if self._turreted is None:
            raise ConfigurationDefectError(
                'scope_not_initialized',
                'whether this scope has a turret is not known: the scope has not been '
                'configured -- run initialize() (a ScopeSession does this at bring-up)',
            )
        return self._turreted

    @api(reads=True)
    def resolve_current_objective(self) -> tuple[str, Objective]:
        """The active objective's id and record, or why it is unknown.

        On a turreted scope, the objective assigned to the slot in the light
        path (``motion.get_turret_slot``), read fresh every time; with no
        turret, the selected one.

        Returns:
            tuple[str, Objective]: The objective id and its catalogue record.

        Raises:
            ObjectiveUnknownError: On a turreted scope, the slot is unknown
                (``'slot_unknown'``), it has no assignment
                (``'slot_unassigned'``), or its assignment is not in the
                catalogue (``'not_in_catalogue'``); with no turret, nothing was
                selected (``'none_selected'``); on any scope, bring-up has not
                said whether it has a turret (``'turret_undecided'``).
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
        # A record that cannot be written must not fail the read that
        # triggered it: the objective is still known and still correct.
        try:
            import modules.config_helpers as config_helpers

            config_helpers.log_resolved_optics(
                objective_id,
                objective.focal_length_mm,
                self._scope.camera.read_binning(),
                capabilities=self._scope.capabilities,
            )
        except Exception:
            logger.warning(
                f'[LVP API  ] could not record the optics for objective {objective_id!r}',
                exc_info=True,
            )

    def _derive(self) -> tuple[str | None, Objective | None, ObjectiveUnknownError | None]:
        if self._turreted is None:
            return None, None, ObjectiveUnknownError('turret_undecided')
        if not self._turreted:
            # The selected objective, from the settings; bring-up refused a
            # stored id the catalogue does not hold, and the one writer
            # (ScopeSession.select_objective) refuses one too.
            objective_id = self._scope.read_setting('objective_id')
            if objective_id is None:
                return None, None, ObjectiveUnknownError('none_selected')
            objective = self._scope.objective_helper.get_objective_info(objective_id=objective_id)
            return objective_id, objective, None
        slot = self._scope.motion.get_turret_slot()
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
    def get_turret_config(self) -> dict:
        """The turret's objective assignments, from the settings.

        Returns:
            dict: Turret position (1-4) to objective ID, None for an
                unassigned slot. A copy: changing it changes no assignment
                (``ScopeSession.assign_turret_objective`` does).

        Raises:
            ConfigError: No session has bound this scope.
        """
        return self._scope.read_setting('turret_objectives')

    @api(reads=True)
    def get_stage_offset(self) -> coord_transformations.StageOffset:
        """The stage offset for coordinate transformations, from the settings.

        Returns:
            Where the plate is mounted on the stage, in um.

        Raises:
            ConfigError: No session has bound this scope.
        """
        stored = self._scope.read_setting('stage_offset')
        return coord_transformations.StageOffset(x_um=float(stored['x']), y_um=float(stored['y']))

    def _frame(
        self,
        labware_id: str | None,
        stage_offset: coord_transformations.StageOffset | None,
    ) -> tuple[WellPlate, coord_transformations.StageOffset]:
        """The plate and offset a conversion is stated in: the ones given, else the selected plate and the live offset."""
        labware = (
            self.get_labware()
            if labware_id is None
            else self._scope.wellplate_loader.get_plate(plate_key=labware_id)
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

        ``plate_to_stage`` inverts it. Stated on the selected plate at the
        live stage offset unless ``labware_id`` or ``stage_offset`` names
        another: a protocol's positions are on the plate it stores, and a
        run keeps the offset it started with.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, a coordinate or an
                offset axis is not a finite number.
            CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id``
                is not a plate the catalogue has.
            ConfigError: No session has bound this scope.
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

        The plate and offset are ``stage_to_plate``'s. Not bounded: a
        position off the stage's travel converts, and the move to it is
        refused.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, a coordinate or an
                offset axis is not a finite number.
            CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id``
                is not a plate the catalogue has.
            ConfigError: No session has bound this scope.
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
        bound to the plate and offset of this call (the arguments are
        ``stage_to_plate``'s), so every position it converts is stated in one
        frame of reference and a later change moves none of them.

        Returns:
            A function of ``(x_um, y_um)`` answering their ``PlatePosition``;
            it raises only as the transform refuses a non-finite coordinate.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, an offset axis is not a
                finite number; asked here, so the function does not.
            CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id``
                is not a plate the catalogue has.
            ConfigError: No session has bound this scope.
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
            ConfigError: No session has bound this scope.
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
