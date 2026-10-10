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
from modules.exceptions import ArgumentRefusedError, ConfigError, ObjectiveUnknownError
from modules.api_surface import api

if TYPE_CHECKING:
    from modules.labware import WellPlate
    from modules.lumascope_api._lumascope import Lumascope


# The axes a plate coordinate is defined for: the plate lies in X and Y.
_PLATE_AXES = ('X', 'Y')


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

    @api
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

    @api
    def is_turreted(self) -> bool:
        """Whether this scope derives its objective from the turret slot.

        The answer bring-up recorded; every objective question asks this
        one, so selection and derivation cannot disagree about the turret.

        Raises:
            ConfigError: Bring-up has not recorded the answer
                (``Lumascope.initialize`` has not run).
        """
        if self._turreted is None:
            raise ConfigError(
                'whether this scope has a turret is not known: the scope has not been '
                'configured -- run initialize() (a ScopeSession does this at bring-up)'
            )
        return self._turreted

    @api
    def resolve_current_objective(self) -> tuple[str, dict]:
        """The active objective's id and metadata, or why it is unknown.

        On a turreted scope, the objective assigned to the slot in the light
        path (``motion.get_turret_slot``), read fresh every time; with no
        turret, the selected one.

        Returns:
            tuple[str, dict]: The objective id and its catalogue entry.

        Raises:
            ObjectiveUnknownError: On a turreted scope, the slot is unknown,
                it has no assignment, or its assignment is not in the
                catalogue; with no turret, nothing was selected; on any
                scope, bring-up has not said whether it has a turret.
        """
        objective_id, info, unknown = self._derive_current_objective()
        if unknown is not None:
            raise unknown
        return objective_id, info

    def _derive_current_objective(
        self,
    ) -> tuple[str | None, dict | None, ObjectiveUnknownError | None]:
        """The one derivation: (id, info, None), or (None, None, why).

        Records the optics whenever the answer is a different objective
        from the last one recorded -- after a turret move, an assignment
        or a selection -- so the scale every later capture used is in the
        log however the objective changed.
        """
        objective_id, info, unknown = self._derive()
        if unknown is None:
            self._record_optics_on_change(objective_id, info)
        return objective_id, info, unknown

    def _record_optics_on_change(self, objective_id: str, info: dict) -> None:
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
                info['focal_length'],
                self._scope.imaging.get_binning_size(),
                capabilities=self._scope.capabilities,
            )
        except Exception:
            logger.warning(
                f'[LVP API  ] could not record the optics for objective {objective_id!r}',
                exc_info=True,
            )

    def _derive(self) -> tuple[str | None, dict | None, ObjectiveUnknownError | None]:
        if self._turreted is None:
            return None, None, ObjectiveUnknownError('turret_undecided')
        if not self._turreted:
            # The selected objective, from the settings; bring-up refused a
            # stored id the catalogue does not hold, and the one writer
            # (ScopeSession.select_objective) refuses one too.
            objective_id = self._scope.read_setting('objective_id')
            if objective_id is None:
                return None, None, ObjectiveUnknownError('none_selected')
            info = self._scope.objective_helper.get_objective_info(objective_id=objective_id)
            return objective_id, info, None
        slot = self._scope.motion.get_turret_slot()
        if slot is None:
            return None, None, ObjectiveUnknownError('slot_unknown')
        objective_id = self.get_turret_config().get(slot)
        if objective_id is None:
            return None, None, ObjectiveUnknownError('slot_unassigned', slot)
        if objective_id not in self._scope.objective_helper.get_objectives_list():
            return None, None, ObjectiveUnknownError('not_in_catalogue', slot)
        info = self._scope.objective_helper.get_objective_info(objective_id=objective_id)
        return objective_id, info, None

    def get_current_objective_id(self) -> str | None:
        """Get the ID of the currently active objective.

        Returns:
            str | None: e.g. '20x Oly', or None when it is not known
                (``resolve_current_objective`` says why).
        """
        return self._derive_current_objective()[0]

    @api
    def get_objective_info(self, objective_id: str | None) -> dict:
        """Get objective metadata by ID.

        Args:
            objective_id: Objective identifier (e.g. "4x Oly", "10x Oly").

        Returns:
            dict: Objective info including focal_length, magnification, etc.

        Raises:
            CatalogueNameRefusedError: ``'objective_not_in_catalogue'``, the
                catalogue holds no such id -- what a protocol saved against
                another catalogue names.
            ObjectiveUnknownError: ``'none_selected'``, the id is null, which
                is what a fresh or half-configured scope stores.

        Refused by name rather than answered with None: nearly every caller
        subscripts the answer, so a None reached the user as a bare
        attribute error naming nothing they could act on.
        """
        return self._scope.objective_helper.get_objective_info(objective_id=objective_id)

    @api
    def get_available_objectives(self) -> list[str]:
        """Get list of all available objective IDs.

        Returns:
            list[str]: Objective identifiers (e.g. ["4x", "10x Oly", "20x Oly"]).
        """
        return self._scope.objective_helper.get_objectives_list()

    def get_current_objective(self) -> dict | None:
        """Get the currently active objective info.

        Returns:
            dict | None: Active objective metadata, or None when it is not
                known (``resolve_current_objective`` says why).
        """
        return self._derive_current_objective()[1]

    @api
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

    @api
    def get_stage_offset(self) -> dict:
        """The stage offset for coordinate transformations, from the settings.

        Returns:
            dict: The offset per axis, in um. A copy: changing it changes
                no setting.

        Raises:
            ConfigError: No session has bound this scope.
        """
        return self._scope.read_setting('stage_offset')

    @api
    def stage_to_plate(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert a stage position (um) to plate coordinates (mm).

        Uses the selected labware and stage offset -- the same
        transform ``get_well_label`` performs before its label lookup,
        exposed for consumers that need the coordinates themselves
        (e.g. image metadata).

        Args:
            sx: Stage X position in um.
            sy: Stage Y position in um.

        Returns:
            (px, py): Plate position in mm.

        Raises:
            ConfigError: No session has bound this scope.
        """
        return self._coordinate_transformer.stage_to_plate(
            labware=self.get_labware(),
            stage_offset=self.get_stage_offset(),
            sx=sx,
            sy=sy,
        )

    @api(in_process=True)
    def plate_transform(self) -> Callable[[float, float], tuple[float, float]]:
        """The stage-to-plate transform bound to the labware and offset selected now.

        ``stage_to_plate`` reads the selected labware and stage offset on
        every call, so a caller converting many positions over time -- a
        recording writing one per frame -- would record its later frames
        against a labware or offset changed mid-way. The transform returned
        here is bound to copies of both taken at the moment of the call, so
        every position it converts is stated in one frame of reference, a
        later change moves none of them, and it cannot raise.

        Returns:
            A function of ``(sx_um, sy_um)`` answering ``(px_mm, py_mm)``.

        Raises:
            ConfigError: No session has bound this scope.
        """
        labware = self.get_labware()
        stage_offset = self.get_stage_offset()
        transformer = self._coordinate_transformer

        def to_plate(sx: float, sy: float) -> tuple[float, float]:
            return transformer.stage_to_plate(
                labware=labware, stage_offset=stage_offset, sx=sx, sy=sy
            )

        return to_plate

    @api
    def plate_to_stage_axis(self, axis: str, plate_mm: float) -> float:
        """Convert one axis of a plate coordinate (mm) to a stage target (um).

        The completing half of ``stage_to_plate``. Per-axis because the
        callers that need it are commanding a single axis, and because
        each stage coordinate depends only on its own plate coordinate --
        so the unused argument below is inert, not a placeholder standing
        in for a value the caller should have supplied.

        Raises:
            ArgumentRefusedError: ``'plate_frame_axis'``, ``axis`` is not
                'X' or 'Y'.
        """
        if axis not in _PLATE_AXES:
            raise ArgumentRefusedError(
                'plate_frame_axis', argument='axis', value=axis, offered=_PLATE_AXES
            )

        sx, sy = self._coordinate_transformer.plate_to_stage(
            labware=self.get_labware(),
            stage_offset=self.get_stage_offset(),
            px=plate_mm if axis == 'X' else 0,
            py=plate_mm if axis == 'Y' else 0,
        )
        return sx if axis == 'X' else sy

    @api
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
            Exception: Re-raises any error encountered reading target
                position; logged before re-raise.
        """
        unknown = self._scope.motion.axes_without_position()
        if 'X' in unknown or 'Y' in unknown:
            return None

        labware = self.get_labware()

        # The all-axes read carries only the axes the scope has; a
        # single-axis read of an absent axis answers 0, which names a well.
        try:
            targets = self._scope.motion.get_target_position()
        except Exception:
            logger.exception('[LVP API  ] Error getting target position.')
            raise
        if 'X' not in targets or 'Y' not in targets:
            return None

        x_target, y_target = self.stage_to_plate(sx=targets['X'], sy=targets['Y'])

        return labware.get_well_label(x=x_target, y=y_target)
