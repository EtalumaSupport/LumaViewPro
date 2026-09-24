# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""RuntimeState -- runtime-mutable scope state, split from ScopeCapabilities.

Design doc sec 2.5 splits the capabilities surface into TWO:

- `scope.capabilities`: IMMUTABLE per Lumascope instance. Reconnect =
  new Lumascope = new capabilities.
- `scope.runtime_state`: MUTABLE, refreshed on driver events (reflash,
  reconnect, etc.). Also hosts the settings-host cluster: labware,
  objective, turret config, stage offset, and the helper objects
  (`_objectives_loader`, `_coordinate_transformer`) that operate on
  those values.

The split exists because firmware version legitimately mutates mid-
session when boards are reflashed; a single frozen surface would lie
post-flash. Sub-APIs read from BOTH surfaces as needed -- capability-
probe gates use the immutable surface, recovery / version gates use
the runtime surface.

The settings-host cluster (labware / objective / turret / stage)
lives here because it's user-config mutable state that callers
adjust during a session; capabilities is hardware-identity state
that doesn't change without a reconnect.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.5 and sec 10.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import modules.coord_transformations as coord_transformations
import modules.objectives_loader as objectives_loader
from lvp_logger import logger
from modules.exceptions import ConfigError, ObjectiveUnknownError

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class RuntimeState:
    """Mutable runtime state on a Lumascope -- the user-config runtime state
    (labware / objective / turret / stage).

    Fields land per design doc sec 2.5 as the underlying firmware +
    driver hooks ship.
    """

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope
        self._labware: Any | None = None
        # The selected objective: the live store on a scope with no turret.
        # On a turreted scope nothing is stored -- the objective is the one
        # assigned to the slot in the light path, derived on every read.
        self._objective: dict | None = None
        self._objective_id: str | None = None
        # Set once at bring-up from the session's one has-a-turret answer.
        # None until then: a scope nobody has configured does not know where
        # its objective comes from, and a False here would let a turret scope
        # answer with a stored objective.
        self._turreted: bool | None = None
        self._turret_config: dict = {}
        # The objective whose optics were last recorded, so the record is
        # written once per change of the active objective, not per read.
        self._logged_objective_id: str | None = None
        self._optics_lock = threading.Lock()
        self._stage_offset: dict | None = None

        self._objectives_loader = objectives_loader.ObjectiveLoader()
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()

    def set_labware(self, labware) -> None:
        """Set the current labware (well plate) for the microscope.

        Args:
            labware: Labware object describing the well plate geometry.
        """
        self._labware = labware

    def get_labware(self) -> Any | None:
        """Get the currently installed labware.

        Returns:
            The current labware object, or None if not set.
        """
        return self._labware

    def set_turreted(self, turreted: bool) -> None:
        """Record whether this scope has a turret, once, at bring-up.

        This method is not part of the L2 API surface: bring-up sets it from
        the session's one has-a-turret answer (the board when it is talking,
        the declared model when it is not), so a turreted scope whose board
        is dead still derives its objective -- and, with no known slot,
        answers unknown -- rather than falling back to a stored one.
        """
        self._turreted = turreted

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

    def set_objective(self, objective_id: str) -> None:
        """Set the active objective by ID, on a scope with no turret.

        Args:
            objective_id: Objective identifier (e.g. "4x", "10x", "20x").

        Raises:
            ConfigError: The id resolves to no objective. State is
                untouched on failure -- resolving before assigning keeps
                the id and the info describing the same objective, so a
                bad id can never leave the pair torn. Also raised on a
                turreted scope, whose objective is the slot's assignment
                and cannot be set beside it, and before bring-up has said
                whether the scope has a turret.
        """
        if self.is_turreted():
            raise ConfigError(
                'a turreted scope has no selected objective to set: the active objective '
                'is the one assigned to the slot in the light path'
            )
        objective = self._objectives_loader.get_objective_info(objective_id=objective_id)
        self._objective_id = objective_id
        self._objective = objective

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
            if self._objective_id is None:
                return None, None, ObjectiveUnknownError('none_selected')
            return self._objective_id, self._objective, None
        slot = self._scope.motion.get_turret_slot()
        if slot is None:
            return None, None, ObjectiveUnknownError('slot_unknown')
        objective_id = self._turret_config.get(slot)
        if objective_id is None:
            return None, None, ObjectiveUnknownError('slot_unassigned', slot)
        if objective_id not in self._objectives_loader.get_objectives_list():
            return None, None, ObjectiveUnknownError('not_in_catalogue', slot)
        info = self._objectives_loader.get_objective_info(objective_id=objective_id)
        return objective_id, info, None

    def get_current_objective_id(self) -> str | None:
        """Get the ID of the currently active objective.

        Returns:
            str | None: e.g. '20x Oly', or None when it is not known
                (``resolve_current_objective`` says why).
        """
        return self._derive_current_objective()[0]

    def get_objective_info(self, objective_id: str) -> dict:
        """Get objective metadata by ID.

        Args:
            objective_id: Objective identifier (e.g. "4x Oly", "10x Oly").

        Returns:
            dict: Objective info including focal_length, magnification, etc.

        Raises:
            ConfigError: The catalogue holds no such id, or the id is null --
                which is what a fresh or half-configured scope stores, and
                what a protocol saved against another catalogue names.
                Refused by name rather than answered with None: nearly every
                caller subscripts the answer, so a None reached the user as
                a bare attribute error naming nothing they could act on.
        """
        return self._objectives_loader.get_objective_info(objective_id=objective_id)

    def get_available_objectives(self) -> list[str]:
        """Get list of all available objective IDs.

        Returns:
            list[str]: Objective identifiers (e.g. ["4x", "10x Oly", "20x Oly"]).
        """
        return self._objectives_loader.get_objectives_list()

    def get_current_objective(self) -> dict | None:
        """Get the currently active objective info.

        Returns:
            dict | None: Active objective metadata, or None when it is not
                known (``resolve_current_objective`` says why).
        """
        return self._derive_current_objective()[1]

    def set_turret_config(self, turret_config: dict[int, str]) -> None:
        """Set the turret objective configuration.

        Args:
            turret_config: Mapping of turret position (1-4) to objective ID.
        """
        self._turret_config = turret_config

    def get_turret_config(self) -> dict:
        """Get the current turret objective configuration.

        Returns:
            dict: Mapping of turret position to objective ID.
        """
        return self._turret_config

    def set_stage_offset(self, stage_offset) -> None:
        """Set the stage offset for coordinate transformations.

        Args:
            stage_offset: Stage offset dict with axis offsets.
        """
        self._stage_offset = stage_offset

    def get_stage_offset(self) -> dict | None:
        """Get the stage offset for coordinate transformations.

        Returns:
            Stage offset dict with axis offsets, or None if unset.
        """
        return self._stage_offset

    def stage_to_plate(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert a stage position (um) to plate coordinates (mm).

        Uses the registered labware and stage offset -- the same
        transform ``get_well_label`` performs before its label lookup,
        exposed for consumers that need the coordinates themselves
        (e.g. image metadata).

        Args:
            sx: Stage X position in um.
            sy: Stage Y position in um.

        Returns:
            (px, py): Plate position in mm.

        Raises:
            NoLabwareSelectedError: If no labware is registered.
        """
        return self._coordinate_transformer.stage_to_plate(
            labware=self.get_labware(),
            stage_offset=self._require_stage_offset(),
            sx=sx,
            sy=sy,
        )

    def plate_transform(self) -> Callable[[float, float], tuple[float, float]] | None:
        """The stage-to-plate transform bound to the labware and offset registered now.

        ``stage_to_plate`` reads the registered labware and stage offset on
        every call, so a caller converting many positions over time -- a
        recording writing one per frame -- would record its later frames
        against a labware selected mid-way, or raise mid-stream when one is
        cleared. The transform returned here is bound to the objects
        registered at the moment of the call, so every position it
        converts is stated in one frame of reference and it cannot raise.

        Returns:
            A function of ``(sx_um, sy_um)`` answering ``(px_mm, py_mm)``,
            or None when no labware or no stage offset is registered: the
            caller then has no plate frame to state positions in, and says
            so, rather than being handed a transform that raises.
        """
        labware = self.get_labware()
        stage_offset = self.get_stage_offset()
        if labware is None or stage_offset is None:
            return None
        transformer = self._coordinate_transformer

        def to_plate(sx: float, sy: float) -> tuple[float, float]:
            return transformer.stage_to_plate(
                labware=labware, stage_offset=stage_offset, sx=sx, sy=sy
            )

        return to_plate

    def _require_stage_offset(self) -> dict:
        """The stage offset, or a refusal naming why a transform cannot run.

        The offset is written once, when the scope initializes. A
        transform attempted before that point would otherwise divide
        None and surface as a TypeError, which tells a user nothing they
        can act on and a REST caller nothing it can branch on.
        """
        stage_offset = self.get_stage_offset()
        if stage_offset is None:
            raise ConfigError(
                'stage offset is not set -- coordinate transforms require an '
                'initialized scope; connect the microscope first'
            )
        return stage_offset

    def plate_to_stage_axis(self, axis: str, plate_mm: float) -> float:
        """Convert one axis of a plate coordinate (mm) to a stage target (um).

        The completing half of ``stage_to_plate``. Per-axis because the
        callers that need it are commanding a single axis, and because
        each stage coordinate depends only on its own plate coordinate --
        so the unused argument below is inert, not a placeholder standing
        in for a value the caller should have supplied.
        """
        if axis not in ('X', 'Y'):
            raise ValueError(f'Plate coordinates are defined for X and Y, got {axis!r}')

        sx, sy = self._coordinate_transformer.plate_to_stage(
            labware=self.get_labware(),
            stage_offset=self._require_stage_offset(),
            px=plate_mm if axis == 'X' else 0,
            py=plate_mm if axis == 'Y' else 0,
        )
        return sx if axis == 'X' else sy

    def get_well_label(self) -> str | None:
        """Get the well label for the current stage XY position.

        Maps the current target X/Y stage position to a plate-frame
        coordinate using the registered labware and stage offset, then
        looks up the matching well label.

        Returns:
            str | None: Well label (e.g. ``"A1"``); ``''`` when the selected
            labware has no wells (the Blank plate); None when X or Y does not
            know its position. Consumers omit the well from filenames and
            metadata for both rather than stamping a fabricated one -- an axis
            that lost its reference keeps answering the last target it had,
            which names a real well the scope may no longer be over. The two
            stay distinct so a caller can say which it was.

        Raises:
            Exception: Re-raises any error encountered reading target
                position; logged before re-raise.
        """
        unknown = self._scope.motion.axes_without_position()
        if 'X' in unknown or 'Y' in unknown:
            return None

        labware = self.get_labware()

        try:
            x_target = self._scope.motion.get_target_position('X')
            y_target = self._scope.motion.get_target_position('Y')
        except Exception:
            logger.exception('[LVP API  ] Error getting target position.')
            raise

        x_target, y_target = self.stage_to_plate(sx=x_target, sy=y_target)

        return labware.get_well_label(x=x_target, y=y_target)
