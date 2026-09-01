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

from typing import TYPE_CHECKING, Any

import modules.coord_transformations as coord_transformations
import modules.objectives_loader as objectives_loader
from lvp_logger import logger
from modules.exceptions import ConfigError

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class RuntimeState:
    """Mutable runtime state on a Lumascope -- firmware versions, feature flags,
    user-config runtime state (labware / objective / turret / stage).

    Fields land per design doc sec 2.5 as the underlying firmware +
    driver hooks ship.
    """

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope
        self.firmware_versions: dict[str, str] = {}
        """Per-board firmware version. Populated when reflash-aware
        hooks ship; until then, callers query via
        `scope.diagnostics.get_motor_info()` etc."""

        self.firmware_features: dict[str, frozenset[str]] = {}
        """Per-subsystem capability set declared by current firmware.
        Empty default; callers treat empty as 'feature unknown' per
        Rule 8 corollary. Populated when FW4.0's `INFO.features` block
        ships."""

        self._labware: Any | None = None
        self._objective: dict | None = None
        self._objective_id: str | None = None
        self._turret_config: dict = {}
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

    def set_objective(self, objective_id: str) -> None:
        """Set the active objective by ID.

        Args:
            objective_id: Objective identifier (e.g. "4x", "10x", "20x").

        Raises:
            ConfigError: The id resolves to no objective. State is
                untouched on failure -- resolving before assigning keeps
                the id and the info describing the same objective, so a
                bad id can never leave the pair torn.
        """
        objective = self._objectives_loader.get_objective_info(objective_id=objective_id)
        if objective is None:
            raise ConfigError(f'unknown objective {objective_id!r}; active objective unchanged')
        self._objective_id = objective_id
        self._objective = objective

    def get_current_objective_id(self) -> str | None:
        """Get the ID of the currently active objective.

        Returns:
            str | None: e.g. '20x Oly', or None if not set.
        """
        return getattr(self, '_objective_id', None)

    def get_objective_info(self, objective_id: str) -> dict:
        """Get objective metadata by ID.

        Args:
            objective_id: Objective identifier (e.g. "4x", "10x", "20x").

        Returns:
            dict: Objective info including focal_length, magnification, etc.
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
            dict | None: Active objective metadata, or None if not set.
        """
        return self._objective

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
            stage_offset=self.get_stage_offset(),
            sx=sx,
            sy=sy,
        )

    def get_well_label(self) -> str:
        """Get the well label for the current stage XY position.

        Maps the current target X/Y stage position to a plate-frame
        coordinate using the registered labware and stage offset, then
        looks up the matching well label.

        Returns:
            str: Well label (e.g. ``"A1"``), or ``''`` when the selected
            labware has no wells (the Blank plate) -- consumers omit the
            well from filenames and metadata rather than stamping a
            fabricated one.

        Raises:
            Exception: Re-raises any error encountered reading target
                position; logged before re-raise.
        """
        labware = self.get_labware()

        try:
            x_target = self._scope.motion.get_target_position('X')
            y_target = self._scope.motion.get_target_position('Y')
        except Exception:
            logger.exception('[LVP API  ] Error getting target position.')
            raise

        x_target, y_target = self.stage_to_plate(sx=x_target, sy=y_target)

        return labware.get_well_label(x=x_target, y=y_target)
