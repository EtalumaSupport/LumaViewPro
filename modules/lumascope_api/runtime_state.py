# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""RuntimeState -- runtime-mutable scope state, split from ScopeCapabilities.

Design doc sec 2.5 splits the capabilities surface into TWO:

- `scope.capabilities`: IMMUTABLE per Lumascope instance. Reconnect =
  new Lumascope = new capabilities.
- `scope.runtime_state`: MUTABLE, refreshed on driver events (reflash,
  reconnect, etc.).

The split exists because firmware version legitimately mutates mid-
session when boards are reflashed; a single frozen surface would lie
post-flash. Sub-APIs read from BOTH surfaces as needed -- capability-
probe gates use the immutable surface, recovery / version gates use
the runtime surface.

This module ships as an empty placeholder per design doc sec 10. Real
content lands when FW4.0 populates `INFO.features` (firmware_features)
and when reconnect-aware versioning hooks are added to the driver
layer (firmware_versions). Until then, both fields stay empty dicts;
callers treat empty as "feature unknown" per Rule 8 corollary.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.5 and sec 10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class RuntimeState:
    """Mutable runtime state on a Lumascope -- firmware versions, feature flags,
    user-config runtime state (labware / objective / turret / stage).

    Fields land per design doc sec 2.5 as the underlying firmware +
    driver hooks ship.
    """

    def __init__(self, scope: 'Lumascope') -> None:
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

    # --- Settings-host forwarders (transitional 8b->8d window) ---
    #
    # These 13 forwarders route through self._scope.X(...) so production
    # callers can migrate from `scope.X(...)` to `scope.runtime_state.X(...)`
    # incrementally. The canonical bodies still live on Lumascope until
    # 8d moves them here atomically; 8f deletes the Lumascope-side
    # forwarders, making RuntimeState the canonical home.

    def set_labware(self, labware) -> None:
        return self._scope.set_labware(labware)

    def get_labware(self) -> 'Any | None':
        return self._scope.get_labware()

    def set_objective(self, objective_id: str) -> None:
        return self._scope.set_objective(objective_id)

    def get_current_objective_id(self) -> str | None:
        return self._scope.get_current_objective_id()

    def get_objective_info(self, objective_id: str) -> dict:
        return self._scope.get_objective_info(objective_id)

    def get_available_objectives(self) -> list[str]:
        return self._scope.get_available_objectives()

    def get_current_objective(self) -> dict | None:
        return self._scope.get_current_objective()

    def set_turret_config(self, turret_config: dict[int, str]) -> None:
        return self._scope.set_turret_config(turret_config)

    def get_turret_config(self) -> dict:
        return self._scope.get_turret_config()

    def set_stage_offset(self, stage_offset) -> None:
        return self._scope.set_stage_offset(stage_offset)

    def get_stage_offset(self) -> 'dict | None':
        return self._scope.get_stage_offset()

    def get_well_label(self) -> str:
        return self._scope.get_well_label()
