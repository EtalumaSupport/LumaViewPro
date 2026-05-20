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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class RuntimeState:
    """Mutable runtime state on a Lumascope -- firmware versions, feature flags.

    Empty placeholder in 4.0.x. Fields land per design doc sec 2.5 as
    the underlying firmware + driver hooks ship.
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
