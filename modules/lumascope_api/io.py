# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IOAPI -- placeholder for USB-to-IO trigger device support.

Reserved per design doc sec 1.1 D2: ships as an empty placeholder in
Wave 7 Phase 1. Methods land in a later wave when the third-board
trigger driver is built. Reserving the name pre-freeze keeps platform
churn zero. EL-0940 TRIGI / TRIGO on the LED controller will surface
here when LVP-side trigger work begins.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 1.1 D2 and roadmap
feature F9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class IOAPI:
    """Trigger-IO sub-API. Empty placeholder for Phase 1."""

    def __init__(self, scope: 'Lumascope') -> None:
        self._scope = scope
