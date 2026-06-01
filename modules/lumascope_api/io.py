# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IOAPI -- placeholder for USB-to-IO trigger device support.

Ships as an empty placeholder. Methods land when the third-board
trigger driver is built. Reserving the name keeps platform churn
zero. EL-0940 TRIGI / TRIGO on the LED controller will surface here
when LVP-side trigger work begins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class IOAPI:
    """Trigger-IO sub-API. Empty placeholder for Phase 1."""

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope
