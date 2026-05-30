# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Lumascope API package.

The composition root Lumascope and its public state enum AxisState live
in _lumascope.py. The six sub-API classes (MotionAPI, IlluminationAPI,
ImagingAPI, DiagnosticsAPI, Capabilities, IOAPI) live in sibling
modules in this package.

Existing imports `from modules.lumascope_api import Lumascope` and
`from modules.lumascope_api import AxisState` keep working via the
re-exports below.
"""

from modules.lumascope_api._lumascope import (
    AxisState,
    Lumascope,
    # Module-level names that tests / callers monkeypatch via
    # `modules.lumascope_api.<name>`. Preserved as re-exports so the
    # patch paths keep working; eligible for retirement once test
    # fixtures stop patching them.
    logger,
    notifications,
    _notify_board_failure,
    _notify_camera_failure,
    _try_connect_board,
)
from modules.lumascope_api.capabilities import Capabilities
from modules.lumascope_api.diagnostics import DiagnosticsAPI
from modules.lumascope_api.illumination import IlluminationAPI
from modules.lumascope_api.imaging import ImagingAPI
from modules.lumascope_api.io import IOAPI
from modules.lumascope_api.motion import MotionAPI
from modules.lumascope_api.runtime_state import RuntimeState

__all__ = [
    'IOAPI',
    'AxisState',
    'Capabilities',
    'DiagnosticsAPI',
    'IlluminationAPI',
    'ImagingAPI',
    'Lumascope',
    'MotionAPI',
    'RuntimeState',
]
