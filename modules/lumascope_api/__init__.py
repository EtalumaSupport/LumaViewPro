# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Lumascope API package.

The composition root Lumascope and its public state enum AxisState live
in _lumascope.py. The sub-API classes (MotionAPI, IlluminationAPI,
ImagingAPI, DiagnosticsAPI, Capabilities, IOAPI, RuntimeState,
ProtocolsAPI) live in sibling modules in this package.

Existing imports `from modules.lumascope_api import Lumascope` and
`from modules.lumascope_api import AxisState` keep working via the
re-exports below.
"""

from modules.lumascope_api._lumascope import (
    AxisState,
    Lumascope,
)
from modules.lumascope_api.capabilities import Capabilities
from modules.lumascope_api.diagnostics import DiagnosticsAPI
from modules.lumascope_api.illumination import IlluminationAPI
from modules.lumascope_api.imaging import ImagingAPI
from modules.lumascope_api.io import IOAPI
from modules.lumascope_api.motion import MotionAPI
from modules.lumascope_api.protocols import ProtocolsAPI
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
    'ProtocolsAPI',
    'RuntimeState',
]
