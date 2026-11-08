# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Lumascope API package.

The composition root Lumascope and its public state enum AxisState live
in _lumascope.py. The sub-API classes (MotionAPI, IlluminationAPI,
CameraAPI, ImagingAPI, DiagnosticsAPI, Capabilities, RuntimeState,
ProtocolsAPI) live in sibling modules in this package.

Existing imports `from _lumascope.modules.lumascope_api import Lumascope` and
`from _lumascope.modules.lumascope_api import AxisState` keep working via the
re-exports below. ``Frame`` is the record a frame listener receives.
"""

from _lumascope.drivers.camera import CameraNodeValue, Frame
from _lumascope.modules.axes import Axis, StageAxis
from _lumascope.modules.lumascope_api._constants import AxisPosition, TurretStatus
from _lumascope.modules.lumascope_api._lumascope import (
    AxisState,
    Lumascope,
)
from _lumascope.modules.lumascope_api.camera import CameraAPI
from _lumascope.modules.lumascope_api.capabilities import Capabilities
from _lumascope.modules.lumascope_api.diagnostics import DiagnosticsAPI
from _lumascope.modules.lumascope_api.illumination import IlluminationAPI
from _lumascope.modules.lumascope_api.imaging import ImagingAPI
from _lumascope.modules.lumascope_api.motion import HomeTarget, Interlock, MotionAPI
from _lumascope.modules.lumascope_api.protocols import ProtocolsAPI
from _lumascope.modules.lumascope_api.runtime_state import RuntimeState
from _lumascope.modules.api_surface import api_fields

# The camera driver defines the records and cannot import the marker, so the
# API publishes their fields here.
api_fields('image', 'seq', 'arrived_ns', 'wall_ns', 'significant_bits', 'chunks')(Frame)
api_fields('value', 'unit')(CameraNodeValue)

__all__ = [
    'Axis',
    'AxisPosition',
    'AxisState',
    'CameraAPI',
    'Capabilities',
    'DiagnosticsAPI',
    'Frame',
    'HomeTarget',
    'IlluminationAPI',
    'ImagingAPI',
    'Interlock',
    'Lumascope',
    'MotionAPI',
    'ProtocolsAPI',
    'RuntimeState',
    'StageAxis',
    'TurretStatus',
]
