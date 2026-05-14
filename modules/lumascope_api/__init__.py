# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Lumascope API package.

Phase 1 of Wave 7 decomposition. The composition root Lumascope and
its public state enum AxisState live in _lumascope.py (the original
6300+-line module, renamed). The six sub-API classes
(MotionAPI, IlluminationAPI, ImagingAPI, DiagnosticsAPI, Capabilities,
IOAPI) live in sibling modules in this package.

Existing imports `from modules.lumascope_api import Lumascope` and
`from modules.lumascope_api import AxisState` keep working via the
re-exports below; no caller migration needed in Phase 1.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2 for the canonical
sub-API specification.
"""

from modules.lumascope_api._lumascope import (
    AxisState,
    Lumascope,
    # Module-level names that pre-Wave 7 tests / callers monkeypatch
    # via modules.lumascope_api.XXX. Preserved as re-exports so the
    # patch paths keep working post-decomposition. Wave 7 Phase 7 may
    # retire these once test fixtures stop patching them.
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

__all__ = [
    'AxisState',
    'Capabilities',
    'DiagnosticsAPI',
    'IlluminationAPI',
    'ImagingAPI',
    'IOAPI',
    'Lumascope',
    'MotionAPI',
]
