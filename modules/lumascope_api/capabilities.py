# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Capabilities -- immutable scope-identity surface.

Phase 1 of Wave 7 decomposition: this module re-exports ScopeCapabilities
from its existing home in modules/scope_capabilities.py under the
canonical name Capabilities so callers can write
`scope.capabilities` (the attribute on Lumascope) and refer to the
type as `Capabilities` (per design doc sec 2.5).

The runtime-mutable split (scope.runtime_state for firmware_versions
etc.) lands in a later Wave 7 phase.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.5.
"""

from __future__ import annotations

from modules.scope_capabilities import ScopeCapabilities as Capabilities

__all__ = ['Capabilities']
