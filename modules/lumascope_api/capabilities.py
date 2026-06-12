# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Capabilities -- immutable scope-identity surface.

Re-exports ScopeCapabilities from modules/scope_capabilities.py under
the canonical name Capabilities so callers can write
`scope.capabilities` (the attribute on Lumascope) and refer to the
type as `Capabilities`.

The runtime-mutable split lives on scope.runtime_state (firmware
versions, etc.).
"""

from __future__ import annotations

from modules.scope_capabilities import ScopeCapabilities as Capabilities

__all__ = ['Capabilities']
