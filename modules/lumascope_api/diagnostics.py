# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""DiagnosticsAPI -- sub-API for hardware diagnostic probes.

Phase 1 of Wave 7 decomposition. Thin delegating facade over the
Lumascope composition root. Bodies still live on Lumascope; later
phases relocate them and migrate callers.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.4 for the canonical
method list. No persistent state -- per-call probes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class DiagnosticsAPI:
    """Diagnostics sub-API. Forwards to Lumascope composition root."""

    def __init__(self, scope: 'Lumascope') -> None:
        self._scope = scope

    # --- Camera probes ---
    def get_camera_temperatures(self, *args, **kwargs):
        return self._scope.get_camera_temperatures(*args, **kwargs)

    def get_camera_diagnostic_info(self, *args, **kwargs):
        return self._scope.get_camera_diagnostic_info(*args, **kwargs)

    def run_camera_bandwidth_test(self, *args, **kwargs):
        return self._scope.run_camera_bandwidth_test(*args, **kwargs)

    def run_grab_lifecycle_benchmark(self, *args, **kwargs):
        return self._scope.run_grab_lifecycle_benchmark(*args, **kwargs)

    def run_pylon_diagnostic_probe(self, *args, **kwargs):
        return self._scope.run_pylon_diagnostic_probe(*args, **kwargs)

    # --- Serial probes ---
    def send_diagnostic_command(self, *args, **kwargs):
        return self._scope.send_diagnostic_command(*args, **kwargs)

    def send_diagnostic_command_multiline(self, *args, **kwargs):
        return self._scope.send_diagnostic_command_multiline(*args, **kwargs)
