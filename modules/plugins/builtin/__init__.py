# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Built-in plugins, registered against ctx.plugins when a session loads its plugins.

These are the LumaViewPro post-processing classes that already shipped
as in-tree implementations (Stitcher, ZProjector, CompositeGeneration,
VideoBuilder) -- they retire INTO ctx.plugins.post_processing during
the D9 migration so the plugin contract is validated on real, shipping
workloads before the intern's first plugin.

The session's PluginRegistry.load() registers each module in
BUILTIN_PLUGINS, AFTER the installed plugins, so a third-party
plugin sharing a name keeps it and the built-in is the one reported as
not loaded (built-ins lose name collisions, which is the right policy
for an opt-in shim).

The legacy invocation paths (UI button handlers, file_dialogs
dispatch) keep working unchanged. The plugin registration is additive
-- it lets a plugin author call Stitcher via the platform contract
without touching the UI handlers.
"""

from __future__ import annotations

from modules.plugins.builtin import stitcher_plugin

# The in-tree plugin modules, in the order they register.
BUILTIN_PLUGINS = (stitcher_plugin,)
