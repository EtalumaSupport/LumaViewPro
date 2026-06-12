# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Built-in plugins registered against ctx.plugins at host startup.

These are the LumaViewPro post-processing classes that already shipped
as in-tree implementations (Stitcher, ZProjector, CompositeGeneration,
VideoBuilder) -- they retire INTO ctx.plugins.post_processing during
the D9 migration so the plugin contract is validated on real, shipping
workloads before the intern's first plugin.

Each built-in is registered by name from
modules.plugins.builtin.register_builtins(ctx), called by
lumaviewpro.build() AFTER load_plugins(ctx) has fired so any
third-party plugin sharing a name is rejected first (built-ins lose
name collisions, which is the right policy for an opt-in shim).

The legacy invocation paths (UI button handlers, file_dialogs
dispatch) keep working unchanged. The plugin registration is additive
-- it lets a plugin author call Stitcher via the platform contract
without touching the UI handlers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger('lvp_logger')


def register_builtins(ctx: Any) -> None:
    """Register the in-tree built-in plugins against ctx.plugins.

    Called once from lumaviewpro.build() after load_plugins(ctx). Each
    built-in is wrapped in try/except so a failure in one (e.g. a
    name collision with a third-party plugin loaded earlier) does
    not block the others.

    Built-ins registered:
        stitcher -- grid stitcher; thin adapter around
                    modules.stitcher.Stitcher.load_folder.
    """
    if ctx is None or not hasattr(ctx, 'plugins'):
        logger.error('[Plugins ] register_builtins called without ctx.plugins')
        return

    from modules.plugins.builtin import stitcher_plugin

    for module, name in ((stitcher_plugin, 'stitcher'),):
        try:
            module.register(ctx)
        except Exception as e:
            logger.warning(
                f'[Plugins ] built-in {name} failed to register: {type(e).__name__}: {e}'
            )
