# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stitcher post-processing plugin -- thin adapter around modules.stitcher.Stitcher.

This module exists to validate the ctx.plugins.post_processing contract
against a real, shipping workload (Phase A canary). It does NOT
reimplement stitching: the processor callable instantiates the same
Stitcher class the UI button path uses and forwards to its load_folder
entry point. The UI button (StitchControls.run_stitcher) keeps working
unchanged; registering here is additive.

The processor contract:
    processor(input_dir, manifest, output_dir) -> ProcessorResult

How the three args map onto Stitcher.load_folder:
    input_dir   -- the protocol folder to stitch (Stitcher writes
                   outputs back inside this folder under per-step
                   subdirs, same as the UI path).
    manifest    -- dict from the host carrying:
                       'has_turret': bool -- forwarded to Stitcher init
                       'tiling_configs_file_loc': str | Path -- path to
                           data/tiling.json (required by the base
                           ProtocolPostProcessor.load_folder).
                   Falls back to safe defaults when keys are missing so
                   harness tests can drive the processor without
                   constructing a full ctx.
    output_dir  -- accepted for contract compliance; Stitcher writes
                   inside input_dir today and ignoring output_dir is
                   intentional. Surfaced in ProcessorResult.metadata
                   so the host knows where to look.

Return shape:
    ProcessorResult.success mirrors Stitcher's {'status': bool}.
    ProcessorResult.message mirrors Stitcher's {'message': str}.
    ProcessorResult.outputs is empty because Stitcher writes its
        artifacts into per-group subdirs whose names are computed
        per-group (the existing path doesn't return a list; surfacing
        each output file would require a Stitcher API change which is
        out of scope for the canary).
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

from modules.plugins import PluginSpec, ProcessorResult


__version__ = '0.1.0'

logger = logging.getLogger('lvp_logger')


# Module-level spec so load_plugins() can discover it via the
# entry_points path AND so register_builtins() can find it by attribute
# without re-instantiating. The platform requires_lvp_version gate
# locks Stitcher canary to 4.0.0+ where ctx.plugins exists.
spec = PluginSpec(
    name='stitcher',
    version=__version__,
    requires_lvp_version='>=4.0.0',
    description=(
        'Grid stitcher for protocol scans -- assembles tile images '
        'into a single TIFF per (well, color, slice) using captured '
        'X/Y positions.'
    ),
    capabilities=('modules.stitcher', 'modules.image_save'),
    subscribes_to=(),
    author='Etaluma',
    url='',
)


def _coerce_path(value: Any) -> pathlib.Path | None:
    """Accept str or Path; return Path, or None when value is falsy."""
    if value is None or value == '':
        return None
    if isinstance(value, pathlib.Path):
        return value
    return pathlib.Path(str(value))


def _stitcher_processor(
    input_dir: str,
    manifest: dict,
    output_dir: str,
) -> ProcessorResult:
    """Plugin processor callable -- adapts Stitcher.load_folder to the
    post_processing contract.

    Imports Stitcher lazily so the plugin module can be imported in
    test harnesses that don't have cv2/pandas/numpy already loaded
    (the harness ctx fixture itself does not need them; only the
    processor invocation does).
    """
    input_path = _coerce_path(input_dir)
    if input_path is None:
        return ProcessorResult(
            success=False,
            message='Stitcher: input_dir not provided.',
        )

    manifest = manifest or {}
    has_turret = bool(manifest.get('has_turret', False))

    tiling_cfg = _coerce_path(manifest.get('tiling_configs_file_loc'))
    if tiling_cfg is None:
        # Default to the repo-shipped tiling.json. Stitcher.load_folder
        # reads this to map tile-group IDs back to grid layout; a
        # missing file causes load_folder to surface a clean failure
        # via its 'status'/'message' return.
        tiling_cfg = pathlib.Path('data') / 'tiling.json'

    from modules.stitcher import Stitcher

    stitcher = Stitcher(has_turret=has_turret)

    try:
        result = stitcher.load_folder(
            path=input_path,
            tiling_configs_file_loc=tiling_cfg,
            popup=None,
        )
    except Exception as e:
        logger.error(
            f'[Plugins ] stitcher: load_folder raised {type(e).__name__}: {e}',
            exc_info=True,
        )
        return ProcessorResult(
            success=False,
            message=f'Stitching failed -- {type(e).__name__}: {e}',
            metadata={'input_dir': str(input_path)},
        )

    status = bool(result.get('status', False))
    message = str(result.get('message', '')) or (
        'Stitching complete.' if status else 'Stitching failed.'
    )
    return ProcessorResult(
        success=status,
        outputs=(),
        message=message,
        metadata={
            'input_dir': str(input_path),
            'output_dir': str(output_dir) if output_dir else '',
            'has_turret': has_turret,
            'tiling_configs_file_loc': str(tiling_cfg),
        },
    )


def register(ctx: Any) -> None:
    """Register the stitcher processor with ctx.plugins.post_processing.

    Called from modules.plugins.builtin.register_builtins (in-tree
    path) and also usable directly from a load_plugins entry_points
    discovery (so an external package could ship its own version
    later by claiming the same plugin name and winning the load
    order).
    """
    ctx.plugins.post_processing.register(spec, _stitcher_processor)
    logger.info(
        f'[Plugins ] {spec.name} v{spec.version} registered with '
        f'ctx.plugins.post_processing (canary)'
    )


def unregister(ctx: Any) -> None:
    """No-op for the canary -- the Phase A registry has no remove
    method, and the built-in is tied to the host's lifetime. Defined
    so load_plugins's partial-failure cleanup path can call it without
    AttributeError if the spec moves to entry_points discovery later.
    """
    return
