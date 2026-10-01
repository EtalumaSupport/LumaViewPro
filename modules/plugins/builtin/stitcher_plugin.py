# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stitcher post-processing plugin -- a thin adapter over the session's stitch.

This module validates the ctx.plugins.post_processing contract against a
real, shipping workload. It does NOT reimplement stitching: the processor
calls ``session.post_processing.stitch``, the member the UI button calls,
so the build runs on the post-processing lane against the installation's
own tiling config and the scope's own turret.

The processor contract:
    processor(input_dir, manifest, output_dir) -> ProcessorResult

How the three args map onto the stitch:
    input_dir   -- the protocol folder to stitch (outputs go back inside
                   this folder under per-step subdirs, same as the UI path).
    manifest    -- accepted for contract compliance; the stitch needs
                   nothing from it.
    output_dir  -- accepted for contract compliance; the stitch writes
                   inside input_dir and ignoring output_dir is intentional.
                   Surfaced in ProcessorResult.metadata so the host knows
                   where to look.

Return shape:
    ProcessorResult.success is True when the stitch returned, False when it
        raised.
    ProcessorResult.message is the stitch's own words, or the raised
        outcome's.
    ProcessorResult.outputs lists every artifact written, the partial set
        included when the stitch was incomplete.
"""

from __future__ import annotations

import functools
import logging
import pathlib
from typing import Any

from modules.exceptions import CaptureError
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
    post_processing: Any,
    input_dir: str,
    manifest: dict,
    output_dir: str,
) -> ProcessorResult:
    """Plugin processor callable -- adapts the session's stitch to the
    post_processing contract. ``post_processing`` is the session's
    ``PostProcessingAPI``, bound at registration."""
    input_path = _coerce_path(input_dir)
    if input_path is None:
        return ProcessorResult(
            success=False,
            message='Stitcher: input_dir not provided.',
        )

    metadata = {
        'input_dir': str(input_path),
        'output_dir': str(output_dir) if output_dir else '',
    }
    try:
        result = post_processing.stitch(input_path)
    except Exception as e:
        logger.error(
            f'[Plugins ] stitcher: stitch raised {type(e).__name__}: {e}',
            exc_info=True,
        )
        # A typed outcome is written for the person and says what was
        # produced; only an untyped one is named by its class.
        if isinstance(e, CaptureError):
            message = str(e)
        else:
            message = f'Stitching failed -- {type(e).__name__}: {e}'
        return ProcessorResult(
            success=False,
            outputs=tuple(getattr(e, 'produced_paths', ())),
            message=message,
            metadata=metadata,
        )

    return ProcessorResult(
        success=True,
        outputs=tuple(result.get('artifact_paths', ())),
        message=str(result.get('message', '')) or 'Stitching complete.',
        metadata=metadata,
    )


def register(ctx: Any) -> None:
    """Register the stitcher processor with ctx.plugins.post_processing.

    Called from modules.plugins.builtin.register_builtins (in-tree
    path) and also usable directly from a load_plugins entry_points
    discovery (so an external package could ship its own version
    later by claiming the same plugin name and winning the load
    order).
    """
    ctx.plugins.post_processing.register(
        spec, functools.partial(_stitcher_processor, ctx.session.post_processing)
    )
    logger.info(
        f'[Plugins ] {spec.name} v{spec.version} registered with '
        f'ctx.plugins.post_processing (canary)'
    )


def unregister(ctx: Any) -> None:
    """No-op for this built-in -- the registry has no remove method,
    and the built-in is tied to the host's lifetime. Defined so
    load_plugins's partial-failure cleanup path can call it without
    AttributeError if the spec moves to entry_points discovery later.
    """
    return
