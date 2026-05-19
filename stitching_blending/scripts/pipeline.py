"""End-to-end stitching pipeline for the standalone prototype."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .alignment import align_neighbor_pairs
    from .blending import blend_tiles
    from .io_utils import read_image, write_csv, write_image, write_json, write_preview_png
    from .metadata import load_tile_metadata, metadata_positions, resolve_tile_path
    from .metrics import compute_metrics, write_metrics
    from .optimization import optimize_tile_positions
    from .overlap import infer_neighbors
    from .paths import OUTPUTS_DIR, ensure_project_dirs
except ImportError:  # pragma: no cover
    from alignment import align_neighbor_pairs
    from blending import blend_tiles
    from io_utils import read_image, write_csv, write_image, write_json, write_preview_png
    from metadata import load_tile_metadata, metadata_positions, resolve_tile_path
    from metrics import compute_metrics, write_metrics
    from optimization import optimize_tile_positions
    from overlap import infer_neighbors
    from paths import OUTPUTS_DIR, ensure_project_dirs


def run_stitching_pipeline(
    metadata_csv: str | Path,
    *,
    output_dir: str | Path | None = None,
    run_name: str = 'synthetic_demo',
    blend_mode: str = 'feather',
    coordinate_mode: str = 'auto',
    pixel_size: float = 1.0,
    max_shift_px: int = 10,
    phase_confidence_threshold: float = 0.08,
    ground_truth_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run metadata placement, alignment, optimization, blending, and metrics."""
    ensure_project_dirs()
    start = time.perf_counter()
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir) if output_dir is not None else OUTPUTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_tile_metadata(metadata_csv)
    metadata = metadata_positions(metadata, coordinate_mode=coordinate_mode, pixel_size=pixel_size)
    tiles = {
        str(row.tile_id): read_image(resolve_tile_path(metadata_csv, row.filepath))
        for row in metadata.itertuples(index=False)
    }

    neighbors = infer_neighbors(metadata)
    alignments = align_neighbor_pairs(
        tiles,
        neighbors,
        max_shift_px=max_shift_px,
        phase_confidence_threshold=phase_confidence_threshold,
    )
    placements = optimize_tile_positions(
        metadata,
        alignments,
        min_confidence=phase_confidence_threshold,
    )
    stitched, normalized_placements = blend_tiles(tiles, placements, mode=blend_mode)

    stitched_path = output_dir / 'stitched.tif'
    preview_path = output_dir / 'preview.png'
    metadata_out = output_dir / 'tile_metadata.csv'
    alignments_out = output_dir / 'pairwise_alignments.csv'
    placements_out = output_dir / 'tile_placements.csv'
    metrics_out = output_dir / 'metrics.json'
    manifest_out = output_dir / 'manifest.json'

    write_image(stitched_path, stitched)
    write_preview_png(preview_path, stitched)
    write_csv(metadata_out, metadata)
    write_csv(alignments_out, alignments)
    write_csv(placements_out, normalized_placements)

    reference = None
    if ground_truth_path is not None:
        ground_truth_path = Path(ground_truth_path)
        if ground_truth_path.exists():
            reference = read_image(ground_truth_path)
            shutil.copy2(ground_truth_path, output_dir / 'ground_truth.tif')

    runtime = time.perf_counter() - start
    metric_values = compute_metrics(
        stitched,
        reference=reference,
        tiles=tiles,
        placements=placements,
        metadata=metadata,
        alignments=alignments,
        runtime_seconds=runtime,
    )
    write_metrics(metrics_out, metric_values)

    manifest = {
        'run_name': run_name,
        'metadata_csv': str(metadata_csv.resolve()),
        'ground_truth_path': str(Path(ground_truth_path).resolve()) if ground_truth_path is not None else None,
        'blend_mode': blend_mode,
        'coordinate_mode': coordinate_mode,
        'pixel_size': pixel_size,
        'max_shift_px': max_shift_px,
        'phase_confidence_threshold': phase_confidence_threshold,
        'outputs': {
            'stitched_image': stitched_path.name,
            'preview_png': preview_path.name,
            'tile_metadata_csv': metadata_out.name,
            'pairwise_alignments_csv': alignments_out.name,
            'tile_placements_csv': placements_out.name,
            'metrics_json': metrics_out.name,
            'manifest_json': manifest_out.name,
        },
    }
    write_json(manifest_out, manifest)

    return {
        'output_dir': output_dir,
        'stitched_path': stitched_path,
        'preview_path': preview_path,
        'metadata_path': metadata_out,
        'alignments_path': alignments_out,
        'placements_path': placements_out,
        'metrics_path': metrics_out,
        'manifest_path': manifest_out,
        'metrics': metric_values,
    }

