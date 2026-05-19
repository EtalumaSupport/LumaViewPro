"""Metrics for stitched microscopy mosaics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .io_utils import read_image, write_json
    from .optimization import placement_rmse
except ImportError:  # pragma: no cover
    from io_utils import read_image, write_json
    from optimization import placement_rmse


def mse(image: np.ndarray, reference: np.ndarray) -> float:
    """Mean squared error over the common image region."""
    image_f, ref_f = _common_region(image, reference)
    return float(np.mean((image_f.astype(np.float64) - ref_f.astype(np.float64)) ** 2))


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB over the common image region."""
    value = mse(image, reference)
    if value <= 0:
        return float('inf')
    dtype = image.dtype if np.issubdtype(image.dtype, np.integer) else reference.dtype
    peak = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(value))


def seam_energy(image: np.ndarray) -> float:
    """Approximate seam energy as mean absolute first derivative."""
    image_f = image.astype(np.float64)
    gy = np.abs(np.diff(image_f, axis=0)).mean() if image.shape[0] > 1 else 0.0
    gx = np.abs(np.diff(image_f, axis=1)).mean() if image.shape[1] > 1 else 0.0
    return float(gx + gy)


def overlap_consistency(
    tiles: dict[str, np.ndarray],
    placements: pd.DataFrame,
) -> float | None:
    """Mean absolute difference where placed tiles overlap."""
    placed = placements.copy()
    placed['x_int'] = np.rint(placed['x_px']).astype(int)
    placed['y_int'] = np.rint(placed['y_px']).astype(int)
    diffs: list[float] = []

    rows = list(placed.itertuples(index=False))
    for i, left in enumerate(rows):
        tile_a = tiles[str(left.tile_id)].astype(np.float64)
        ax0, ay0 = int(left.x_int), int(left.y_int)
        ax1, ay1 = ax0 + tile_a.shape[1], ay0 + tile_a.shape[0]
        for right in rows[i + 1 :]:
            tile_b = tiles[str(right.tile_id)].astype(np.float64)
            bx0, by0 = int(right.x_int), int(right.y_int)
            bx1, by1 = bx0 + tile_b.shape[1], by0 + tile_b.shape[0]
            x0, x1 = max(ax0, bx0), min(ax1, bx1)
            y0, y1 = max(ay0, by0), min(ay1, by1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop_a = tile_a[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0]
            crop_b = tile_b[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0]
            diffs.append(float(np.mean(np.abs(crop_a - crop_b))))
    if not diffs:
        return None
    return float(np.mean(diffs))


def compute_metrics(
    stitched: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    tiles: dict[str, np.ndarray] | None = None,
    placements: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    alignments: pd.DataFrame | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute registration, image-quality, seam, runtime, and memory metrics."""
    result: dict[str, Any] = {
        'runtime_seconds': runtime_seconds,
        'approx_memory_bytes': int(stitched.nbytes + sum(tile.nbytes for tile in (tiles or {}).values())),
        'seam_energy': seam_energy(stitched),
    }

    if reference is not None:
        result['mse'] = mse(stitched, reference)
        result['psnr_db'] = psnr(stitched, reference)
        result.update(_ssim_metric(stitched, reference))
    else:
        result.update({'mse': None, 'psnr_db': None, 'ssim': None, 'ssim_note': 'reference unavailable'})

    if tiles is not None and placements is not None:
        result['overlap_consistency_error'] = overlap_consistency(tiles, placements)
    else:
        result['overlap_consistency_error'] = None

    if placements is not None and metadata is not None:
        result['registration_rmse_px'] = placement_rmse(placements, metadata)
        result['placement_mode'] = _placement_mode(placements)
    else:
        result['registration_rmse_px'] = None
        result['placement_mode'] = None

    if alignments is not None and len(alignments) > 0:
        result['alignment_acceptance_rate'] = float(alignments['accepted'].mean())
        result['alignment_count'] = int(len(alignments))
    else:
        result['alignment_acceptance_rate'] = None
        result['alignment_count'] = 0

    return result


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Write metrics JSON."""
    write_json(path, metrics)


def _placement_mode(placements: pd.DataFrame) -> str | None:
    if 'placement_mode' not in placements.columns or placements.empty:
        return None
    modes = sorted(set(str(mode) for mode in placements['placement_mode']))
    return modes[0] if len(modes) == 1 else ','.join(modes)


def _common_region(image: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(image.shape[0], reference.shape[0])
    w = min(image.shape[1], reference.shape[1])
    return image[:h, :w], reference[:h, :w]


def _ssim_metric(image: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:  # pragma: no cover - depends on optional import state
        return {'ssim': None, 'ssim_note': f'scikit-image unavailable: {exc}'}

    image_c, ref_c = _common_region(image, reference)
    channel_axis = -1 if image_c.ndim == 3 else None
    dtype = image_c.dtype if np.issubdtype(image_c.dtype, np.integer) else ref_c.dtype
    data_range = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    return {
        'ssim': float(structural_similarity(image_c, ref_c, data_range=data_range, channel_axis=channel_axis)),
        'ssim_note': None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Recompute basic metrics for a stitched output directory.')
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    stitched = read_image(args.output_dir / 'stitched.tif')
    reference_path = args.output_dir / 'ground_truth.tif'
    reference = read_image(reference_path) if reference_path.exists() else None
    metrics = compute_metrics(stitched, reference=reference)
    write_metrics(args.output_dir / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

