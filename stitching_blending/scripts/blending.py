"""Tile compositing and seam blending methods."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

try:
    from .io_utils import cast_preserving_dtype
except ImportError:  # pragma: no cover
    from io_utils import cast_preserving_dtype


def _tile_weights(shape: tuple[int, ...], mode: str) -> np.ndarray:
    h, w = shape[:2]
    if mode == 'average':
        return np.ones((h, w), dtype=np.float32)

    if mode == 'distance':
        mask = np.ones((h, w), dtype=np.uint8)
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0
        weights = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        max_weight = float(weights.max())
        return weights / max_weight if max_weight > 0 else np.ones((h, w), dtype=np.float32)

    if mode == 'feather':
        y = np.minimum(np.arange(h) + 1, np.arange(h, 0, -1)).astype(np.float32)
        x = np.minimum(np.arange(w) + 1, np.arange(w, 0, -1)).astype(np.float32)
        weights = np.minimum.outer(y, x)
        weights /= max(float(weights.max()), 1.0)
        return weights.astype(np.float32)

    raise ValueError(f'Unsupported blend mode: {mode}')


def blend_tiles(
    tiles: dict[str, np.ndarray],
    placements: pd.DataFrame,
    *,
    mode: str = 'feather',
) -> tuple[np.ndarray, pd.DataFrame]:
    """Blend tiles into one mosaic without overwriting overlap pixels."""
    if not tiles:
        raise ValueError('No tiles supplied')

    dtype = next(iter(tiles.values())).dtype
    positions = placements.copy()
    positions['x_int'] = np.rint(positions['x_px']).astype(int)
    positions['y_int'] = np.rint(positions['y_px']).astype(int)
    min_x = int(positions['x_int'].min())
    min_y = int(positions['y_int'].min())
    positions['x_int'] -= min_x
    positions['y_int'] -= min_y

    max_x = 0
    max_y = 0
    sample = next(iter(tiles.values()))
    channels = sample.shape[2:] if sample.ndim > 2 else ()
    for row in positions.itertuples(index=False):
        tile = tiles[str(row.tile_id)]
        max_x = max(max_x, int(row.x_int) + tile.shape[1])
        max_y = max(max_y, int(row.y_int) + tile.shape[0])

    canvas_shape = (max_y, max_x, *channels)
    accumulator = np.zeros(canvas_shape, dtype=np.float64)
    weights = np.zeros((max_y, max_x), dtype=np.float64)

    for row in positions.itertuples(index=False):
        tile = tiles[str(row.tile_id)].astype(np.float64)
        y0 = int(row.y_int)
        x0 = int(row.x_int)
        y1 = y0 + tile.shape[0]
        x1 = x0 + tile.shape[1]
        tile_weight = _tile_weights(tile.shape, mode).astype(np.float64)
        if tile.ndim == 3:
            accumulator[y0:y1, x0:x1, :] += tile * tile_weight[:, :, None]
        else:
            accumulator[y0:y1, x0:x1] += tile * tile_weight
        weights[y0:y1, x0:x1] += tile_weight

    safe_weights = np.maximum(weights, 1e-12)
    if accumulator.ndim == 3:
        blended = accumulator / safe_weights[:, :, None]
    else:
        blended = accumulator / safe_weights
    blended[weights == 0] = 0
    return cast_preserving_dtype(blended, dtype), positions

