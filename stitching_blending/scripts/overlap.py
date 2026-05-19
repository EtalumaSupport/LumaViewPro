"""Neighbor inference and expected-overlap extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeighborPair:
    source_tile_id: str
    target_tile_id: str
    direction: str
    expected_dx_px: float
    expected_dy_px: float


def infer_neighbors(metadata: pd.DataFrame) -> list[NeighborPair]:
    """Infer right/down neighbor pairs from row/col or coordinate ordering."""
    pairs: list[NeighborPair] = []
    if {'row', 'col'} <= set(metadata.columns):
        by_grid = {(int(row.row), int(row.col)): row for row in metadata.itertuples(index=False)}
        for (row_idx, col_idx), row in by_grid.items():
            for direction, key in [('right', (row_idx, col_idx + 1)), ('down', (row_idx + 1, col_idx))]:
                target = by_grid.get(key)
                if target is None:
                    continue
                pairs.append(
                    NeighborPair(
                        source_tile_id=str(row.tile_id),
                        target_tile_id=str(target.tile_id),
                        direction=direction,
                        expected_dx_px=float(target.metadata_x_px - row.metadata_x_px),
                        expected_dy_px=float(target.metadata_y_px - row.metadata_y_px),
                    )
                )
        return pairs

    # Fallback: use nearest larger-x and larger-y coordinates as adjacency.
    for row in metadata.itertuples(index=False):
        others = metadata[metadata['tile_id'] != row.tile_id].copy()
        right = others[others['metadata_x_px'] > row.metadata_x_px]
        if not right.empty:
            right = right.assign(
                distance=(right['metadata_x_px'] - row.metadata_x_px).abs()
                + (right['metadata_y_px'] - row.metadata_y_px).abs()
            ).sort_values('distance')
            target = right.iloc[0]
            pairs.append(
                NeighborPair(
                    str(row.tile_id),
                    str(target.tile_id),
                    'right',
                    float(target.metadata_x_px - row.metadata_x_px),
                    float(target.metadata_y_px - row.metadata_y_px),
                )
            )
        down = others[others['metadata_y_px'] > row.metadata_y_px]
        if not down.empty:
            down = down.assign(
                distance=(down['metadata_x_px'] - row.metadata_x_px).abs()
                + (down['metadata_y_px'] - row.metadata_y_px).abs()
            ).sort_values('distance')
            target = down.iloc[0]
            pairs.append(
                NeighborPair(
                    str(row.tile_id),
                    str(target.tile_id),
                    'down',
                    float(target.metadata_x_px - row.metadata_x_px),
                    float(target.metadata_y_px - row.metadata_y_px),
                )
            )
    return pairs


def expected_overlap_slices(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    pair: NeighborPair,
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    """Return source/target slices for the expected overlap strips.

    Overlap width is the tile dimension minus the expected center-to-center
    displacement. No-overlap neighbors return ``None`` and are skipped by image
    alignment.
    """
    src_h, src_w = source_shape[:2]
    tgt_h, tgt_w = target_shape[:2]
    if pair.direction == 'right':
        dx = int(round(abs(pair.expected_dx_px)))
        overlap = min(src_w, tgt_w, src_w - dx)
        if overlap <= 0:
            return None
        height = min(src_h, tgt_h)
        return (slice(0, height), slice(src_w - overlap, src_w)), (slice(0, height), slice(0, overlap))
    if pair.direction == 'down':
        dy = int(round(abs(pair.expected_dy_px)))
        overlap = min(src_h, tgt_h, src_h - dy)
        if overlap <= 0:
            return None
        width = min(src_w, tgt_w)
        return (slice(src_h - overlap, src_h), slice(0, width)), (slice(0, overlap), slice(0, width))
    raise ValueError(f'Unsupported neighbor direction: {pair.direction}')


def extract_overlap_strips(
    source: np.ndarray,
    target: np.ndarray,
    pair: NeighborPair,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the expected overlap image strips for one neighbor pair."""
    slices = expected_overlap_slices(source.shape, target.shape, pair)
    if slices is None:
        return None
    source_slice, target_slice = slices
    return source[source_slice], target[target_slice]

