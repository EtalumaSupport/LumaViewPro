"""Global tile-position optimization from pairwise shift constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg


def optimize_tile_positions(
    metadata_with_positions: pd.DataFrame,
    alignments: pd.DataFrame,
    *,
    min_confidence: float = 0.08,
    min_edges: int = 1,
) -> pd.DataFrame:
    """Solve globally consistent tile positions with least squares.

    Each accepted edge contributes equations of the form:
    ``x_target - x_source = measured_dx`` and
    ``y_target - y_source = measured_dy``.
    A high-weight anchor fixes the first tile to its metadata position, making
    the otherwise translation-invariant graph identifiable.
    """
    metadata = metadata_with_positions.copy()
    tile_ids = [str(tile_id) for tile_id in metadata['tile_id']]
    index = {tile_id: i for i, tile_id in enumerate(tile_ids)}

    accepted = alignments.copy()
    if not accepted.empty:
        accepted = accepted[(accepted['accepted']) & (accepted['confidence'] >= min_confidence)]

    if accepted.empty or len(accepted) < min_edges:
        return _metadata_fallback(metadata, 'metadata_fallback')

    def solve_axis(axis: str) -> np.ndarray:
        rows = []
        rhs = []
        weights = []
        for edge in accepted.itertuples(index=False):
            row = np.zeros(len(tile_ids), dtype=np.float64)
            row[index[str(edge.target_tile_id)]] = 1.0
            row[index[str(edge.source_tile_id)]] = -1.0
            rows.append(row)
            rhs.append(float(getattr(edge, f'd{axis}_px')))
            weights.append(max(float(edge.confidence), min_confidence))

        anchor = np.zeros(len(tile_ids), dtype=np.float64)
        anchor[0] = 1.0
        rows.append(anchor)
        rhs.append(float(metadata.iloc[0][f'metadata_{axis}_px']))
        weights.append(100.0)

        matrix = np.vstack(rows)
        weight = np.sqrt(np.asarray(weights))[:, None]
        weighted_matrix = sparse.csr_matrix(matrix * weight)
        weighted_rhs = np.asarray(rhs) * weight[:, 0]
        result = linalg.lsqr(weighted_matrix, weighted_rhs, atol=1e-8, btol=1e-8)
        return result[0]

    x = solve_axis('x')
    y = solve_axis('y')
    placements = metadata[['tile_id', 'metadata_x_px', 'metadata_y_px']].copy()
    placements['x_px'] = x
    placements['y_px'] = y
    placements['placement_mode'] = 'optimized'
    return placements


def _metadata_fallback(metadata: pd.DataFrame, mode: str) -> pd.DataFrame:
    placements = metadata[['tile_id', 'metadata_x_px', 'metadata_y_px']].copy()
    placements['x_px'] = placements['metadata_x_px']
    placements['y_px'] = placements['metadata_y_px']
    placements['placement_mode'] = mode
    return placements


def placement_rmse(placements: pd.DataFrame, metadata: pd.DataFrame) -> float | None:
    """Return RMSE against true synthetic positions when available."""
    if not {'true_x_px', 'true_y_px'} <= set(metadata.columns):
        return None
    merged = placements.merge(metadata[['tile_id', 'true_x_px', 'true_y_px']], on='tile_id')
    err = (merged['x_px'] - merged['true_x_px']) ** 2 + (merged['y_px'] - merged['true_y_px']) ** 2
    return float(np.sqrt(np.mean(err)))

