"""Metadata loading and coordinate placement utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_tile_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    """Load tile metadata and validate the minimum required fields."""
    metadata = pd.read_csv(metadata_csv)
    required = {'tile_id', 'filepath'}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f'Metadata missing required columns: {sorted(missing)}')
    return metadata


def resolve_tile_path(metadata_csv: str | Path, filepath: str | Path) -> Path:
    """Resolve a tile filepath relative to its metadata CSV."""
    path = Path(filepath)
    if path.is_absolute():
        return path
    return Path(metadata_csv).resolve().parent / path


def metadata_positions(
    metadata: pd.DataFrame,
    *,
    coordinate_mode: str = 'auto',
    pixel_size: float = 1.0,
) -> pd.DataFrame:
    """Convert metadata coordinates into pixel placements.

    Stage coordinates are treated as pixel-equivalent by default. Supplying
    ``pixel_size`` converts physical stage units to pixels.
    """
    frame = metadata.copy()
    if coordinate_mode == 'auto':
        if {'nominal_x_px', 'nominal_y_px'} <= set(frame.columns):
            coordinate_mode = 'pixel'
        elif {'stage_x', 'stage_y'} <= set(frame.columns):
            coordinate_mode = 'stage'
        else:
            raise ValueError('Metadata needs nominal_x_px/nominal_y_px or stage_x/stage_y')

    if coordinate_mode == 'pixel':
        frame['metadata_x_px'] = frame['nominal_x_px'].astype(float)
        frame['metadata_y_px'] = frame['nominal_y_px'].astype(float)
    elif coordinate_mode == 'stage':
        frame['metadata_x_px'] = frame['stage_x'].astype(float) / float(pixel_size)
        frame['metadata_y_px'] = frame['stage_y'].astype(float) / float(pixel_size)
    else:
        raise ValueError(f'Unsupported coordinate_mode: {coordinate_mode}')

    return frame


def infer_grid_shape(metadata: pd.DataFrame) -> tuple[int, int] | None:
    """Return grid shape from row/col metadata when available."""
    if {'row', 'col'} <= set(metadata.columns):
        return int(metadata['row'].max()) + 1, int(metadata['col'].max()) + 1
    return None


def normalize_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """Shift position columns so the minimum optimized position is zero."""
    frame = positions.copy()
    min_x = float(np.floor(frame['x_px'].min()))
    min_y = float(np.floor(frame['y_px'].min()))
    frame['x_px'] = frame['x_px'] - min_x
    frame['y_px'] = frame['y_px'] - min_y
    return frame

