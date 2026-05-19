"""Synthetic microscopy-like tile generation.

The generator creates a known ground-truth image, then samples overlapping
tiles from it. The metadata carries both nominal stage placement and true
pixel placement, so downstream registration error can be measured directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


@dataclass(frozen=True)
class SyntheticConfig:
    grid_shape: tuple[int, int] = (3, 3)
    tile_shape: tuple[int, int] = (160, 160)
    overlap_px: int | tuple[int, int] = 40
    noise_sigma: float = 2.0
    brightness_variation: float = 0.12
    max_translation_error_px: int = 4
    dtype: str = 'uint16'
    seed: int = 7


def _as_overlap_pair(overlap_px: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(overlap_px, tuple):
        return int(overlap_px[0]), int(overlap_px[1])
    return int(overlap_px), int(overlap_px)


def generate_ground_truth(
    shape: tuple[int, int],
    *,
    dtype: str | np.dtype = 'uint16',
    seed: int = 7,
) -> np.ndarray:
    """Generate a deterministic microscopy-like grayscale image.

    The image mixes smooth illumination, random texture, and blurred circular
    objects. The precise biology is not modeled; the goal is repeatable image
    structure with enough texture for registration.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]

    gradient = 0.15 + 0.25 * (xx / max(width - 1, 1)) + 0.15 * (yy / max(height - 1, 1))
    texture = ndimage.gaussian_filter(rng.normal(0, 1, shape), sigma=2.0)
    texture = (texture - texture.min()) / max(texture.max() - texture.min(), 1e-6)
    image = gradient + 0.18 * texture

    for _ in range(max(40, (height * width) // 3500)):
        cy = rng.uniform(0, height)
        cx = rng.uniform(0, width)
        radius = rng.uniform(5, 18)
        amplitude = rng.uniform(0.35, 0.95)
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * radius**2)))
        image += amplitude * blob

    image = ndimage.gaussian_filter(image, sigma=0.8)
    image -= image.min()
    image /= max(image.max(), 1e-6)

    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(image * info.max).astype(dtype)
    return image.astype(dtype)


def generate_synthetic_dataset(
    output_dir: str | Path,
    config: SyntheticConfig | None = None,
) -> dict[str, Path | pd.DataFrame]:
    """Generate tiles, metadata CSV, and the ground-truth image."""
    config = config or SyntheticConfig()
    output_dir = Path(output_dir)
    tiles_dir = output_dir / 'tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = config.grid_shape
    tile_h, tile_w = config.tile_shape
    overlap_x, overlap_y = _as_overlap_pair(config.overlap_px)
    stride_x = tile_w - overlap_x
    stride_y = tile_h - overlap_y
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError('overlap must be smaller than tile dimensions')

    margin = max(int(config.max_translation_error_px) + 8, 8)
    gt_h = margin * 2 + tile_h + stride_y * (rows - 1)
    gt_w = margin * 2 + tile_w + stride_x * (cols - 1)
    ground_truth = generate_ground_truth((gt_h, gt_w), dtype=config.dtype, seed=config.seed)
    ground_truth_path = output_dir / 'ground_truth.tif'
    tifffile.imwrite(ground_truth_path, ground_truth)

    rng = np.random.default_rng(config.seed + 1)
    records: list[dict[str, object]] = []
    dtype = np.dtype(config.dtype)
    max_value = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0

    for row in range(rows):
        for col in range(cols):
            nominal_x = margin + col * stride_x
            nominal_y = margin + row * stride_y
            err_x = int(rng.integers(-config.max_translation_error_px, config.max_translation_error_px + 1))
            err_y = int(rng.integers(-config.max_translation_error_px, config.max_translation_error_px + 1))
            true_x = nominal_x + err_x
            true_y = nominal_y + err_y
            brightness = float(rng.uniform(1.0 - config.brightness_variation, 1.0 + config.brightness_variation))

            tile = ground_truth[true_y : true_y + tile_h, true_x : true_x + tile_w].astype(np.float32)
            tile *= brightness
            if config.noise_sigma > 0:
                tile += rng.normal(0.0, config.noise_sigma, tile.shape)
            tile = np.clip(np.rint(tile), 0, max_value).astype(dtype)

            tile_id = f'r{row:02d}_c{col:02d}'
            tile_rel = Path('tiles') / f'{tile_id}.tif'
            tifffile.imwrite(output_dir / tile_rel, tile)
            records.append(
                {
                    'tile_id': tile_id,
                    'row': row,
                    'col': col,
                    'filepath': tile_rel.as_posix(),
                    'nominal_x_px': nominal_x,
                    'nominal_y_px': nominal_y,
                    'stage_x': float(nominal_x),
                    'stage_y': float(nominal_y),
                    'true_x_px': true_x,
                    'true_y_px': true_y,
                    'translation_error_x_px': err_x,
                    'translation_error_y_px': err_y,
                    'brightness_scale': brightness,
                }
            )

    metadata = pd.DataFrame.from_records(records)
    metadata_path = output_dir / 'tile_metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    return {
        'output_dir': output_dir,
        'tiles_dir': tiles_dir,
        'ground_truth_path': ground_truth_path,
        'metadata_path': metadata_path,
        'metadata': metadata,
    }

