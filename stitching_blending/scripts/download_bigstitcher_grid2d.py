"""Download and describe the BigStitcher 2D grid example dataset."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import tifffile

try:
    from .paths import PUBLIC_DATA_DIR, ensure_project_dirs
except ImportError:  # pragma: no cover
    from paths import PUBLIC_DATA_DIR, ensure_project_dirs


GRID2D_URL = 'https://preibischlab.mdc-berlin.de/BigStitcher/Grid_2d.zip'
DATASET_DIR = PUBLIC_DATA_DIR / 'bigstitcher_grid2d'
ZIP_PATH = DATASET_DIR / 'Grid_2d.zip'
METADATA_PATH = DATASET_DIR / 'metadata.csv'
# BigStitcher describes the import as a "2-by-3 grid"; the aligned reference
# XML has two regular-grid x positions and three y positions, so the pipeline
# row/col representation is 3 rows x 2 columns.
EXPECTED_GRID_SHAPE = (3, 2)
EXPECTED_OVERLAP_FRACTION = 0.10


def download_bigstitcher_grid2d(*, force: bool = False) -> Path:
    """Download and extract Grid_2d.zip, skipping existing data unless forced."""
    ensure_project_dirs()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    image_files = _tile_image_paths(DATASET_DIR)
    if image_files and ZIP_PATH.exists() and not force:
        return DATASET_DIR

    if force and DATASET_DIR.exists():
        for child in DATASET_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists() or force:
        urllib.request.urlretrieve(GRID2D_URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATASET_DIR)
    return DATASET_DIR


def infer_bigstitcher_grid2d_metadata(dataset_dir: str | Path = DATASET_DIR) -> pd.DataFrame:
    """Infer pipeline-compatible metadata for the BigStitcher 2D grid dataset.

    BigStitcher documents this sample as a 2-by-3 grid with 10% overlap. The
    aligned reference XML resolves that as two columns by three rows. The raw
    archive contains six TIFFs sorted by acquisition number and they are mapped
    row-major: MAX_73..MAX_74 on row 0, MAX_75..MAX_76 on row 1, and
    MAX_77..MAX_78 on row 2.
    """
    dataset_dir = Path(dataset_dir)
    image_paths = _tile_image_paths(dataset_dir)
    expected_tiles = EXPECTED_GRID_SHAPE[0] * EXPECTED_GRID_SHAPE[1]
    if len(image_paths) != expected_tiles:
        raise ValueError(f'Expected {expected_tiles} TIFF tiles, found {len(image_paths)} in {dataset_dir}')

    first = tifffile.imread(image_paths[0])
    tile_h, tile_w = _spatial_shape(first)
    stride_x = tile_w * (1.0 - EXPECTED_OVERLAP_FRACTION)
    stride_y = tile_h * (1.0 - EXPECTED_OVERLAP_FRACTION)
    channel = _channel_label(first)

    records: list[dict[str, object]] = []
    for index, path in enumerate(image_paths):
        row = index // EXPECTED_GRID_SHAPE[1]
        col = index % EXPECTED_GRID_SHAPE[1]
        records.append(
            {
                'tile_id': path.stem,
                'filepath': path.relative_to(dataset_dir).as_posix(),
                'row': row,
                'col': col,
                'nominal_x_px': col * stride_x,
                'nominal_y_px': row * stride_y,
                'channel': channel,
                'expected_overlap_fraction': EXPECTED_OVERLAP_FRACTION,
            }
        )
    return pd.DataFrame.from_records(records)


def write_bigstitcher_metadata(dataset_dir: str | Path = DATASET_DIR) -> Path:
    """Write metadata.csv compatible with the existing stitching pipeline."""
    dataset_dir = Path(dataset_dir)
    metadata = infer_bigstitcher_grid2d_metadata(dataset_dir)
    output = dataset_dir / 'metadata.csv'
    metadata.to_csv(output, index=False)
    return output


def _tile_image_paths(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.glob('*.tif')
        if path.is_file() and not path.name.startswith('._')
    )


def _spatial_shape(image) -> tuple[int, int]:
    if image.ndim == 2:
        return int(image.shape[0]), int(image.shape[1])
    if image.ndim == 3 and image.shape[0] in (3, 4):
        return int(image.shape[1]), int(image.shape[2])
    return int(image.shape[0]), int(image.shape[1])


def _channel_label(image) -> str:
    if image.ndim == 3 and image.shape[0] in (3, 4):
        return f'planar_{image.shape[0]}ch'
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return f'interleaved_{image.shape[-1]}ch'
    return 'single'


def main() -> None:
    parser = argparse.ArgumentParser(description='Download BigStitcher Grid_2d and write metadata.csv.')
    parser.add_argument('--force', action='store_true', help='Redownload and re-extract the dataset.')
    args = parser.parse_args()

    dataset_dir = download_bigstitcher_grid2d(force=args.force)
    metadata_path = write_bigstitcher_metadata(dataset_dir)
    metadata = pd.read_csv(metadata_path)
    print(f'Dataset directory: {dataset_dir}')
    print(f'Metadata CSV: {metadata_path}')
    print(f'Tiles: {len(metadata)}')
    print(f'Grid: {EXPECTED_GRID_SHAPE[0]} rows x {EXPECTED_GRID_SHAPE[1]} columns')
    print(f'Expected overlap: {EXPECTED_OVERLAP_FRACTION:.0%}')


if __name__ == '__main__':
    main()
