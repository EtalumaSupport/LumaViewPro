"""CLI for generating synthetic microscopy tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .paths import SYNTHETIC_DATA_DIR, ensure_project_dirs
    from .synthetic import SyntheticConfig, generate_synthetic_dataset
except ImportError:  # pragma: no cover
    from paths import SYNTHETIC_DATA_DIR, ensure_project_dirs
    from synthetic import SyntheticConfig, generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic microscopy tile data.')
    parser.add_argument('--output-dir', type=Path, default=SYNTHETIC_DATA_DIR / 'demo')
    parser.add_argument('--rows', type=int, default=3)
    parser.add_argument('--cols', type=int, default=3)
    parser.add_argument('--tile-size', type=int, default=160)
    parser.add_argument('--overlap', type=int, default=40)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    ensure_project_dirs()
    config = SyntheticConfig(
        grid_shape=(args.rows, args.cols),
        tile_shape=(args.tile_size, args.tile_size),
        overlap_px=args.overlap,
        seed=args.seed,
    )
    result = generate_synthetic_dataset(args.output_dir, config)
    print(f'Synthetic metadata: {result["metadata_path"]}')
    print(f'Ground truth: {result["ground_truth_path"]}')


if __name__ == '__main__':
    main()

