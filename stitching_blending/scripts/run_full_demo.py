"""Generate synthetic data and run the full stitching demo."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .paths import OUTPUTS_DIR, SYNTHETIC_DATA_DIR, ensure_project_dirs
    from .pipeline import run_stitching_pipeline
    from .synthetic import SyntheticConfig, generate_synthetic_dataset
except ImportError:  # pragma: no cover
    from paths import OUTPUTS_DIR, SYNTHETIC_DATA_DIR, ensure_project_dirs
    from pipeline import run_stitching_pipeline
    from synthetic import SyntheticConfig, generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the full synthetic stitching/blending demo.')
    parser.add_argument('--run-name', default='synthetic_demo')
    parser.add_argument('--rows', type=int, default=3)
    parser.add_argument('--cols', type=int, default=3)
    parser.add_argument('--tile-size', type=int, default=160)
    parser.add_argument('--overlap', type=int, default=40)
    parser.add_argument('--blend-mode', choices=['average', 'feather', 'distance'], default='feather')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    ensure_project_dirs()
    dataset_dir = SYNTHETIC_DATA_DIR / args.run_name
    output_dir = OUTPUTS_DIR / args.run_name
    config = SyntheticConfig(
        grid_shape=(args.rows, args.cols),
        tile_shape=(args.tile_size, args.tile_size),
        overlap_px=args.overlap,
        seed=args.seed,
    )
    synthetic = generate_synthetic_dataset(dataset_dir, config)
    result = run_stitching_pipeline(
        Path(synthetic['metadata_path']),
        output_dir=output_dir,
        run_name=args.run_name,
        blend_mode=args.blend_mode,
        max_shift_px=max(10, config.max_translation_error_px * 3),
        ground_truth_path=Path(synthetic['ground_truth_path']),
    )
    print(f'Synthetic data: {dataset_dir}')
    print(f'Output directory: {result["output_dir"]}')
    print(f'Stitched image: {result["stitched_path"]}')
    print(f'Metrics: {result["metrics_path"]}')


if __name__ == '__main__':
    main()

