"""Run the standalone stitching pipeline on BigStitcher Grid_2d."""

from __future__ import annotations

import argparse

try:
    from .download_bigstitcher_grid2d import (
        DATASET_DIR,
        METADATA_PATH,
        download_bigstitcher_grid2d,
        write_bigstitcher_metadata,
    )
    from .paths import OUTPUTS_DIR
    from .pipeline import run_stitching_pipeline
except ImportError:  # pragma: no cover
    from download_bigstitcher_grid2d import (
        DATASET_DIR,
        METADATA_PATH,
        download_bigstitcher_grid2d,
        write_bigstitcher_metadata,
    )
    from paths import OUTPUTS_DIR
    from pipeline import run_stitching_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description='Stitch the BigStitcher Grid_2d sample dataset.')
    parser.add_argument('--download', action='store_true', help='Download/extract the dataset before stitching.')
    parser.add_argument('--force-download', action='store_true', help='Redownload even if files already exist.')
    parser.add_argument('--blend-mode', choices=['feather', 'distance', 'average'], default='feather')
    parser.add_argument('--max-shift-px', type=int, default=80)
    parser.add_argument('--phase-confidence-threshold', type=float, default=0.08)
    args = parser.parse_args()

    if args.download or args.force_download or not METADATA_PATH.exists():
        download_bigstitcher_grid2d(force=args.force_download)
        write_bigstitcher_metadata(DATASET_DIR)

    result = run_stitching_pipeline(
        METADATA_PATH,
        output_dir=OUTPUTS_DIR / 'bigstitcher_grid2d',
        run_name='bigstitcher_grid2d',
        blend_mode=args.blend_mode,
        max_shift_px=args.max_shift_px,
        phase_confidence_threshold=args.phase_confidence_threshold,
    )
    print(f'Output directory: {result["output_dir"]}')
    print(f'Stitched image: {result["stitched_path"]}')
    print(f'Metrics: {result["metrics_path"]}')


if __name__ == '__main__':
    main()
