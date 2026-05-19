"""CLI for stitching an existing tile metadata CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .pipeline import run_stitching_pipeline
except ImportError:  # pragma: no cover
    from pipeline import run_stitching_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description='Run standalone stitching on a metadata CSV.')
    parser.add_argument('metadata_csv', type=Path)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--run-name', default='stitching_run')
    parser.add_argument('--blend-mode', choices=['average', 'feather', 'distance'], default='feather')
    parser.add_argument('--max-shift-px', type=int, default=10)
    parser.add_argument('--phase-confidence-threshold', type=float, default=0.08)
    parser.add_argument('--ground-truth-path', type=Path, default=None)
    args = parser.parse_args()

    result = run_stitching_pipeline(
        args.metadata_csv,
        output_dir=args.output_dir,
        run_name=args.run_name,
        blend_mode=args.blend_mode,
        max_shift_px=args.max_shift_px,
        phase_confidence_threshold=args.phase_confidence_threshold,
        ground_truth_path=args.ground_truth_path,
    )
    print(f'Output directory: {result["output_dir"]}')
    print(f'Metrics: {result["metrics_path"]}')


if __name__ == '__main__':
    main()

