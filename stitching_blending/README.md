# Standalone Stitching + Blending Prototype

This directory contains an isolated prototype for microscopy tile stitching and seam blending. It does not import or modify the LumaViewPro GUI, plugin system, microscope hardware code, or production stitcher.

## Layout

```text
stitching_blending/
  scripts/              # all prototype Python code
  data/synthetic/       # generated synthetic datasets
  data/public/          # reserved for future public datasets
  outputs/              # stitched results and metrics
  tests/                # prototype tests only
```

## Run The Full Synthetic Demo

```bash
python stitching_blending/scripts/run_full_demo.py
```

Optional example:

```bash
python stitching_blending/scripts/run_full_demo.py --run-name demo_2x2 --rows 2 --cols 2 --tile-size 128 --overlap 32 --blend-mode feather
```

Outputs are saved in:

```text
stitching_blending/outputs/<run_name>/
```

The generated synthetic tiles are saved in:

```text
stitching_blending/data/synthetic/<run_name>/
```

## Run Stitching On An Existing Metadata CSV

```bash
python stitching_blending/scripts/run_stitching.py path/to/tile_metadata.csv --output-dir stitching_blending/outputs/my_run
```

Metadata file paths may be absolute or relative to the metadata CSV.

## BigStitcher Grid 2D Dataset

Download and prepare the public BigStitcher 2D multi-tile sample:

```bash
python stitching_blending/scripts/download_bigstitcher_grid2d.py
```

This downloads `Grid_2d.zip`, extracts the six raw TIFF tiles into:

```text
stitching_blending/data/public/bigstitcher_grid2d/
```

and writes:

```text
stitching_blending/data/public/bigstitcher_grid2d/metadata.csv
```

The metadata adapter maps `MAX_73.tif` through `MAX_78.tif` row-major into a 3 row x 2 column grid. BigStitcher describes the import as a 2-by-3 grid; the aligned reference XML has two regular-grid x positions and three y positions, so this prototype represents it as 3 rows x 2 columns. It uses the documented 10% expected overlap to compute nominal pixel positions. The raw TIFFs are planar multi-channel images, so the prototype I/O converts them from `C,H,W` to `H,W,C` while preserving dtype.

Run stitching:

```bash
python stitching_blending/scripts/run_bigstitcher_grid2d.py
```

Outputs are saved in:

```text
stitching_blending/outputs/bigstitcher_grid2d/
```

This runner reuses the standalone pipeline: phase correlation is attempted first, NCC fallback is used when phase confidence is low, accepted pairwise links are globally optimized, and the final mosaic is blended with feather blending by default. BigStitcher's reference workflow for this dataset used phase correlation, ignored links with correlation below `0.7`, and then ran global optimization; this prototype mirrors the same broad workflow but keeps its own confidence scale and thresholds.

## Run Tests

```bash
pytest stitching_blending/tests
```

## Output Files

Each run writes:

- `stitched.tif`: dtype-preserving stitched mosaic.
- `preview.png`: contrast-scaled preview.
- `tile_metadata.csv`: metadata used by the run.
- `pairwise_alignments.csv`: neighbor shift estimates and confidence scores.
- `tile_placements.csv`: final tile positions.
- `metrics.json`: quality, registration, seam, runtime, and memory metrics.
- `manifest.json`: run settings and output artifact names.

## Metrics

- `mse`: mean squared error versus the synthetic ground truth over the common image region. Lower is better.
- `psnr_db`: peak signal-to-noise ratio versus ground truth. Higher is better.
- `ssim`: structural similarity when `scikit-image` is available. Higher is better.
- `registration_rmse_px`: root mean squared placement error versus true synthetic tile positions. Lower is better.
- `seam_energy`: approximate image-gradient energy across the stitched mosaic. Lower usually indicates smoother seams.
- `overlap_consistency_error`: mean absolute difference between overlapping tile pixels after placement. Lower is better.
- `runtime_seconds`: end-to-end pipeline runtime.
- `approx_memory_bytes`: approximate memory held by loaded tiles plus stitched image.
- `alignment_acceptance_rate`: fraction of neighbor edges accepted by confidence and max-shift checks.
- `placement_mode`: `optimized` when pairwise alignments were used, or `metadata_fallback` when placement relied on metadata.
