# Stitching Benchmarking

This folder contains a self-contained offline benchmark for synthetic tiling
datasets generated from `LS800imageAtlas`. The original atlas is treated as
read-only input. Generated tiles, stitched outputs, and reports are written next
to it under `generated_tiles/`, `results/`, and `reports/`.

## Quick Dry Run

```bash
.venv/bin/python stitching_benchmarking/benchmark_tools/stitch_benchmark.py pipeline \
  --source-limit 1 \
  --grids 3 \
  --tiers clean \
  --methods lvp_current_overlap lvp_simple_grid fft_phase_correlation coarse_to_fine_ncc opencv_feature \
  --run-limit 1
```

The visual grid report will be written under:

```text
stitching_benchmarking/reports/grids/
```

The aggregate outputs are:

```text
stitching_benchmarking/reports/benchmark_summary.csv
stitching_benchmarking/reports/benchmark_summary.html
```

## Full Benchmark

```bash
.venv/bin/python stitching_benchmarking/benchmark_tools/stitch_benchmark.py pipeline
```

This generates `3x3`, `5x5`, and `7x7` grids at 10% overlap for all 44 source
images and all perturbation tiers: `clean`, `realistic`, `stress`, and
`failure`.

## 96-Well 10x10 Timing Check

The default full benchmark measures `3x3`, `5x5`, and `7x7` grids. To directly
stress the use case of 100 tiles per well, run a focused `10x10` benchmark:

```bash
.venv/bin/python stitching_benchmarking/benchmark_tools/stitch_benchmark.py pipeline \
  --grids 10 \
  --tiers clean realistic stress \
  --methods lvp_current_overlap lvp_simple_grid fft_phase_correlation coarse_to_fine_ncc
```

The CSV report includes:

```text
total_ms
read_ms
registration_ms / registration_blend_ms
blend_ms
write_ms
peak_rss_mb
total_ms_per_tile
estimated_96well_10x10_total_min
speedup_vs_lvp_current_overlap
```

`estimated_96well_10x10_total_min` is a plate-scale estimate from measured
per-tile runtime: `runtime_per_tile * 96 wells * 100 tiles`. Treat it as a
planning estimate; the direct `10x10` rows are the best basis for choosing the
fastest method.

## Methods

- `lvp_current_overlap`: current LVP overlap registration plus average blend.
- `lvp_simple_grid`: existing simple grid placement fallback baseline.
- `fft_phase_correlation`: experimental OpenCV phase-correlation registration.
- `coarse_to_fine_ncc`: experimental downsampled NCC search plus local refine.
- `opencv_feature`: optional OpenCV feature stitch comparison.

Experimental methods automatically fall back to `lvp_current_overlap`, then to
`lvp_simple_grid`. Fallbacks are marked in CSV/HTML and printed on visual grid
panels as `FALLBACK: source -> fallback`.

## Metrics

Each benchmark row records method status, fallback information, timing, peak RSS,
peak `tracemalloc`, SSIM, PSNR, NRMSE, NMI, seam-band MAE, registration score
summary, and tile-position error against known synthetic offsets.

## Research References

- Fiji Grid/Collection Stitching: overlapping tile positions, blending,
  downsampling, and virtual memory patterns.
  https://imagej.net/plugins/grid-collection-stitching
- BigStitcher: pairwise phase correlation, overlap thresholds, downsampling, and
  global optimization.
  https://imagej.net/plugins/bigstitcher
- MIST: microscopy stitching with stage models, overlap ranges, multicore
  execution, and GPU support.
  https://www.nature.com/articles/s41598-017-04567-y
- OpenCV phase correlation: Fourier-domain translational shift estimation.
  https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html
- scikit-image metrics: SSIM, PSNR, NRMSE, and NMI.
  https://scikit-image.org/docs/stable/api/skimage.metrics.html
