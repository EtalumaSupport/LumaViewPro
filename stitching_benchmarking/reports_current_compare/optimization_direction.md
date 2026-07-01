# Stitching Optimization Direction

Rows benchmarked: 2112
Fallback rows: 0

## Baseline Interpretation

`lvp_current_overlap` is the reference method. Experimental methods are useful
only if they are faster than this baseline while staying close on SSIM/NRMSE and
tile-position error, and while avoiding fallbacks.

## Native Method Summary

| method | runs | median_total_ms | median_peak_rss_mb | median_ssim | median_nrmse | median_speedup_vs_lvp | median_est_96well_10x10_min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lvp_simple_grid | 528 | 131.1 | 1219 | 0.8528 | 0.4097 | 10.84 | 0.8519 |
| fft_phase_correlation | 528 | 145.4 | 1221 | 0.8772 | 0.3506 | 9.902 | 0.9411 |
| coarse_to_fine_ncc | 528 | 513.4 | 1221 | 0.8676 | 0.4296 | 2.983 | 3.292 |
| lvp_current_overlap | 528 | 1508 | 1223 | 0.8884 | 0.3396 | 1 | 9.658 |

## Recommended Direction

1. Use the current LVP overlap stitcher as the quality baseline.
2. Prefer FFT phase correlation if it consistently shows speedup with low tile
   error on `realistic` and `stress` tiers.
3. Prefer coarse-to-fine NCC if it preserves LVP-like registration quality but
   beats current local NCC on larger `5x5` and `7x7` grids.
4. Parallelize neighbor-pair registration only after the report confirms
   registration dominates total time.
5. Treat GPU acceleration as later-stage research for this MacBook; the first
   practical target is CPU/OpenCV/vectorized registration plus lower allocation
   pressure.

## Citations Used For Direction

- Fiji Grid/Collection Stitching: https://imagej.net/plugins/grid-collection-stitching
- BigStitcher: https://imagej.net/plugins/bigstitcher
- MIST: https://www.nature.com/articles/s41598-017-04567-y
- OpenCV phase correlation: https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html
- scikit-image metrics: https://scikit-image.org/docs/stable/api/skimage.metrics.html
