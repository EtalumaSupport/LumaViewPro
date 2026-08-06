# Stitching

## Fast Preview Stitch

Fast Preview Stitch is for a quick visual check of tile placement and coverage. When the stage metadata indicates real overlap, it uses bounded FFT phase correlation to estimate the local offset. It is faster, but can be less reliable on weak, repetitive, or low-detail overlap.

## Quality Stitch

Quality Stitch is for the final derived mosaic. With the same real overlap, it uses the recorded stage geometry plus bounded local normalized cross-correlation (NCC), which searches the plausible local offset more carefully. It is slower and is expected to produce more reliable alignment when tile content makes registration difficult.

## Important behavior

Both modes first infer overlap from the recorded stage positions. If that inference is 0% overlap, neither mode attempts image registration: both place tiles from stage geometry and preserve source pixels and channel colors. Fast Preview therefore does know when real overlap exists; it uses the same stage-derived overlap decision as Quality, but applies the faster FFT registration route instead of the more careful NCC route.
