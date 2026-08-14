# Enhance clump splitting: restart note

**Status:** exploration only; do not merge into `4.0.0-beta`.

**Working branch:** `mariam-enhance-fix`

## Current state

The standard Quick Enhance work is implemented on this branch:

- preserves a single-channel fluorescence colour (the supplied test image remains green);
- applies a conservative, signal-gated one-pixel sharpen to the derived visual image;
- leaves the source untouched;
- hides the derived output path after completion.

The supplied external examples were `0green_s.tiff` (raw) and
`0green_s_enhanced.tif` (the prior derived output). The standard visual
Enhance result is suitable for inspection only, never quantitative analysis.

## Clump-splitting direction agreed so far

We want a second, derived visual result for touching beads:

1. **Main viewer / split visual:** green image with natural-looking, smooth
   dark valleys between accepted bead instances. No cyan markings in this
   image.
2. **Saved audit companion:** the same derived green image with cyan
   boundaries at the exact locations used for splitting.
3. **Source:** never overwritten or changed.

The first hard black-trench prototype was rejected because it looked drawn on.
The preferred rendering direction is a smooth, tapered valley with a deeper
center where a split is accepted. This alone is not sufficient: a full split
also requires correctly finding separate bead centers (seeds).

## What the next prototype must do

This is instance segmentation plus visual rendering, not stronger sharpening.

1. Detect candidate bead centers from more than one cue: intensity peaks,
   multi-scale blob shape, and distance-transform peaks.
2. Estimate typical bead size from isolated beads. Use it to flag unusually
   large or elongated merged clumps that may need multiple centers.
3. Use marker-controlled watershed to form complete boundaries between
   accepted centers.
4. Render each accepted internal boundary with a smooth, local-background
   valley: a deep enough center to separate the visual objects, with tapered
   shoulders rather than a binary black line.
5. Apply confidence checks. A low-evidence candidate must remain unsplit
   rather than silently inventing a bead.
6. Save recipe/provenance that records the split settings and output role.

Do not treat the split visual or cyan audit image as measurement input. A
separate label mask may be added later if counting/measurement is needed.

## Ask the mentor for this

Please send **5–10 original, unprocessed bead images**. For each image, a
small screenshot or note identifying **2–4 clumps** is enough; full masks are
not required.

For each marked clump, ask them to provide:

- `should split` or `should remain one object`;
- expected bead/object count, if known;
- confidence (`certain` / `uncertain` is enough) and any biological/optical
  reason a clump should not be split.

Ask for a deliberately varied set:

- typical isolated beads and normal touching pairs;
- shallow-valley touching pairs like `0green_s.tiff`;
- larger/denser clumps;
- dim, noisy, or slightly out-of-focus fields;
- negative controls where apparent internal texture must **not** become a
  split.

For every raw image, request or preserve the acquisition context if available:

- original TIFF (not a screenshot or previously enhanced export);
- channel/dye, objective/magnification, pixel size, camera/exposure/gain;
- bit depth and whether colour is a display false colour;
- whether all images share the same acquisition settings.

Suggested message:

> I’m validating a visual clump-separation feature for bead images. Could you
> send 5–10 original, unprocessed TIFFs spanning isolated beads, touching
> pairs, dense clumps, weak valleys, and a few cases that should not be split?
> For 2–4 clumps per image, a screenshot with `split` / `do not split` and the
> expected count if known is enough. If available, please include channel,
> objective/magnification, pixel size, exposure/gain, bit depth, and whether
> the supplied colour is false colour. The raw images will remain unchanged;
> this is to validate a derived visual output.

## Resume checklist

1. Put the raw mentor examples and their notes in an agreed non-repository
   location; do not commit image data unless explicitly authorized.
2. Create a compact manifest with source filename, marked region, expected
   result, and acquisition group.
3. Prototype the detector and renderer only on the manifest examples.
4. Review grids showing raw, normal Enhance, split visual, and cyan audit
   companion for every marked region.
5. Measure: expected splits found, false splits, expected no-splits preserved,
   and output determinism.
6. Implement only after the grid passes the agreed examples. Add focused tests
   and preserve the current non-destructive Quick Enhance contract.

## Evidence from this session

- Current branch commits for colour/sharpening/path hiding end at `fed2fb6b`.
- The raw test image has saturated regions, so a single local peak can cover
  more than one apparent bead. This is why one manually expected split was
  absent from the first watershed prototype.
- Local CLAHE and multi-scale contrast prototypes either failed to deepen the
  tested valley or amplified background texture; do not add them to standard
  Quick Enhance.
