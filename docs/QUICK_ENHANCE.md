# Quick Enhance

Quick Enhance creates a derived TIFF for visual inspection. It never changes the source image. Do not use derived files for quantitative analysis.

## Use it

1. Choose Image to inspect and save one image, or Choose Folder to save every supported image in that folder.
2. Select a mode.
3. Save Image or Save Folder. Use Show Output Folder when complete.

Choose Image enables Before / After and Update Preview. A folder is a batch operation, so it has no single-image preview.

## Modes

- **Auto (Recommended)** — Detects each image independently. BF, phase, and darkfield use the gentle brightfield adjustment. Composite, fluorescence, and unknown images use neutral automatic contrast.
- **Brightfield / Phase** — Forces the gentle brightfield adjustment (automatic contrast plus light brightening). Use when Auto cannot identify a BF, phase, or darkfield file.
- **Contrast Only** — Applies automatic contrast only. Use when you do not want the brightfield adjustment.

## Buttons

- **Before / After** switches the single-image preview between the source and its enhanced version.
- **Update Preview** redraws the single-image preview using the selected mode.
- **Show Output Folder** opens the folder containing the most recently saved derived TIFFs.
