# Quick Enhance

Quick Enhance creates a derived TIFF for visual inspection. It never changes the source image. Do not use derived files for quantitative analysis. For AI-assisted, validated quantitative enhancement workflows, use LumaQuant Pro.

## Use it

1. Choose Image to inspect and save one image, or Choose Folder to save every supported image in that folder.
2. Select Quick Enhance Image or Quick Enhance Folder. Use Show Output Folder when complete.

Choose Image enables Before / After and Update Preview. A folder is a batch operation, so it has no single-image preview.

## Fixed recipe

Quick Enhance applies the same non-AI recipe to every selected image:

1. **Global illumination correction** fits a field-scale brightness plane and divides by it multiplicatively. It does not follow or subtract individual bright structures, avoiding dark halos around them.
2. **Auto levels** maps the 1st–99th percentile into the available image range, followed by gentle midtone brightening.

This is classical presentation processing, not AI denoising or restoration. LumaQuant Pro remains the separate AI-assisted cleanup workflow.

## Buttons

- **Before / After** switches the single-image preview between the source and its enhanced version.
- **Update Preview** redraws the single-image preview using the selected mode.
- **Show Output Folder** opens the folder containing the most recently saved derived TIFFs.
