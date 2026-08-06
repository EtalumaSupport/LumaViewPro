# Quick Enhance

Quick Enhance creates a derived TIFF for visual inspection. It never changes the source image. Do not use derived files for quantitative analysis. For AI-assisted, validated quantitative enhancement workflows, use LumaQuant Pro.

## Use it

1. Select **Enhance**.
2. In the native picker, choose either one supported image or a folder.
3. Enhancement starts immediately. Every output is saved as a derived TIFF, and the main viewer briefly shows each completed image. The panel reports `Image x of y` while a folder runs, then the saved output path.

## Fixed recipe

Quick Enhance applies the same non-AI recipe to every selected image:

1. **Global illumination correction** fits a field-scale brightness plane and divides by it multiplicatively. It does not follow or subtract individual bright structures, avoiding dark halos around them.
2. **Auto levels** maps the 1st–99th percentile into the available image range, followed by gentle midtone brightening.

This is classical presentation processing, not AI denoising or restoration. LumaQuant Pro remains the separate AI-assisted cleanup workflow.

## Progress

Folder processing runs one image at a time. Unsupported or unreadable images are skipped; source images and prior derived outputs are never overwritten.
