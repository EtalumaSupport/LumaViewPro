# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Shared composite image builder.

Merges multiple microscope channels (transmitted, fluorescence, luminescence)
into a single 3-channel RGB composite image. Used by both the live composite
capture path and the post-capture composite generation path.
"""

import numpy as np

import modules.image_utils as image_utils

# Canonical RGB color mapping -- single source of truth for channel-to-RGB index.
# Index 0 = Red, 1 = Green, 2 = Blue (standard RGB ordering).
# Callers using BGR (OpenCV) must convert at their boundaries.
CHANNEL_RGB_INDEX = {
    'Red': 0,
    'Green': 1,
    'Blue': 2,
    'Lumi': 2,  # Luminescence renders in the blue channel
}


def build_composite(
    channel_images: dict,
    significant_bits: int,
    transmitted_image: np.ndarray | None = None,
    brightness_thresholds: dict | None = None,
) -> np.ndarray:
    """Build an 8-bit-per-channel RGB composite from grayscale channel images.

    A composite is a viewing product: it merges several channels' intensities
    into distinct color planes for display, and once merged the per-channel
    values are no longer separable for quantitative use. Its output is therefore
    always 8-bit RGB -- the form every viewer and monitor renders directly.
    Storing a composite at the camera's native 12-bit depth (right-aligned in a
    16-bit container) leaves the brightest pixel at ~6% of full scale, so any
    viewer that ignores the depth tag shows it near-black. 8-bit sidesteps that
    entirely; the raw single-channel captures keep their full depth elsewhere.

    Downconversion is owned here, at the one place both the live-capture and the
    post-processing orchestrators funnel through, so neither can reintroduce a
    depth-carrying composite.

    Args:
        channel_images: Dict mapping channel name ('Red', 'Green', 'Blue', 'Lumi')
            to a 2D grayscale array at the capture's native depth.
        significant_bits: The meaningful bit depth of the input channels (12 for a
            Mono12 capture, 8 for 8-bit); drives the downconvert scale. Ignored for
            inputs already 8-bit.
        transmitted_image: Optional 2D grayscale array for transmitted channel
            (BF/PC/DF), at the same native depth. Used as the base with fluorescence
            overlaid.
        brightness_thresholds: Dict mapping channel name to threshold value on the
            OUTPUT 8-bit scale (absolute, not percentage). Pixels below threshold are
            not composited onto the transmitted image. Only used with transmitted_image.

    Returns:
        3-channel uint8 RGB array of shape (H, W, 3).
    """
    if brightness_thresholds is None:
        brightness_thresholds = {}

    # Every channel (and the transmitted base) must share one canvas before they
    # can be blended into RGB; a per-layer stitch divergence otherwise surfaces
    # as a cryptic numpy broadcast error mid-blend.
    labeled = list(channel_images.items())
    if transmitted_image is not None:
        labeled.append(('transmitted', transmitted_image))
    image_utils.require_uniform_geometry(labeled, operation='composite this tile-group')

    # Downconvert every input to the 8-bit output scale up front, so the blend
    # below is a single dtype and the thresholds compare against 8-bit values.
    channel_images = {
        name: image_utils.convert_to_8bit(img, significant_bits)
        for name, img in channel_images.items()
    }
    if transmitted_image is not None:
        transmitted_image = image_utils.convert_to_8bit(transmitted_image, significant_bits)

    dtype = np.uint8

    # Determine image dimensions from first available image
    if transmitted_image is not None:
        h, w = transmitted_image.shape[:2]
    else:
        first_img = next(iter(channel_images.values()))
        h, w = first_img.shape[:2]

    if transmitted_image is not None:
        # Start with transmitted channel replicated across all 3 RGB channels
        img = np.repeat(transmitted_image[:, :, None].astype(dtype), 3, axis=2)
        mask_changed = np.zeros((h, w), dtype=bool)

        for channel_name, img_gray in channel_images.items():
            channel_index = CHANNEL_RGB_INDEX.get(channel_name)
            if channel_index is None:
                continue

            threshold = brightness_thresholds.get(channel_name, 0)
            above_threshold = img_gray > threshold

            # Pixels above threshold that haven't been modified yet:
            # clear all RGB channels, then set the target channel
            not_changed = above_threshold & (~mask_changed)
            # Pixels above threshold that have already been modified:
            # only update the target channel (additive RGB blending)
            changed = above_threshold & mask_changed

            img[not_changed, 0] = 0
            img[not_changed, 1] = 0
            img[not_changed, 2] = 0
            img[not_changed, channel_index] = img_gray[not_changed]
            mask_changed[not_changed] = True

            img[changed, channel_index] = img_gray[changed]
    else:
        # No transmitted channel -- assign each channel directly
        img = np.zeros((h, w, 3), dtype=dtype)
        for channel_name, img_gray in channel_images.items():
            channel_index = CHANNEL_RGB_INDEX.get(channel_name)
            if channel_index is None:
                continue
            img[:, :, channel_index] = img_gray

    return img
