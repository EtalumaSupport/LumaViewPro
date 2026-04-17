# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Shared composite image builder.

Merges multiple microscope channels (transmitted, fluorescence, luminescence)
into a single 3-channel RGB composite image. Used by both the live composite
capture path and the post-capture composite generation path.
"""

import numpy as np

# Canonical RGB color mapping — single source of truth for channel-to-RGB index.
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
    transmitted_image: np.ndarray = None,
    brightness_thresholds: dict = None,
    dtype: np.dtype = np.uint8,
    max_value: float = 255,
) -> np.ndarray:
    """Build a composite RGB image from individual channel grayscale images.

    Args:
        channel_images: Dict mapping channel name ('Red', 'Green', 'Blue', 'Lumi')
            to a 2D grayscale numpy array.
        transmitted_image: Optional 2D grayscale array for transmitted channel
            (BF/PC/DF). Used as the base image with fluorescence overlaid.
        brightness_thresholds: Dict mapping channel name to threshold value
            (absolute, not percentage). Pixels below threshold are not composited
            onto the transmitted image. Only used when transmitted_image is provided.
        dtype: Output array dtype (np.uint8 or np.uint16).
        max_value: Maximum pixel value for this dtype (255 for 8-bit, 4095 for 12-bit).

    Returns:
        3-channel RGB numpy array of shape (H, W, 3).
    """
    if brightness_thresholds is None:
        brightness_thresholds = {}

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
        # No transmitted channel — assign each channel directly
        img = np.zeros((h, w, 3), dtype=dtype)
        for channel_name, img_gray in channel_images.items():
            channel_index = CHANNEL_RGB_INDEX.get(channel_name)
            if channel_index is None:
                continue
            img[:, :, channel_index] = img_gray

    return img
