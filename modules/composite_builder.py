# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Shared composite image builder.

Merges multiple microscope channels (transmitted, fluorescence, luminescence)
into a single 3-channel RGB composite image. Used by both the live composite
capture path and the post-capture composite generation path.
"""

import math
from typing import NamedTuple

import numpy as np

import modules.image_utils as image_utils
from modules.layer_record import Color

# Largest value an 8-bit pixel can take. The composite's blend operates on the
# 8-bit output scale (the inputs are downconverted below before any threshold
# is consulted), so this is the scale a brightness percentage maps onto.
_MAX_8BIT_VALUE = 255


class CompositeChannel(NamedTuple):
    """One channel's input to a composite: its image and the colour it is
    shown in, carried together so no channel reaches the build without one."""

    image: np.ndarray
    color: Color


def brightness_cutoff_from_percent(pct: float) -> int:
    """The 8-bit pixel value a channel must reach to blend, from a percentage.

    Lives here because this module owns the 8-bit output scale the cutoff is
    expressed in; a caller computing its own mapping would be a second copy of
    that ownership, free to drift from the comparison it feeds.

    The result is an integer and is compared with ``>=``. A fractional cutoff
    compared with ``>`` cannot express the top of the scale -- 100 percent of
    255 is 255, which no 8-bit pixel exceeds, so the whole channel silently
    dropped out of the composite at the highest setting the UI offers.

    The floor of 1 is what keeps 0 percent meaning "any pixel carrying signal"
    rather than "every pixel, including true black": a cutoff of 0 with ``>=``
    would admit black and erase the transmitted base frame-wide.

    Args:
        pct: Brightness threshold as a percentage of full scale, 0 to 100.
            The caller is responsible for the range; see the refusal in
            ``CompositeGeneration._create_composite_image``.

    Returns:
        int: Minimum 8-bit pixel value that blends, in 1..255.
    """
    return max(1, math.ceil(pct * _MAX_8BIT_VALUE / 100))


def build_composite(
    channels: dict[str, CompositeChannel],
    significant_bits: int,
    transmitted_image: np.ndarray | None = None,
    brightness_thresholds: dict | None = None,
) -> np.ndarray:
    """Build an 8-bit-per-channel RGB composite from grayscale channel images.

    A composite is a viewing product: it merges several channels' intensities
    into one colour image for display, and once merged the per-channel
    values are no longer separable for quantitative use. Its output is therefore
    always 8-bit RGB -- the form every viewer and monitor renders directly.
    Storing a composite at the camera's native 12-bit depth (right-aligned in a
    16-bit container) leaves the brightest pixel at ~6% of full scale, so any
    viewer that ignores the depth tag shows it near-black. 8-bit sidesteps that
    entirely; the raw single-channel captures keep their full depth elsewhere.

    Downconversion is owned here, at the one place both the live-capture and the
    post-processing orchestrators funnel through, so neither can reintroduce a
    depth-carrying composite.

    The channels add: each contributes its intensity times its colour, summed
    in a wider accumulator and clipped to 8 bits, so two channels that share
    a colour add instead of the later one overwriting the earlier.

    Args:
        channels: Dict mapping channel name to its ``CompositeChannel``: a 2D
            grayscale array at the capture's native depth and its colour.
        significant_bits: The meaningful bit depth of the input channels (12 for a
            Mono12 capture, 8 for 8-bit); drives the downconvert scale. Ignored for
            inputs already 8-bit.
        transmitted_image: Optional 2D grayscale array for transmitted channel
            (BF/PC/DF), at the same native depth. Its grey is the base; a pixel
            where any channel reaches its cutoff is that pixel's sum of the
            channels that reach it, in place of the base.
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
    labeled = [(name, channel.image) for name, channel in channels.items()]
    if transmitted_image is not None:
        labeled.append(('transmitted', transmitted_image))
    image_utils.require_uniform_geometry(labeled, operation='composite this tile-group')

    # Downconvert every input to the 8-bit output scale up front, so the blend
    # below is a single dtype and the thresholds compare against 8-bit values.
    channel_images = {
        name: image_utils.convert_to_8bit(channel.image, significant_bits)
        for name, channel in channels.items()
    }
    if transmitted_image is not None:
        transmitted_image = image_utils.convert_to_8bit(transmitted_image, significant_bits)

    # Determine image dimensions from first available image
    if transmitted_image is not None:
        h, w = transmitted_image.shape[:2]
    else:
        first_img = next(iter(channel_images.values()))
        h, w = first_img.shape[:2]

    # 16 bits hold 257 channels at full scale in one plane; a composite takes
    # one image per channel key, and a scope lights at most eight. Narrower
    # than 32 bits because a stitched composite can be hundreds of megapixels.
    total = np.zeros((h, w, 3), dtype=np.uint16)
    colored = np.empty((h, w, 3), dtype=np.uint8)
    reached_any = np.zeros((h, w), dtype=bool)
    for name, img_gray in channel_images.items():
        if transmitted_image is not None:
            # Inclusive, against the integer cutoff brightness_cutoff_from_percent
            # produces: the cutoff is the lowest value that blends, so the top of
            # the scale stays expressible. An exclusive test cannot admit 255.
            reached = img_gray >= brightness_thresholds.get(name, 1)
            reached_any |= reached
            img_gray = img_gray * reached
        total += image_utils.mono_to_rgb_falsecolor(img_gray, channels[name].color, colored)

    np.minimum(total, _MAX_8BIT_VALUE, out=total)
    img = total.astype(np.uint8)
    if transmitted_image is not None:
        img[~reached_any] = transmitted_image[~reached_any, None]
    return img
