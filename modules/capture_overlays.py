# Copyright Etaluma, Inc.
"""The overlays a still can be saved with: the bullseye colour map and the
crosshairs.

They live here, below the GUI, because the saved overlay copy is part of
what a capture produces: a script or a REST caller asking for it gets the
same pixels the Capture button does. The live view draws its own bullseye
into a reused buffer (``ScopeDisplay.transform_to_bullseye_prealloc``) from
the same table, so the screen and the file cannot disagree about the map.
"""

import numpy as np
import skimage.draw

# Ten-level-wide bands: green every twenty levels, blue at mid-scale and
# red at the top, so focus and saturation read off the image at a glance.
# (start_exclusive, end_inclusive, R, G, B)
_BULLSEYE_BANDS = (
    (5, 15, 0, 255, 0),
    (25, 35, 0, 255, 0),
    (45, 55, 0, 255, 0),
    (65, 75, 0, 255, 0),
    (85, 95, 0, 255, 0),
    (105, 115, 0, 255, 0),
    (125, 135, 0, 0, 255),
    (145, 155, 0, 255, 0),
    (165, 175, 0, 255, 0),
    (185, 195, 0, 255, 0),
    (205, 215, 0, 255, 0),
    (225, 235, 0, 255, 0),
    (245, 255, 255, 0, 0),
)


def _build_bullseye_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for start, end, r, g, b in _BULLSEYE_BANDS:
        lut[start + 1 : end + 1] = [r, g, b]
    return lut


# Built once at import: every live frame indexes it, and a table that is
# never rebuilt cannot be caught half-written by the display thread.
BULLSEYE_LUT = _build_bullseye_lut()


def transform_to_bullseye(image: np.ndarray) -> np.ndarray:
    """The 8-bit mono ``image`` mapped through the bullseye table, as RGB."""
    return BULLSEYE_LUT[image]


def add_crosshairs(image: np.ndarray) -> np.ndarray:
    """A copy of ``image`` with centre crosshairs and four radiating circles.

    Returns a copy: when no bullseye is drawn first, the input is the raw
    capture itself, and drawing into it would put the overlay into the
    unmarked file as well if the two saves were ever reordered.
    """
    image = image.copy()
    height, width = image.shape[0], image.shape[1]
    center_x = round(width / 2)
    center_y = round(height / 2)

    # 2 pixels wide; the trailing axis, when present, is colour.
    image[:, center_x - 1 : center_x + 1, ...] = 255
    image[center_y - 1 : center_y + 1, :, ...] = 255

    num_circles = 4
    circle_spacing = round(min(height, width) / 2 / num_circles)
    for i in range(num_circles):
        radius = (i + 1) * circle_spacing
        # Two adjacent perimeters make each circle 2 pixels wide.
        for r in (radius, radius + 1):
            rr, cc = skimage.draw.circle_perimeter(center_y, center_x, radius=r, shape=image.shape)
            image[rr, cc] = 255

    return image
