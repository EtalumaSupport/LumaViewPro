# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import cv2
import numpy as np
from kivy.graphics.texture import Texture

import modules.image_utils as image_utils


def image_to_texture(image, existing: Texture | None = None) -> Texture:
    """Convert an 8-bit numpy image to a Kivy Texture.

    If ``existing`` is provided and its size matches, blit into it and
    return it (avoids allocating a new GDI texture). Otherwise allocate
    a new Texture. Callers in tight UI loops (e.g. cell-count slider
    scrub) should pass their current widget texture to avoid leaking
    a GDI handle per frame.

    The input must already be 8-bit: the blit copies raw bytes as ubyte,
    so a wider payload would be read back as garbage. Downconvert with
    convert_to_8bit(image, significant_bits) at the depth-aware caller.
    """
    if image.dtype != np.uint8:
        raise ValueError(
            f'image_to_texture needs an 8-bit display array; got {image.dtype}. '
            'Downconvert with convert_to_8bit(image, significant_bits) first -- '
            'blitting a wider payload as ubyte reads back as garbage.'
        )
    # Vertical flip
    image = cv2.flip(image, 0)

    if image.ndim == 3 and image.shape[2] == 4:
        # TIFF data is RGB-native. Kivy's reusable display texture is BGR,
        # so discard alpha while converting the RGB components to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif not image_utils.is_color_image(image=image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    buf = image.tobytes()
    size = (image.shape[1], image.shape[0])

    if existing is not None and existing.size == size:
        existing.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return existing

    image_texture = Texture.create(size=size, colorfmt='bgr')
    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return image_texture


def image_file_to_texture(image_file) -> Texture:
    image = image_utils.image_file_to_image(image_file=image_file)

    if image is None:
        return None

    texture = image_to_texture(image=image)
    return texture
