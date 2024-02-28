
import cv2
from kivy.graphics.texture import Texture

import image_utils

from lvp_logger import logger


def image_to_texture(image) -> Texture:
    # Vertical flip
    image = cv2.flip(image, 0)

    buf = image.tostring()

    image_texture = Texture.create(
        size=(image.shape[1], image.shape[0]), colorfmt='bgr')

    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    return image_texture


def image_file_to_texture(image_file) -> Texture:
    image = image_utils.image_file_to_image(image_file=image_file)
    
    if image is None:
        return None

    texture = image_to_texture(image=image)
    return texture
