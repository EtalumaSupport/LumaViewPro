
import cv2
import numpy as np

from kivy.graphics.texture import Texture

from lvp_logger import logger

def image_file_to_image(image_file):
    logger.info(f'[LVP image_utils  ] Loading: {image_file}')
    if not cv2.haveImageReader(image_file):
        logger.error(f'[LVP image_utils  ] - Image not supported by OpenCV')
        return

    num_images = cv2.imcount(image_file)
    logger.info(f'[LVP image_utils  ] - {num_images} images detected')

    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    if image is None:
        logger.error(f'[LVP image_utils  ] - Unable to load file')
        return

    return image


def image_to_texture(image) -> Texture:
    # Vertical flip
    image = cv2.flip(image, 0)

    buf = image.tostring()

    image_texture = Texture.create(
        size=(image.shape[1], image.shape[0]), colorfmt='bgr')

    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    return image_texture


def image_file_to_texture(image_file) -> Texture:
    image = image_file_to_image(image_file=image_file)
    
    if image is None:
        return None

    texture = image_to_texture(image=image)
    return texture


def rgb_image_to_gray(image):

    def _is_grayscale(image):
        shape = image.shape
        if (len(shape) <= 2) or (shape[2] == 1):
            return True

        return False
    
    def _values_in_one_plane(image):
        count = 0
        for color_plane_idx in range(image.shape[2]):
            image_view = image[:,:,color_plane_idx]
            if np.any(image_view):
                count += 1

        if count <= 1:
            return True
        else:
            return False

    if _is_grayscale(image=image):
        return image

    if _values_in_one_plane(image=image):
        return np.amax(image, axis=2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
