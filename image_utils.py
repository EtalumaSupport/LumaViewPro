import pathlib

import cv2
import numpy as np
import tifffile as tf

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


def convert_12bit_to_8bit(image):
    if image.dtype == 'uint8':
        return image
    
    new_image = image.copy()
    return (new_image // 16).astype(np.uint8)


def convert_12bit_to_16bit(image):
    if image.dtype == 'uint8':
        return image
    
    new_image = image.copy()
    return (new_image * 16)


def write_ome_tiff(data, file_loc: pathlib.Path, channel: str):
    pixel_size=0.29
    # pixel_size = common_utils.get_pixel_size(focal_length= )

    with tf.TiffWriter(str(file_loc), bigtiff=False) as tif:
        metadata={
            'axes': 'YXS',
            'SignificantBits': data.itemsize*8,
            'PhysicalSizeX': pixel_size,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixel_size,
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': [channel]},
            'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
        }

        options=dict(
            photometric='rgb',
            tile=(128, 128),
            compression='jpeg',
            resolutionunit='CENTIMETER',
            maxworkers=2
        )

        tif.write(
            data,
            resolution=(1e4 / pixel_size, 1e4 / pixel_size),
            metadata=metadata,
            **options
        )
