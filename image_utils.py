import pathlib

import cv2
import numpy as np
import tifffile as tf

import modules.common_utils as common_utils
import image_utils

from lvp_logger import logger


def is_color_image(image) -> bool:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return True
    
    return False


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


def write_ome_tiff(
    data,
    file_loc: pathlib.Path,
    channel: str,
    focal_length: float,
    plate_pos_mm: dict[str, float],
    z_pos_um: float,
    exposure_time_ms: float,
    gain_db: float,
    ill_ma: float
):
    pixel_size = round(common_utils.get_pixel_size(focal_length=focal_length), common_utils.max_decimal_precision('pixel_size'))

    use_color = image_utils.is_color_image(data)

    if use_color:
        photometric = 'rgb'
        axes = 'YXS'
    else:
        photometric = 'minisblack'
        axes = 'YX'

    with tf.TiffWriter(str(file_loc), bigtiff=False) as tif:
        metadata={
            'axes': axes,
            'SignificantBits': data.itemsize*8,
            'PhysicalSizeX': pixel_size,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixel_size,
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': [channel]},
            'Plane': {
                'PositionX': plate_pos_mm['x'],
                'PositionY': plate_pos_mm['y'],
                'PositionZ': z_pos_um,
                'PositionXUnit': 'mm',
                'PositionYUnit': 'mm',
                'PositionZUnit': 'um',
                'ExposureTime': exposure_time_ms,
                'ExposureTimeUnit': 'ms',
                'Gain': gain_db,
                'GainUnit': 'dB',
                'Illumination': ill_ma,
                'IlluminationUnit': 'mA'
            }
        }

        options=dict(
            photometric=photometric,
            tile=(128, 128),
            compression='lzw',
            resolutionunit='CENTIMETER',
            maxworkers=2
        )

        tif.write(
            data,
            resolution=(1e4 / pixel_size, 1e4 / pixel_size),
            metadata=metadata,
            **options
        )
