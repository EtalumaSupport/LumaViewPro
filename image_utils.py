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


def add_scale_bar(image, objective: dict):
    height, width = image.shape[0], image.shape[1]

    dtype = image.dtype
    is_color = is_color_image(image=image)

    scale_bar_length = min(100, int(width/10))
    scale_bar_thickness = min(3, int(height/300))
    scale_bar_bottom_offset = int(height/40)
    scale_bar_right_offset = int(width/40)

    if dtype == np.uint8:
        scale_bar_value = 2**8-1
    else: # 12-bit
        scale_bar_value = 2**12-1

    x_end = width - scale_bar_right_offset
    x_start = x_end - scale_bar_length
    y_start = scale_bar_bottom_offset
    y_end = y_start + scale_bar_thickness

    if is_color:
        image[y_start:y_end+1,x_start:x_end+1,:] = scale_bar_value
    else:
        image[y_start:y_end+1,x_start:x_end+1] = scale_bar_value

    pixel_size_um = common_utils.get_pixel_size(focal_length=objective['focal_length'])
    scale_bar_length_um = round(scale_bar_length * pixel_size_um)
    text_x_pos = x_start
    text_y_pos = y_end + 5
    font_scale = max(0.4, width/4000)
    cv2.putText(
        img=image, 
        text=f"{scale_bar_length_um}um, {objective['magnification']}x",
        org=(text_x_pos, text_y_pos),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(scale_bar_value,scale_bar_value,scale_bar_value),
        thickness=1,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=True
    )

    return image


def add_timestamp(image, timestamp_str: str):
    height, width = image.shape[0], image.shape[1]

    dtype = image.dtype

    text_color_bg = (0,0,0)
    font_scale = max(0.75, width/2000)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    text_size, _ = cv2.getTextSize(
        text=timestamp_str,
        fontFace=font_face,
        fontScale=font_scale,
        thickness=font_thickness
    )
    text_w, text_h = text_size

    bottom_offset = int(height/40)
    left_offset = int(width/40)

    top_offset = height - bottom_offset

    if dtype == np.uint8:
        text_intensity = 2**8-1
    else: # 16-bit
        text_intensity = 2**16-1

    cv2.rectangle(
        image,
        (left_offset, top_offset),
        (left_offset+text_w, top_offset+text_h),
        text_color_bg,
        -1
    )

    cv2.putText(
        img=image, 
        text=f"{timestamp_str}",
        org=(left_offset, int(top_offset + text_h + font_scale - 1)),
        fontFace=font_face,
        fontScale=font_scale,
        color=(text_intensity,text_intensity,text_intensity),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=False
    )

    return image
